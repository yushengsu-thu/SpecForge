"""Same-precision fast path for the DSpark target-head objective.

The reference objective (``OnlineDSparkModel._dspark_objective_chunk_terms``)
projects every (anchor, block offset) row through the frozen target ``lm_head``
twice per chunk (the draft hidden and the teacher's aligned target hidden),
upcasts both full-vocab logits to fp32, materializes two softmaxes and an L1
difference, and runs under activation checkpointing, so all of that forward
work is repeated in the backward. For Qwen3-8B (vocab 151,936) that is five
full-vocab GEMMs and roughly 35 full-vocab passes per chunk.

This path keeps the same math and dtypes (bf16 GEMMs with fp32 accumulation,
fp32 probabilities, fp32 L1 and CE) but

* evaluates the teacher distribution once per *sequence position* instead of
  once per anchor row (512 anchors x 16 offsets of a typical sample map onto
  ~1,000 distinct positions, so the teacher GEMM shrinks several times) and
  keeps it in a per-micro-batch table that the chunks gather from;
* evaluates the draft side in one ``autograd.Function`` with an analytic
  backward, ``grad_logits = w * [g_ce (p - onehot) + g_l1 p (s - <s, p>)]``
  with ``s = sign(p - t)``, so nothing is recomputed and only the bf16 logits
  are saved for the backward;
* lets the chunk loop save activations instead of checkpointing.

The per-row CE is ``lse - logit[label]`` in fp32 instead of ``F.cross_entropy``
on the bf16 logits (which rounds each row's loss to bf16); the L1 term and
every gradient path are the same operations as the reference.

``SPECFORGE_DSPARK_FAST_HEAD_COMPILE=1`` runs the elementwise forward statistics
and the analytic logit gradient through ``torch.compile(dynamic=True)``, which
fuses the full-vocab passes (same math; measured 2x faster than the eager fast
path on H200 with the Qwen3-8B head, with lower peak memory).
"""

from __future__ import annotations

import os
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSparkTeacherTable(NamedTuple):
    """Teacher statistics for every (batch, position) of one micro-batch."""

    probs: torch.Tensor  # [batch * seq, vocab] fp32 softmax of the target head
    ids: torch.Tensor  # [batch * seq] argmax token
    top1: torch.Tensor  # [batch * seq] fp32 max probability
    seq_len: int


def build_teacher_table(
    head_weight: torch.Tensor,
    target_last_hidden_states: torch.Tensor,
    *,
    rows_per_chunk: int = 4096,
    max_table_bytes: Optional[int] = None,
) -> Optional[DSparkTeacherTable]:
    """Project every target position through the frozen head once.

    Returns ``None`` when the fp32 table would exceed ``max_table_bytes`` so the
    caller can fall back to the reference per-anchor objective.
    """

    batch_size, seq_len, hidden_size = target_last_hidden_states.shape
    vocab_size = head_weight.shape[0]
    rows = batch_size * seq_len
    if max_table_bytes is not None and rows * vocab_size * 4 > max_table_bytes:
        return None
    flat = target_last_hidden_states.reshape(rows, hidden_size)
    probs = torch.empty(rows, vocab_size, dtype=torch.float32, device=flat.device)
    ids = torch.empty(rows, dtype=torch.long, device=flat.device)
    top1 = torch.empty(rows, dtype=torch.float32, device=flat.device)
    with torch.no_grad():
        for start in range(0, rows, rows_per_chunk):
            stop = min(rows, start + rows_per_chunk)
            logits = F.linear(flat[start:stop], head_weight)
            # Same operations as the reference: softmax of the fp32-upcast bf16
            # logits, argmax of the logits.
            torch.softmax(logits.float(), dim=-1, out=probs[start:stop])
            ids[start:stop] = logits.argmax(dim=-1)
            top1[start:stop] = probs[start:stop].max(dim=-1).values
    return DSparkTeacherTable(probs=probs, ids=ids, top1=top1, seq_len=seq_len)


def _forward_stats_eager(logits, labels, teacher_probs):
    lp = logits.float()
    lse = torch.logsumexp(lp, dim=-1)
    target_logit = lp.gather(1, labels.unsqueeze(1)).squeeze(1)
    ce = lse - target_logit
    probs = lp.sub_(lse.unsqueeze(1)).exp_()  # fp32 probabilities, in place
    p_max, pred = probs.max(dim=-1)
    l1 = probs.sub_(teacher_probs).abs_().sum(dim=-1)  # consumes ``probs``
    return ce, l1, pred, p_max, lse


def _forward_stats_functional(logits, labels, teacher_probs):
    lp = logits.float()
    lse = torch.logsumexp(lp, dim=-1)
    target_logit = lp.gather(1, labels.unsqueeze(1)).squeeze(1)
    ce = lse - target_logit
    probs = torch.exp(lp - lse.unsqueeze(1))
    p_max, pred = probs.max(dim=-1)
    l1 = (probs - teacher_probs).abs().sum(dim=-1)
    return ce, l1, pred, p_max, lse


def _backward_grad_eager(logits, lse, labels, coef_ce, coef_l1, teacher_probs):
    rows = logits.shape[0]
    probs = logits.float().sub_(lse.unsqueeze(1)).exp_()  # recompute p
    # d|p - t| / dp = sign(p - t), with sign(0) = 0 like torch.abs' backward.
    factor = torch.sign(probs - teacher_probs)
    sign_dot_p = (factor * probs).sum(dim=-1)  # <s, p> per row
    # factor <- coef_ce + coef_l1 * (s - <s, p>); grad_logits = p * factor
    factor.sub_(sign_dot_p.unsqueeze(1)).mul_(coef_l1.unsqueeze(1)).add_(
        coef_ce.unsqueeze(1)
    )
    grad_logits = probs.mul_(factor)
    del factor
    grad_logits[torch.arange(rows, device=logits.device), labels] -= coef_ce
    return grad_logits.to(logits.dtype)


def _backward_grad_functional(logits, lse, labels, coef_ce, coef_l1, teacher_probs):
    probs = torch.exp(logits.float() - lse.unsqueeze(1))
    sign = torch.sign(probs - teacher_probs)
    sign_dot_p = (sign * probs).sum(dim=-1)
    grad_logits = probs * (
        coef_ce.unsqueeze(1) + coef_l1.unsqueeze(1) * (sign - sign_dot_p.unsqueeze(1))
    )
    grad_logits = grad_logits.scatter_add(
        1, labels.unsqueeze(1), (-coef_ce).unsqueeze(1).to(grad_logits.dtype)
    )
    return grad_logits.to(logits.dtype)


_COMPILE = os.environ.get("SPECFORGE_DSPARK_FAST_HEAD_COMPILE", "0") == "1"
if _COMPILE:
    _forward_stats = torch.compile(_forward_stats_functional, dynamic=True)
    _backward_grad = torch.compile(_backward_grad_functional, dynamic=True)
else:
    _forward_stats = _forward_stats_eager
    _backward_grad = _backward_grad_eager


class _DSparkHeadLoss(torch.autograd.Function):
    """Draft-side head objective with a fused, recompute-free backward.

    Forward returns the CE and L1 numerators (differentiable) plus detached
    per-row CE, L1, argmax ids and max probabilities for the telemetry.
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,  # [rows, hidden] bf16
        latent: Optional[torch.Tensor],  # [rows, rank] bf16 Markov latents or None
        head_weight: torch.Tensor,  # [vocab, hidden] bf16, frozen
        markov_weight: Optional[torch.Tensor],  # [vocab, rank] bf16 or None
        labels: torch.Tensor,  # [rows] long
        loss_weights: torch.Tensor,  # [rows] fp32
        teacher_probs: torch.Tensor,  # [rows, vocab] fp32
    ):
        logits = F.linear(hidden, head_weight)
        if latent is not None:
            # The reference adds the bf16 Markov bias to the bf16 base logits.
            logits = logits + F.linear(latent, markov_weight)
        ce, l1, pred, p_max, lse = _forward_stats(logits, labels, teacher_probs)
        ce_num = (ce * loss_weights).sum()
        l1_num = (l1 * loss_weights).sum()
        ctx.save_for_backward(logits, lse, labels, loss_weights, teacher_probs)
        ctx.latent = latent
        ctx.head_weight = head_weight
        ctx.markov_weight = markov_weight
        ctx.mark_non_differentiable(ce, l1, pred, p_max)
        return ce_num, l1_num, ce, l1, pred, p_max

    @staticmethod
    def backward(ctx, grad_ce_num, grad_l1_num, *_unused):
        logits, lse, labels, loss_weights, teacher_probs = ctx.saved_tensors
        latent, head_weight, markov_weight = ctx.latent, ctx.head_weight, ctx.markov_weight
        need_hidden, need_latent, _, need_markov = ctx.needs_input_grad[:4]
        if grad_ce_num is None:
            grad_ce_num = lse.new_zeros(())
        if grad_l1_num is None:
            grad_l1_num = lse.new_zeros(())
        coef_ce = grad_ce_num * loss_weights  # [rows]
        coef_l1 = grad_l1_num * loss_weights
        # fp32 logit gradient cast to the bf16 logits' dtype before the linear
        # backward, like the reference.
        grad_logits = _backward_grad(logits, lse, labels, coef_ce, coef_l1, teacher_probs)
        grad_hidden = grad_logits @ head_weight if need_hidden else None
        grad_latent = None
        grad_markov = None
        if latent is not None:
            if need_latent:
                grad_latent = grad_logits @ markov_weight
            if need_markov:
                grad_markov = grad_logits.t() @ latent
        return grad_hidden, grad_latent, None, grad_markov, None, None, None


def dspark_head_loss(
    hidden: torch.Tensor,
    latent: Optional[torch.Tensor],
    head: nn.Linear,
    markov_weight: Optional[torch.Tensor],
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    teacher_probs: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    return _DSparkHeadLoss.apply(
        hidden, latent, head.weight, markov_weight, labels, loss_weights, teacher_probs
    )


def fast_head_supported(model) -> bool:
    """Whether ``model`` (an OnlineDSparkModel) can use the fast objective."""

    head = model.lm_head
    if not isinstance(head, nn.Linear) or head.bias is not None:
        return False
    if head.weight.requires_grad or head.weight.dtype != torch.bfloat16:
        return False
    if getattr(model.draft_model, "confidence_head", None) is not None:
        return False
    markov = getattr(model.draft_model, "markov_head", None)
    if markov is not None and getattr(markov, "markov_head_type", None) != "vanilla":
        return False
    if model.dspark_l1_loss_alpha <= 0:
        return False
    return True


__all__ = [
    "DSparkTeacherTable",
    "build_teacher_table",
    "dspark_head_loss",
    "fast_head_supported",
]
