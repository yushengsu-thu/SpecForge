# coding=utf-8
"""MoE load balancing: the loss math, the router health metrics, and the hooks the DFlash-family model calls."""

from typing import Dict, List, Optional, Protocol, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def count_tokens_per_expert(
    indices: torch.Tensor, mask: Optional[torch.Tensor], num_experts: int
) -> torch.Tensor:
    """Kept slots per expert, fp32 [E]; ``mask`` [T] marks the kept tokens."""
    weights = None
    if mask is not None:
        weights = mask.to(torch.float32).repeat_interleave(indices.shape[-1])
    counts = torch.bincount(indices.reshape(-1), weights=weights, minlength=num_experts)
    return counts.float()


def load_balance_loss(
    scores: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``E * sum_e share_e * P_e`` over the kept tokens; ``1`` when balanced."""
    # The MoE "aux loss" in torchtitan's, Megatron's and DeepSeek-V3's units:
    # share_e is expert e's part of the kept slots (K per token), so f = E * share.
    # Per-token L1 normalization: a no-op for softmax, needed for sigmoid scores.
    probs = F.normalize(scores, p=1, dim=-1)
    kept = probs.new_ones(probs.shape[0]) if mask is None else mask.to(probs.dtype)
    mean_prob = (probs * kept.unsqueeze(-1)).sum(dim=0) / kept.sum().clamp(min=1)
    share = tokens_per_expert / tokens_per_expert.sum().clamp(min=1)
    return probs.shape[-1] * (share * mean_prob).sum()


def router_z_loss(
    logits: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """ST-MoE's router z-loss: ``mean_t logsumexp(logits_t) ** 2`` over the kept tokens."""
    # Penalizes large router logits so the softmax never saturates; ST-MoE and
    # OLMoE weight it by 1e-3. Unlike the main and aux losses its gradient does not
    # vanish at saturation, so it also pulls a collapsed router back.
    z = torch.logsumexp(logits.float(), dim=-1).square()
    if mask is None:
        return z.mean()
    kept = mask.to(z.dtype)
    return (z * kept).sum() / kept.sum().clamp(min=1)


def dispatch_balance(tokens_per_expert: torch.Tensor) -> torch.Tensor:
    """A dispatch's balance against itself: 1 when uniform, E / K when collapsed."""
    # load_balance_loss with the load standing in for the probabilities.
    share = tokens_per_expert / tokens_per_expert.sum().clamp(min=1)
    return share.shape[-1] * share.square().sum()


def dispatch_maxvio(tokens_per_expert: torch.Tensor) -> torch.Tensor:
    """DeepSeek's MaxVio: the busiest expert's load over the balanced load, minus 1."""
    # 0 when uniform, E - 1 when every slot went to one expert; 0 for an empty step.
    share = tokens_per_expert / tokens_per_expert.sum().clamp(min=1)
    return (share.shape[-1] * share.max() - 1).clamp(min=0)


def dead_experts(tokens_per_expert: torch.Tensor, fraction: float = 0.25) -> torch.Tensor:
    """How many experts got under ``fraction`` of their balanced share of the slots."""
    share = tokens_per_expert / tokens_per_expert.sum().clamp(min=1)
    return (share < fraction / share.shape[-1]).sum().to(share.dtype)


@torch.no_grad()
def router_metrics(
    logits: torch.Tensor,
    scores: torch.Tensor,
    indices: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    selection_bias: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Router health over the kept tokens, detached: how peaked the scores are and
    how large the logits grew, independently of where the dispatch sent the tokens.

    ``top1_prob``: mean top score (1/E uniform, 1 saturated). ``prob_balance``:
    ``E * sum_e share_e^2`` with the mean score shares standing in for the dispatch
    (1 uniform, E collapsed) -- what the aux loss cannot see once the selection
    bias has evened out the counts. ``logit_gap``: mean top-1 minus top-2 logit.
    ``logit_absmax``: the largest |logit| of a kept token. ``bias_flip_frac``: the
    kept tokens whose top-k the selection bias moved off the scores' top-k.
    ``entropy``: mean per-token entropy of the scores (ln E uniform, 0 saturated).
    ``logsumexp_sq``: mean logsumexp(logits)^2 (ST-MoE's z-loss as a metric: the
    logit scale). ``top1_logit``: mean top logit. ``mean_prob/e{k}``: expert k's
    mean score, so the dominant expert can be named.
    """
    logits = logits.detach().float()
    scores = scores.detach().float()
    num_experts, top_k = scores.shape[-1], indices.shape[-1]
    kept = scores.new_ones(scores.shape[0]) if mask is None else mask.to(scores.dtype)
    n = kept.sum().clamp(min=1)
    k2 = min(2, num_experts)
    top_scores = scores.topk(k2, dim=-1).values
    top_logits = logits.topk(k2, dim=-1).values
    mean_score = (scores * kept.unsqueeze(-1)).sum(dim=0) / n
    share = mean_score / mean_score.sum().clamp(min=1e-12)
    gap = top_logits[:, 0] - top_logits[:, -1] if k2 == 2 else torch.zeros_like(kept)
    if selection_bias is None:
        moved = torch.zeros_like(kept)
    else:
        unbiased = scores.topk(top_k, dim=-1).indices.sort(dim=-1).values
        picked = indices.sort(dim=-1).values
        moved = (unbiased != picked).any(dim=-1).to(scores.dtype)
    entropy = -(scores * scores.clamp_min(1e-12).log()).sum(dim=-1)
    out = {
        "top1_prob": (top_scores[:, 0] * kept).sum() / n,
        "prob_balance": num_experts * share.square().sum(),
        "logit_gap": (gap * kept).sum() / n,
        "logit_absmax": (logits.abs().amax(dim=-1) * kept).amax(),
        "bias_flip_frac": (moved * kept).sum() / n,
        "entropy": (entropy * kept).sum() / n,
        "logsumexp_sq": (torch.logsumexp(logits, dim=-1).square() * kept).sum() / n,
        "top1_logit": (top_logits[:, 0] * kept).sum() / n,
    }
    for expert in range(num_experts):
        out[f"mean_prob/e{expert}"] = mean_score[expert]
    return out


# Hooks the DFlash-family model calls; routers and layers are found by their state.


class _RouterLike(Protocol):
    tokens_per_expert: torch.Tensor
    own_tokens_per_expert: torch.Tensor
    selection_bias: Optional[torch.Tensor]
    aux_loss: Optional[torch.Tensor]
    aux_loss_coef: float
    metrics: Optional[Dict[str, torch.Tensor]]
    z_loss: Optional[torch.Tensor]
    z_loss_coef: float

    def update_expert_bias(self, tokens_per_expert: torch.Tensor) -> None: ...


class _MoELayerLike(Protocol):
    gate: _RouterLike
    token_mask: Optional[torch.Tensor]


def set_moe_token_mask(model: nn.Module, mask: Optional[torch.Tensor]) -> None:
    """Hand every MoE layer this batch's ``[B, S]`` kept-slot mask."""
    for layer in _moe_layers(model):
        layer.token_mask = mask


def take_moe_aux_loss(
    model: nn.Module,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Pop this forward's load-balance loss: (weighted sum over layers, raw mean)."""
    # Weighted like its objective: by denominator under loss_terms, equally otherwise.
    layers = _moe_layers(model)
    terms = [
        (layer.gate.aux_loss_coef, layer.gate.aux_loss)
        for layer in layers
        if layer.gate.aux_loss is not None
    ]
    for layer in layers:
        layer.gate.aux_loss = None
        layer.token_mask = None
    if not terms:
        return None
    # One coefficient per layer, as in torchtitan and Megatron; the metric stays a mean.
    aux = torch.stack([coef * loss for coef, loss in terms]).sum()
    raw = torch.stack([loss for _, loss in terms]).mean()
    return aux, raw


def take_moe_router_metrics(model: nn.Module) -> Optional[Dict[str, torch.Tensor]]:
    """Pop this forward's router metrics: each under ``moe/`` as the mean over the
    MoE layers, and per layer under ``moe/layer{i}/``."""
    routers = _routers(model)
    recorded = [
        (index, router.metrics)
        for index, router in enumerate(routers)
        if getattr(router, "metrics", None) is not None
    ]
    for router in routers:
        router.metrics = None
    if not recorded:
        return None
    out: Dict[str, torch.Tensor] = {}
    for name in recorded[0][1]:
        out[f"moe/{name}"] = torch.stack([m[name] for _, m in recorded]).mean()
        for index, m in recorded:
            out[f"moe/layer{index}/{name}"] = m[name]
    return out


def take_moe_z_loss(
    model: nn.Module,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Pop this forward's router z-loss: (weighted sum over layers, raw mean)."""
    layers = _moe_layers(model)
    terms = [
        (layer.gate.z_loss_coef, layer.gate.z_loss)
        for layer in layers
        if getattr(layer.gate, "z_loss", None) is not None
    ]
    for layer in layers:
        layer.gate.z_loss = None
    if not terms:
        return None
    z = torch.stack([coef * loss for coef, loss in terms]).sum()
    raw = torch.stack([loss for _, loss in terms]).mean()
    return z, raw


def step_moe_balancing(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Optimizer step boundary: pool counts once, move the biases, score the step."""
    routers = _routers(model)
    if not routers:
        return {}
    # One collective for every layer; all ranks ran the same forwards, so it lines up.
    counts = torch.stack([router.tokens_per_expert for router in routers])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts)
    for router, pooled in zip(routers, counts):
        if router.selection_bias is not None:
            router.update_expert_bias(pooled)
        router.tokens_per_expert.zero_()  # the next step's count window
        router.own_tokens_per_expert.zero_()
    # The step's pooled load: the collapse signal, unlike the per-micro-batch loss.
    per_layer = {
        "step_balance": [dispatch_balance(c) for c in counts],
        "maxvio": [dispatch_maxvio(c) for c in counts],
        "dead_experts": [dead_experts(c) for c in counts],
        # The biases after this step's move: how much they are compensating.
        "bias_range": [
            (
                router.selection_bias.max() - router.selection_bias.min()
                if router.selection_bias is not None
                else counts.new_zeros(())
            )
            for router in routers
        ],
    }
    out: Dict[str, torch.Tensor] = {}
    for name, values in per_layer.items():
        out[f"moe/{name}"] = torch.stack(values).mean()
        for index, value in enumerate(values):
            out[f"moe/layer{index}/{name}"] = value
    # Per expert and layer: the step's dispatch share and the bias, to name the
    # expert that took the tokens.
    for index, (router, pooled) in enumerate(zip(routers, counts)):
        share = pooled / pooled.sum().clamp(min=1)
        for expert in range(share.shape[-1]):
            out[f"moe/layer{index}/share/e{expert}"] = share[expert]
            if router.selection_bias is not None:
                out[f"moe/layer{index}/bias/e{expert}"] = router.selection_bias[expert]
    return out


def _routers(model: nn.Module) -> List[_RouterLike]:
    return [m for m in model.modules() if hasattr(m, "tokens_per_expert")]


def _moe_layers(model: nn.Module) -> List[_MoELayerLike]:
    return [m for m in model.modules() if hasattr(m, "token_mask")]
