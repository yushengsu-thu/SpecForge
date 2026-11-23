# coding=utf-8
"""Top-k routers for draft MoE layers: selection, the selection bias, the counts, the metrics."""

from typing import Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from specforge.modeling.draft.moe.balance import (
    count_tokens_per_expert,
    load_balance_loss,
    router_metrics,
    router_z_loss,
)

ScoreFn = Callable[[torch.Tensor], torch.Tensor]


def softmax_scores(logits: torch.Tensor) -> torch.Tensor:
    """``softmax(logits)`` over experts: the default router scores."""
    return logits.softmax(dim=-1)


class TopKRouter(nn.Module):
    """Top-k router; ``weight`` [E, H] matches Qwen3.5-MoE's ``gate``."""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        top_k: int,
        norm_topk_prob: bool,
        initializer_range: float,
        score_fn: ScoreFn = softmax_scores,
        selection_bias: bool = False,
        route_scale: float = 1.0,
        aux_loss_coef: float = 0.0,
        aux_loss_type: str = "microbatch_wise",
        bias_update_rate: float = 0.001,
        z_loss_coef: float = 0.0,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                f"top_k={top_k} must be between 1 and num_experts={num_experts}"
            )
        if aux_loss_type not in ("microbatch_wise", "batch_wise"):
            raise ValueError(f"unknown aux_loss_type {aux_loss_type!r}")
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.score_fn = score_fn
        # DeepSeek's selection bias: shifts the top-k picks, not the weights; fp32.
        self.register_buffer(
            "selection_bias",
            torch.zeros(num_experts, dtype=torch.float32) if selection_bias else None,
        )
        # The step's kept-slot counts: bias update, step balance.
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )
        # The same counts by the router's own top-k, without the selection bias:
        # the load the aux loss balances. DeepSeek-V3 routes with the bias
        # (eq. 16) but takes the balance loss's f_i from the unbiased scores
        # (eq. 18); on the biased picks the loss is flat as soon as the bias has
        # balanced them, however skewed the router is. Equal to tokens_per_expert
        # without a bias.
        self.register_buffer(
            "own_tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.route_scale = route_scale
        self.aux_loss_coef = aux_loss_coef
        self.aux_loss_type = aux_loss_type
        self.bias_update_rate = bias_update_rate
        # ST-MoE's router z-loss, logsumexp(logits)^2: bounds the logits so the
        # softmax cannot saturate (P -> 1 kills the router's gradient, and its own
        # gradient 2 * lse * P stays alive there). 0 keeps it as a metric only.
        self.z_loss_coef = z_loss_coef
        # This micro-batch's load-balance (aux) loss, kept in training for the trainer.
        self.aux_loss: Optional[torch.Tensor] = None
        self.z_loss: Optional[torch.Tensor] = None
        # This micro-batch's router health metrics (detached), kept in training.
        self.metrics: Optional[Dict[str, torch.Tensor]] = None
        self.reset_parameters(initializer_range)

    def _apply(self, fn, recurse=True):
        # Keep the buffers fp32, values included, through casts like .to(bfloat16).
        names = ("selection_bias", "tokens_per_expert", "own_tokens_per_expert")
        kept = {name: getattr(self, name) for name in names}
        super()._apply(fn, recurse)
        for name, old in kept.items():
            if old is not None:
                new = getattr(self, name)
                setattr(self, name, new.float() if old.is_meta else old.to(new.device))
        return self

    def reset_parameters(self, std: float) -> None:
        # HF's default _init_weights skips bare Parameters and buffers.
        nn.init.normal_(self.weight, std=std)
        for buffer in (
            self.selection_bias,
            self.tokens_per_expert,
            self.own_tokens_per_expert,
        ):
            if buffer is not None:
                nn.init.zeros_(buffer)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x`` [T, H] -> fp32 scores [T, E], indices [T, K], fp32 weights [T, K]."""
        self.aux_loss = None  # per-forward state: never carry a stale loss into eval
        self.metrics = None
        self.z_loss = None
        # Logits in x's dtype as in Qwen; DeepSeek-V4 projects in fp32.
        logits = F.linear(x, self.weight).float()
        scores = self.score_fn(logits)
        choice = scores if self.selection_bias is None else scores + self.selection_bias
        indices = choice.topk(self.top_k, dim=-1).indices
        weights = scores.gather(-1, indices)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        if self.training:
            counts = count_tokens_per_expert(indices, mask, self.num_experts)
            self.tokens_per_expert += counts
            # The aux loss sees the router's own picks, not the biased ones.
            if self.selection_bias is None:
                own_counts = counts
            else:
                own = scores.topk(self.top_k, dim=-1).indices
                own_counts = count_tokens_per_expert(own, mask, self.num_experts)
            self.own_tokens_per_expert += own_counts
            # Always kept in training: with coefficient 0 it is the router's own
            # balance metric (moe/step_balance scores the biased dispatch).
            self.aux_loss = load_balance_loss(
                scores, self._aux_loss_load(own_counts), mask
            )
            self.z_loss = router_z_loss(logits, mask)
            self.metrics = router_metrics(
                logits, scores, indices, mask, self.selection_bias
            )
        return scores, indices, weights * self.route_scale

    def _aux_loss_load(self, own_counts: torch.Tensor) -> torch.Tensor:
        """The load the aux loss balances: this micro-batch's own counts, or the step's."""
        if self.aux_loss_type != "batch_wise":
            return own_counts
        # Batch-wise: the step's counts so far, pooled over ranks; the loss uses shares.
        load = self.own_tokens_per_expert.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(load)
        return load

    @torch.no_grad()
    def update_expert_bias(self, tokens_per_expert: torch.Tensor) -> None:
        """One DeepSeek-V3 aux-loss-free step from the pooled kept-slot counts."""
        gap = tokens_per_expert.mean() - tokens_per_expert
        delta = self.bias_update_rate * torch.sign(gap)
        # Zero-mean delta: the same top-k picks, and the bias never drifts.
        self.selection_bias += delta - delta.mean()
