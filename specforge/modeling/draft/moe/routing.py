"""Routing and differentiable Qwen auxiliary balancing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import MoEConfig


@dataclass
class RoutingResult:
    """Expert IDs and weights are [tokens, top_k]; probabilities cover all experts."""

    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    probabilities: torch.Tensor


class TopKRouter(nn.Module):
    """Keep router and shared-expert gates distinct in both math and state names."""

    def __init__(self, config: MoEConfig, hidden_size: int):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> RoutingResult:
        # Match Qwen: projection follows model precision, softmax is always FP32.
        logits = F.linear(hidden_states, self.weight)
        probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
        weights, indices = probabilities.topk(self.config.num_experts_per_tok, dim=-1)
        if self.config.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.config.routed_scaling_factor
        return RoutingResult(indices, weights.to(hidden_states.dtype), probabilities)


def auxiliary_load_balance(
    routing: RoutingResult,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Qwen/Switch loss using local forward tokens, without distributed side effects.

    Counts include every selected slot, matching Transformers' top-k convention:
    the uniform-routing baseline is k. Parameter gradients are synchronized by
    the training backend, not by the router. Padding must be removed by callers
    that pass padded sequences; DFlash's query tensors are the routed tokens.
    """
    probabilities = routing.probabilities
    num_tokens, num_experts = probabilities.shape
    counts = torch.bincount(
        routing.expert_indices.reshape(-1), minlength=num_experts
    ).float()
    tokens_per_expert = counts / max(num_tokens, 1)
    mean_probability = probabilities.sum(dim=0) / max(num_tokens, 1)
    loss = num_experts * (tokens_per_expert * mean_probability).sum()
    fractions = counts / max(routing.expert_indices.numel(), 1)
    metrics = {
        "router_entropy": -(
            probabilities.detach() * probabilities.detach().clamp_min(1e-20).log()
        )
        .sum(-1)
        .mean(),
        "expert_load_max": fractions.max(),
        "expert_active_fraction": (counts > 0).float().mean(),
    }
    return loss, metrics
