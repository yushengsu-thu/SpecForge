"""SwiGLU experts and local token dispatch; no expert parallel collectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import MoEConfig
from .routing import RoutingResult


def swiglu(gate_up: torch.Tensor, limit: float | None = None) -> torch.Tensor:
    """Clamp seam for future families; the Qwen preset leaves it disabled."""
    gate, up = gate_up.chunk(2, dim=-1)
    gate = F.silu(gate)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return gate * up


class SharedExpert(nn.Module):
    """One always-active expert in the official Qwen checkpoint layout."""

    def __init__(self, hidden_size: int, intermediate_size: int, limit: float | None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.limit = limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat(
            (self.gate_proj(hidden_states), self.up_proj(hidden_states)), dim=-1
        )
        return self.down_proj(swiglu(gate_up, self.limit))


class RoutedExperts(nn.Module):
    """Fused [E,2I,H]/[E,H,I] parameters are native official Qwen state keys."""

    def __init__(self, config: MoEConfig, hidden_size: int):
        super().__init__()
        self.config = config
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                config.n_routed_experts, 2 * config.moe_intermediate_size, hidden_size
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                config.n_routed_experts, hidden_size, config.moe_intermediate_size
            )
        )

    def forward(
        self, hidden_states: torch.Tensor, routing: RoutingResult
    ) -> torch.Tensor:
        if self.config.dispatch == "grouped_mm":
            return self._grouped_mm(hidden_states, routing)
        return self._reference(hidden_states, routing)

    def _reference(
        self, hidden_states: torch.Tensor, routing: RoutingResult
    ) -> torch.Tensor:
        indices = routing.expert_indices.reshape(-1)
        order = indices.argsort(stable=True)
        counts = torch.bincount(indices, minlength=self.config.n_routed_experts)
        token_indices = torch.arange(
            hidden_states.shape[0], device=hidden_states.device
        ).repeat_interleave(self.config.num_experts_per_tok)[order]
        weights = routing.expert_weights.reshape(-1)[order]
        output = torch.zeros_like(hidden_states)
        offset = 0
        # One CPU synchronization for the readable reference implementation.
        for expert, count in enumerate(counts.tolist()):
            if count:
                tokens = token_indices[offset : offset + count]
                gate_up = F.linear(hidden_states[tokens], self.gate_up_proj[expert])
                values = F.linear(
                    swiglu(gate_up, self.config.swiglu_limit), self.down_proj[expert]
                )
                values = values * weights[offset : offset + count, None]
                output.index_add_(0, tokens, values.to(output.dtype))
            offset += count
        return output

    def _grouped_mm(
        self, hidden_states: torch.Tensor, routing: RoutingResult
    ) -> torch.Tensor:
        grouped_mm = getattr(F, "grouped_mm", None)
        if grouped_mm is None:
            raise RuntimeError(
                "moe_dispatch='grouped_mm' requires torch.nn.functional.grouped_mm"
            )
        if not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16:
            raise RuntimeError(
                "moe_dispatch='grouped_mm' requires CUDA BF16 activations"
            )
        if (
            self.gate_up_proj.dtype != torch.bfloat16
            or self.down_proj.dtype != torch.bfloat16
        ):
            raise RuntimeError(
                "moe_dispatch='grouped_mm' requires BF16 expert parameters"
            )
        num_tokens = hidden_states.shape[0]
        indices = routing.expert_indices.reshape(-1)
        weights = routing.expert_weights.reshape(-1)
        tokens = torch.arange(
            num_tokens, device=hidden_states.device
        ).repeat_interleave(self.config.num_experts_per_tok)
        counts = torch.bincount(indices, minlength=self.config.n_routed_experts)
        # Empty expert groups have caused kernel/backward restrictions in PyTorch.
        # One zero-weight dummy token per empty expert keeps every group nonempty.
        empty_experts = (counts == 0).nonzero().flatten()
        indices = torch.cat((indices, empty_experts))
        tokens = torch.cat((tokens, torch.full_like(empty_experts, num_tokens)))
        weights = torch.cat((weights, weights.new_zeros(empty_experts.numel())))
        order = indices.argsort(stable=True)
        tokens = tokens[order]
        weights = weights[order]
        offsets = (
            torch.bincount(indices, minlength=self.config.n_routed_experts)
            .cumsum(0)
            .to(torch.int32)
        )
        padded = torch.cat(
            (hidden_states, hidden_states.new_zeros(1, hidden_states.shape[-1])), dim=0
        )
        dispatched = padded[tokens]
        gate_up = grouped_mm(
            dispatched, self.gate_up_proj.transpose(-2, -1), offs=offsets
        )
        activations = swiglu(gate_up, self.config.swiglu_limit)
        values = grouped_mm(activations, self.down_proj.transpose(-2, -1), offs=offsets)
        output = torch.zeros_like(padded)
        output.index_add_(0, tokens, (values * weights[:, None]).to(output.dtype))
        return output[:num_tokens]
