# coding=utf-8
"""Routed experts and the shared expert: SwiGLU FFNs with an optional clamp."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import MoEConfig


def swiglu(
    gate: torch.Tensor, up: torch.Tensor, limit: Optional[float] = None
) -> torch.Tensor:
    """``silu(gate) * up``.

    ``limit`` first clamps ``gate`` from above and ``up`` to ``[-limit,
    limit]`` (DeepSeek-V4's ``swiglu_limit``).
    """

    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


class SharedExpert(nn.Module):
    """Dense SwiGLU expert applied to every token."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(swiglu(gate, up, self.swiglu_limit))


class RoutedExperts(nn.Module):
    """``E`` SwiGLU experts stored in the fused HF layout.

    ``gate_up_proj`` is ``[E, 2I, H]`` with each expert's gate rows before its
    up rows, and ``down_proj`` is ``[E, H, I]``. ``forward`` returns every
    token's routing-weighted sum over its selected experts.
    """

    def __init__(self, config: MoEConfig, hidden_size: int) -> None:
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.intermediate_size = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.dispatch = config.dispatch
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, hidden_size, self.intermediate_size)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """``hidden_states`` ``[T, H]``; ``indices``/``weights`` ``[T, K]``."""

        dispatch = resolve_expert_dispatch(self.dispatch, hidden_states)
        return EXPERT_DISPATCH[dispatch](self, hidden_states, indices, weights)

    def extra_repr(self) -> str:
        return (
            f"experts={self.num_experts}, intermediate={self.intermediate_size}, "
            f"dispatch={self.dispatch}, swiglu_limit={self.swiglu_limit}"
        )


def grouped_mm_experts(
    experts: RoutedExperts,
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Run every expert as one grouped GEMM per projection.

    Token-expert assignments are sorted by expert so each expert owns a
    contiguous slice (experts without tokens are empty groups). Routing
    weights scale the ``[S, I]`` activations before the down projection,
    which keeps the ``[S, H]`` output out of autograd's saved tensors. Each
    token's ``K`` contributions are summed in fp32 without atomics.
    """

    num_tokens, top_k = indices.shape
    flat_experts = indices.reshape(-1)
    order = torch.argsort(flat_experts, stable=True)
    # scatter_add counts without bincount's device-to-host size check.
    counts = torch.zeros(
        experts.num_experts, dtype=torch.long, device=flat_experts.device
    ).scatter_add_(0, flat_experts, torch.ones_like(flat_experts))
    offsets = counts.cumsum(0).to(torch.int32)
    token_ids = torch.div(order, top_k, rounding_mode="floor")

    routed = hidden_states.index_select(0, token_ids)
    gate_up = F.grouped_mm(routed, experts.gate_up_proj.transpose(-2, -1), offs=offsets)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = swiglu(gate, up, experts.swiglu_limit)
    combine = weights.reshape(-1).index_select(0, order).unsqueeze(-1)
    activated = activated * combine.to(activated.dtype)
    expert_out = F.grouped_mm(
        activated, experts.down_proj.transpose(-2, -1), offs=offsets
    )
    expert_out = torch.empty_like(expert_out).index_copy(0, order, expert_out)
    summed = expert_out.view(num_tokens, top_k, -1).sum(dim=1, dtype=torch.float32)
    return summed.to(hidden_states.dtype)


def loop_experts(
    experts: RoutedExperts,
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Reference path: one dense GEMM pair per expert that received tokens.

    Portable to devices without grouped GEMM; reading the hit experts costs a
    host synchronization per call.
    """

    output = torch.zeros(
        hidden_states.shape, dtype=torch.float32, device=hidden_states.device
    )
    hit = torch.unique(indices).tolist()
    for expert in hit:
        token_ids, slot = torch.where(indices == expert)
        gate, up = F.linear(
            hidden_states[token_ids], experts.gate_up_proj[expert]
        ).chunk(2, dim=-1)
        activated = swiglu(gate, up, experts.swiglu_limit)
        activated = activated * weights[token_ids, slot, None].to(activated.dtype)
        expert_out = F.linear(activated, experts.down_proj[expert])
        output = output.index_add(0, token_ids, expert_out.float())
    return output.to(hidden_states.dtype)


#: ``moe_dispatch`` implementations of :meth:`RoutedExperts.forward`.
EXPERT_DISPATCH: Dict[str, Callable[..., torch.Tensor]] = {
    "grouped_mm": grouped_mm_experts,
    "loop": loop_experts,
}


def resolve_expert_dispatch(name: str, hidden_states: torch.Tensor) -> str:
    """Resolve ``auto`` to grouped GEMM where PyTorch provides it."""

    if name != "auto":
        if name == "grouped_mm" and not hasattr(F, "grouped_mm"):
            raise RuntimeError(
                "moe_dispatch='grouped_mm' needs torch.nn.functional.grouped_mm "
                f"(torch {torch.__version__} lacks it); use 'loop' or 'auto'"
            )
        return name
    supported = hasattr(F, "grouped_mm") and hidden_states.device.type in (
        "cuda",
        "cpu",
    )
    return "grouped_mm" if supported else "loop"


__all__ = [
    "EXPERT_DISPATCH",
    "RoutedExperts",
    "SharedExpert",
    "grouped_mm_experts",
    "loop_experts",
    "resolve_expert_dispatch",
    "swiglu",
]
