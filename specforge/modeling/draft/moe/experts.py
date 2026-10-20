"""Packed Qwen expert storage, with interchangeable PyTorch execution paths."""

import torch
from torch import nn
from torch.nn import functional as F

from .config import MoEConfig


def swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float | None) -> torch.Tensor:
    """Optional pre-activation clamp; the official Qwen default is unclamped."""
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


class SharedExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, limit: float | None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.limit = limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x), self.limit))


class PackedExperts(nn.Module):
    """Official Qwen packed names: the two expert tensors have no ``.weight``."""

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
        self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        if self.config.dispatch == "grouped_mm":
            return self._grouped_mm(x, indices, weights)
        result = torch.zeros_like(x)
        # Empty selections still participate in autograd. A packed expert bank
        # consequently gets zero gradients for unused slices, including DDP.
        for expert in range(self.config.n_routed_experts):
            token, slot = torch.where(indices == expert)
            gate, up = F.linear(x[token], self.gate_up_proj[expert]).chunk(2, dim=-1)
            hidden = swiglu(gate, up, self.config.swiglu_limit)
            output = F.linear(hidden, self.down_proj[expert])
            result = result.index_add(
                0, token, (output * weights[token, slot, None]).to(x.dtype)
            )
        return result

    def _grouped_mm(
        self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        grouped_mm = getattr(F, "grouped_mm", None)
        if (
            grouped_mm is None
            or x.device.type != "cuda"
            or self.gate_up_proj.dtype != torch.bfloat16
            or torch.cuda.get_device_capability(x.device)[0] < 8
        ):
            raise ValueError(
                "moe_dispatch='grouped_mm' requires PyTorch grouped_mm, CUDA SM80+, and BF16 expert weights; use 'reference' otherwise"
            )
        if (
            x.shape[-1] % 8
            or self.config.moe_intermediate_size % 8
            or self.config.n_routed_experts >= 1024
        ):
            raise ValueError(
                "grouped_mm requires hidden/intermediate dimensions divisible by 8 and fewer than 1024 experts"
            )
        if indices.numel() == 0:
            # Preserve a differentiable zero for the complete packed bank.
            return x * 0 + (self.gate_up_proj.sum() + self.down_proj.sum()) * 0
        order = indices.flatten().argsort(stable=True)
        token = order // indices.shape[-1]
        counts = torch.bincount(
            indices.flatten(), minlength=self.config.n_routed_experts
        )
        offsets = counts.cumsum(0).to(torch.int32)
        sorted_x = x[token].to(self.gate_up_proj.dtype)
        gate, up = grouped_mm(
            sorted_x, self.gate_up_proj.transpose(-2, -1), offs=offsets
        ).chunk(2, dim=-1)
        hidden = swiglu(gate, up, self.config.swiglu_limit)
        output = grouped_mm(hidden, self.down_proj.transpose(-2, -1), offs=offsets)
        output = output * weights.flatten()[order, None].to(output.dtype)
        # Restore unique (token, top-k slot) assignments before reduction.
        # A single CUDA index_add over repeated token IDs has nondeterministic
        # BF16 atomic accumulation; a fixed top-k reduction avoids that race.
        per_slot = output[order.argsort()].reshape(
            x.shape[0], indices.shape[-1], x.shape[-1]
        )
        return per_slot.sum(dim=1, dtype=x.dtype)
