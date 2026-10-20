# coding=utf-8
"""Expert routing: score functions, router-logit centering and top-k weights."""

from __future__ import annotations

from typing import Callable, Dict, NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import MoEConfig


def _sqrtsoftplus(logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(logits).sqrt()


#: Router score functions, applied to fp32 logits.
SCORE_FUNCTIONS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "softmax": lambda logits: logits.softmax(dim=-1),
    "sigmoid": torch.sigmoid,
    "sqrtsoftplus": _sqrtsoftplus,
}


class RouterOutput(NamedTuple):
    """Routing for ``T`` tokens: top-``K`` of ``E`` experts per token."""

    #: ``[T, K]`` fp32 combine weights.
    weights: torch.Tensor
    #: ``[T, K]`` int64 selected expert ids.
    indices: torch.Tensor
    #: ``[T, E]`` fp32 scores before any selection bias.
    scores: torch.Tensor


def center_router_logits(
    logits: torch.Tensor, sample_size: Optional[int]
) -> torch.Tensor:
    """Subtract each expert's mean logit within every routing sample.

    ``logits`` is ``[batch, tokens, experts]``. A sample is ``sample_size``
    consecutive tokens of one row, or the whole row when ``sample_size`` is
    ``None``.
    """

    batch, tokens, experts = logits.shape
    if sample_size is None:
        return logits - logits.mean(dim=1, keepdim=True)
    if tokens % sample_size:
        raise ValueError(
            f"router centering needs whole samples of {sample_size} tokens, "
            f"got {tokens} tokens per row"
        )
    samples = logits.view(batch, tokens // sample_size, sample_size, experts)
    return (samples - samples.mean(dim=2, keepdim=True)).view(batch, tokens, experts)


class TopKRouter(nn.Module):
    """Linear router choosing the top-k experts and their combine weights.

    Logits are taken in the activation dtype, then scored in fp32. The
    optional fp32 ``e_score_correction_bias`` (DeepSeek's aux-loss-free
    balancing) shifts scores for expert selection only; combine weights
    always come from the unbiased scores.
    """

    def __init__(self, config: MoEConfig, hidden_size: int) -> None:
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.score_function = config.score_function
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.router_center = config.router_center
        self.router_center_size = config.router_center_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, hidden_size))
        bias = None
        if config.selection_bias:
            bias = torch.zeros(self.num_experts, dtype=torch.float32)
        self.register_buffer("e_score_correction_bias", bias)

    def _apply(self, fn, recurse: bool = True):
        # Bias updates sit far below bf16 resolution: keep it fp32 through
        # module-wide casts such as ``model.to(torch.bfloat16)``.
        bias = self.e_score_correction_bias
        module = super()._apply(fn, recurse)
        moved = self.e_score_correction_bias
        if bias is not None and moved.dtype != torch.float32:
            self.e_score_correction_bias = bias.to(
                device=moved.device, dtype=torch.float32
            )
        return module

    def forward(self, hidden_states: torch.Tensor) -> RouterOutput:
        """Route ``[batch, tokens, hidden]`` (or ``[tokens, hidden]``) inputs."""

        logits = F.linear(hidden_states, self.weight).float()
        if self.router_center == "sample":
            rows = logits if logits.dim() == 3 else logits.unsqueeze(0)
            logits = center_router_logits(rows, self.router_center_size)
        scores = SCORE_FUNCTIONS[self.score_function](
            logits.reshape(-1, self.num_experts)
        )
        selection = scores
        if self.e_score_correction_bias is not None:
            selection = scores + self.e_score_correction_bias
        indices = torch.topk(selection, self.top_k, dim=-1).indices
        weights = scores.gather(-1, indices)
        if self.norm_topk_prob:
            denominator = weights.sum(dim=-1, keepdim=True)
            weights = weights / denominator.clamp_min(torch.finfo(weights.dtype).tiny)
        if self.routed_scaling_factor != 1.0:
            weights = weights * self.routed_scaling_factor
        return RouterOutput(weights=weights, indices=indices, scores=scores)

    def extra_repr(self) -> str:
        return (
            f"experts={self.num_experts}, top_k={self.top_k}, "
            f"score={self.score_function}, center={self.router_center}"
        )


__all__ = [
    "RouterOutput",
    "SCORE_FUNCTIONS",
    "TopKRouter",
    "center_router_logits",
]
