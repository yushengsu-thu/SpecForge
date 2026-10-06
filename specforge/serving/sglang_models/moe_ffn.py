# coding=utf-8
"""MoE FFN of a SpecForge draft layer, for serving (plain PyTorch).

SpecForge's MoE-FFN drafters (``specforge/modeling/draft/moe``) keep the
DFlash-family decoder and replace each layer's dense MLP with
``n_routed_experts`` routed experts plus an optional ungated shared expert,
routed DeepSeek-style (score function, aux-loss-free selection bias,
renormalized combine weights times ``routed_scaling_factor``). Exports keep
the official DeepSeek checkpoint naming::

    layers.N.mlp.gate.weight                     [E, hidden]
    layers.N.mlp.gate.bias                       [E]  (fp32 selection bias, noaux_tc)
    layers.N.mlp.experts.{i}.w1.weight           [inter, hidden]   (gate_proj)
    layers.N.mlp.experts.{i}.w2.weight           [hidden, inter]   (down_proj)
    layers.N.mlp.experts.{i}.w3.weight           [inter, hidden]   (up_proj)
    layers.N.mlp.shared_experts.w{1,2,3}.weight

and ``config.json`` carries the routing recipe in the DeepSeek HF vocabulary
(``n_routed_experts``, ``num_experts_per_tok``, ``moe_intermediate_size``,
``n_shared_experts``, ``scoring_func``, ``norm_topk_prob``,
``routed_scaling_factor``, ``n_group``, ``topk_group``, ``topk_method``,
``swiglu_limit``), which ``MoEConfig.serving_fields()`` writes at export.

The math mirrors the trainer's ``TopKRouter`` / ``NoAuxTCController`` /
``GroupedExperts`` / ``SwiGLUSharedExpert``: routing in fp32, experts in the
model dtype through ``torch._grouped_mm`` with on-device offsets (no host
sync, CUDA-graph safe), per-expert loop as the portable fallback. This module
imports no SGLang code, so it doubles as the CPU reference in
``scripts/gates/check_dspark_moe_sglang_equivalence.py``.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

SCORE_FUNCTIONS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "softmax": lambda logits: logits.softmax(dim=-1),
    "sigmoid": torch.sigmoid,
    # DeepSeek-V4 scoring.
    "sqrtsoftplus": lambda logits: F.softplus(logits).sqrt(),
}

#: Preset defaults, used only for keys the export config does not state
#: (``MoEConfig.serving_fields()`` normally writes all of them).
PRESET_DEFAULTS: Dict[str, Dict[str, object]] = {
    "deepseek_v4": dict(
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        n_shared_experts=1,
        swiglu_limit=10.0,
        topk_method="noaux_tc",
    ),
    "qwen3": dict(
        scoring_func="softmax",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        n_shared_experts=0,
        swiglu_limit=0.0,
        topk_method="noaux_tc",
    ),
}


def _cfg(config, key: str, default):
    value = getattr(config, key, None)
    return default if value is None else value


def routed_expert_count(config) -> int:
    """``n_routed_experts`` (DeepSeek) or ``num_experts`` (Qwen) of a config; 0 = dense."""
    for key in ("n_routed_experts", "num_experts"):
        value = _cfg(config, key, 0)
        if value:
            return int(value)
    return 0


def swiglu_clamped(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """SwiGLU in fp32 with the DeepSeek-V4 activation clamp (``limit`` 0 = off)."""
    gate = gate.float()
    up = up.float()
    if limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    return F.silu(gate) * up


def group_limited_mask(
    selection: torch.Tensor, n_group: int, topk_group: int
) -> torch.Tensor:
    tokens, n_experts = selection.shape
    grouped = selection.view(tokens, n_group, n_experts // n_group)
    group_scores = grouped.topk(min(2, grouped.shape[-1]), dim=-1).values.sum(-1)
    keep = group_scores.topk(topk_group, dim=-1).indices
    mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, keep, True)
    return grouped.masked_fill(~mask.unsqueeze(-1), float("-inf")).view(
        tokens, n_experts
    )


class DraftMoEGate(nn.Module):
    """Router projection plus the aux-loss-free selection bias (``gate.bias``)."""

    def __init__(self, n_experts: int, hidden_size: int, use_bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_experts, hidden_size))
        if use_bias:
            # Selection-only bias; kept fp32 like the trainer's controller buffer.
            self.bias = nn.Parameter(
                torch.zeros(n_experts, dtype=torch.float32), requires_grad=False
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), self.weight.float())


class DraftRoutedExperts(nn.Module):
    """All routed experts as three stacked parameters (``w1``/``w3`` gate/up
    ``[E, inter, hidden]``, ``w2`` down ``[E, hidden, inter]``)."""

    def __init__(self, n_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        e, d, i = n_experts, hidden_size, intermediate_size
        self.w1 = nn.Parameter(torch.empty(e, i, d))
        self.w2 = nn.Parameter(torch.empty(e, d, i))
        self.w3 = nn.Parameter(torch.empty(e, i, d))


class DraftSharedExpert(nn.Module):
    """Ungated SwiGLU shared expert (DeepSeek ``shared_experts.w{1,2,3}``)."""

    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float):
        super().__init__()
        self.swiglu_limit = float(swiglu_limit)
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = swiglu_clamped(self.w1(x), self.w3(x), self.swiglu_limit)
        return self.w2(h.to(x.dtype))


class DraftMoEFFN(nn.Module):
    """``y = experts(x, gate(x)) + shared_experts(x)`` for one draft layer."""

    def __init__(self, config) -> None:
        super().__init__()
        preset = PRESET_DEFAULTS.get(str(_cfg(config, "moe_preset", "")), {})

        def get(key, default):
            return _cfg(config, key, preset.get(key, default))

        self.hidden_size = int(config.hidden_size)
        self.n_experts = routed_expert_count(config)
        self.topk = int(_cfg(config, "num_experts_per_tok", 0))
        self.intermediate_size = int(_cfg(config, "moe_intermediate_size", 0))
        if self.n_experts <= 0 or self.topk <= 0 or self.intermediate_size <= 0:
            raise ValueError(
                "MoE draft config needs n_routed_experts (or num_experts), "
                "num_experts_per_tok and moe_intermediate_size; got "
                f"{self.n_experts}, {self.topk}, {self.intermediate_size}."
            )
        if self.topk > self.n_experts:
            raise ValueError("num_experts_per_tok must not exceed n_routed_experts")
        if _cfg(config, "hidden_act", "silu") != "silu":
            raise ValueError("MoE draft experts support only silu (SwiGLU)")

        self.scoring_func = str(get("scoring_func", "softmax"))
        if self.scoring_func not in SCORE_FUNCTIONS:
            raise ValueError(
                f"unknown scoring_func {self.scoring_func!r}; "
                f"known: {sorted(SCORE_FUNCTIONS)}"
            )
        self.score_fn = SCORE_FUNCTIONS[self.scoring_func]
        self.norm_topk_prob = bool(get("norm_topk_prob", True))
        self.routed_scaling_factor = float(get("routed_scaling_factor", 1.0))
        self.n_group = int(get("n_group", 1))
        self.topk_group = int(get("topk_group", 1))
        if self.n_group <= 0 or self.n_experts % self.n_group:
            raise ValueError("n_group must divide n_routed_experts")
        if not 0 < self.topk_group <= self.n_group:
            raise ValueError("topk_group must be in [1, n_group]")
        self.swiglu_limit = float(get("swiglu_limit", 0.0))
        use_bias = str(get("topk_method", "greedy")) == "noaux_tc"

        self.gate = DraftMoEGate(self.n_experts, self.hidden_size, use_bias)
        # ``experts.w{1,2,3}`` stacked [E, out, in]; the loader stacks the
        # per-expert checkpoint tensors ``experts.{i}.w{1,2,3}.weight``.
        self.experts = DraftRoutedExperts(
            self.n_experts, self.hidden_size, self.intermediate_size
        )

        n_shared = int(get("n_shared_experts", 0) or 0)
        if n_shared not in (0, 1):
            raise ValueError("n_shared_experts must be 0 or 1 for MoE drafts")
        shared_width = int(
            _cfg(config, "shared_expert_intermediate_size", self.intermediate_size)
        )
        self.shared_experts: Optional[DraftSharedExpert] = (
            DraftSharedExpert(self.hidden_size, shared_width, self.swiglu_limit)
            if n_shared
            else None
        )
        self.grouped_mm = hasattr(torch, "_grouped_mm")
        if not self.grouped_mm:
            logger.warning(
                "torch._grouped_mm is unavailable; MoE draft experts fall back to a "
                "per-expert loop with a host sync (not CUDA-graph safe)."
            )

    def describe(self) -> str:
        return (
            f"{self.n_experts} routed experts, top-{self.topk}, width "
            f"{self.intermediate_size}, shared={self.shared_experts is not None}, "
            f"scoring={self.scoring_func}, bias={self.gate.bias is not None}, "
            f"renorm={self.norm_topk_prob}, scale={self.routed_scaling_factor:.2f}, "
            f"swiglu_limit={self.swiglu_limit:.1f}"
        )

    def route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (weights [T,k] fp32, indices [T,k] long, counts [E] long)."""
        scores = self.score_fn(self.gate(x))
        selection = (
            scores if self.gate.bias is None else scores + self.gate.bias.float()
        )
        if self.topk_group < self.n_group:
            selection = group_limited_mask(selection, self.n_group, self.topk_group)
        indices = selection.topk(self.topk, dim=-1).indices
        weights = scores.gather(1, indices)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        flat = indices.flatten()
        counts = torch.zeros(
            self.n_experts, dtype=torch.long, device=x.device
        ).scatter_add_(0, flat, torch.ones_like(flat))
        return weights, indices, counts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.hidden_size)
        weights, indices, counts = self.route(x)
        flat_expert = indices.flatten()
        order = flat_expert.argsort(stable=True)
        token_of = order // self.topk
        x_sorted = x.index_select(0, token_of)
        w_sorted = weights.reshape(-1, 1).index_select(0, order).float()
        w1, w2, w3 = self.experts.w1, self.experts.w2, self.experts.w3
        if x.is_cuda and self.grouped_mm:
            offs = counts.cumsum(0).to(torch.int32)
            gate = torch._grouped_mm(x_sorted, w1.transpose(-1, -2), offs=offs)
            up = torch._grouped_mm(x_sorted, w3.transpose(-1, -2), offs=offs)
            h = w_sorted * swiglu_clamped(gate, up, self.swiglu_limit)
            y_routed = torch._grouped_mm(h.to(x.dtype), w2.transpose(-1, -2), offs=offs)
        else:
            counts_list = counts.tolist()
            parts = []
            offset = 0
            for i, n in enumerate(counts_list):
                if n == 0:
                    continue
                seg = x_sorted[offset : offset + n]
                h = w_sorted[offset : offset + n] * swiglu_clamped(
                    F.linear(seg, w1[i]),
                    F.linear(seg, w3[i]),
                    self.swiglu_limit,
                )
                parts.append(F.linear(h.to(seg.dtype), w2[i]))
                offset += n
            y_routed = (
                torch.cat(parts, dim=0)
                if parts
                else torch.zeros(0, self.hidden_size, dtype=x.dtype, device=x.device)
            )
        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        y = y.index_add(0, token_of, y_routed.float()).to(x.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y.view(shape)


_PER_EXPERT_KEY = re.compile(
    r"^(?P<base>(?:.*\.)?experts)\.(?P<idx>\d+)\.(?P<w>w[123])\.weight$"
)


def stack_expert_weights(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> List[Tuple[str, torch.Tensor]]:
    """Turn ``experts.{i}.w{1,2,3}.weight`` entries into stacked ``experts.w{1,2,3}``.

    Every other entry passes through unchanged. Raises if an expert index is
    missing, so a truncated export cannot load half of a layer silently.
    """
    passthrough: List[Tuple[str, torch.Tensor]] = []
    groups: Dict[Tuple[str, str], Dict[int, torch.Tensor]] = {}
    for name, tensor in weights:
        m = _PER_EXPERT_KEY.match(name)
        if m is None:
            passthrough.append((name, tensor))
            continue
        groups.setdefault((m["base"], m["w"]), {})[int(m["idx"])] = tensor
    for (base, w), members in groups.items():
        n = max(members) + 1
        if sorted(members) != list(range(n)):
            raise ValueError(
                f"{base}.*.{w}.weight is missing expert indices: have {sorted(members)}"
            )
        passthrough.append((f"{base}.{w}", torch.stack([members[i] for i in range(n)])))
    return passthrough


def merge_gate_up(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> List[Tuple[str, torch.Tensor]]:
    """Turn stacked ``experts.w1`` / ``experts.w3`` pairs into ``experts.w13``.

    The fused MoE kernel reads gate and up projections from one ``[E, 2N, K]``
    tensor (gate rows first, not interleaved). Entries without a partner pass
    through unchanged, so a dense draft or a half-loaded export is reported
    by the strict check rather than silently merged.
    """
    pending: Dict[str, Dict[str, torch.Tensor]] = {}
    out: List[Tuple[str, torch.Tensor]] = []
    for name, tensor in weights:
        if name.endswith(".w1") or name.endswith(".w3"):
            pending.setdefault(name[:-3], {})[name[-2:]] = tensor
        else:
            out.append((name, tensor))
    for base, parts in pending.items():
        if "w1" in parts and "w3" in parts:
            out.append((f"{base}.w13", torch.cat([parts["w1"], parts["w3"]], dim=1)))
        else:
            out.extend((f"{base}.{k}", v) for k, v in parts.items())
    return out


def verify_moe_weights(
    model: nn.Module, provided_names: Set[str], class_name: str
) -> None:
    """Fail loudly when the checkpoint's FFN entries and the class disagree.

    ``provided_names`` are the (stacked) checkpoint names without a ``model.``
    prefix. Every ``*.mlp.*`` parameter of the model must have been provided
    and every provided ``*.mlp.*`` name must exist on the model; otherwise an
    MoE export loaded into a dense class (or vice versa) would serve random
    FFNs without any error.
    """
    params = {name for name, _ in model.named_parameters() if ".mlp." in name}
    provided = {name for name in provided_names if ".mlp." in name}
    missing = sorted(params - provided)
    unexpected = sorted(provided - params)
    if missing:
        shown = ", ".join(missing[:8])
        raise ValueError(
            f"{class_name}: {len(missing)} FFN parameter(s) were not in the "
            f"checkpoint and would serve uninitialised: {shown}"
            f"{' ...' if len(missing) > 8 else ''}. Check that config.json's "
            "`architectures` matches the export."
        )
    if unexpected:
        shown = ", ".join(unexpected[:8])
        raise ValueError(
            f"{class_name}: {len(unexpected)} checkpoint FFN weight(s) do not map "
            f"to any parameter of this architecture: {shown}"
            f"{' ...' if len(unexpected) > 8 else ''}. Check that config.json's "
            "`architectures` matches the export (an MoE drafter needs an "
            "MoE-capable draft class)."
        )
