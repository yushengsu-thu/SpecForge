"""Resolve one MoE architecture independently of a draft model family."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable

from torch import nn


@dataclass(frozen=True)
class MoEConfig:
    """Structural values are checkpoint-visible; training controls stay separate.

    The Qwen preset describes only an FFN. Attention, residual connections and
    draft heads remain the responsibility of the enclosing draft model.
    """

    n_routed_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    scoring_func: str = "softmax"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    shared_expert_gate: bool = True
    swiglu_limit: float | None = None
    layout: str = "qwen3_5_moe"
    aux_loss_coeff: float = 0.001
    dispatch: str = "reference"
    router_center: str = "none"

    def __post_init__(self) -> None:
        from .layout import validate_layout

        for name in (
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        shared_size = self.shared_expert_intermediate_size
        if (
            not isinstance(shared_size, int)
            or isinstance(shared_size, bool)
            or shared_size < 0
        ):
            raise ValueError(
                "shared_expert_intermediate_size must be a nonnegative integer"
            )
        if self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("num_experts_per_tok cannot exceed n_routed_experts")
        if self.scoring_func != "softmax":
            raise ValueError(
                "Only the Qwen softmax scoring function is currently supported"
            )
        if self.norm_topk_prob is not True:
            raise ValueError("qwen3_5_moe requires norm_topk_prob=true")
        if not isinstance(self.shared_expert_gate, bool):
            raise ValueError("shared_expert_gate must be a boolean")
        for name, nonnegative in (
            ("aux_loss_coeff", True),
            ("routed_scaling_factor", False),
            ("swiglu_limit", False),
        ):
            value = getattr(self, name)
            if name == "swiglu_limit" and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite real number")
            invalid = value < 0 if nonnegative else value <= 0
            if invalid:
                condition = "nonnegative" if nonnegative else "positive"
                raise ValueError(f"{name} must be {condition}")
        if self.dispatch not in ("reference", "grouped_mm"):
            raise ValueError("moe_dispatch must be 'reference' or 'grouped_mm'")
        if self.router_center not in (None, "none", "sample"):
            raise ValueError("moe_router_center must be 'none' or 'sample'")
        validate_layout(self.layout)


def _get(config: Any, key: str, default: Any = None) -> Any:
    return (
        config.get(key, default)
        if isinstance(config, Mapping)
        else getattr(config, key, default)
    )


def _set(config: Any, key: str, value: Any) -> None:
    if isinstance(config, dict):
        config[key] = value
    else:
        setattr(config, key, value)


def resolve_moe_config(config: Any) -> MoEConfig | None:
    """Resolve the explicit preset plus top-level structural overrides.

    Resolved structural fields are written back so a saved draft config is
    self-contained for serving consumers that do not know SpecForge presets.
    A teacher's expert count is never enough to silently select a draft FFN.
    """
    preset = _get(config, "moe_preset")
    routed_count = _get(config, "n_routed_experts")
    hf_count = _get(config, "num_experts")
    if routed_count is not None and hf_count is not None and routed_count != hf_count:
        raise ValueError(
            "n_routed_experts and num_experts must agree when both are provided"
        )
    count = routed_count if routed_count is not None else hf_count
    if preset is None:
        if count is not None:
            raise ValueError("Draft expert counts require an explicit moe_preset")
        return None
    if preset != "qwen3_5_moe":
        raise ValueError(f"Unsupported moe_preset {preset!r}; expected 'qwen3_5_moe'")
    training = _get(config, "dflash_config", {}) or {}
    if not isinstance(training, Mapping):
        raise ValueError("dflash_config must be a mapping")
    known = {"moe_aux_loss_coeff", "moe_dispatch", "moe_router_center"}
    unknown = {key for key in training if key.startswith("moe_")} - known
    if unknown:
        raise ValueError(f"Unknown MoE training options: {sorted(unknown)}")
    defaults = MoEConfig()
    values = {
        name: _get(config, name, getattr(defaults, name))
        for name in (
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "scoring_func",
            "norm_topk_prob",
            "routed_scaling_factor",
            "shared_expert_gate",
            "swiglu_limit",
        )
    }
    values["n_routed_experts"] = defaults.n_routed_experts if count is None else count
    values["layout"] = _get(config, "moe_layout", defaults.layout)
    values["aux_loss_coeff"] = training.get(
        "moe_aux_loss_coeff", defaults.aux_loss_coeff
    )
    values["dispatch"] = training.get("moe_dispatch", defaults.dispatch)
    values["router_center"] = training.get("moe_router_center", defaults.router_center)
    resolved = MoEConfig(**values)
    hidden_act = _get(config, "hidden_act", "silu")
    if hidden_act not in ("silu", "swish"):
        raise ValueError("qwen3_5_moe requires a SwiGLU hidden_act ('silu' or 'swish')")
    for name, value in values.items():
        if name not in ("aux_loss_coeff", "dispatch", "router_center", "layout"):
            _set(config, name, value)
    _set(config, "num_experts", resolved.n_routed_experts)
    _set(config, "moe_layout", resolved.layout)
    return resolved


def build_ffn(config: Any, *, dense: Callable[[Any], nn.Module]) -> nn.Module:
    """Keep dense construction unchanged unless a draft explicitly selects MoE."""
    from .layer import MoELayer

    resolved = resolve_moe_config(config)
    return (
        dense(config)
        if resolved is None
        else MoELayer(resolved, _get(config, "hidden_size"))
    )
