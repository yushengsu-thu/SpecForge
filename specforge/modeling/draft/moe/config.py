"""Resolve the draft FFN topology separately from its execution options."""

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MoEConfig:
    """The supported Qwen shared-expert FFN contract.

    A preset selects mathematics and checkpoint layout, never teacher dimensions.
    Topology fields must be explicit so enabling MoE cannot silently allocate a
    teacher-sized expert bank. Runtime options live in ``dflash_config``.
    """

    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    norm_topk_prob: bool = True
    aux_loss_coeff: float = 0.0
    dispatch: str = "reference"
    swiglu_limit: float | None = None


def _positive_int(config: Any, name: str) -> int:
    value = getattr(config, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"MoE config.{name} must be a positive integer, got {value!r}")
    return value


def resolve_moe_config(config: Any) -> MoEConfig:
    if getattr(config, "moe_preset", None) != "qwen3_5_moe":
        raise ValueError("MoE supports only moe_preset='qwen3_5_moe'")
    options = getattr(config, "dflash_config", None) or {}
    if not isinstance(options, dict):
        raise ValueError("MoE dflash_config must be a dictionary")
    allowed = {"moe_aux_loss_coeff", "moe_dispatch", "moe_router_center"}
    misplaced = sorted(name for name in allowed if hasattr(config, name))
    if misplaced:
        raise ValueError(f"MoE execution options belong in dflash_config: {misplaced}")
    unknown = {key for key in options if key.startswith("moe_")} - allowed
    if unknown:
        raise ValueError(f"Unknown MoE execution options: {sorted(unknown)}")
    if options.get("moe_router_center", "none") != "none":
        raise ValueError(
            "moe_router_center supports only 'none' (official Qwen routing)"
        )
    if getattr(config, "hidden_act", "silu") not in {"silu", "swish"}:
        raise ValueError("qwen3_5_moe requires silu/swish SwiGLU")
    # Fail explicitly on future-family semantics rather than accepting a config
    # whose forward or checkpoint format this preset cannot implement.
    for name, expected in (
        ("scoring_func", "softmax"),
        ("routed_scaling_factor", 1.0),
        ("n_shared_experts", 1),
        ("topk_method", "greedy"),
    ):
        if getattr(config, name, expected) != expected:
            raise ValueError(f"qwen3_5_moe requires {name}={expected!r}")
    experts = _positive_int(config, "n_routed_experts")
    official_experts = getattr(config, "num_experts", experts)
    if official_experts != experts:
        raise ValueError("num_experts conflicts with n_routed_experts")
    top_k = _positive_int(config, "num_experts_per_tok")
    if top_k > experts:
        raise ValueError("num_experts_per_tok cannot exceed n_routed_experts")
    norm_topk = getattr(config, "norm_topk_prob", True)
    if not isinstance(norm_topk, bool):
        raise ValueError("norm_topk_prob must be a boolean")
    coeff = options.get("moe_aux_loss_coeff", 0.0)
    if (
        isinstance(coeff, bool)
        or not isinstance(coeff, (int, float))
        or not math.isfinite(coeff)
        or coeff < 0
    ):
        raise ValueError("moe_aux_loss_coeff must be finite and non-negative")
    dispatch = options.get("moe_dispatch", "reference")
    if dispatch not in {"reference", "grouped_mm"}:
        raise ValueError("moe_dispatch must be 'reference' or 'grouped_mm'")
    limit = getattr(config, "swiglu_limit", None)
    if limit is not None and (
        isinstance(limit, bool)
        or not isinstance(limit, (int, float))
        or not math.isfinite(limit)
        or limit <= 0
    ):
        raise ValueError("swiglu_limit must be finite and positive or null")
    return MoEConfig(
        n_routed_experts=experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=_positive_int(config, "moe_intermediate_size"),
        shared_expert_intermediate_size=_positive_int(
            config, "shared_expert_intermediate_size"
        ),
        norm_topk_prob=norm_topk,
        aux_loss_coeff=float(coeff),
        dispatch=dispatch,
        swiglu_limit=limit,
    )
