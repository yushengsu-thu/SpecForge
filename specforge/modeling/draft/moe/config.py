# coding=utf-8
"""Draft MoE configuration: presets, architecture overrides and training knobs.

A draft config opts into MoE FFNs with ``moe_preset``.  The preset names an
upstream MoE family and fixes its structure (score function, top-k weighting,
shared-expert gating, balancing scheme, weight layout).  Sizes and other
architecture keys sit at the top level of the draft config and override the
preset; knobs that only shape training live in the method config as ``moe_*``
(``dflash_config`` for the DFlash family)::

    {
      "moe_preset": "qwen3_5_moe",
      "n_routed_experts": 512, "num_experts_per_tok": 10,
      "moe_intermediate_size": 512, "shared_expert_intermediate_size": 2048,
      "dflash_config": {"moe_aux_loss_coeff": 0.001,
                        "moe_dispatch": "grouped_mm",
                        "moe_router_center": "sample"}
    }

A config without ``moe_preset`` stays dense.  One that carries MoE fields
without a preset is rejected instead of silently training a dense FFN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

MOE_PRESET_KEY = "moe_preset"
SCORE_FUNCTION_NAMES = ("softmax", "sigmoid", "sqrtsoftplus")
DISPATCH_NAMES = ("auto", "grouped_mm", "loop")
ROUTER_CENTER_MODES = ("none", "sample")

#: Top-level draft-config keys (upstream HF config names) -> MoEConfig fields.
ARCHITECTURE_KEYS: Dict[str, str] = {
    "n_routed_experts": "n_routed_experts",
    "num_experts_per_tok": "num_experts_per_tok",
    "moe_intermediate_size": "moe_intermediate_size",
    "shared_expert_intermediate_size": "shared_expert_intermediate_size",
    "norm_topk_prob": "norm_topk_prob",
    "scoring_func": "score_function",
    "routed_scaling_factor": "routed_scaling_factor",
    "swiglu_limit": "swiglu_limit",
}
#: Qwen-family spelling of ``n_routed_experts``; both must agree when present.
NUM_EXPERTS_ALIAS = "num_experts"
#: Method-config keys -> MoEConfig fields.
TRAINING_KEYS: Dict[str, str] = {
    "moe_aux_loss_coeff": "aux_loss_coeff",
    "moe_bias_update_rate": "bias_update_rate",
    "moe_dispatch": "dispatch",
    "moe_router_center": "router_center",
}
_MOE_DECLARING_KEYS = (
    "n_routed_experts",
    NUM_EXPERTS_ALIAS,
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
)


def _require_int(name: str, value: Any, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"MoE {name} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_float(name: str, value: Any, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"MoE {name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        bound = "> 0" if positive else ">= 0"
        raise ValueError(f"MoE {name} must be finite and {bound}, got {value!r}")
    return value


def _require_choice(name: str, value: Any, choices: Tuple[str, ...]) -> str:
    if value not in choices:
        raise ValueError(f"MoE {name} must be one of {list(choices)}, got {value!r}")
    return value


@dataclass(frozen=True)
class MoEConfig:
    """Resolved configuration shared by every MoE layer of one draft.

    Architecture fields fix the function a layer computes and its weights;
    ``router_center`` is set with the training knobs but applies in every
    forward. The remaining training fields only change how layers optimize.
    """

    preset: str
    layout: str
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    #: 0 disables the shared expert.
    shared_expert_intermediate_size: int = 0
    #: Scale the shared expert by ``sigmoid(Linear(hidden, 1))``.
    shared_expert_gate: bool = False
    score_function: str = "softmax"
    #: Renormalize the selected top-k scores to sum to one.
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    #: Clamp SwiGLU inputs (gate from above, up to +-limit); None disables.
    swiglu_limit: Optional[float] = None
    #: Carry an fp32 per-expert bias that shifts top-k selection only.
    selection_bias: bool = False
    #: "sample" subtracts every expert's mean router logit within each sample.
    router_center: str = "none"
    #: Tokens per centering sample (a drafted block); None means one batch row.
    router_center_size: Optional[int] = None
    aux_loss_coeff: float = 0.0
    bias_update_rate: float = 0.0
    dispatch: str = "auto"

    def __post_init__(self) -> None:
        experts = _require_int("n_routed_experts", self.n_routed_experts, 1)
        top_k = _require_int("num_experts_per_tok", self.num_experts_per_tok, 1)
        if top_k > experts:
            raise ValueError(
                f"MoE num_experts_per_tok={top_k} exceeds n_routed_experts={experts}"
            )
        _require_int("moe_intermediate_size", self.moe_intermediate_size, 1)
        shared = _require_int(
            "shared_expert_intermediate_size", self.shared_expert_intermediate_size, 0
        )
        for name in ("shared_expert_gate", "norm_topk_prob", "selection_bias"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(
                    f"MoE {name} must be a boolean, got {getattr(self, name)!r}"
                )
        if self.shared_expert_gate and shared == 0:
            raise ValueError("MoE shared_expert_gate requires a shared expert")
        _require_choice("scoring_func", self.score_function, SCORE_FUNCTION_NAMES)
        _require_float(
            "routed_scaling_factor", self.routed_scaling_factor, positive=True
        )
        if self.swiglu_limit is not None:
            _require_float("swiglu_limit", self.swiglu_limit, positive=True)
        _require_choice("moe_router_center", self.router_center, ROUTER_CENTER_MODES)
        if self.router_center_size is not None:
            _require_int("router_center_size", self.router_center_size, 1)
        _require_float("moe_aux_loss_coeff", self.aux_loss_coeff)
        if _require_float("moe_bias_update_rate", self.bias_update_rate) > 0 and (
            not self.selection_bias
        ):
            raise ValueError(
                f"MoE moe_bias_update_rate needs a selection-bias preset; "
                f"{self.preset!r} has none"
            )
        _require_choice("moe_dispatch", self.dispatch, DISPATCH_NAMES)


_CONFIG_FIELDS = frozenset(field.name for field in fields(MoEConfig))


@dataclass(frozen=True)
class MoEPreset:
    """An upstream MoE family: its fixed structure and official weight layout."""

    name: str
    layout: str
    #: MoEConfig field defaults; top-level and ``moe_*`` keys override them.
    defaults: Mapping[str, Any]
    #: MoEConfig fields the draft config must set (sizes are never guessed).
    required: Tuple[str, ...] = (
        "n_routed_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
    )


MOE_PRESETS: Dict[str, MoEPreset] = {}


def register_moe_preset(preset: MoEPreset) -> MoEPreset:
    """Register ``preset`` under its name; names are unique."""

    if preset.name in MOE_PRESETS:
        raise ValueError(f"MoE preset {preset.name!r} is already registered")
    unknown = set(preset.defaults) | set(preset.required)
    unknown -= _CONFIG_FIELDS
    unknown -= {"preset", "layout"}
    if unknown:
        raise ValueError(
            f"MoE preset {preset.name!r} names unknown fields {sorted(unknown)}"
        )
    MOE_PRESETS[preset.name] = preset
    return preset


# Qwen3-Next / Qwen3.5 / Qwen3.8: softmax scores, renormalized top-k, one
# sigmoid-gated shared expert, Switch aux loss 0.001, fused checkpoint names.
#
# The other upstream families map onto the same fields; registering them needs
# their layout (and, for DeepSeek-V4, FP4/FP8 conversion) in ``layout.py``:
#   qwen3_moe   - shared_expert_intermediate_size 0, layout per-expert
#                 ``experts.{i}.{gate,up,down}_proj.weight``.
#   deepseek_v4 - ungated shared expert, score_function "sqrtsoftplus",
#                 routed_scaling_factor 2.5, swiglu_limit 10, selection_bias
#                 (``gate.bias``, fp32) with bias_update_rate, layout
#                 ``experts.{i}.w1/w2/w3`` with FP4 experts and FP8 elsewhere.
register_moe_preset(
    MoEPreset(
        name="qwen3_5_moe",
        layout="hf_fused",
        defaults={
            "shared_expert_gate": True,
            "score_function": "softmax",
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.0,
            "swiglu_limit": None,
            "selection_bias": False,
            "aux_loss_coeff": 1e-3,
        },
        required=(
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
        ),
    )
)


def _lookup(config: Any) -> Callable[[str], Any]:
    if isinstance(config, Mapping):
        return config.get
    return lambda key: getattr(config, key, None)


def _method_block_size(
    lookup: Callable[[str], Any], method_config: Mapping[str, Any]
) -> Optional[int]:
    block_size = lookup("block_size")
    if block_size is None:
        block_size = method_config.get("block_size")
    return block_size


def resolve_moe_config(
    config: Any,
    *,
    method_config: Optional[Mapping[str, Any]] = None,
    block_size: Optional[int] = None,
) -> Optional[MoEConfig]:
    """Resolve a draft config's MoE FFN, or ``None`` for a dense draft.

    ``config`` is a draft ``PretrainedConfig`` or mapping. Training knobs are
    read from ``method_config`` (default ``config.dflash_config``).
    ``block_size`` sizes ``moe_router_center: "sample"`` samples and defaults to
    the DFlash-family block size: a drafter proposes one block per step, so
    centering per block keeps training and inference routing identical.
    """

    lookup = _lookup(config)
    if method_config is None:
        method_config = lookup("dflash_config") or {}
    training = {key: value for key, value in method_config.items() if key[:4] == "moe_"}
    preset_name = lookup(MOE_PRESET_KEY)
    if preset_name is None:
        declared = [key for key in _MOE_DECLARING_KEYS if lookup(key) is not None]
        if declared or training:
            raise ValueError(
                f"draft config sets MoE fields {declared + sorted(training)} "
                f"without {MOE_PRESET_KEY!r}; name one of {sorted(MOE_PRESETS)} "
                "or remove them for a dense FFN"
            )
        return None
    preset = MOE_PRESETS.get(preset_name)
    if preset is None:
        raise ValueError(
            f"unknown {MOE_PRESET_KEY} {preset_name!r}; available: "
            f"{sorted(MOE_PRESETS)}"
        )
    unknown = sorted(set(training) - set(TRAINING_KEYS))
    if unknown:
        raise ValueError(
            f"unknown MoE training keys {unknown}; supported: {sorted(TRAINING_KEYS)}"
        )

    values: Dict[str, Any] = dict(preset.defaults)
    for key, field_name in ARCHITECTURE_KEYS.items():
        value = lookup(key)
        if value is not None:
            values[field_name] = value
    if values.get("swiglu_limit") == 0:
        # Upstream configs spell "no clamp" as swiglu_limit: 0.
        values["swiglu_limit"] = None
    alias = lookup(NUM_EXPERTS_ALIAS)
    if alias is not None:
        if values.setdefault("n_routed_experts", alias) != alias:
            raise ValueError(
                f"draft config {NUM_EXPERTS_ALIAS}={alias!r} disagrees with "
                f"n_routed_experts={values['n_routed_experts']!r}"
            )
    missing = [name for name in preset.required if values.get(name) is None]
    if missing:
        raise ValueError(
            f"{MOE_PRESET_KEY} {preset.name!r} requires draft config keys {missing}"
        )
    for key, field_name in TRAINING_KEYS.items():
        if key in training:
            values[field_name] = training[key]
    if values.get("router_center") == "sample":
        values["router_center_size"] = (
            block_size
            if block_size is not None
            else _method_block_size(lookup, method_config)
        )
    return MoEConfig(preset=preset.name, layout=preset.layout, **values)


__all__ = [
    "ARCHITECTURE_KEYS",
    "MOE_PRESETS",
    "MOE_PRESET_KEY",
    "MoEConfig",
    "MoEPreset",
    "TRAINING_KEYS",
    "register_moe_preset",
    "resolve_moe_config",
]
