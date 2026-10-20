# coding=utf-8
"""Mixture-of-Experts FFNs for draft models.

Every operation a draft MoE layer needs lives here, one concern per module:
``routing`` (score functions, router-logit centering, top-k weights),
``experts`` (routed experts, shared expert, SwiGLU clamp), ``balance`` (aux
loss, selection bias), ``layout`` (weight naming) and ``config`` (presets).

Draft models adopt it through three seams:

* construction: ``self.mlp = build_ffn(config, dense=kernels.make_mlp)``
  (DFlash family), or ``MoELayer(resolve_moe_config(config), hidden_size)``
  for drafts with their own blocks;
* initialization: ``_init_weights`` calls ``module.reset_parameters(std)``
  on every :class:`MoELayer`;
* training forward: ``apply_pending_balance_updates(model)`` first, outside
  activation checkpointing.

The trainer folds the per-layer aux losses and ``moe/*`` metrics into each
step (``finalize_step_output``), and every draft-weight read or write converts
names through the preset's layout (:class:`MoEStateCodec`).
"""

from .balance import (
    BalanceTracker,
    MoEStepTerms,
    apply_pending_balance_updates,
    pop_moe_step_terms,
)
from .config import (
    MOE_PRESET_KEY,
    MOE_PRESETS,
    MoEConfig,
    MoEPreset,
    register_moe_preset,
    resolve_moe_config,
)
from .experts import RoutedExperts, SharedExpert, swiglu
from .layer import MoELayer, build_ffn, moe_layers, named_moe_layers
from .layout import (
    MOE_LAYOUTS,
    MoEStateCodec,
    MoEWeightLayout,
    get_moe_layout,
    register_moe_layout,
)
from .routing import SCORE_FUNCTIONS, RouterOutput, TopKRouter

__all__ = [
    "BalanceTracker",
    "MOE_LAYOUTS",
    "MOE_PRESETS",
    "MOE_PRESET_KEY",
    "MoEConfig",
    "MoELayer",
    "MoEPreset",
    "MoEStateCodec",
    "MoEStepTerms",
    "MoEWeightLayout",
    "RoutedExperts",
    "RouterOutput",
    "SCORE_FUNCTIONS",
    "SharedExpert",
    "TopKRouter",
    "apply_pending_balance_updates",
    "build_ffn",
    "get_moe_layout",
    "moe_layers",
    "named_moe_layers",
    "pop_moe_step_terms",
    "register_moe_layout",
    "register_moe_preset",
    "resolve_moe_config",
    "swiglu",
]
