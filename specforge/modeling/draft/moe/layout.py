# coding=utf-8
"""Weight naming: convert MoE layer state between module and checkpoint names.

Inside the module every MoE layer uses the fused HF names, which are also the
official Qwen3.5-MoE / Qwen3.8 checkpoint names (``E`` experts, hidden ``H``,
expert width ``I``)::

    gate.weight                        [E, H]
    gate.e_score_correction_bias       [E]  fp32, selection-bias presets only
    experts.gate_up_proj               [E, 2I, H]  gate rows, then up rows
    experts.down_proj                  [E, H, I]
    shared_expert.{gate,up,down}_proj.weight
    shared_expert_gate.weight          [1, H]

A preset names the layout of its upstream checkpoints, and every place that
reads or writes draft weights converts through :class:`MoEStateCodec`, so
SpecForge checkpoints and exports carry upstream names whatever the module
layout is. A layout converts one layer's tensors keyed relative to the layer:
a per-expert Qwen3-MoE layout splits ``experts.gate_up_proj`` into
``experts.{i}.gate_proj.weight`` / ``experts.{i}.up_proj.weight``; a
DeepSeek-V4 layout renames ``gate.e_score_correction_bias`` to ``gate.bias``,
``experts.{i}.w1/w2/w3`` and ``shared_experts.*``, and quantizes experts to
FP4 with ``.scale`` tensors on the way out.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
from torch import nn

from .config import MoEConfig

TensorDict = Dict[str, torch.Tensor]


class MoEWeightLayout:
    """One upstream checkpoint naming of an MoE layer's tensors."""

    name: str = ""
    #: Module and checkpoint names coincide; codecs skip the conversion.
    is_identity: bool = False

    def to_checkpoint(self, tensors: TensorDict, config: MoEConfig) -> TensorDict:
        """Module-relative names -> checkpoint-relative names for one layer."""
        raise NotImplementedError

    def from_checkpoint(self, tensors: TensorDict, config: MoEConfig) -> TensorDict:
        """Checkpoint-relative names -> module-relative names for one layer."""
        raise NotImplementedError

    def config_fields(self, config: MoEConfig) -> Dict[str, Any]:
        """Upstream config fields describing ``config`` for exported configs."""
        raise NotImplementedError


class FusedLayout(MoEWeightLayout):
    """HF fused naming (Qwen3.5-MoE, Qwen3.8): identical to the module."""

    name = "hf_fused"
    is_identity = True

    def to_checkpoint(self, tensors: TensorDict, config: MoEConfig) -> TensorDict:
        return dict(tensors)

    def from_checkpoint(self, tensors: TensorDict, config: MoEConfig) -> TensorDict:
        return dict(tensors)

    def config_fields(self, config: MoEConfig) -> Dict[str, Any]:
        return {
            "num_experts": config.n_routed_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "moe_intermediate_size": config.moe_intermediate_size,
            "shared_expert_intermediate_size": config.shared_expert_intermediate_size,
            "norm_topk_prob": config.norm_topk_prob,
            "router_aux_loss_coef": config.aux_loss_coeff,
        }


MOE_LAYOUTS: Dict[str, MoEWeightLayout] = {}


def register_moe_layout(layout: MoEWeightLayout) -> MoEWeightLayout:
    """Register ``layout`` under its name; names are unique."""

    if not layout.name or layout.name in MOE_LAYOUTS:
        raise ValueError(f"MoE layout name {layout.name!r} is empty or registered")
    MOE_LAYOUTS[layout.name] = layout
    return layout


def get_moe_layout(name: str) -> MoEWeightLayout:
    layout = MOE_LAYOUTS.get(name)
    if layout is None:
        raise ValueError(
            f"unknown MoE layout {name!r}; available: {sorted(MOE_LAYOUTS)}"
        )
    return layout


register_moe_layout(FusedLayout())

# Wrapper modules that insert themselves into module names but not into the
# draft state-dict keys: FSDP1, activation checkpointing, torch.compile.
_WRAPPER_SEGMENTS = frozenset(
    {"_fsdp_wrapped_module", "_checkpoint_wrapped_module", "_orig_mod"}
)


def _state_prefix(module_name: str) -> str:
    return ".".join(
        part for part in module_name.split(".") if part not in _WRAPPER_SEGMENTS
    )


class MoEStateCodec:
    """Converts a draft's state dict between module and checkpoint names.

    Keys are relative to the draft module the codec was built from, the
    ``draft_state_dict`` convention. Dense drafts and identity layouts return
    the input mapping unchanged.
    """

    def __init__(self, layers: Mapping[str, MoEConfig]) -> None:
        self._layers: Dict[str, Tuple[MoEConfig, MoEWeightLayout]] = {
            prefix: (config, get_moe_layout(config.layout))
            for prefix, config in layers.items()
        }

    @classmethod
    def for_model(cls, model: nn.Module) -> "MoEStateCodec":
        """Codec for ``model``'s MoE layers (wrapper-free names)."""

        from .layer import named_moe_layers

        if not isinstance(model, nn.Module):
            return cls({})
        return cls(
            {
                _state_prefix(name): layer.moe_config
                for name, layer in named_moe_layers(model)
            }
        )

    @property
    def is_identity(self) -> bool:
        return all(layout.is_identity for _, layout in self._layers.values())

    def encode(self, state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Module names -> checkpoint names."""
        return self._convert(state_dict, to_checkpoint=True)

    def decode(self, state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Checkpoint names -> module names."""
        return self._convert(state_dict, to_checkpoint=False)

    def _convert(
        self, state_dict: Mapping[str, Any], *, to_checkpoint: bool
    ) -> Mapping[str, Any]:
        if self.is_identity:
            return state_dict
        converted = dict(state_dict)
        for prefix, (config, layout) in self._layers.items():
            head = f"{prefix}." if prefix else ""
            local = {
                key[len(head) :]: converted.pop(key)
                for key in [key for key in converted if key.startswith(head)]
            }
            if not local:
                continue
            convert = layout.to_checkpoint if to_checkpoint else layout.from_checkpoint
            converted.update(
                {head + key: value for key, value in convert(local, config).items()}
            )
        return converted


__all__ = [
    "FusedLayout",
    "MOE_LAYOUTS",
    "MoEStateCodec",
    "MoEWeightLayout",
    "get_moe_layout",
    "register_moe_layout",
]
