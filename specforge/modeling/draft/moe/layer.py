# coding=utf-8
"""The MoE FFN layer and the constructor seam draft models adopt it through."""

from __future__ import annotations

from typing import Any, Callable, List, Tuple

import torch
from torch import nn
from transformers import initialization as init

from .balance import BalanceTracker
from .config import MoEConfig, resolve_moe_config
from .experts import RoutedExperts, SharedExpert
from .routing import TopKRouter


class MoELayer(nn.Module):
    """Top-k routed SwiGLU experts plus an optional, optionally gated, shared expert.

    Parameter names follow the fused HF layout (see :mod:`.layout`). Hosts
    initialize it with :meth:`reset_parameters` from ``_init_weights`` and,
    for training, call :func:`apply_pending_balance_updates` first in their
    forward; in training mode each forward records its routing for load
    balancing.
    """

    def __init__(self, config: MoEConfig, hidden_size: int) -> None:
        super().__init__()
        if not isinstance(config, MoEConfig):
            raise TypeError(f"MoELayer needs a MoEConfig, got {type(config).__name__}")
        self.moe_config = config
        self.hidden_size = int(hidden_size)
        self.gate = TopKRouter(config, self.hidden_size)
        self.experts = RoutedExperts(config, self.hidden_size)
        self.shared_expert = None
        self.shared_expert_gate = None
        if config.shared_expert_intermediate_size:
            self.shared_expert = SharedExpert(
                self.hidden_size,
                config.shared_expert_intermediate_size,
                config.swiglu_limit,
            )
            if config.shared_expert_gate:
                self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)
        self.balance = BalanceTracker(config)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 0.02) -> None:
        """Normal(0, ``std``) weights (HF ``initializer_range``); zero selection bias."""

        for parameter in self.parameters():
            init.normal_(parameter, mean=0.0, std=std)
        if self.gate.e_score_correction_bias is not None:
            init.zeros_(self.gate.e_score_correction_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Map ``[batch, tokens, hidden]`` (or ``[tokens, hidden]``) to the same shape."""

        routing = self.gate(hidden_states)
        tokens = hidden_states.reshape(-1, self.hidden_size)
        output = self.experts(tokens, routing.indices, routing.weights)
        if self.shared_expert is not None:
            shared = self.shared_expert(tokens)
            if self.shared_expert_gate is not None:
                shared = torch.sigmoid(self.shared_expert_gate(tokens)) * shared
            output = output + shared
        if self.training:
            self.balance.record(routing)
        return output.view_as(hidden_states)

    def extra_repr(self) -> str:
        return f"preset={self.moe_config.preset}, layout={self.moe_config.layout}"


def build_ffn(config: Any, *, dense: Callable[[Any], nn.Module]) -> nn.Module:
    """A decoder layer's FFN: an :class:`MoELayer` when the draft config names a
    ``moe_preset``, otherwise ``dense(config)`` exactly as before."""

    moe_config = resolve_moe_config(config)
    if moe_config is None:
        return dense(config)
    return MoELayer(moe_config, int(config.hidden_size))


def named_moe_layers(model: nn.Module) -> List[Tuple[str, MoELayer]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, MoELayer)
    ]


def moe_layers(model: nn.Module) -> List[MoELayer]:
    return [module for module in model.modules() if isinstance(module, MoELayer)]


__all__ = ["MoELayer", "build_ffn", "moe_layers", "named_moe_layers"]
