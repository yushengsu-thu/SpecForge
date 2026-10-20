"""Draft-independent MoE layer and the model/trainer loss boundary."""

from __future__ import annotations

import torch
from torch import nn
from transformers import initialization as init

from .config import MoEConfig
from .experts import RoutedExperts, SharedExpert
from .layout import validate_layout
from .routing import TopKRouter, auxiliary_load_balance


class MoELayer(nn.Module):
    """Qwen-style FFN: selected routed experts plus an always-active shared expert.

    ``forward`` keeps the dense FFN tensor contract. The trainer drains the
    auxiliary objective once through ``collect_moe_auxiliary_loss``. Nonreentrant
    activation checkpointing retains the auxiliary graph from the first forward;
    recomputation runs the same tensor operations but never appends another loss.
    """

    def __init__(self, config: MoEConfig, hidden_size: int):
        super().__init__()
        if not isinstance(config, MoEConfig):
            raise TypeError(
                "MoELayer requires resolve_moe_config(config) to select a preset"
            )
        if (
            not isinstance(hidden_size, int)
            or isinstance(hidden_size, bool)
            or hidden_size <= 0
        ):
            raise ValueError("hidden_size must be a positive integer")
        validate_layout(config.layout)
        self.config = config
        self.hidden_size = hidden_size
        self.gate = TopKRouter(config, hidden_size)
        self.experts = RoutedExperts(config, hidden_size)
        self.shared_expert = None
        self.shared_expert_gate = None
        if config.shared_expert_intermediate_size:
            self.shared_expert = SharedExpert(
                hidden_size, config.shared_expert_intermediate_size, config.swiglu_limit
            )
            if config.shared_expert_gate:
                self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)
        self._capture_auxiliary = True
        self._auxiliary_loss: torch.Tensor | None = None
        self._auxiliary_metrics: dict[str, torch.Tensor] = {}
        self.reset_parameters(0.02)

    def reset_parameters(self, std: float) -> None:
        """Initialize fused tensors while preserving parameters already loaded by HF.

        HF can initialize a partially loaded parent after its present children
        have been read. Its initializer respects their loaded-parameter flags.
        """
        with torch.no_grad():
            for parameter in self.parameters():
                init.normal_(parameter, mean=0.0, std=std)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"MoELayer expected hidden size {self.hidden_size}, got {hidden_states.shape[-1]}"
            )
        flat = hidden_states.reshape(-1, self.hidden_size)
        if flat.shape[0] == 0:
            raise ValueError("MoELayer requires at least one routed token")
        router_input = hidden_states
        if self.config.router_center == "sample":
            if hidden_states.ndim not in (2, 3):
                raise ValueError(
                    "sample router centering requires [tokens, hidden] or [batch, tokens, hidden]"
                )
            # Optional SpecForge training extension; not part of official Qwen.
            # Expert functions still receive the unmodified hidden activations.
            router_input = hidden_states - hidden_states.mean(dim=-2, keepdim=True)
        routing = self.gate(router_input.reshape(-1, self.hidden_size))
        output = self.experts(flat, routing)
        if self.shared_expert is not None:
            shared = self.shared_expert(flat)
            if self.shared_expert_gate is not None:
                shared = shared * torch.sigmoid(self.shared_expert_gate(flat))
            output = output + shared
        if self.training and torch.is_grad_enabled():
            # Tensor work is unconditional during recomputation; only storage is
            # switched off after the trainer drains the first-forward objective.
            auxiliary, metrics = auxiliary_load_balance(routing)
            weighted = auxiliary * self.config.aux_loss_coeff
            if self._capture_auxiliary:
                self._auxiliary_loss = weighted
                self._auxiliary_metrics = {
                    "aux_loss": auxiliary.detach(),
                    "aux_loss_weighted": weighted.detach(),
                    **{key: value.detach() for key, value in metrics.items()},
                }
        return output.reshape_as(hidden_states)


def apply_pending_balance_updates(model: nn.Module) -> None:
    """Start a forward outside checkpoint regions and clear previous records.

    Qwen uses differentiable auxiliary balancing, so there are no selection-bias
    mutations. This boundary is where a future bias-balancing family can apply
    pending updates exactly once, without checkpoint-recomputation side effects.
    """
    for module in model.modules():
        if isinstance(module, MoELayer):
            module._auxiliary_loss = None
            module._auxiliary_metrics = {}
            module._capture_auxiliary = True


def collect_moe_auxiliary_loss(
    model: nn.Module,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    """Drain each layer's weighted scalar once and aggregate detached telemetry."""
    losses = []
    metrics: dict[str, list[torch.Tensor]] = {}
    for module in model.modules():
        if not isinstance(module, MoELayer):
            continue
        if module._auxiliary_loss is not None:
            losses.append(module._auxiliary_loss)
            for key, value in module._auxiliary_metrics.items():
                metrics.setdefault(key, []).append(value)
        module._auxiliary_loss = None
        module._auxiliary_metrics = {}
        module._capture_auxiliary = False
    if not losses:
        return None, {}
    auxiliary = torch.stack(losses).sum()
    aggregated = {
        f"moe/{key}": (
            torch.stack(values).sum()
            if key.startswith("aux_loss")
            else torch.stack(values).mean()
        )
        for key, values in metrics.items()
    }
    return auxiliary, aggregated
