# coding=utf-8
"""Load balancing for draft MoE layers.

Presets pick the schemes a draft trains with:

* **Aux loss** (Switch Transformer). Per layer and training forward,
  ``aux = E * sum_e f_e * P_e``: ``f_e`` is the fraction of the ``T * K``
  routed assignments sent to expert ``e`` and ``P_e`` its mean normalized
  score, so a perfectly balanced layer scores 1.0. The trainer adds
  ``moe_aux_loss_coeff * sum_layers aux`` to the step objective
  (``finalize_step_output``).
* **Selection bias** (DeepSeek's aux-loss-free balancing). An fp32 per-expert
  bias shifts only top-k selection. Training forwards accumulate expert load;
  the next training forward first applies ``bias += rate * sign(mean - load)``
  with the load summed over ranks (:func:`apply_pending_balance_updates`).

The update is deferred to the start of the next training forward, outside any
activation-checkpointed region, so a recomputed forward routes with exactly
the bias its original forward used. Recomputed forwards record nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from .config import MoEConfig
from .routing import RouterOutput


def expert_load(indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """``[E]`` int64 count of routed assignments per expert, without host syncs."""

    flat = indices.reshape(-1)
    return torch.zeros(num_experts, dtype=torch.long, device=flat.device).scatter_add_(
        0, flat, torch.ones_like(flat)
    )


def switch_aux_loss(scores: torch.Tensor, load: torch.Tensor) -> torch.Tensor:
    """Switch Transformer balance loss for one layer; 1.0 when balanced.

    ``scores`` is ``[T, E]``; non-softmax scores are normalized per token
    before averaging, as in DeepSeek's sequence-wise balance loss.
    """

    probs = scores / scores.sum(dim=-1, keepdim=True)
    fraction = load.to(probs.dtype) / load.sum().clamp_min(1)
    return scores.shape[-1] * torch.dot(fraction, probs.mean(dim=0))


@torch.no_grad()
def selection_bias_update(bias: torch.Tensor, load: torch.Tensor, rate: float) -> None:
    """Raise the bias of under-loaded experts and lower it for over-loaded ones."""

    load = load.to(bias.dtype)
    bias.add_(torch.sign(load.mean() - load), alpha=rate)


def _recomputing() -> bool:
    """True inside a backward pass, i.e. an activation-checkpoint recompute."""

    current_task = getattr(torch._C, "_current_graph_task_id", None)
    return current_task is not None and current_task() != -1


class RoutingRecord(NamedTuple):
    """One layer's routing in one training forward."""

    #: Switch aux loss with grad, or None when the coefficient is zero.
    aux_loss: Optional[torch.Tensor]
    #: ``[E]`` int64 routed assignments per expert.
    load: torch.Tensor


class BalanceTracker:
    """Per-layer routing records and the load awaiting a selection-bias update."""

    def __init__(self, config: MoEConfig) -> None:
        self.num_experts = config.n_routed_experts
        self.aux_loss_coeff = config.aux_loss_coeff
        self.bias_update_rate = config.bias_update_rate
        self.records: List[RoutingRecord] = []
        self.pending_load: Optional[torch.Tensor] = None

    def record(self, routing: RouterOutput) -> None:
        # Compute in every pass so an activation-checkpoint recompute saves the
        # same tensors as its original forward, but keep only the original.
        load = expert_load(routing.indices, self.num_experts)
        aux_loss = None
        if self.aux_loss_coeff > 0:
            aux_loss = switch_aux_loss(routing.scores, load)
        if _recomputing():
            return
        self.records.append(RoutingRecord(aux_loss=aux_loss, load=load))
        if self.bias_update_rate > 0:
            pending = self.pending_load
            self.pending_load = load if pending is None else pending + load

    def take_records(self) -> List[RoutingRecord]:
        records, self.records = self.records, []
        return records


@dataclass(frozen=True)
class MoEStepTerms:
    """What a training forward's MoE layers add to the training step."""

    #: ``sum over layer forwards of coeff * aux`` with grad, or None.
    objective: Optional[torch.Tensor]
    #: ``moe/*`` metrics as ``(numerator, denominator)`` pairs.
    ratio_metrics: Dict[str, Tuple[torch.Tensor, torch.Tensor]]


def pop_moe_step_terms(model: nn.Module) -> Optional[MoEStepTerms]:
    """Collect and clear the routing records of ``model``'s MoE layers.

    Metrics average over layer forwards: ``moe/aux_loss`` and
    ``moe/max_load`` (busiest expert's load over the balanced load) are 1.0
    when balanced; ``moe/unused_experts`` is the fraction of experts that
    received no token. Returns None when nothing was recorded.
    """

    from .layer import moe_layers

    objective = None
    aux_sum = max_load_sum = unused_sum = None
    forwards = 0
    for layer in moe_layers(model):
        num_experts = layer.balance.num_experts
        for record in layer.balance.take_records():
            forwards += 1
            load = record.load.float()
            max_load = load.max() * num_experts / load.sum().clamp_min(1)
            unused = (record.load == 0).float().mean()
            max_load_sum = max_load if max_load_sum is None else max_load_sum + max_load
            unused_sum = unused if unused_sum is None else unused_sum + unused
            if record.aux_loss is None:
                continue
            if not record.aux_loss.requires_grad and layer.gate.weight.requires_grad:
                raise RuntimeError(
                    "MoE aux loss lost its gradient; a reentrant activation "
                    "checkpoint runs the routed forward without grad. Use "
                    "non-reentrant checkpointing (use_reentrant=False)."
                )
            weighted = layer.balance.aux_loss_coeff * record.aux_loss
            objective = weighted if objective is None else objective + weighted
            detached = record.aux_loss.detach()
            aux_sum = detached if aux_sum is None else aux_sum + detached
    if forwards == 0:
        return None
    # new_full fills on the device; a host tensor would force a H2D copy.
    count = max_load_sum.new_full((), float(forwards))
    ratio_metrics = {
        "moe/max_load": (max_load_sum, count),
        "moe/unused_experts": (unused_sum, count),
    }
    if aux_sum is not None:
        ratio_metrics["moe/aux_loss"] = (aux_sum, count)
    return MoEStepTerms(objective=objective, ratio_metrics=ratio_metrics)


def apply_pending_balance_updates(model: nn.Module) -> None:
    """Start a training forward of ``model``'s MoE layers.

    Applies the selection-bias updates deferred from previous training
    forwards (load summed across ranks, so every rank keeps an identical
    bias) and drops routing records no trainer consumed. Call it first in a
    draft's training forward, outside activation checkpointing; it is a
    no-op in eval mode and for dense drafts.
    """

    if not model.training:
        return
    from .layer import moe_layers

    layers = moe_layers(model)
    pending = []
    for layer in layers:
        layer.balance.records.clear()
        if layer.balance.pending_load is not None:
            pending.append(layer)
    if not pending:
        return
    loads = torch.stack([layer.balance.pending_load for layer in pending])
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(loads)
    for layer, load in zip(pending, loads):
        bias = layer.gate.e_score_correction_bias
        selection_bias_update(
            bias, load.to(bias.device), layer.balance.bias_update_rate
        )
        layer.balance.pending_load = None


__all__ = [
    "BalanceTracker",
    "MoEStepTerms",
    "RoutingRecord",
    "apply_pending_balance_updates",
    "expert_load",
    "pop_moe_step_terms",
    "selection_bias_update",
    "switch_aux_loss",
]
