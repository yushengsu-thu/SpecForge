# coding=utf-8
"""SGLang draft classes for DFlash-family drafters with an MoE FFN.

``DFlashMoEDraftModel``, ``DFlash2MoEDraftModel`` and ``Qwen3MoEDSparkModel``
are SGLang's ``DFlashDraftModel`` / ``DFlash2DraftModel`` / ``Qwen3DSparkModel``
with every decoder layer's dense MLP replaced by :class:`DraftMoEFFN`
(``moe_ffn``), the official per-expert checkpoint tensors stacked into the
grouped-GEMM layout at load time, and a strict check that the checkpoint's FFN
entries and the class agree. Attention, convolutions (DFlash2), candidate
selector (DFlash2) and Markov / confidence heads (DSpark) are inherited
unchanged, as is the speculative worker. The draft is replicated (TP1 draft
worker); expert parallelism is not implemented.

Registered through ``SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang_models``.
Written against SGLang v0.5.19+ (``DFlashDecoderLayer(config, layer_id,
attention_conv, mlp_conv, quant_config, prefix)``); the constructor passes the
base layer's arguments through, so older signatures without the convolution
arguments also work.
"""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import torch
from sglang.srt.models.dflash import (
    DFlash2DraftModel,
    DFlashDecoderLayer,
    DFlashDraftModel,
)
from sglang.srt.models.dspark import Qwen3DSparkModel
from torch import nn

from .moe_ffn import (
    DraftMoEFFN,
    routed_expert_count,
    stack_expert_weights,
    verify_moe_weights,
)

logger = logging.getLogger(__name__)


class DFlashMoEDecoderLayer(DFlashDecoderLayer):
    """``DFlashDecoderLayer`` with the dense MLP replaced by :class:`DraftMoEFFN`."""

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        # The base layer built a dense MLP the export does not carry; swap it
        # for the MoE FFN before any weights are loaded.
        del self.mlp
        self.mlp = build_draft_moe_ffn(config)


class _MoEDraftMixin:
    """Shared constructor check, logging and strict expert loading."""

    decoder_layer_cls = DFlashMoEDecoderLayer

    def __init__(self, config, *args, **kwargs) -> None:
        if routed_expert_count(config) <= 0:
            raise ValueError(
                f"{type(self).__name__} requires n_routed_experts > 0 in the draft "
                "config; use the dense draft class for dense drafts."
            )
        super().__init__(config, *args, **kwargs)
        ffn = self.layers[0].mlp
        logger.info(
            "MoE draft (%s, backend=%s): %s",
            type(self).__name__,
            "fused" if isinstance(ffn, FusedDraftMoEFFN) else "grouped_mm",
            ffn.describe(),
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Per-expert checkpoint tensors -> stacked module parameters, then the
        # base loader; then refuse an FFN that only partially matched.
        stacked = stack_expert_weights(weights)
        if isinstance(self.layers[0].mlp, FusedDraftMoEFFN):
            stacked = merge_gate_up(stacked)
        provided = {name.removeprefix("model.") for name, _ in stacked}
        super().load_weights(iter(stacked))
        verify_moe_weights(self, provided, type(self).__name__)


class DFlashMoEDraftModel(_MoEDraftMixin, DFlashDraftModel):
    """DFlash draft with a DeepSeek-/Qwen3-style MoE FFN."""


class DFlash2MoEDraftModel(_MoEDraftMixin, DFlash2DraftModel):
    """DFlash2 draft (grouped convolutions + candidate selector) with an MoE FFN."""


class Qwen3MoEDSparkModel(_MoEDraftMixin, Qwen3DSparkModel):
    """DSpark draft with Qwen3-style attention and an MoE FFN."""


EntryClass = [DFlashMoEDraftModel, DFlash2MoEDraftModel, Qwen3MoEDSparkModel]
