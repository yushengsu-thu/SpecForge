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
import os
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
    merge_gate_up,
    routed_expert_count,
    stack_expert_weights,
    to_native_names,
    verify_moe_weights,
)

logger = logging.getLogger(__name__)

#: ``SPECFORGE_DRAFT_MOE_BACKEND``: ``fused`` (SGLang's fused MoE Triton kernel,
#: default) or ``grouped_mm`` (the plain-PyTorch reference in ``moe_ffn``).
MOE_BACKEND_ENV = "SPECFORGE_DRAFT_MOE_BACKEND"

try:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    _FUSED_AVAILABLE = True
    _FUSED_IMPORT_ERROR: Exception | None = None
except Exception as _err:  # pragma: no cover - depends on the SGLang build
    _FUSED_AVAILABLE = False
    _FUSED_IMPORT_ERROR = _err


class FusedDraftMoEFFN(DraftMoEFFN):
    """``DraftMoEFFN`` whose routed experts run through SGLang's fused MoE kernel.

    Routing (score function, selection bias, renormalisation, scaling) stays in
    :meth:`DraftMoEFFN.route`, so the serving recipe is unchanged; only the
    expert GEMMs, the SwiGLU and the weighted combine move into
    ``fused_experts`` (permute + grouped GEMM + activation + unpermute, with
    tuned per-shape Triton configs). The experts are stored in the kernel's
    layout: ``w13`` ``[E, 2N, K]`` (gate rows then up rows) and ``w2``
    ``[E, K, N]``; :func:`merge_gate_up` builds ``w13`` at load time.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        if self.swiglu_limit not in (0.0, 10.0):
            # The kernel implements only DeepSeek-V4's clamp at 10.
            raise ValueError(
                "fused MoE backend supports swiglu_limit 0 or 10, "
                f"got {self.swiglu_limit}"
            )
        e, d, i = self.n_experts, self.hidden_size, self.intermediate_size
        del self.experts.w1
        del self.experts.w3
        self.experts.w13 = nn.Parameter(torch.empty(e, 2 * i, d))
        self._runner_config = MoeRunnerConfig(
            num_experts=e,
            num_local_experts=e,
            hidden_size=d,
            intermediate_size_per_partition=i,
            top_k=self.topk,
            params_dtype=torch.get_default_dtype(),
            activation="silu",
            is_gated=True,
            inplace=False,
            no_combine=False,
            # route() already renormalises and scales the combine weights.
            routed_scaling_factor=None,
            gate_up_interleaved=False,
            swiglu_limit=self.swiglu_limit if self.swiglu_limit > 0 else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.hidden_size)
        weights, indices, _ = self.route(x)
        y = fused_experts(
            x,
            self.experts.w13,
            self.experts.w2,
            StandardTopKOutput(weights, indices.to(torch.int32), None),
            self._runner_config,
        )
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y.view(shape)


def moe_backend() -> str:
    backend = os.environ.get(MOE_BACKEND_ENV, "fused").strip().lower()
    if backend not in ("fused", "grouped_mm"):
        raise ValueError(
            f"{MOE_BACKEND_ENV} must be 'fused' or 'grouped_mm', got {backend!r}"
        )
    if backend == "fused" and not _FUSED_AVAILABLE:
        logger.warning(
            "fused MoE backend unavailable (%s); using grouped_mm",
            _FUSED_IMPORT_ERROR,
        )
        backend = "grouped_mm"
    return backend


def build_draft_moe_ffn(config) -> DraftMoEFFN:
    """The layer FFN for the selected backend (see :data:`MOE_BACKEND_ENV`)."""
    return FusedDraftMoEFFN(config) if moe_backend() == "fused" else DraftMoEFFN(config)


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
        stacked = stack_expert_weights(to_native_names(weights))
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


class Qwen3MoeDSparkModel(Qwen3MoEDSparkModel):
    """Alias: the architecture name SpecForge's ``kan/moe-3-qwen38`` exports
    write (Qwen3.8-27B DSpark MoE drafter, ``qwen3_5_moe`` preset, Qwen
    checkpoint naming, folded router centering in ``gate.bias``)."""


EntryClass = [
    DFlashMoEDraftModel,
    DFlash2MoEDraftModel,
    Qwen3MoEDSparkModel,
    Qwen3MoeDSparkModel,
]
