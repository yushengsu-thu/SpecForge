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

import dataclasses
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
    shared_expert_as_experts,
    stack_expert_weights,
    to_native_names,
    verify_moe_weights,
)

logger = logging.getLogger(__name__)

#: ``SPECFORGE_DRAFT_MOE_BACKEND``: ``fused`` (SGLang's fused MoE Triton kernel,
#: default) or ``grouped_mm`` (the plain-PyTorch reference in ``moe_ffn``).
MOE_BACKEND_ENV = "SPECFORGE_DRAFT_MOE_BACKEND"
#: ``SPECFORGE_DRAFT_MOE_EXPERT_DTYPE``: ``bf16`` (default) or ``fp8``. With
#: ``fp8`` the fused backend quantises the routed experts to float8 e4m3 with
#: one scale per output channel right after loading and runs the kernel's
#: W8A8 path (activations quantised per token on the fly). Halves the expert
#: bytes each draft step reads; the export stays bf16.
MOE_EXPERT_DTYPE_ENV = "SPECFORGE_DRAFT_MOE_EXPERT_DTYPE"
#: ``SPECFORGE_DRAFT_MOE_FUSE_SHARED``: ``1`` folds the shared expert into the
#: fused kernel as ``S // N`` extra experts every token routes to (combine
#: weight = the sigmoid gate, or 1), removing its separate GEMMs. Exact up to
#: summation order. Off by default.
MOE_FUSE_SHARED_ENV = "SPECFORGE_DRAFT_MOE_FUSE_SHARED"
#: ``SPECFORGE_DRAFT_MOE_FUSE_SHARED_MAX_TOKENS``: the folded shared expert is
#: used for steps of at most this many tokens (small batches, where the extra
#: kernel launches dominate); larger steps keep the separate shared GEMMs,
#: which are more efficient there. Default 64.
MOE_FUSE_SHARED_MAX_TOKENS_ENV = "SPECFORGE_DRAFT_MOE_FUSE_SHARED_MAX_TOKENS"

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

    _FP8_MAX = 448.0  # float8_e4m3fn

    @staticmethod
    def _quantize_per_channel(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``w [E, N, K]`` (bf16) -> (e4m3 ``[E, N, K]``, fp32 scale ``[E, N]``) with
        one absmax scale per expert output channel, the layout the fused kernel
        reads for ``per_channel_quant=True``."""
        scale = (
            w.float().abs().amax(dim=-1).clamp_min(1e-12) / FusedDraftMoEFFN._FP8_MAX
        )
        q = (w.float() / scale.unsqueeze(-1)).clamp(
            -FusedDraftMoEFFN._FP8_MAX, FusedDraftMoEFFN._FP8_MAX
        )
        return q.to(torch.float8_e4m3fn), scale.contiguous()

    def fuse_shared_expert(self) -> None:
        """Fold the shared expert into the routed-expert tensors (see
        :func:`shared_expert_as_experts`). Must run before any fp8 quantisation."""
        if self.shared_experts is None or getattr(self, "_fused_shared", 0):
            return
        if getattr(self, "_fp8", False):
            raise RuntimeError(
                "fuse_shared_expert must run before quantize_experts_fp8"
            )
        w13x, w2x = shared_expert_as_experts(
            self.shared_experts, self.intermediate_size
        )
        m = w13x.shape[0]
        w13 = torch.cat([self.experts.w13.data, w13x.to(self.experts.w13.dtype)], dim=0)
        w2 = torch.cat([self.experts.w2.data, w2x.to(self.experts.w2.dtype)], dim=0)
        del self.experts.w13
        del self.experts.w2
        self.experts.w13 = nn.Parameter(w13, requires_grad=False)
        self.experts.w2 = nn.Parameter(w2, requires_grad=False)
        # The separate shared expert stays for large steps (see forward).
        self._shared_gate = self.shared_experts.gate  # Linear [1, H] or None
        self._fused_shared = m
        self._fuse_shared_max_tokens = int(
            os.environ.get(MOE_FUSE_SHARED_MAX_TOKENS_ENV, "64")
        )
        # One runner config per path: the kernel sizes its token alignment
        # from top_k, so the folded path advertises top_k + m.
        self._runner_config_folded = dataclasses.replace(
            self._runner_config,
            top_k=self.topk + m,
            num_experts=self.n_experts + m,
            num_local_experts=self.n_experts + m,
        )
        torch.cuda.empty_cache()

    def quantize_experts_fp8(self) -> None:
        """Replace the bf16 routed experts by fp8 weights plus per-channel scales.

        Called after the checkpoint is loaded and verified; the shared expert
        and the router stay bf16/fp32.
        """
        if getattr(self, "_fp8", False):
            return
        w13, s13 = self._quantize_per_channel(self.experts.w13.data)
        w2, s2 = self._quantize_per_channel(self.experts.w2.data)
        del self.experts.w13
        del self.experts.w2
        self.experts.register_buffer("w13", w13)
        self.experts.register_buffer("w2", w2)
        self.experts.register_buffer("w13_scale", s13)
        self.experts.register_buffer("w2_scale", s2)
        self._fp8 = True
        torch.cuda.empty_cache()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.hidden_size)
        weights, indices, _ = self.route(x)
        m = getattr(self, "_fused_shared", 0)
        tokens = x.shape[0]
        fold = bool(m) and tokens <= self._fuse_shared_max_tokens
        runner_config = self._runner_config_folded if fold else self._runner_config
        if fold:
            # Every token also visits the m shared-expert pieces, weighted by
            # the per-token sigmoid gate (or 1 when the shared expert is ungated).
            extra_ids = torch.arange(
                self.n_experts, self.n_experts + m, device=x.device, dtype=indices.dtype
            ).expand(tokens, m)
            gate = (
                torch.sigmoid(self._shared_gate(x).float())
                if self._shared_gate is not None
                else torch.ones(tokens, 1, device=x.device, dtype=torch.float32)
            )
            indices = torch.cat([indices, extra_ids], dim=1)
            weights = torch.cat([weights, gate.expand(tokens, m)], dim=1)
        kwargs = {}
        if getattr(self, "_fp8", False):
            kwargs = dict(
                use_fp8_w8a8=True,
                per_channel_quant=True,
                w1_scale=self.experts.w13_scale,
                w2_scale=self.experts.w2_scale,
            )
        y = fused_experts(
            x,
            self.experts.w13,
            self.experts.w2,
            StandardTopKOutput(weights, indices.to(torch.int32), None),
            runner_config,
            **kwargs,
        )
        if self.shared_experts is not None and not fold:
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
        if os.environ.get(MOE_FUSE_SHARED_ENV, "0").strip() == "1":
            if not isinstance(self.layers[0].mlp, FusedDraftMoEFFN):
                raise ValueError(f"{MOE_FUSE_SHARED_ENV}=1 needs the fused backend")
            for layer in self.layers:
                layer.mlp.fuse_shared_expert()
            m = getattr(self.layers[0].mlp, "_fused_shared", 0)
            if m:
                logger.info(
                    "MoE draft: shared expert folded into the fused kernel as %d "
                    "extra expert(s); kernel now sees E=%d, top-%d",
                    m,
                    self.layers[0].mlp.n_experts + m,
                    self.layers[0].mlp.topk + m,
                )
        dtype = os.environ.get(MOE_EXPERT_DTYPE_ENV, "bf16").strip().lower()
        if dtype == "fp8":
            if not isinstance(self.layers[0].mlp, FusedDraftMoEFFN):
                raise ValueError(
                    f"{MOE_EXPERT_DTYPE_ENV}=fp8 needs the fused backend "
                    f"({MOE_BACKEND_ENV}=fused)"
                )
            before = torch.cuda.memory_allocated()
            for layer in self.layers:
                layer.mlp.quantize_experts_fp8()
            logger.info(
                "MoE draft experts quantised to fp8 e4m3 (per-channel scales); "
                "freed %.1f GB",
                (before - torch.cuda.memory_allocated()) / 1e9,
            )
        elif dtype != "bf16":
            raise ValueError(
                f"{MOE_EXPERT_DTYPE_ENV} must be bf16 or fp8, got {dtype!r}"
            )


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
