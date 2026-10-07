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

``SPECFORGE_DRAFT_MOE_BACKEND`` picks how the FFN runs (see :func:`moe_backend`):
``fused`` (default: :class:`FusedDraftMoEFFN`, SGLang's fused MoE Triton kernel
driven by SpecForge's own router), ``sglang`` (:class:`SglangMoEFFN`, SGLang's
``TopK`` + ``FusedMoE`` modules and linear layers, the path a native SGLang
Qwen-MoE block runs) or ``grouped_mm`` (the plain-PyTorch reference).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
from typing import Dict, Iterable, List, Set, Tuple

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
    to_sglang_module_entries,
    verify_moe_weights,
)

logger = logging.getLogger(__name__)

#: ``SPECFORGE_DRAFT_MOE_BACKEND``: ``fused`` (SGLang's fused MoE Triton kernel
#: behind SpecForge's router, default), ``sglang`` (SGLang's own ``TopK`` +
#: ``FusedMoE`` modules, Qwen recipe only) or ``grouped_mm`` (the plain-PyTorch
#: reference in ``moe_ffn``).
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

try:
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import TopK
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8Config

    _SGLANG_MODULES_AVAILABLE = True
    _SGLANG_MODULES_IMPORT_ERROR: Exception | None = None
except Exception as _err:  # pragma: no cover - depends on the SGLang build
    _SGLANG_MODULES_AVAILABLE = False
    _SGLANG_MODULES_IMPORT_ERROR = _err


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


class SglangSharedExpert(nn.Module):
    """SwiGLU shared expert on SGLang's linear layers: ``gate_up_proj``
    ``[2S, H]`` (gate rows then up rows, a ``MergedColumnParallelLinear``),
    fused ``SiluAndMul`` and ``down_proj`` ``[H, S]``, optionally gated per
    token by ``sigmoid(gate(x))``. What ``Qwen2MoeSparseMoeBlock``'s
    ``shared_expert`` / ``shared_expert_gate`` run in SGLang."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, gated: bool, prefix: str = ""
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=None,
            reduce_results=False,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()
        self.gate = nn.Linear(hidden_size, 1, bias=False) if gated else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gate_up_proj(x)
        y, _ = self.down_proj(self.act_fn(h))
        if self.gate is not None:
            y = torch.sigmoid(self.gate(x).float()).to(y.dtype) * y
        return y


class SglangMoEFFN(DraftMoEFFN):
    """``DraftMoEFFN`` rebuilt from SGLang's own MoE modules.

    This is the path a native SGLang Qwen-MoE block runs
    (``Qwen2MoeSparseMoeBlock``: ``TopK`` + ``FusedMoE`` + a separate
    sigmoid-gated shared MLP), i.e. what an SGLang-side ``FusedMoE`` loader for
    the same export executes. The routed-expert GEMMs are the same fused Triton
    kernel and tuned per-shape config as :class:`FusedDraftMoEFFN`; the
    difference is that top-k selection, the shared expert's SwiGLU and the
    weighted combine run in SGLang's fused kernels, which halves the number of
    launches per layer (18 vs 40 at 7 tokens for the Qwen3.8-27B DSpark MoE
    shape).

    Supports the Qwen recipe only (``qwen3_5_moe``, or ``qwen3`` without a
    selection bias): softmax scoring, renormalised top-k, scale 1, no expert
    groups, no SwiGLU clamp. The router logits (fp32, with the folded router
    centering ``gate.bias`` added) come from :class:`DraftMoEGate` exactly as in
    the reference, so routing stays identical to training; only SGLang's
    ``TopK`` kernel does the selection and renormalisation. The draft is
    replicated (TP1), so checkpoint tensors are copied straight into the
    modules (:meth:`load_native_tensors`) instead of going through SGLang's
    sharded weight loaders.

    ``expert_dtype="fp8"`` (``SPECFORGE_DRAFT_MOE_EXPERT_DTYPE=fp8``) builds the
    ``FusedMoE`` with SGLang's ``W8A8Fp8`` MoE method (float8 e4m3 expert
    weights with one scale per output channel, activations quantised per token
    on the fly) and quantises the bf16 checkpoint experts at load time; the
    router and the shared expert stay bf16/fp32.
    """

    def __init__(self, config, layer_id: int = 0, expert_dtype: str = "bf16") -> None:
        super().__init__(config)
        if expert_dtype not in ("bf16", "fp8"):
            raise ValueError(
                f"{MOE_EXPERT_DTYPE_ENV} must be bf16 or fp8, got {expert_dtype!r}"
            )
        self._fp8 = expert_dtype == "fp8"
        if not _SGLANG_MODULES_AVAILABLE:
            raise RuntimeError(
                f"{MOE_BACKEND_ENV}=sglang needs SGLang's MoE modules "
                f"({_SGLANG_MODULES_IMPORT_ERROR})"
            )
        unsupported = []
        if self.scoring_func != "softmax":
            unsupported.append(f"scoring_func={self.scoring_func}")
        if self.gate.bias_mode == "selection":
            unsupported.append("topk_method=noaux_tc (selection bias)")
        if self.n_group != 1 or self.topk_group != 1:
            unsupported.append("expert groups")
        if self.swiglu_limit != 0.0:
            unsupported.append(f"swiglu_limit={self.swiglu_limit}")
        if self.routed_scaling_factor != 1.0:
            unsupported.append(f"routed_scaling_factor={self.routed_scaling_factor}")
        if unsupported:
            raise ValueError(
                f"{MOE_BACKEND_ENV}=sglang supports the Qwen MoE recipe only "
                f"(softmax, renormalised top-k, no selection bias); this draft "
                f"needs: {', '.join(unsupported)}. Use {MOE_BACKEND_ENV}=fused."
            )
        e, d, i = self.n_experts, self.hidden_size, self.intermediate_size
        prefix = f"layers.{layer_id}.mlp"
        del self.experts
        self.router = TopK(
            top_k=self.topk, renormalize=self.norm_topk_prob, layer_id=layer_id
        )
        self.experts = FusedMoE(
            num_experts=e,
            hidden_size=d,
            intermediate_size=i,
            layer_id=layer_id,
            top_k=self.topk,
            params_dtype=torch.get_default_dtype(),
            quant_config=(
                W8A8Fp8Config(is_checkpoint_fp8_serialized=True) if self._fp8 else None
            ),
            prefix=f"{prefix}.experts",
            inplace=False,
            # Only read by runner backends that route inside the kernel
            # (flashinfer TRT-LLM on Blackwell): softmax -> top-k -> renormalise,
            # the Qwen recipe; the Triton runner uses TopK's output instead.
            routing_method_type=RoutingMethodType.RenormalizeNaive,
            # w13 holds the gate rows then the up rows (merge_gate_up), not
            # interleaved pairs; runner backends that care (TRT-LLM) read this.
            gate_up_interleaved=False,
        )
        if self.shared_experts is not None:
            gated = self.shared_experts.gate is not None
            width = self.shared_experts.w1.out_features
            del self.shared_experts
            self.shared_experts = SglangSharedExpert(
                d, width, gated, prefix=f"{prefix}.shared_experts"
            )

    def route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same contract as :meth:`DraftMoEFFN.route`, computed by SGLang's
        ``TopK`` kernel on the reference router logits."""
        out = self.router(x, self.gate(x))
        if getattr(out, "topk_ids", None) is None:
            # Runner backends that route inside the kernel (flashinfer TRT-LLM)
            # bypass TopK; report the reference routing instead.
            return super().route(x)
        indices = out.topk_ids.long()
        flat = indices.flatten()
        counts = torch.zeros(
            self.n_experts, dtype=torch.long, device=x.device
        ).scatter_add_(0, flat, torch.ones_like(flat))
        return out.topk_weights.float(), indices, counts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.hidden_size)
        shared = self.shared_experts(x) if self.shared_experts is not None else None
        y = self.experts(x, self.router(x, self.gate(x)))
        if isinstance(y, tuple):  # some SGLang builds return (out, bias)
            y = y[0]
        if shared is not None:
            y = y + shared
        return y.view(shape)

    @staticmethod
    def _copy_padded_expert(
        name: str, param: torch.Tensor, tensor: torch.Tensor
    ) -> bool:
        """Some runner backends (flashinfer TRT-LLM) pad the expert width to a
        multiple of 128: ``w13_weight`` is ``[E, 2N', K]`` with the gate rows
        at ``[0:N]`` and the up rows at ``[N':N'+N]``, ``w2_weight`` is
        ``[E, K, N']``. Copy the checkpoint's ``N`` columns/rows into place and
        zero the padding; returns False for any other mismatch."""
        if param.dtype != tensor.dtype and param.dtype not in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            return False
        if (
            name == "experts.w13_weight"
            and param.dim() == 3
            and param.shape[0] == tensor.shape[0]
            and param.shape[2] == tensor.shape[2]
            and param.shape[1] > tensor.shape[1]
        ):
            n, n_pad = tensor.shape[1] // 2, param.shape[1] // 2
            param.zero_()
            param[:, :n].copy_(tensor[:, :n].to(param.dtype))
            param[:, n_pad : n_pad + n].copy_(tensor[:, n:].to(param.dtype))
            return True
        if (
            name == "experts.w2_weight"
            and param.dim() == 3
            and param.shape[:2] == tensor.shape[:2]
            and param.shape[2] > tensor.shape[2]
        ):
            param.zero_()
            param[:, :, : tensor.shape[2]].copy_(tensor.to(param.dtype))
            return True
        return False

    def load_native_tensors(self, entries: Dict[str, torch.Tensor]) -> Set[str]:
        """Copy one layer's checkpoint entries (names relative to the FFN, after
        :func:`stack_expert_weights` and :func:`merge_gate_up`) into the SGLang
        modules; returns the parameter names (relative to the FFN) that were
        filled, for the strict checkpoint check."""
        mapped = to_sglang_module_entries(entries)
        if "experts.w13_weight" in mapped and getattr(
            self.experts, "use_flashinfer_trtllm_moe", False
        ):
            # The flashinfer TRT-LLM runner's bf16 weight preparation
            # (reorder_rows_for_gated_act_gemm + shuffle) expects w13 as
            # [up rows; gate rows]; merge_gate_up gives [gate; up], so swap the
            # halves. Verified by the equivalence gate on GB300 (without the
            # swap the routed output is off by ~60%).
            w13 = mapped["experts.w13_weight"]
            n = w13.shape[1] // 2
            mapped["experts.w13_weight"] = torch.cat([w13[:, n:], w13[:, :n]], dim=1)
        params = dict(self.named_parameters())
        unknown = sorted(set(mapped) - set(params))
        if unknown:
            raise ValueError(
                f"checkpoint FFN entries do not map to {type(self).__name__} "
                f"parameters: {unknown[:8]}; module has {sorted(params)[:8]} ..."
            )
        provided: Set[str] = set()
        with torch.no_grad():
            for name, tensor in mapped.items():
                param = params[name]
                if tuple(param.shape) != tuple(tensor.shape):
                    if self._copy_padded_expert(name, param, tensor):
                        provided.add(name)
                        continue
                    raise ValueError(
                        f"{name}: checkpoint shape {tuple(tensor.shape)} does not "
                        f"match the parameter shape {tuple(param.shape)}"
                    )
                if self._fp8 and name in ("experts.w13_weight", "experts.w2_weight"):
                    # bf16 checkpoint -> e4m3 weights + per-output-channel
                    # scales, the layout SGLang's W8A8Fp8 MoE method serves.
                    q, scale = FusedDraftMoEFFN._quantize_per_channel(
                        tensor.to(param.device)
                    )
                    param.copy_(q)
                    scale_param = params[f"{name}_scale"]
                    scale_param.copy_(scale.unsqueeze(-1).to(scale_param.dtype))
                    provided.add(f"{name}_scale")
                else:
                    param.copy_(tensor.to(param.dtype))
                provided.add(name)
        return provided


def moe_backend() -> str:
    backend = os.environ.get(MOE_BACKEND_ENV, "fused").strip().lower()
    if backend not in ("fused", "grouped_mm", "sglang"):
        raise ValueError(
            f"{MOE_BACKEND_ENV} must be 'fused', 'sglang' or 'grouped_mm', "
            f"got {backend!r}"
        )
    if backend == "fused" and not _FUSED_AVAILABLE:
        logger.warning(
            "fused MoE backend unavailable (%s); using grouped_mm",
            _FUSED_IMPORT_ERROR,
        )
        backend = "grouped_mm"
    if backend == "sglang" and not _SGLANG_MODULES_AVAILABLE:
        raise RuntimeError(
            f"{MOE_BACKEND_ENV}=sglang needs SGLang's MoE modules "
            f"({_SGLANG_MODULES_IMPORT_ERROR})"
        )
    return backend


def backend_name(ffn: nn.Module) -> str:
    if isinstance(ffn, SglangMoEFFN):
        return "sglang"
    if isinstance(ffn, FusedDraftMoEFFN):
        return "fused"
    return "grouped_mm"


def build_draft_moe_ffn(config, layer_id: int = 0) -> DraftMoEFFN:
    """The layer FFN for the selected backend (see :data:`MOE_BACKEND_ENV`)."""
    backend = moe_backend()
    if backend == "sglang":
        expert_dtype = os.environ.get(MOE_EXPERT_DTYPE_ENV, "bf16").strip().lower()
        return SglangMoEFFN(config, layer_id, expert_dtype)
    if backend == "fused":
        return FusedDraftMoEFFN(config)
    return DraftMoEFFN(config)


class DFlashMoEDecoderLayer(DFlashDecoderLayer):
    """``DFlashDecoderLayer`` with the dense MLP replaced by :class:`DraftMoEFFN`."""

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        # The base layer built a dense MLP the export does not carry; swap it
        # for the MoE FFN before any weights are loaded.
        layer_id = kwargs.get("layer_id", args[0] if args else 0)
        del self.mlp
        self.mlp = build_draft_moe_ffn(config, int(layer_id))


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
            backend_name(ffn),
            ffn.describe(),
        )

    _LAYER_FFN_KEY = re.compile(r"^(?:model\.)?layers\.(?P<idx>\d+)\.mlp\.(?P<key>.+)$")

    def _load_sglang_ffn_tensors(
        self, stacked: List[Tuple[str, torch.Tensor]]
    ) -> Tuple[List[Tuple[str, torch.Tensor]], Set[str]]:
        """``sglang`` backend: SGLang's ``FusedMoE`` and linear parameters carry
        weight loaders with their own (sharded) signatures, so each layer's FFN
        entries are copied straight into the modules (replicated TP1 draft).
        Returns the entries left for the base loader and the FFN parameter
        names that were filled."""
        rest: List[Tuple[str, torch.Tensor]] = []
        per_layer: Dict[int, Dict[str, torch.Tensor]] = {}
        for name, tensor in stacked:
            m = self._LAYER_FFN_KEY.match(name)
            if m is None:
                rest.append((name, tensor))
                continue
            per_layer.setdefault(int(m["idx"]), {})[m["key"]] = tensor
        provided: Set[str] = set()
        for idx, entries in per_layer.items():
            if idx >= len(self.layers):
                raise ValueError(
                    f"checkpoint has FFN weights for layer {idx} but the draft "
                    f"has {len(self.layers)} layers"
                )
            for key in self.layers[idx].mlp.load_native_tensors(entries):
                provided.add(f"layers.{idx}.mlp.{key}")
        return rest, provided

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Per-expert checkpoint tensors -> stacked module parameters, then the
        # base loader; then refuse an FFN that only partially matched.
        stacked = stack_expert_weights(to_native_names(weights))
        ffn = self.layers[0].mlp
        if isinstance(ffn, (FusedDraftMoEFFN, SglangMoEFFN)):
            stacked = merge_gate_up(stacked)
        provided: Set[str] = set()
        if isinstance(ffn, SglangMoEFFN):
            stacked, provided = self._load_sglang_ffn_tensors(stacked)
        provided |= {name.removeprefix("model.") for name, _ in stacked}
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
        if dtype == "fp8" and isinstance(ffn, SglangMoEFFN):
            # Quantised while loading (see SglangMoEFFN.load_native_tensors).
            logger.info(
                "MoE draft experts are fp8 e4m3 with per-channel scales (SGLang "
                "W8A8Fp8 MoE method); router and shared expert stay bf16"
            )
        elif dtype == "fp8":
            if not isinstance(ffn, FusedDraftMoEFFN):
                raise ValueError(
                    f"{MOE_EXPERT_DTYPE_ENV}=fp8 needs the fused or sglang backend "
                    f"({MOE_BACKEND_ENV}=fused|sglang)"
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
