#!/usr/bin/env python3
"""Equivalence gate: serving ``DraftMoEFFN`` vs SpecForge's ``MoELayer``.

Builds SpecForge's MoE FFN with random weights (a small config and the real
DeepSeek-V4-Flash sizes, for the ``deepseek_v4`` and ``qwen3`` presets),
converts its state to the official checkpoint naming, loads it into the
serving FFN through the same expert-stacking logic the SGLang loader uses, and
compares routing (identical top-k, same combine weights) and outputs on random
inputs for both dispatch paths. ``--backend fused`` / ``--backend sglang`` run
the same checks against the two SGLang-kernel backends of
``specforge.serving.sglang_models.dflash_moe`` and add the Qwen3.8-27B DSpark
MoE recipe (``qwen3_5_moe``: 512 experts of width 512, top-10, sigmoid-gated
shared expert, folded router centering) at its real size, with the plain
``DraftMoEFFN`` (validated against the trainer in ``tests/test_serving``) as
the reference.

The serving FFN is ``specforge.serving.sglang_models.moe_ffn`` (plain
PyTorch), so this runs anywhere SpecForge imports; the grouped_mm cases need a
CUDA device and are skipped on CPU. ``--sglang-patch`` compares against the
``sglang.srt.models.dspark_moe`` module of the v0.5.18 patch instead.

    python scripts/gates/check_dspark_moe_sglang_equivalence.py
"""

import argparse
import sys

import torch

from specforge.modeling.draft.moe import MoEConfig, MoELayer, to_checkpoint_state_dict

PRESETS = {
    "deepseek_v4": dict(
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        n_shared_experts=1,
        swiglu_limit=10.0,
    ),
    "qwen3": dict(
        scoring_func="softmax",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        n_shared_experts=0,
        swiglu_limit=0.0,
    ),
}


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _serving_module(
    use_patch: bool, backend: str = "grouped_mm", moe_runner_backend: str = "triton"
):
    if use_patch:
        from sglang.srt.models import dspark_moe as mod

        return mod.DraftMoEFFN, mod.stack_expert_weights
    from specforge.serving.sglang_models import moe_ffn as mod

    if backend == "grouped_mm":
        return mod.DraftMoEFFN, mod.stack_expert_weights
    # The fused kernel needs SGLang's runtime context and a TP group, which a
    # server has and a standalone process must publish itself.
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    from specforge.serving.sglang_models.dflash_moe import (
        FusedDraftMoEFFN,
        SglangMoEFFN,
    )

    publish(
        ServerArgs(
            model_path="dummy",
            attention_backend="triton",
            moe_runner_backend=moe_runner_backend,
        ),
        role="scheduler",
    )
    # Seeds the MoE runtime flags (runner backend, ...) from the published
    # args, as the scheduler does; without it the runner stays "auto".
    from sglang.srt.layers.moe.utils import initialize_moe_config

    initialize_moe_config()
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:29599",
        local_rank=0,
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )

    def stack(weights):
        return mod.merge_gate_up(mod.stack_expert_weights(weights))

    return (SglangMoEFFN if backend == "sglang" else FusedDraftMoEFFN), stack


def _load_ffn(ffn, stacked):
    """Copy stacked checkpoint entries (names relative to the FFN) into a
    serving FFN of any backend."""
    if hasattr(ffn, "load_native_tensors"):
        ffn.load_native_tensors(stacked)
        return
    params = dict(ffn.named_parameters())
    assert set(stacked) == set(params), (
        sorted(set(stacked) - set(params)),
        sorted(set(params) - set(stacked)),
    )
    with torch.no_grad():
        for name, tensor in stacked.items():
            params[name].copy_(tensor.to(params[name].dtype))


def build_qwen35_pair(
    e, k, inter, hidden, shared, device, dtype, serving_cls, fp8=False
):
    """Kan's Qwen3.8-27B DSpark MoE recipe (``qwen3_5_moe``): softmax over
    ``x W^T + gate.bias`` (folded router centering), top-k renormalised, no
    scaling, sigmoid-gated shared expert, Qwen checkpoint naming. Reference =
    the plain-PyTorch ``DraftMoEFFN``; serving = ``serving_cls`` on the same
    random checkpoint."""
    from specforge.serving.sglang_models import moe_ffn as mod

    cfg = _Cfg(
        hidden_size=hidden,
        num_experts=e,
        num_experts_per_tok=k,
        moe_intermediate_size=inter,
        shared_expert_intermediate_size=shared,
        n_shared_experts=1,
        moe_preset="qwen3_5_moe",
        scoring_func="softmax",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        moe_router_bias=True,
        n_group=1,
        topk_group=1,
        topk_method="greedy",
        hidden_act="silu",
        swiglu_limit=0.0,
    )
    gen = torch.Generator().manual_seed(0)

    def rn(*shape, scale):
        return (torch.randn(*shape, generator=gen) * scale).to(dtype)

    w = {
        "gate.weight": rn(e, hidden, scale=0.02),
        "gate.bias": rn(e, scale=0.5).float(),
        "shared_expert.gate_proj.weight": rn(shared, hidden, scale=0.02),
        "shared_expert.up_proj.weight": rn(shared, hidden, scale=0.02),
        "shared_expert.down_proj.weight": rn(hidden, shared, scale=0.02),
        "shared_expert_gate.weight": rn(1, hidden, scale=0.02),
    }
    for i in range(e):
        w[f"experts.{i}.gate_proj.weight"] = rn(inter, hidden, scale=0.02)
        w[f"experts.{i}.up_proj.weight"] = rn(inter, hidden, scale=0.02)
        w[f"experts.{i}.down_proj.weight"] = rn(hidden, inter, scale=0.02)
    stacked = dict(mod.stack_expert_weights(mod.to_native_names(w.items())))
    with torch.device(device):
        ref = mod.DraftMoEFFN(cfg).to(dtype=dtype)
        if serving_cls.__name__ == "SglangMoEFFN":
            # Build under the model dtype instead of casting afterwards: a
            # blanket .to(dtype) would also cast the fp8 expert parameters
            # (and their fp32 scales) SGLang's W8A8 method created.
            prev = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                sg = serving_cls(cfg, 0, "fp8" if fp8 else "bf16")
            finally:
                torch.set_default_dtype(prev)
        else:
            sg = serving_cls(cfg).to(dtype=dtype)
    for m in (ref, sg):
        m.gate.bias.data = m.gate.bias.data.float()
    _load_ffn(ref, stacked)
    _load_ffn(sg, dict(mod.merge_gate_up(list(stacked.items()))))
    if fp8 and hasattr(sg, "quantize_experts_fp8"):
        sg.quantize_experts_fp8()
    elif hasattr(sg.experts, "quant_method"):
        # What SGLang's model loader does after load_weights (fp8 re-wrap,
        # TRT-LLM weight shuffling, ...).
        sg.experts.quant_method.process_weights_after_loading(sg.experts)
    return ref.eval(), sg.eval()


def compare_ffn(ref, sg, tokens, hidden, device, dtype, label, fp8_tolerance=False):
    """Like :func:`compare` for two serving FFNs (both expose ``route``)."""
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    with torch.no_grad():
        y_ref, y_sg = ref(x), sg(x)
        w_ref, i_ref, _ = ref.route(x)
        w_sg, i_sg, _ = sg.route(x)
    same_idx = torch.equal(i_ref.sort(-1).values, i_sg.sort(-1).values)
    w_diff = (w_ref.sort(-1).values - w_sg.sort(-1).values).abs().max().item()
    diff = (y_ref.float() - y_sg.float()).abs()
    scale = y_ref.float().abs().mean().item()
    print(
        f"[{label}] tokens={tokens} same_topk={same_idx} max|dw|={w_diff:.2e} "
        f"max|dy|={diff.max().item():.3e} mean|dy|={diff.mean().item():.3e} "
        f"mean|y|={scale:.3e}"
    )
    if fp8_tolerance:
        # e4m3 experts: routing identical, mean relative error at the e4m3
        # level (<= 10%); the real criterion is the end-to-end accept length.
        return (
            same_idx and w_diff < 1e-5 and diff.mean().item() <= 0.1 * max(scale, 1e-3)
        )
    return (
        same_idx and w_diff < 1e-5 and diff.max().item() <= 2e-2 * max(scale, 1e-3) * 10
    )


def build_pair(
    preset,
    e,
    k,
    inter,
    hidden,
    device,
    dtype,
    bias_scale,
    serving,
    fp8=False,
    fuse_shared=False,
):
    DraftMoEFFN, stack_expert_weights = serving
    recipe = PRESETS[preset]
    moe_cfg = MoEConfig(
        preset=preset,
        n_routed_experts=e,
        num_experts_per_tok=k,
        moe_intermediate_size=inter,
        router="topk",
        balance="noaux_tc",
        experts_backend="grouped",
        shared_expert="swiglu",
        shared_expert_gate="none",
        dispatch="grouped_mm",
        **recipe,
    )
    ref = MoELayer(moe_cfg, hidden)
    ref.reset_parameters(0.02)
    if bias_scale:
        # Simulate a trained selection bias so selection != raw score order.
        ref.gate.balance.bias.normal_(0, bias_scale)
    ref = ref.to(device=device, dtype=dtype).eval()
    ref.gate.balance.bias.data = ref.gate.balance.bias.data.float()

    state = to_checkpoint_state_dict(
        {k_: v.detach() for k_, v in ref.state_dict().items()}
    )
    keys = sorted(state)
    assert "experts.0.w1.weight" in keys and "gate.bias" in keys, keys[:5]
    if recipe["n_shared_experts"]:
        assert "shared_experts.w1.weight" in keys
    else:
        assert not any(key.startswith("shared_experts.") for key in keys), keys

    # What MoEConfig.serving_fields() writes into the export's config.json.
    sg_cfg = _Cfg(
        hidden_size=hidden,
        n_routed_experts=e,
        num_experts_per_tok=k,
        moe_intermediate_size=inter,
        n_group=1,
        topk_group=1,
        topk_method="noaux_tc",
        hidden_act="silu",
        moe_preset=preset,
        **recipe,
    )
    with torch.device(device):
        sg = DraftMoEFFN(sg_cfg).to(dtype=dtype)
    sg.gate.bias.data = sg.gate.bias.data.float()
    stacked = dict(stack_expert_weights(list(state.items())))
    sg_params = dict(sg.named_parameters())
    assert set(stacked) == set(sg_params), (
        sorted(set(stacked) - set(sg_params)),
        sorted(set(sg_params) - set(stacked)),
    )
    with torch.no_grad():
        for name, tensor in stacked.items():
            sg_params[name].copy_(tensor.to(sg_params[name].dtype))
    if fuse_shared:
        sg.fuse_shared_expert()
    if fp8:
        sg.quantize_experts_fp8()
    return ref.eval(), sg.eval()


def compare(ref, sg, tokens, hidden, device, dtype, label, fp8_tolerance=False):
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    with torch.no_grad():
        y_ref = ref(x)
        y_sg = sg(x)
        # Routing must be identical, not just close.
        w_ref, i_ref = ref.gate(x).weights, ref.gate(x).indices
        w_sg, i_sg, _ = sg.route(x)
    same_idx = torch.equal(i_ref.sort(-1).values, i_sg.sort(-1).values)
    w_diff = (w_ref.sort(-1).values - w_sg.sort(-1).values).abs().max().item()
    diff = (y_ref.float() - y_sg.float()).abs()
    scale = y_ref.float().abs().mean().item()
    print(
        f"[{label}] tokens={tokens} same_topk={same_idx} max|dw|={w_diff:.2e} "
        f"max|dy|={diff.max().item():.3e} mean|dy|={diff.mean().item():.3e} "
        f"mean|y|={scale:.3e}"
    )
    if fp8_tolerance:
        # W8A8 experts: float8 e4m3 keeps 3 mantissa bits, so every quantised
        # weight and activation carries up to ~6% relative error, and with
        # random-sign weights a dot product does not average it out. The gate
        # therefore only checks routing identity and that the mean relative
        # error stays at the e4m3 level (<= 10%); the real acceptance criterion
        # for fp8 experts is the end-to-end accept length.
        return (
            same_idx and w_diff < 1e-5 and diff.mean().item() <= 0.1 * max(scale, 1e-3)
        )
    ok = (
        same_idx and w_diff < 1e-5 and diff.max().item() <= 2e-2 * max(scale, 1e-3) * 10
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sglang-patch", action="store_true")
    ap.add_argument(
        "--backend",
        choices=("grouped_mm", "fused", "sglang"),
        default="grouped_mm",
        help="fused = SGLang's fused MoE kernel through "
        "specforge.serving.sglang_models.dflash_moe (CUDA + SGLang runtime); "
        "sglang = SGLang's TopK + FusedMoE modules (Qwen recipe only)",
    )
    ap.add_argument(
        "--fp8",
        action="store_true",
        help="with --backend fused: quantise the routed experts to fp8 e4m3 "
        "(per-channel) after loading, i.e. the SPECFORGE_DRAFT_MOE_EXPERT_DTYPE=fp8 path",
    )
    ap.add_argument(
        "--fuse-shared",
        action="store_true",
        help="with --backend fused: fold the shared expert into the kernel's "
        "expert tensors (SPECFORGE_DRAFT_MOE_FUSE_SHARED=1 path)",
    )
    ap.add_argument(
        "--moe-runner-backend",
        default="triton",
        choices=("triton", "flashinfer_trtllm"),
        help="SGLang MoE runner for the sglang backend's FusedMoE (flashinfer_trtllm "
        "= the TRT-LLM MoE kernels on Blackwell, which route inside the kernel)",
    )
    ap.add_argument("--presets", nargs="+", default=sorted(PRESETS))
    args = ap.parse_args()
    if args.fuse_shared and args.backend != "fused":
        ap.error("--fuse-shared requires --backend fused")
    if args.fp8 and args.backend == "grouped_mm":
        ap.error("--fp8 requires --backend fused or sglang")
    if args.moe_runner_backend != "triton" and args.backend != "sglang":
        ap.error("--moe-runner-backend applies to --backend sglang only")
    serving = _serving_module(args.sglang_patch, args.backend, args.moe_runner_backend)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    ok = True
    torch.manual_seed(0)
    if args.backend == "sglang":
        # SGLang's TopK has no selection bias; the trainer presets below all
        # use noaux_tc, so only the Qwen recipe case applies.
        print(
            "[deepseek_v4, qwen3] skipped: --backend sglang supports the Qwen recipe only"
        )
        args.presets = []
    for preset in args.presets:
        # Small config, with a non-trivial selection bias.
        ref, sg = build_pair(
            preset,
            8,
            2,
            64,
            128,
            device,
            dtype,
            0.3,
            serving,
            args.fp8,
            args.fuse_shared,
        )
        if device == "cuda":
            tag = f"{preset}/small/{args.backend}{'-fp8' if args.fp8 else ''}{'-fusedshared' if args.fuse_shared else ''}"
            ok &= compare(ref, sg, 1, 128, device, dtype, tag, fp8_tolerance=args.fp8)
            ok &= compare(ref, sg, 37, 128, device, dtype, tag, fp8_tolerance=args.fp8)
        if args.backend == "grouped_mm":
            # Loop path (force by moving to CPU) vs reference on CPU; the fused
            # kernel is CUDA-only.
            ref_cpu, sg_cpu = ref.to("cpu"), sg.to("cpu")
            ref_cpu.experts.grouped_mm = False
            ok &= compare(
                ref_cpu,
                sg_cpu,
                23,
                128,
                "cpu",
                dtype,
                f"{preset}/small/loop-cpu",
                fp8_tolerance=args.fp8,
            )
        # Real sizes: DSV4-Flash arm (64 x 2048 top-6) and Qwen3.8-27B arm (16 x 4352 top-4).
        e, k, inter, hidden = (
            (64, 6, 2048, 4096) if preset == "deepseek_v4" else (16, 4, 4352, 5120)
        )
        if device == "cuda":
            ref, sg = build_pair(
                preset,
                e,
                k,
                inter,
                hidden,
                device,
                dtype,
                1.0,
                serving,
                args.fp8,
                args.fuse_shared,
            )
            ok &= compare(
                ref,
                sg,
                8,
                hidden,
                device,
                dtype,
                f"{preset}/real/grouped_mm",
                fp8_tolerance=args.fp8,
            )
            ok &= compare(
                ref,
                sg,
                256,
                hidden,
                device,
                dtype,
                f"{preset}/real/grouped_mm",
                fp8_tolerance=args.fp8,
            )
        else:
            print(f"[{preset}/real] skipped on CPU")
    if args.backend in ("fused", "sglang") and device == "cuda":
        serving_cls, _ = serving
        tag = (
            f"qwen3_5_moe/{args.backend}{'-fp8' if args.fp8 else ''}"
            f"{'-' + args.moe_runner_backend if args.moe_runner_backend != 'triton' else ''}"
        )
        ref, sg = build_qwen35_pair(
            8, 3, 64, 128, 128, device, dtype, serving_cls, args.fp8
        )
        ok &= compare_ffn(ref, sg, 1, 128, device, dtype, f"{tag}/small", args.fp8)
        ok &= compare_ffn(ref, sg, 37, 128, device, dtype, f"{tag}/small", args.fp8)
        del ref, sg
        # Kan's Qwen3.8-27B DSpark MoE drafter: 512 x 512 top-10, shared 2048.
        ref, sg = build_qwen35_pair(
            512, 10, 512, 5120, 2048, device, dtype, serving_cls, args.fp8
        )
        for tokens in (7, 56, 448):
            ok &= compare_ffn(
                ref, sg, tokens, 5120, device, dtype, f"{tag}/real", args.fp8
            )
        del ref, sg
    # Stacking must reject a truncated export.
    _, stack_expert_weights = serving
    try:
        stack_expert_weights(
            [
                ("layers.0.mlp.experts.0.w1.weight", torch.zeros(2, 2)),
                ("layers.0.mlp.experts.2.w1.weight", torch.zeros(2, 2)),
            ]
        )
        print("[stack] FAIL: missing expert index not rejected")
        ok = False
    except ValueError as err:
        print(f"[stack] rejects gaps: {str(err)[:80]}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
