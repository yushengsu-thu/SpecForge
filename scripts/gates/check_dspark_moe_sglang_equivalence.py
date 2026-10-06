#!/usr/bin/env python3
"""Equivalence gate: serving ``DraftMoEFFN`` vs SpecForge's ``MoELayer``.

Builds SpecForge's MoE FFN with random weights (a small config and the real
DeepSeek-V4-Flash sizes, for the ``deepseek_v4`` and ``qwen3`` presets),
converts its state to the official checkpoint naming, loads it into the
serving FFN through the same expert-stacking logic the SGLang loader uses, and
compares routing (identical top-k, same combine weights) and outputs on random
inputs for both dispatch paths.

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


def _serving_module(use_patch: bool, backend: str = "grouped_mm"):
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

    from specforge.serving.sglang_models.dflash_moe import FusedDraftMoEFFN

    publish(
        ServerArgs(model_path="dummy", attention_backend="triton"), role="scheduler"
    )
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

    return FusedDraftMoEFFN, stack


def build_pair(preset, e, k, inter, hidden, device, dtype, bias_scale, serving):
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
    return ref.eval(), sg.eval()


def compare(ref, sg, tokens, hidden, device, dtype, label):
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
    ok = (
        same_idx and w_diff < 1e-5 and diff.max().item() <= 2e-2 * max(scale, 1e-3) * 10
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sglang-patch", action="store_true")
    ap.add_argument(
        "--backend",
        choices=("grouped_mm", "fused"),
        default="grouped_mm",
        help="fused = SGLang's fused MoE kernel through "
        "specforge.serving.sglang_models.dflash_moe (CUDA + SGLang runtime)",
    )
    ap.add_argument("--presets", nargs="+", default=sorted(PRESETS))
    args = ap.parse_args()
    serving = _serving_module(args.sglang_patch, args.backend)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    ok = True
    torch.manual_seed(0)
    for preset in args.presets:
        # Small config, with a non-trivial selection bias.
        ref, sg = build_pair(preset, 8, 2, 64, 128, device, dtype, 0.3, serving)
        if device == "cuda":
            tag = f"{preset}/small/{args.backend}"
            ok &= compare(ref, sg, 1, 128, device, dtype, tag)
            ok &= compare(ref, sg, 37, 128, device, dtype, tag)
        if args.backend == "grouped_mm":
            # Loop path (force by moving to CPU) vs reference on CPU; the fused
            # kernel is CUDA-only.
            ref_cpu, sg_cpu = ref.to("cpu"), sg.to("cpu")
            ref_cpu.experts.grouped_mm = False
            ok &= compare(
                ref_cpu, sg_cpu, 23, 128, "cpu", dtype, f"{preset}/small/loop-cpu"
            )
        # Real sizes: DSV4-Flash arm (64 x 2048 top-6) and Qwen3.8-27B arm (16 x 4352 top-4).
        e, k, inter, hidden = (
            (64, 6, 2048, 4096) if preset == "deepseek_v4" else (16, 4, 4352, 5120)
        )
        if device == "cuda":
            ref, sg = build_pair(
                preset, e, k, inter, hidden, device, dtype, 1.0, serving
            )
            ok &= compare(
                ref, sg, 8, hidden, device, dtype, f"{preset}/real/grouped_mm"
            )
            ok &= compare(
                ref, sg, 256, hidden, device, dtype, f"{preset}/real/grouped_mm"
            )
        else:
            print(f"[{preset}/real] skipped on CPU")
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
