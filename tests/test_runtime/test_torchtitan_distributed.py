"""Two-GPU numerical gate using real DFlash, DFlash2, and DSpark objectives.

Run in the optional TorchTitan environment (no model downloads):

    torchrun --standalone --nproc_per_node=2 -m \
        tests.test_runtime.test_torchtitan_distributed --output /tmp/titan-gate

The unwrapped reference evaluates every rank's distinct microbatches on one
GPU. DFlash uses its global optimizer-window numerator/denominator; DSpark uses
its existing global token denominator per microbatch. BF16 collective rounding is
allowed for cross-backend parity; same-backend checkpoint continuation must be
bitwise identical, including FP32 masters and Adam moments.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from datetime import timedelta
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist
from torch import nn

ACCUMULATION = 2
HIDDEN = 32
VOCAB = 64
LR = 0.002


def _build_model(family: str):
    from transformers import Qwen3Config

    from specforge.algorithms.common.dflash_family_model import (
        OnlineDFlashModel,
        OnlineDSparkModel,
    )
    from specforge.modeling.draft.dflash import DFlashDraftModel
    from specforge.modeling.draft.dflash2 import DFlash2DraftModel
    from specforge.modeling.draft.dspark import DSparkDraftModel

    torch.manual_seed(173)
    config = Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        num_target_layers=4,
        head_dim=8,
        max_position_embeddings=128,
        vocab_size=VOCAB,
        attention_bias=False,
        attention_dropout=0.0,
        layer_types=["full_attention"] * 2,
        dflash_config={
            "block_size": 4,
            "mask_token_id": VOCAB - 1,
            "target_layer_ids": [1, 2],
            "conv_group_size": 4,
            "conv_kernel_size": 2,
            "selector_rank": 4,
            "selector_top_k": 4,
            "markov_rank": 4,
            "markov_head_type": "gated",
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
        },
    )
    config._attn_implementation = "eager"
    draft_cls = {
        "dflash": DFlashDraftModel,
        "dflash2": DFlash2DraftModel,
        "dspark": DSparkDraftModel,
    }[family]
    draft = draft_cls(config)
    if family == "dflash2":
        # Exercise all selector factors on the first update, rather than only
        # the default zero-initialized successor table.
        with torch.no_grad():
            draft.candidate_selector.successor_codebook.normal_(std=0.05)
    kwargs = dict(
        draft_model=draft,
        target_lm_head=nn.Linear(HIDDEN, VOCAB, bias=False).requires_grad_(False),
        target_embed_tokens=nn.Embedding(VOCAB, HIDDEN).requires_grad_(False),
        mask_token_id=VOCAB - 1,
        block_size=4,
        num_anchors=64,
        attention_backend="eager",
        objective_chunk_blocks=0,
    )
    if family == "dspark":
        model = OnlineDSparkModel(**kwargs)
    else:
        model = OnlineDFlashModel(
            **kwargs, selector_loss_alpha=1.0, teacher_metrics=False
        )
    return model.to(device=torch.cuda.current_device(), dtype=torch.bfloat16)


def _strategy(family, model):
    from specforge.training.strategies.base import (
        DFlashTrainStrategy,
        DSparkTrainStrategy,
    )

    return (DSparkTrainStrategy if family == "dspark" else DFlashTrainStrategy)(model)


def _optimizer(model):
    from specforge.optimizer import BF16Optimizer

    return BF16Optimizer(
        model,
        lr=LR,
        max_grad_norm=0.25,
        warmup_ratio=0.0,
        total_steps=8,
        lr_scheduler="constant",
    )


def _batch(family, source_rank, micro, window):
    from specforge.runtime.contracts import TrainBatch

    seq = 24
    generator = torch.Generator().manual_seed(
        400 + source_rank * 19 + micro + window * 71
    )
    # Both ranks have valid anchors, but substantially different denominators.
    loss_mask = torch.zeros(1, seq, dtype=torch.long)
    supervised = 19 - source_rank * 8 - micro * 2
    loss_mask[:, :supervised] = 1
    return TrainBatch(
        sample_ids=[f"r{source_rank}-m{micro}-w{window}"],
        strategy="dspark" if family == "dspark" else "dflash",
        tensors={
            "input_ids": torch.randint(0, VOCAB - 1, (1, seq), generator=generator),
            "loss_mask": loss_mask,
            "hidden_states": torch.randn(1, seq, 2 * HIDDEN, generator=generator).to(
                torch.bfloat16
            ),
            "target_last_hidden_states": torch.randn(
                1, seq, HIDDEN, generator=generator
            ).to(torch.bfloat16),
        },
        metadata={},
    )


def _seed_forward(source_rank, micro, window):
    torch.cuda.manual_seed(8300 + window * 100 + source_rank * 10 + micro)


def _context(window):
    from specforge.training.strategies.base import StepContext

    return StepContext(
        global_step=window, total_steps=8, collect_detailed_metrics=False
    )


def _dspark_global_denominator(micro, window):
    """Independent fixture oracle: all valid anchors, four next-token labels."""
    total = 0
    for source_rank in range(dist.get_world_size()):
        masks = _batch("dspark", source_rank, micro, window).tensors["loss_mask"]
        for row in masks.tolist():
            for anchor in range(len(row) - 1):
                if not (row[anchor] and row[anchor + 1]):
                    continue
                for offset in range(1, 5):
                    index = anchor + offset
                    if index >= len(row) or not row[index]:
                        break
                    total += 1
    return total


def _local_optimizer_state(optimizer, draft):
    names = {
        id(param): name.replace("_fsdp_wrapped_module.", "")
        for name, param in draft.named_parameters()
    }
    result = {}
    for param, master in zip(optimizer.model_params, optimizer.fp32_params):
        name = names[id(param)]
        state = optimizer.optimizer.state[master]
        if master.numel() and "exp_avg" not in state:
            raise AssertionError(f"nonempty parameter has no Adam state: {name}")
        result[name] = {
            "master": master.detach().cpu().clone(),
            "exp_avg": state.get("exp_avg", torch.zeros_like(master))
            .detach()
            .cpu()
            .clone(),
            "exp_avg_sq": state.get("exp_avg_sq", torch.zeros_like(master))
            .detach()
            .cpu()
            .clone(),
            "step": float(state["step"].item()) if "step" in state else 0.0,
        }
    return result


def _global_optimizer_state(backend, draft, full_shapes):
    local = _local_optimizer_state(backend.optimizer, draft)
    shards = [None] * dist.get_world_size()
    dist.all_gather_object(shards, local)
    result = {}
    for name, shape in full_shapes.items():
        result[name] = {}
        for key in ("master", "exp_avg", "exp_avg_sq"):
            # FSDP1's original-parameter slices and FSDP2's dim-0 shards both
            # reconstruct in rank order; empty FSDP1/FSDP2 local slices are OK.
            full = torch.cat([shard[name][key].reshape(-1) for shard in shards])
            expected_numel = shape.numel()
            if full.numel() != expected_numel:
                raise AssertionError(
                    f"{name}: gathered {full.numel()} values, expected {expected_numel}"
                )
            result[name][key] = full.reshape(shape)
        nonempty = [
            shard[name]["step"] for shard in shards if shard[name]["master"].numel()
        ]
        if len(set(nonempty)) != 1:
            raise AssertionError(f"inconsistent Adam step for {name}: {nonempty}")
        result[name]["step"] = nonempty[0]
    return result


def _flatten(state, key):
    return torch.cat([state[name][key].reshape(-1).float() for name in sorted(state)])


def _compare(actual, expected, initial, actual_norm, expected_norm):
    if set(actual) != set(expected):
        raise AssertionError("optimizer parameter names changed")
    metrics = {}
    for key, limit in (("exp_avg", 0.035), ("exp_avg_sq", 0.07)):
        left, right = _flatten(actual, key), _flatten(expected, key)
        relative_l2 = (left - right).norm() / right.norm().clamp_min(1e-12)
        metrics[f"{key}_relative_l2"] = float(relative_l2)
        if relative_l2 > limit:
            raise AssertionError(f"{key} relative error {relative_l2} > {limit}")
    initial_flat = torch.cat(
        [initial[name].float().reshape(-1) for name in sorted(actual)]
    )
    delta = _flatten(actual, "master") - initial_flat
    reference_delta = _flatten(expected, "master") - initial_flat
    update_error = (delta - reference_delta).norm() / reference_delta.norm().clamp_min(
        1e-12
    )
    cosine = torch.nn.functional.cosine_similarity(delta, reference_delta, dim=0)
    metrics["fp32_update_relative_l2"] = float(update_error)
    metrics["fp32_update_cosine"] = float(cosine)
    norm_error = abs(actual_norm - expected_norm) / max(abs(expected_norm), 1e-12)
    metrics["grad_norm_relative_error"] = norm_error
    if update_error > 0.08 or cosine < 0.995 or norm_error > 0.035:
        raise AssertionError(f"BF16 backend/reference mismatch: {metrics}")
    for name in actual:
        if actual[name]["step"] != expected[name]["step"]:
            raise AssertionError(f"Adam step differs for {name}")
    return metrics


def _reference(family, initial_state, window=0):
    model = _build_model(family)
    model.load_state_dict(initial_state)
    optimizer = _optimizer(model.draft_model)
    optimizer.configure_grad_norm_reduction(enabled=False)
    strategy = _strategy(family, model)
    terms, denominators, rank_gradients = [], [], []
    for source_rank in range(dist.get_world_size()):
        for micro in range(ACCUMULATION):
            _seed_forward(source_rank, micro, window)
            # DSpark owns a scalar all-reduce inside its objective. The serial
            # oracle substitutes the independently counted global denominator;
            # reducing with the peer here would count this same fixture twice.
            collective = contextlib.nullcontext()
            if family == "dspark":
                expected_denominator = _dspark_global_denominator(micro, window)

                def reduce_denominator(tensor, **kwargs):
                    if tensor.numel() != 1:
                        raise AssertionError("unexpected DSpark reference collective")
                    tensor.fill_(expected_denominator)

                collective = mock.patch.object(
                    dist, "all_reduce", side_effect=reduce_denominator
                )
            with collective:
                output = strategy.forward_loss(
                    _batch(family, source_rank, micro, window), _context(window)
                )
            if output.loss_terms is None:
                terms.append(output.loss)
                micro_loss = output.loss
            else:
                numerator, denominator = output.loss_terms
                terms.append(numerator)
                denominators.append(denominator.detach())
                micro_loss = numerator
            (micro_loss / ACCUMULATION).backward()
        rank_gradients.append(
            {
                name: parameter.grad.detach().clone()
                for name, parameter in model.draft_model.named_parameters()
                if parameter.requires_grad
            }
        )
        model.zero_grad(set_to_none=True)
    denominator = sum(denominators) if denominators else len(terms)
    loss = sum(terms) / denominator
    # Match the explicit precision boundaries: accumulate BF16 gradients per
    # rank, DP-average once, then apply SpecForge's global normalization. This
    # independently computes the same global objective without asking a
    # distributed wrapper to reduce the reference model.
    with torch.no_grad():
        for name, parameter in model.draft_model.named_parameters():
            if not parameter.requires_grad:
                continue
            averaged = (
                torch.stack([gradients[name].float() for gradients in rank_gradients])
                .mean(dim=0)
                .to(parameter.dtype)
            )
            if denominators:
                averaged.mul_(
                    (dist.get_world_size() * ACCUMULATION / denominator).to(
                        parameter.dtype
                    )
                )
            parameter.grad = averaged
    norm = float(optimizer.step().item())
    return {
        "optimizer": _local_optimizer_state(optimizer, model.draft_model),
        "norm": norm,
        "loss": float(loss.detach().item()),
        "denominators": [float(value.item()) for value in denominators],
    }


def _prepare(family, initial_state, backend_name, sharding):
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
    from specforge.training.controller import TrainerCore
    from specforge.training.torchtitan_backend import TorchTitanTrainingBackend

    model = _build_model(family)
    model.load_state_dict(initial_state)
    backend_type = (
        TorchTitanTrainingBackend
        if backend_name == "torchtitan"
        else FSDPTrainingBackend
    )
    backend = backend_type(
        ParallelConfig(
            world_size=dist.get_world_size(),
            fsdp_process_group=dist.group.WORLD,
            sharding_strategy=sharding,
        ),
        optimizer_factory=_optimizer,
    )
    wrapped = backend.prepare_model(model, optimizer_target=model.draft_model)
    strategy = _strategy(family, wrapped)
    return (
        model,
        backend,
        TrainerCore(strategy, backend, accumulation_steps=ACCUMULATION),
    )


def _step(family, core, window):
    for micro in range(ACCUMULATION):
        _seed_forward(dist.get_rank(), micro, window)
        result = core.train_step(
            _batch(family, dist.get_rank(), micro, window), _context(window)
        )
        if result.optimizer_stepped != (micro == ACCUMULATION - 1):
            raise AssertionError("optimizer advanced outside the accumulation boundary")
    return result


def _check_frozen(model, initial_state):
    from torch.distributed.tensor import DTensor

    for name in ("lm_head", "embed_tokens"):
        table = getattr(model, name).weight
        if isinstance(table, DTensor) or table.requires_grad or table.grad is not None:
            raise AssertionError(f"teacher {name} was sharded or received gradients")
        torch.testing.assert_close(
            table.cpu(), initial_state[f"{name}.weight"], rtol=0, atol=0
        )


def _check_export(core, full_state, initial_state):
    from torch.distributed.tensor import DTensor

    exported = core.strategy.checkpoint_state_filter(full_state)
    if dist.get_rank() != 0:
        # FSDP1 may retain ignored frozen teacher entries even with rank0_only.
        # Neither backend may materialize a full draft on a non-writing rank.
        if exported:
            raise AssertionError("full draft state must be rank-zero-only")
        return
    expected = {
        key.removeprefix("draft_model.")
        for key in initial_state
        if key.startswith("draft_model.")
    }
    if set(exported) != expected:
        raise AssertionError(f"export keys changed: {set(exported) ^ expected}")
    if any(
        isinstance(value, DTensor) or value.device.type != "cpu"
        for value in exported.values()
    ):
        raise AssertionError("export must contain ordinary full CPU tensors")


def _case(family, sharding, output_dir):
    initial_model = _build_model(family)
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in initial_model.state_dict().items()
    }
    initial_draft = {
        name: value.detach().cpu().clone()
        for name, value in initial_model.draft_model.named_parameters()
        if value.requires_grad
    }
    full_shapes = {name: value.shape for name, value in initial_draft.items()}
    del initial_model
    reference = _reference(family, initial_state)
    if family != "dspark" and len(set(reference["denominators"])) < 2:
        raise AssertionError(
            "fixture failed to exercise unequal global loss denominators"
        )
    report = {
        "family": family,
        "sharding": sharding,
        "reference_norm": reference["norm"],
        "reference_loss": reference["loss"],
        "loss_denominators": reference["denominators"],
    }
    snapshots = {}
    for backend_name in ("fsdp", "torchtitan"):
        model, backend, core = _prepare(family, initial_state, backend_name, sharding)
        result = _step(family, core, 0)
        norm = result.grad_norm
        snapshot = _global_optimizer_state(backend, model.draft_model, full_shapes)
        snapshots[backend_name] = (snapshot, norm)
        report[backend_name] = _compare(
            snapshot, reference["optimizer"], initial_draft, norm, reference["norm"]
        )
        _check_frozen(model, initial_state)
        state = backend.state_dict()
        _check_export(core, state["model"], initial_state)
        if backend_name == "fsdp":
            continue

        case_dir = output_dir / f"{family}-{sharding.lower()}"
        case_dir.mkdir(parents=True, exist_ok=True)
        if dist.get_rank() == 0:
            torch.save(state["model"], case_dir / "model.pt")
        torch.save(
            {"optimizer": state["optimizer"], "rng": state["rng"]},
            case_dir / f"rank{dist.get_rank()}.pt",
        )
        dist.barrier()
        # A fresh backend loads ordinary full weights before sharding, followed
        # by its rank-local FP32 masters/moments/RNG, as Trainer resume does.
        continued_result = _step(family, core, 1)
        continued = _global_optimizer_state(backend, model.draft_model, full_shapes)
        continued_full = backend.state_dict()["model"]
        saved_model = torch.load(
            case_dir / "model.pt", map_location="cpu", weights_only=False
        )
        resumed_model, resumed_backend, resumed_core = _prepare(
            family, saved_model, "torchtitan", sharding
        )
        rank_state = torch.load(
            case_dir / f"rank{dist.get_rank()}.pt",
            map_location="cpu",
            weights_only=False,
        )
        resumed_backend.load_state_dict(rank_state)
        torch.testing.assert_close(
            torch.get_rng_state(), rank_state["rng"]["torch"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            torch.cuda.get_rng_state(), rank_state["rng"]["cuda"], rtol=0, atol=0
        )
        resumed_result = _step(family, resumed_core, 1)
        resumed = _global_optimizer_state(
            resumed_backend, resumed_model.draft_model, full_shapes
        )
        for name in continued:
            for key in ("master", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    resumed[name][key],
                    continued[name][key],
                    rtol=0,
                    atol=0,
                    msg=f"resume mismatch: {name}/{key}",
                )
            if resumed[name]["step"] != continued[name]["step"]:
                raise AssertionError(f"resumed Adam step differs: {name}")
        if resumed_result.grad_norm != continued_result.grad_norm:
            raise AssertionError("resumed gradient norm is not bitwise identical")
        resumed_full = resumed_backend.state_dict()["model"]
        for name in continued_full:
            torch.testing.assert_close(
                resumed_full[name], continued_full[name], rtol=0, atol=0
            )
        _check_frozen(resumed_model, initial_state)
        _check_export(resumed_core, resumed_full, initial_state)
        report["resume_bitwise_equal"] = True
        report["plain_cpu_export"] = True
        report["frozen_teacher_tables_unchanged"] = True
    report["torchtitan_vs_fsdp"] = _compare(
        snapshots["torchtitan"][0],
        snapshots["fsdp"][0],
        initial_draft,
        snapshots["torchtitan"][1],
        snapshots["fsdp"][1],
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("dflash", "dflash2", "dspark"),
        default=["dflash", "dflash2", "dspark"],
    )
    parser.add_argument(
        "--sharding",
        nargs="+",
        choices=("FULL_SHARD", "SHARD_GRAD_OP"),
        default=["FULL_SHARD", "SHARD_GRAD_OP"],
    )
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", 0)) != 2:
        raise RuntimeError(
            "launch this correctness gate with torchrun --nproc_per_node=2"
        )
    # Keep this gate focused on backend numerics; fused kernels have separate
    # correctness gates and are benchmarked independently.
    os.environ["SPECFORGE_DFLASH2_FUSED_CONV"] = "0"
    os.environ["SPECFORGE_DFLASH_FUSED_HEAD"] = "0"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for family in args.families:
            for sharding in args.sharding:
                result = _case(family, sharding, args.output)
                results.append(result)
                if dist.get_rank() == 0:
                    print(json.dumps(result, sort_keys=True), flush=True)
                    (args.output / "results.json").write_text(
                        json.dumps(results, indent=2) + "\n"
                    )
                dist.barrier()
    except BaseException:
        # A peer may already be blocked in a collective. Let torchrun terminate
        # peers immediately instead of hiding the traceback in NCCL teardown.
        raise
    else:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
