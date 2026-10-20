"""Multi-GPU gate for per-decoder FSDP, auxiliary loss, and accumulation.

Run with ``torchrun --standalone --nproc-per-node=8
tests/test_runtime/moe_distributed_gate.py --report /tmp/moe-fsdp.json``.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import Qwen3Config

from specforge.distributed import destroy_distributed, init_distributed
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.strategies.base import StepOutput, finalize_step_output


class Composite(nn.Module):
    def __init__(self, draft):
        super().__init__()
        self.draft_model = draft

    def forward(self, **inputs):
        return self.draft_model(**inputs)


def case(sharding, checkpointing):
    rank = dist.get_rank()
    torch.manual_seed(29)
    config = Qwen3Config(
        architectures=["DFlashDraftModel"],
        block_size=4,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=2,
        num_target_layers=4,
        head_dim=64,
        max_position_embeddings=64,
        vocab_size=512,
        layer_types=["full_attention", "full_attention"],
        moe_preset="qwen3_5_moe",
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        dflash_config={
            "target_layer_ids": [1],
            "mask_token_id": 0,
            "moe_aux_loss_coeff": 0.1,
            "moe_dispatch": "grouped_mm",
        },
    )
    config._attn_implementation = "sdpa"
    draft = DFlashDraftModel(config).to(device=rank, dtype=torch.bfloat16).train()
    if checkpointing:
        draft.gradient_checkpointing_enable({"use_reentrant": False})
    composite = Composite(draft)
    pc = ParallelConfig.from_distributed(sharding_strategy=sharding)
    pc.sharding_strategy = sharding
    backend = FSDPTrainingBackend(
        pc,
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.1),
    )
    wrapped = backend.prepare_model(composite, optimizer_target=draft)
    assert len(backend.auto_wrap_block_classes) == 1
    fsdp_units = sum(isinstance(module, FSDP) for module in wrapped.modules())
    assert fsdp_units == 3, fsdp_units
    with FSDP.summon_full_params(wrapped):
        before = {
            name: value.detach().clone()
            for name, value in draft.named_parameters()
            if ".mlp.gate.weight" in name or ".mlp.experts." in name
        }
    metrics = []
    for microbatch in range(2):
        torch.manual_seed(120 + rank * 10 + microbatch)
        inputs = {
            "position_ids": torch.arange(12, device=rank)[None, :],
            "noise_embedding": torch.randn(
                1, 4, 128, device=rank, dtype=torch.bfloat16
            ),
            "target_hidden": torch.randn(1, 8, 128, device=rank, dtype=torch.bfloat16),
        }
        output = wrapped(**inputs)
        task = (output.float() - 0.2).square().mean()
        result = finalize_step_output(StepOutput(task, {}), wrapped)
        assert result.metrics["moe/aux_loss_weighted"].item() > 0.0
        metrics.append({key: value.item() for key, value in result.metrics.items()})
        backend.backward(result.loss / 2, is_boundary=microbatch == 1)
    grad_norms = {"router": 0.0, "experts": 0.0}
    for name, parameter in draft.named_parameters():
        if parameter.grad is None:
            continue
        assert torch.isfinite(parameter.grad).all(), name
        key = "router" if ".mlp.gate.weight" in name else "experts"
        if ".mlp.gate.weight" in name or ".mlp.experts." in name:
            grad_norms[key] += parameter.grad.float().square().sum().item()
    tensor = torch.tensor(list(grad_norms.values()), device=rank)
    dist.all_reduce(tensor)
    assert (tensor > 0).all(), tensor
    backend.step()
    with FSDP.summon_full_params(wrapped):
        deltas = {
            name: (parameter.detach() - before[name]).float().abs().max().item()
            for name, parameter in draft.named_parameters()
            if name in before
        }
    assert any(
        value > 0 for name, value in deltas.items() if ".mlp.gate.weight" in name
    )
    assert any(value > 0 for name, value in deltas.items() if ".mlp.experts." in name)
    report = {
        "sharding": sharding,
        "checkpointing": checkpointing,
        "world_size": dist.get_world_size(),
        "fsdp_units": fsdp_units,
        "accumulation_steps": 2,
        "microbatch_metrics": metrics,
        "global_router_gradient_l2": tensor[0].sqrt().item(),
        "global_expert_gradient_l2": tensor[1].sqrt().item(),
        "parameter_max_deltas": deltas,
    }
    if rank == 0:
        print(json.dumps(report), flush=True)
    del wrapped, backend, draft, composite
    torch.cuda.empty_cache()
    dist.barrier()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    init_distributed(timeout=120, tp_size=1)
    try:
        reports = [
            case(sharding, checkpointing)
            for sharding in ("FULL_SHARD", "SHARD_GRAD_OP")
            for checkpointing in (False, True)
        ]
        if dist.get_rank() == 0 and args.report is not None:
            args.report.write_text(json.dumps(reports, indent=2) + "\n")
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
