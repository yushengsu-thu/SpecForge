"""Correctness-only optimizer hook collector; all measured times are invalid.

Pass ordinary benchmark arguments. COLLECT_DRIVER selects the frozen driver;
COLLECT_DIRECTORY is a fresh directory for rank-zero full-tensor snapshots.
Optional PROBE_PRESERVE_CASTS / PROBE_DIVISION_ROUNDING match root's ablations.
This wrapper never replaces Trainer's loop, backward, clipping, or optimizer.
"""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

for environment_name, config_name in (
    ("PROBE_PRESERVE_CASTS", "emulate_precision_casts"),
    ("PROBE_DIVISION_ROUNDING", "eager_numerics.division_rounding"),
):
    if environment_name in os.environ:
        import torch._inductor.config as inductor_config

        if "." in config_name:
            setattr(inductor_config.eager_numerics, "division_rounding", os.environ[environment_name] == "1")
        else:
            setattr(inductor_config, config_name, os.environ[environment_name] == "1")

driver_path = Path(os.environ["COLLECT_DRIVER"]).resolve()
output_directory = Path(os.environ["COLLECT_DIRECTORY"]).resolve()
sys.path.insert(0, str(driver_path.parent))
spec = importlib.util.spec_from_file_location("collected_benchmark", driver_path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
original_run_titan = benchmark.run_titan
disabled_graph_passes = []
if os.environ.get("COLLECT_FLOAT32") == "1":
    original_build_model = benchmark.recipes._build_model
    original_make_batches = benchmark.recipes._make_batches

    def build_float32(args, config, device, dtype):
        return original_build_model(args, config, device, torch.float32)

    def batches_float32(args, config, rank, device, dtype):
        return original_make_batches(args, config, rank, device, torch.float32)

    benchmark.recipes._build_model = build_float32
    benchmark.recipes._make_batches = batches_float32
if "COLLECT_TINY_HEAD_DIM" in os.environ:
    original_resolve_config = benchmark.recipes.resolve_config

    def resolve_config(*args, **kwargs):
        config, source = original_resolve_config(*args, **kwargs)
        if not kwargs.get("tiny"):
            raise ValueError("COLLECT_TINY_HEAD_DIM requires --tiny")
        config["head_dim"] = int(os.environ["COLLECT_TINY_HEAD_DIM"])
        config["hidden_size"] = config["head_dim"] * config["num_attention_heads"]
        config["intermediate_size"] = 2 * config["hidden_size"]
        return config, source

    benchmark.recipes.resolve_config = resolve_config


def full_copy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if isinstance(value, DTensor):
            value = value.full_tensor()
        return value.cpu().clone() if dist.get_rank() == 0 else None
    return value


def collect_run_titan(*args, **kwargs):
    if os.environ.get("COLLECT_FLOAT32") == "1":
        from specforge.training.torchtitan.runtime import SpecForgeTitanTrainer

        original_init = SpecForgeTitanTrainer.__init__

        def init_float32(self, config):
            config.training.dtype = "float32"
            config.training.mixed_precision_param = "float32"
            config.training.mixed_precision_reduce = "float32"
            original_init(self, config)

        SpecForgeTitanTrainer.__init__ = init_float32
        try:
            train, trainer = original_run_titan(*args, **kwargs)
        finally:
            SpecForgeTitanTrainer.__init__ = original_init
    else:
        train, trainer = original_run_titan(*args, **kwargs)
    if os.environ.get("COLLECT_DISABLE_GRAPH_PASSES") == "1":
        from specforge.training.torchtitan import graph as graph_adapter
        from torchtitan.experiments.graph_trainer.passes import _get_pass_name

        original_construct = graph_adapter.construct_default_graph_passes

        def record_disabled_passes(traced_result, config, **pass_kwargs):
            passes = original_construct(traced_result, config, **pass_kwargs)
            keep = {
                "annotate_flex_attention_for_regional_inductor_pass",
                "regional_inductor_pass",
            } if os.environ.get("COLLECT_KEEP_FLEX_PASSES") == "1" else set()
            disabled_graph_passes[:] = [_get_pass_name(fn) for fn in passes
                                       if _get_pass_name(fn) not in keep]
            config.compile.disable_passes = list(disabled_graph_passes)
            return passes

        graph_adapter.construct_default_graph_passes = record_disabled_passes
    names = {}
    parameters = {}
    for model in trainer.model_parts:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            name = ".".join(part for part in name.split(".") if part not in (
                "_orig_mod", "_checkpoint_wrapped_module"))
            if name in parameters or id(parameter) in names:
                raise AssertionError(f"Duplicate canonical parameter {name}")
            names[id(parameter)] = name
            parameters[name] = parameter
    optimizer_by_parameter = {}
    for optimizer in trainer.optimizers.optimizers:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if id(parameter) in optimizer_by_parameter:
                    raise AssertionError("Parameter appears in multiple optimizer groups")
                optimizer_by_parameter[id(parameter)] = optimizer
    if set(names) != set(optimizer_by_parameter):
        raise AssertionError("Optimizer/model parameter mapping is incomplete")
    output_directory.mkdir(parents=True, exist_ok=True)
    if dist.get_rank() == 0:
        (output_directory / "collector.py").write_bytes(Path(__file__).read_bytes())

    def snapshot(phase):
        if trainer.step > 2:
            return
        result = {"step": trainer.step, "phase": phase, "parameters": {},
                  "gradients": {}, "adam_state": {}, "parameter_layout": {}}
        for name, parameter in sorted(parameters.items()):
            result["parameters"][name] = full_copy(parameter)
            result["parameter_layout"][name] = dict(
                dtype=str(parameter.dtype), global_shape=list(parameter.shape),
                placements=str(parameter.placements) if isinstance(parameter, DTensor) else None,
            )
            if phase in ("before_clip", "before_adam"):
                result["gradients"][name] = full_copy(parameter.grad)
            optimizer = optimizer_by_parameter[id(parameter)]
            result["adam_state"][name] = {
                key: full_copy(value)
                for key, value in sorted(optimizer.state.get(parameter, {}).items())
            }
        if phase == "before_adam":
            losses = torch.stack(trainer.recorded_losses).detach().clone()
            group = trainer.parallel_dims.get_optional_mesh("batch", include_singleton_axes=True).get_group()
            dist.all_reduce(losses, op=dist.ReduceOp.SUM, group=group)
            result["global_micro_losses"] = losses.cpu().tolist()
        if dist.get_rank() == 0:
            path = output_directory / f"step{trainer.step}-{phase}.pt"
            if path.exists():
                raise FileExistsError(path)
            torch.save(result, path)
            print(json.dumps({"correctness_snapshot": str(path), "parameters": len(parameters)}), flush=True)

    trainer.optimizers.register_step_pre_hook(lambda optimizer, args, kwargs: snapshot("before_adam"))
    trainer.optimizers.register_step_post_hook(lambda optimizer, args, kwargs: snapshot("after_adam"))
    # Scratch-only interception to separate backward errors from the shared
    # clipping scalar. Delegate the entire original clip implementation.
    import torchtitan.distributed.utils as distributed_utils

    original_clip = distributed_utils.clip_grad_norm_

    def capture_clip(*args, **kwargs):
        snapshot("before_clip")
        return original_clip(*args, **kwargs)

    def collected_train():
        distributed_utils.clip_grad_norm_ = capture_clip
        try:
            return train()
        finally:
            distributed_utils.clip_grad_norm_ = original_clip

    return collected_train, trainer


benchmark.run_titan = collect_run_titan
benchmark.main()
if int(os.environ.get("RANK", "0")) == 0:
    result_path = Path(sys.argv[sys.argv.index("--output") + 1])
    result = json.loads(result_path.read_text())
    if os.environ.get("COLLECT_FLOAT32") == "1":
        result["runtime"]["compute_dtype"] = "float32"
    result["correctness_collector"] = {
        "timings_are_not_performance_results": True,
        "snapshot_directory": str(output_directory),
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preserve_casts_override": os.environ.get("PROBE_PRESERVE_CASTS"),
        "division_rounding_override": os.environ.get("PROBE_DIVISION_ROUNDING"),
        "disabled_default_graph_passes": disabled_graph_passes,
        "float32_override": os.environ.get("COLLECT_FLOAT32") == "1",
        "capture_boundary": "native optimizer-container hooks, after native gradient clipping",
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
