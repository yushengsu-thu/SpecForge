"""Checkpoint adaptation at the FFN boundary, independent of model strategy."""

import torch


def _per_expert_names(module, prefix):
    return {
        f"{prefix}experts.{expert}.{projection}.weight"
        for expert in range(module.moe_config.n_routed_experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }


def _validate_layout(module, keys, prefix):
    expected = _per_expert_names(module, prefix)
    bank_prefix = prefix + "experts."
    present = {
        key
        for key in keys
        if key.startswith(bank_prefix)
        and key[len(bank_prefix) :].split(".")[0].isdigit()
    }
    if present and (
        present != expected
        or any(
            key in keys
            for key in (bank_prefix + "gate_up_proj", bank_prefix + "down_proj")
        )
    ):
        raise ValueError(
            f"{prefix}MoE checkpoint must contain one complete packed or per-expert bank"
        )
    return present


def normalize_qwen_expert_keys(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Pack original Qwen3-Next per-expert weights into the native Qwen layout.

    This accepts only a complete bank; partial or mixed layouts fail instead of
    quietly leaving expert slices uninitialized. Unrelated model keys are kept.
    """
    del local_metadata, strict, missing_keys, unexpected_keys
    try:
        present = _validate_layout(module, state_dict, prefix)
    except ValueError as error:
        error_msgs.append(str(error))
        return
    if not present:
        return
    packed_names = (prefix + "experts.gate_up_proj", prefix + "experts.down_proj")
    gates, downs = [], []
    for expert in range(module.moe_config.n_routed_experts):
        base = f"{prefix}experts.{expert}."
        gates.append(
            torch.cat(
                [
                    state_dict[base + "gate_proj.weight"],
                    state_dict[base + "up_proj.weight"],
                ],
                dim=0,
            )
        )
        downs.append(state_dict[base + "down_proj.weight"])
    state_dict[packed_names[0]] = torch.stack(gates)
    state_dict[packed_names[1]] = torch.stack(downs)
    for key in present:
        del state_dict[key]


def normalize_moe_checkpoint_state(model, state_dict):
    """Return canonical expert keys for export without mutating caller state."""
    from .layer import MoELayer

    normalized = dict(state_dict)
    errors = []
    for name, module in model.named_modules():
        if isinstance(module, MoELayer):
            normalize_qwen_expert_keys(
                module, normalized, name + "." if name else "", {}, True, [], [], errors
            )
    if errors:
        raise ValueError("; ".join(errors))
    return normalized


def prepare_hf_moe_load(model, state_dict, checkpoint_files, load_config):
    """Use the pinned Transformers streaming loader for original Next banks.

    Modern HF loading bypasses nn.Module load hooks. Attach converters to this
    load only; native saves stay packed and no global conversion registry is
    changed. Header/index inspection validates completeness before conversion.
    """
    from dataclasses import replace

    from safetensors import safe_open
    from transformers.core_model_loading import (
        Concatenate,
        MergeModulelist,
        WeightConverter,
    )

    from .layer import MoELayer

    layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, MoELayer)
    ]
    if not layers:
        return load_config
    metadata = load_config.sharded_metadata or {}
    keys = set(metadata.get("all_checkpoint_keys", ()))
    if state_dict is not None:
        keys.update(state_dict)
    if not keys:
        for path in checkpoint_files or ():
            if str(path).endswith(".safetensors"):
                with safe_open(path, framework="pt") as handle:
                    keys.update(handle.keys())
            else:
                keys.update(
                    torch.load(path, map_location="meta", weights_only=True, mmap=True)
                )
    conversions = []
    for name, module in layers:
        prefix = name + "." if name else ""
        if not _validate_layout(module, keys, prefix):
            continue
        if load_config.hf_quantizer is not None:
            raise ValueError(
                "Original per-expert MoE conversion supports unquantized weights only"
            )
        conversions.extend(
            [
                WeightConverter(
                    source_patterns=[
                        prefix + "experts.*.gate_proj.weight",
                        prefix + "experts.*.up_proj.weight",
                    ],
                    target_patterns=prefix + "experts.gate_up_proj",
                    operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
                ),
                WeightConverter(
                    source_patterns=prefix + "experts.*.down_proj.weight",
                    target_patterns=prefix + "experts.down_proj",
                    operations=[MergeModulelist(dim=0)],
                ),
            ]
        )
    if not conversions:
        return load_config
    return replace(
        load_config, weight_mapping=[*(load_config.weight_mapping or []), *conversions]
    )
