"""Checkpoint layout contract for draft MoE layers.

The supported Qwen family uses its official fused names directly at runtime.
This identity adapter is intentional: PyTorch/FSDP state dictionaries and
Transformers' meta-device loader see the same names and shapes, without hooks
that might rename sharded parameters. Future families should translate full
checkpoint tensors here, before loading or after FSDP gathers them; a local
parameter shard must never be treated as a complete expert tensor.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

QWEN3_5_MOE_LAYOUT = "qwen3_5_moe"


def validate_layout(layout: str) -> None:
    if layout != QWEN3_5_MOE_LAYOUT:
        raise ValueError(
            f"unsupported draft MoE weight layout {layout!r}; "
            f"expected {QWEN3_5_MOE_LAYOUT!r}"
        )


def canonicalize_moe_state_dict(
    state_dict: Mapping[str, Any], *, layout: str = QWEN3_5_MOE_LAYOUT
) -> dict[str, Any]:
    """Validate complete native expert tensors and preserve their identity.

    Other draft weights pass through unchanged. This function belongs at a
    full-state I/O boundary, not in a module state-dict hook: FSDP's hooks may
    still be processing parameter shards when nested module hooks execute.
    """
    validate_layout(layout)
    result = dict(state_dict)
    for key, value in result.items():
        if re.search(r"(?:^|\.)experts\.\d+\.(?:gate|up|down)_proj\.weight$", key):
            raise ValueError(f"{layout} requires fused expert tensors, got {key!r}")
        down_suffix = "experts.down_proj"
        if key == down_suffix or key.endswith("." + down_suffix):
            gate_up_key = key[: -len(down_suffix)] + "experts.gate_up_proj"
            if gate_up_key not in result:
                raise ValueError(f"missing fused expert tensor {gate_up_key!r}")
        suffix = "experts.gate_up_proj"
        if key != suffix and not key.endswith("." + suffix):
            continue
        if value.ndim != 3 or value.shape[1] % 2:
            raise ValueError(
                f"{key} must have shape [experts, 2 * intermediate, hidden], "
                f"got {tuple(value.shape)}"
            )
        prefix = key[: -len(suffix)]
        down_key = prefix + "experts.down_proj"
        if down_key not in result:
            raise ValueError(f"missing fused expert tensor {down_key!r}")
        expected = (value.shape[0], value.shape[2], value.shape[1] // 2)
        if tuple(result[down_key].shape) != expected:
            raise ValueError(
                f"{down_key} must have shape {expected}, "
                f"got {tuple(result[down_key].shape)}"
            )
        gate_key = prefix + "gate.weight"
        if gate_key in result:
            expected_gate = (value.shape[0], value.shape[2])
            if tuple(result[gate_key].shape) != expected_gate:
                raise ValueError(
                    f"{gate_key} must have shape {expected_gate}, "
                    f"got {tuple(result[gate_key].shape)}"
                )
    return result


__all__ = [
    "QWEN3_5_MOE_LAYOUT",
    "canonicalize_moe_state_dict",
    "validate_layout",
]
