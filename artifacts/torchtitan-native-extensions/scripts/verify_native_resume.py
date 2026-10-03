"""CPU-only comparison of native DCP state and the exported draft weights."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", required=True)
parser.add_argument("--resumed", required=True)
parser.add_argument("--step", type=int, required=True)
parser.add_argument("--allow-fixture-path-differences", action="store_true")
parser.add_argument("--output", required=True)
args = parser.parse_args()
counts = {"tensors": 0, "non_tensor_leaves": 0}
excluded = []


def checkpoint(path):
    path = Path(path)
    converted = path / f"verification-step{args.step}.pt"
    dcp_to_torch_save(path / "checkpoint" / f"step-{args.step}", converted)
    return torch.load(converted, map_location="cpu", weights_only=False)


def compare(left, right, path="root"):
    allowed = {
        "root.train_state.specforge_resume_contract.specforge_model_provenance.1.1.1",
        "root.train_state.specforge_resume_contract.stream_path",
        "root.train_state.specforge_resume_contract.ledger_path",
    }
    if args.allow_fixture_path_differences and path in allowed and left != right:
        assert isinstance(left, str) and isinstance(right, str), path
        excluded.append(path)
    elif isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert torch.equal(left, right), path
        counts["tensors"] += 1
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            compare(left[key], right[key], path + "." + str(key))
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (value, other) in enumerate(zip(left, right)):
            compare(value, other, path + f".{index}")
    else:
        assert left == right, (path, left, right)
        counts["non_tensor_leaves"] += 1


compare(checkpoint(args.baseline), checkpoint(args.resumed))
dcp_counts = dict(counts)
compare(
    load_file(str(Path(args.baseline) / "draft/model.safetensors")),
    load_file(str(Path(args.resumed) / "draft/model.safetensors")),
    "hf_export",
)
result = {
    "passed": True,
    "comparison": "bitwise",
    "step": args.step,
    "baseline": args.baseline,
    "resumed": args.resumed,
    "dcp": dcp_counts,
    "hf_export_tensors": counts["tensors"] - dcp_counts["tensors"],
    "total": counts,
    "excluded_fixture_identity_paths": excluded,
    "scope": "model, optimizer, scheduler, rank RNG, data progress, other DCP state and exported draft",
}
Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
