import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

root = Path("/scratch/specforge-torchtitan-20261003/graph-probes")
states = {}
for name in ("graph-final-dflash2", "graph-resume"):
    path = root / (name + "-state.pt")
    dcp_to_torch_save(root / name / "checkpoint/step-3", path)
    states[name] = torch.load(path, map_location="cpu", weights_only=False)

counts = {"tensors": 0, "values": 0}
differences = []


def compare(left, right, key=""):
    if isinstance(left, torch.Tensor):
        counts["tensors"] += 1
        if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
            differences.append(key)
    elif isinstance(left, dict):
        if set(left) != set(right):
            differences.append(key + "/keys")
        for name in left.keys() & right.keys():
            compare(left[name], right[name], key + "/" + str(name))
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            differences.append(key + "/length")
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, key + "/" + str(index))
    else:
        counts["values"] += 1
        if left != right:
            differences.append(key)


compare(states["graph-final-dflash2"], states["graph-resume"])


def draft_state(name):
    result = {}
    for path in (root / name / "draft").glob("*.safetensors"):
        shard = load_file(path)
        assert not result.keys() & shard.keys()
        result.update(shard)
    return result


exports = {name: draft_state(name) for name in ("graph-final-dflash2", "graph-resume")}
assert all(set(state) == set(exports["graph-final-dflash2"]) for state in exports.values())
export_equal = all(torch.equal(value, exports["graph-resume"][key]) for key, value in exports["graph-final-dflash2"].items())
report = {"checkpoint_bitwise_equal": not differences, "differences": differences, "compared": counts, "export_keys": len(exports["graph-final-dflash2"]), "resume_export_bitwise_equal": export_equal}
(root / "graph-verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
assert not differences and export_equal
