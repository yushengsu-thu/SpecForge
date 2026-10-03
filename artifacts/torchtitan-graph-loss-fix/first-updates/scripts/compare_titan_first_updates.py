"""Compare correctness-only first-update snapshots on CPU, without model imports."""

import argparse
import json
import math
from pathlib import Path

import torch


def compare_tensors(reference, candidate):
    if set(reference) != set(candidate):
        raise ValueError("Canonical parameter/state key mismatch")
    details = []
    sum_diff_sq = sum_ref_sq = 0.0
    for name in sorted(reference):
        left, right = reference[name], candidate[name]
        if left is None or right is None:
            if left is not right:
                raise ValueError(f"Missing gradient differs: {name}")
            continue
        if not isinstance(left, torch.Tensor):
            if left != right:
                raise ValueError(f"Non-tensor state differs: {name}")
            continue
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Tensor shape/dtype differs: {name}")
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError(f"Non-finite snapshot: {name}")
        left, right = left.double(), right.double()
        delta = right - left
        ref_sq, diff_sq = left.square().sum().item(), delta.square().sum().item()
        sum_ref_sq += ref_sq
        sum_diff_sq += diff_sq
        details.append(dict(name=name, max_abs=delta.abs().max().item(),
                            l2_relative=math.sqrt(diff_sq / max(ref_sq, 1e-60)),
                            reference_norm=math.sqrt(ref_sq),
                            changed_elements=int((left != right).sum()), elements=left.numel()))
    return dict(
        tensor_count=len(details), bitwise_equal=all(item["changed_elements"] == 0 for item in details),
        max_abs=max((item["max_abs"] for item in details), default=0.0),
        l2_relative=math.sqrt(sum_diff_sq / max(sum_ref_sq, 1e-60)),
        per_parameter=details,
        top_relative_errors=sorted(details, key=lambda item: item["l2_relative"], reverse=True)[:20],
        top_absolute_errors=sorted(details, key=lambda item: item["max_abs"], reverse=True)[:20],
    )


def flatten_state(state):
    return {f"{name}/{key}": value for name, values in state.items() for key, value in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(reference=str(args.reference), candidate=str(args.candidate), windows=[])
    for step in (1, 2):
        snapshots = {}
        for kind, directory in (("ref", args.reference), ("candidate", args.candidate)):
            for phase in ("before_adam", "after_adam"):
                snapshots[(kind, phase)] = torch.load(directory / f"step{step}-{phase}.pt", map_location="cpu", weights_only=True)
        before_ref, before_candidate = snapshots[("ref", "before_adam")], snapshots[("candidate", "before_adam")]
        after_ref, after_candidate = snapshots[("ref", "after_adam")], snapshots[("candidate", "after_adam")]
        ref_losses, candidate_losses = before_ref["global_micro_losses"], before_candidate["global_micro_losses"]
        window = dict(step=step, reference_micro_losses=ref_losses, candidate_micro_losses=candidate_losses,
                      loss_max_abs=max(abs(left - right) for left, right in zip(ref_losses, candidate_losses)),
                      parameters_before=compare_tensors(before_ref["parameters"], before_candidate["parameters"]),
                      clipped_gradients=compare_tensors(before_ref["gradients"], before_candidate["gradients"]),
                      parameters_after=compare_tensors(after_ref["parameters"], after_candidate["parameters"]),
                      adam_before=compare_tensors(flatten_state(before_ref["adam_state"]), flatten_state(before_candidate["adam_state"])),
                      adam_after=compare_tensors(flatten_state(after_ref["adam_state"]), flatten_state(after_candidate["adam_state"])))
        deltas = {
            kind: {name: snapshots[(kind, "after_adam")]["parameters"][name] - before
                   for name, before in snapshots[(kind, "before_adam")]["parameters"].items()}
            for kind in ("ref", "candidate")
        }
        window["parameter_updates"] = compare_tensors(deltas["ref"], deltas["candidate"])
        if all((directory / f"step{step}-before_clip.pt").exists()
               for directory in (args.reference, args.candidate)):
            preclip = [torch.load(directory / f"step{step}-before_clip.pt", map_location="cpu", weights_only=True)
                       for directory in (args.reference, args.candidate)]
            window["unclipped_gradients"] = compare_tensors(preclip[0]["gradients"], preclip[1]["gradients"])
        report["windows"].append(window)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "windows": [
        {key: value if not isinstance(value, dict) else {metric: value[metric] for metric in (
            "bitwise_equal", "max_abs", "l2_relative")}
         for key, value in window.items() if key not in ("reference_micro_losses", "candidate_micro_losses")}
        for window in report["windows"]]}, indent=2))


if __name__ == "__main__":
    main()
