#!/usr/bin/env python3
"""Accept-length tables (request average tau) per task and width: dense-1-8-only-5-epoch ep1..ep5 next to the earlier
measurements on this node (dense 2-epoch run, dense-1/8 32-epoch run at epoch 12, 5-layer MoE 8x1536 top-1 at epoch 6)."""
import json, os

D = "/scratch/pr1003/eval5ep/results"
REFS = [("dense 2ep (2-epoch run)", "/scratch/dense-eval/results", "dense2ep-ep2"),
        ("dense-1/8 ep12 (32-epoch run)", "/scratch/SpecForge-spark8b/experiments/qwen3-8b-dspark-moe-8expert-1activate/bench/results_epoch12", "dense18"),
        ("5-layer MoE 8x1536 ep6 (PR #1031)", "/scratch/ep6-serving/results", "pr1031-ep6")]
COLS = [(f"5-epoch run ep{k}", D, f"dense5ep-ep{k}") for k in range(1, 6)] + REFS
TASKS = [("gsm8k", "GSM8K"), ("math500", "MATH-500"), ("humaneval", "HumanEval"), ("mbpp", "MBPP"), ("mt-bench", "MT-Bench")]


def load(dirname, stem, width):
    p = os.path.join(dirname, f"accept_{stem}_v{width}.json")
    return json.load(open(p)) if os.path.exists(p) else None


out = {}
for width in (8, 16):
    cols = [(label, load(d, s, width)) for label, d, s in COLS]
    if not any(r is not None for (label, d, s), (_, r) in zip(COLS, cols) if d == D):
        continue  # nothing measured at this width yet
    cols = [(label, r) for label, r in cols if r is not None]
    print(f"\n## Accept length, temperature 0, verify width {width} (request average)\n")
    print("| Task | " + " | ".join(label for label, _ in cols) + " |")
    print("|---|" + "---|" * len(cols))
    rows = {}
    for key, name in TASKS:
        vals = [r["datasets"][key]["request_macro_tau"] for _, r in cols]
        rows[name] = vals
        print(f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    for key, name in (("request_macro_tau", "Pooled (request average)"), ("benchmark_equal_weight_macro_tau", "Equal-weight mean of 5 tasks"),
                      ("micro_tau", "Pooled (tokens / verify steps)")):
        vals = [r["pooled"][key] for _, r in cols]
        rows[name] = vals
        print(f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    errs = [(label, len(r.get("errors", []))) for label, r in cols if r.get("errors")]
    if errs:
        print("\nRequest errors: " + ", ".join(f"{l}: {n}" for l, n in errs))
    out[width] = {"columns": [label for label, _ in cols], "rows": rows,
                  "elapsed_s": {label: r.get("elapsed_s") for label, r in cols},
                  "length_limited": {label: r["pooled"].get("length_limited_requests") for label, r in cols}}
json.dump(out, open("/scratch/pr1003/eval5ep/summary.json", "w"), indent=1)
