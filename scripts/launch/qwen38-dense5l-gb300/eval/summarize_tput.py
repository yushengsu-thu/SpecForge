#!/usr/bin/env python3
"""Per-dataset output tokens/s at c=1/8/32 from logs/tput_<label>_c<c>.json; cell = tok/s (speed-up over no draft) [drafts/step]."""
import json, os, glob
E = "/scratch/dense5l-eval/logs"
LABELS = [("none", "no draft (autoregressive)"), ("dense5l-4ep", "dense 4ep"), ("official-v1", "official v1"), ("moe256-4ep", "MoE-256exp 4ep"), ("moe16x-5ep", "MoE-16x1024 5ep")]
DS = [("gsm8k", "GSM8K (5-shot, non-thinking, max 1024)"), ("math500", "MATH500 (thinking, max 8192)"), ("mt_bench", "MT-Bench (2 turns, thinking, max 4096)"), ("aime26", "AIME26 (thinking, max 32768)"), ("traces", "training traces (thinking, max 8192)")]
CS = [1, 8, 32]
res = {}
for lab, _ in LABELS:
    for c in CS:
        p = f"{E}/tput_{lab}_c{c}.json"
        if os.path.exists(p):
            try: res[(lab, c)] = json.load(open(p))["datasets"]
            except Exception: pass
print("### Output tokens/s on the accept-length datasets at concurrency 1 / 8 / 32 (greedy, bf16 target, one H200 per server; prompts per dataset: c=1 -> 64/32/32x2/8/32, c=8 and c=32 -> 256/128/80x2/30/128; 32 running requests, Mamba cache 32 slots); cell = tok/s (speed-up over no draft) [accepted drafts per step]\n")
for ds, title in DS:
    print(f"| {title} | " + " | ".join(f"c={c}" for c in CS) + " |"); print("|---|" + "---|" * len(CS))
    for lab, name in LABELS:
        if not any((lab, c) in res for c in CS): continue
        cells = []
        for c in CS:
            d = res.get((lab, c), {}).get(ds)
            if not d: cells.append("-"); continue
            t = d["tokens_per_s"]; base = res.get(("none", c), {}).get(ds, {}).get("tokens_per_s")
            s = f"{t:.1f}"
            if lab != "none" and base: s += f" ({t/base:.2f}x)"
            if d.get("drafts_per_step") is not None: s += f" [{d['drafts_per_step']:.2f}]"
            if d.get("errors"): s += f" errors={d['errors']}"
            cells.append(s)
        print(f"| {name} | " + " | ".join(cells) + " |")
    print()
