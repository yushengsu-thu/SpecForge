#!/usr/bin/env python3
"""Per-family report: (A) accept length per epoch and (B) throughput at c=1/8/32 on the five datasets, each family in
its own tables, always next to dense 4ep and official v1 (and the no-draft baseline when measured)."""
import json, os, sys
E = "/scratch/dense5l-eval/logs"
DS = [("gsm8k", "GSM8K (5-shot, non-thinking)"), ("math500", "MATH500 (thinking)"), ("mt_bench", "MT-Bench (2 turns, thinking)"), ("aime26", "AIME26 (thinking, 32k)"), ("traces", "training traces (thinking)")]
FAMILIES = [("256-expert MoE (RadixArk/qwen38-dspark-moe-256exp)", "moe256", [1, 2, 3, 4], "moe256-4ep", "256exp 4ep"),
            ("16x1024 MoE (RadixArk/qwen38-dspark-moe-16x1024)", "moe16x", [1, 2, 3, 4, 5], "moe16x-5ep", "16x1024 5ep")]
REF = [("dense5l-4ep", "dense 4ep"), ("official-v1", "official v1")]
def load(p):
    try: return json.load(open(p))["datasets"]
    except Exception: return None
def acc(lab):
    return load(f"{E}/eval_{lab}.json")
def fmt_acc(d, ds):
    r = (d or {}).get(ds); return "-" if not r or r.get("drafts_per_step") is None else f"{r['drafts_per_step']:.2f}"
def fmt_tput(d, ds, base):
    r = (d or {}).get(ds)
    if not r: return "-"
    s = f"{r['tokens_per_s']:.1f}"
    b = (base or {}).get(ds, {}).get("tokens_per_s")
    if b: s += f" ({r['tokens_per_s']/b:.2f}x)"
    s += f" [{r['drafts_per_step']:.2f}]" if r.get("drafts_per_step") is not None else ""
    if r.get("errors"): s += f" err={r['errors']}"
    return s
for title, prefix, epochs, best, bestname in FAMILIES:
    print(f"## {title}\n")
    cols = [(f"{lab}", name) for lab, name in REF] + [(f"{prefix}-{e}ep", f"{bestname.split()[0]} {e}ep") for e in epochs]
    print("### A. Accept length: accepted drafts per verify step (aggregate accept length minus the bonus token); c=8, greedy, bf16 target, one H200\n")
    print("| dataset | " + " | ".join(n for _, n in cols) + " |"); print("|---|" + "---|" * len(cols))
    accs = {lab: acc(lab) for lab, _ in cols}
    for ds, name in DS: print(f"| {name} | " + " | ".join(fmt_acc(accs[lab], ds) for lab, _ in cols) + " |")
    print("\n### B. Throughput: output tokens/s on the same five datasets (speed-up over no draft) [accepted drafts per step]; greedy, bf16 target, one H200, 32 running requests; prompts per dataset: c=1 -> 64/32/32x2/8/32, c=8 and c=32 -> 256/128/80x2/30/128\n")
    tcols = REF + [(best, bestname)]
    for c in (1, 8, 32):
        base = load(f"{E}/tput_none_c{c}.json")
        data = {lab: load(f"{E}/tput_{lab}_c{c}.json") for lab, _ in tcols}
        if not any(data.values()): print(f"(c={c}: not measured yet)\n"); continue
        hdr = (["no draft"] if base else []) + [n for _, n in tcols]
        print(f"| c={c} | " + " | ".join(hdr) + " |"); print("|---|" + "---|" * len(hdr))
        for ds, name in DS:
            cells = ([f"{base[ds]['tokens_per_s']:.1f}" if base and base.get(ds) else "-"] if base else []) + [fmt_tput(data[lab], ds, base) for lab, _ in tcols]
            print(f"| {name} | " + " | ".join(cells) + " |")
        print()
