#!/usr/bin/env python3
"""Tables for the epoch-12 MoE vs dense-1/8 benchmark:  summarize.py <results dir> [--md out.md] [--json out.json]"""
import argparse, glob, json, os
DS = [("gsm8k", "GSM8K"), ("math500", "MATH500"), ("humaneval", "HumanEval"), ("mbpp", "MBPP"), ("mt-bench", "MT-Bench")]
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--md"); ap.add_argument("--json")
a = ap.parse_args()
acc = {}
for f in sorted(glob.glob(os.path.join(a.dir, "accept_*.json"))):
    r = json.load(open(f)); acc[(r["model"], int(r["width"]))] = r
e2e = {}
for f in sorted(glob.glob(os.path.join(a.dir, "e2e_*.json"))):
    name = os.path.basename(f)[4:-5]                      # <label>_<dataset>_c<c>
    label, rest = name.split("_", 1); ds, c = rest.rsplit("_c", 1)
    e2e[(label, ds, int(c))] = json.load(open(f))
models = [m for m in ("dense-1/8", "MoE") if any(k[0] == m for k in acc)]
L = []
L.append("## Acceptance length (temperature 0, jiapingW protocol)\n")
L.append("tau = completion_tokens / verify rounds per request (bonus token included); macro = mean over requests, micro = total tokens / total rounds.\n")
for w in (16, 8):
    if not any(k[1] == w for k in acc): continue
    L.append(f"\n### D16/V{w}\n")
    hdr = "| Benchmark | Requests | " + " | ".join(f"{m} macro tau | {m} micro tau" for m in models) + (" | MoE vs dense (macro) |" if len(models) == 2 else " |")
    L.append(hdr); L.append("|" + "---|" * (2 + 2 * len(models) + (1 if len(models) == 2 else 0)))
    rows = [(k, n) for k, n in DS] + [("__pooled__", "**All requests, pooled**")]
    for key, name in rows:
        cells, macros = [], []
        req = None
        for m in models:
            r = acc.get((m, w))
            d = (r["pooled"] if key == "__pooled__" else r["datasets"].get(key)) if r else None
            if d:
                req = d["requests"]; macros.append(d["request_macro_tau"])
                cells += [f"{d['request_macro_tau']:.3f}", f"{d['micro_tau']:.3f}"]
            else:
                cells += ["-", "-"]; macros.append(None)
        gap = ""
        if len(models) == 2 and None not in macros:
            gap = f" {100 * (macros[1] / macros[0] - 1):+.1f}% |"
        L.append(f"| {name} | {req if req is not None else '-'} | " + " | ".join(cells) + (" |" + gap if len(models) == 2 else " |"))
    eq = []
    for m in models:
        r = acc.get((m, w))
        if r: eq.append(f"{m}: equal-weight macro {r['pooled']['benchmark_equal_weight_macro_tau']:.3f}, micro {r['pooled']['benchmark_equal_weight_micro_tau']:.3f}, length-limited {r['pooled']['length_limited_requests']}, errors {len(r['errors'])}")
    L.append("\n" + "; ".join(eq) + "\n")
if e2e:
    L.append("\n## End-to-end serving throughput (SGLang, CUDA graphs on, 512 new tokens, greedy)\n")
    labels = sorted({k[0] for k in e2e}, key=lambda x: (x != "baseline", x))
    for ds in ("gsm8k", "mt-bench"):
        L.append(f"\n### {ds}\n")
        L.append("| Config | " + " | ".join(f"c={c} tok/s | c={c} speed-up" for c in (1, 8, 32)) + " | accept len (c=1) |")
        L.append("|" + "---|" * 8)
        for lab in labels:
            cells = []
            for c in (1, 8, 32):
                r = e2e.get((lab, ds, c)); b = e2e.get(("baseline", ds, c))
                if r:
                    cells.append(f"{r['throughput_tokens_per_second']:.1f}")
                    cells.append(f"{r['throughput_tokens_per_second'] / b['throughput_tokens_per_second']:.2f}x" if b else "-")
                else:
                    cells += ["-", "-"]
            r1 = e2e.get((lab, ds, 1)); al = (r1 or {}).get("average_acceptance_length")
            L.append(f"| {lab} | " + " | ".join(cells) + f" | {al:.2f} |" if al else f"| {lab} | " + " | ".join(cells) + " | - |")
text = "\n".join(L) + "\n"
print(text)
if a.md: open(a.md, "w").write(text)
if a.json:
    json.dump({"accept": {f"{m}/V{w}": r for (m, w), r in acc.items()},
               "e2e": {f"{l}/{d}/c{c}": r for (l, d, c), r in e2e.items()}}, open(a.json, "w"), indent=2)
