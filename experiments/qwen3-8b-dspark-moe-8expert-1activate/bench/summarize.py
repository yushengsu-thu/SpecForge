#!/usr/bin/env python3
"""Tables for the epoch-12 MoE vs dense-1/8 benchmark:  summarize.py <results dir> [--md out.md] [--json out.json]
   [--e2e-note TEXT] [--compare-e2e DIR --compare-name NAME]  (DIR = an earlier e2e run, tabulated as a cross-check)"""
import argparse, glob, json, os
DS = [("gsm8k", "GSM8K"), ("math500", "MATH500"), ("humaneval", "HumanEval"), ("mbpp", "MBPP"), ("mt-bench", "MT-Bench")]
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--md"); ap.add_argument("--json")
ap.add_argument("--e2e-note"); ap.add_argument("--compare-e2e"); ap.add_argument("--compare-name", default="the earlier run")
a = ap.parse_args()
acc = {}
for f in sorted(glob.glob(os.path.join(a.dir, "accept_*.json"))):
    r = json.load(open(f)); acc[(r["model"], int(r["width"]))] = r
def load_e2e(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "e2e_*.json"))):
        name = os.path.basename(f)[4:-5]                  # <label>_<dataset>_c<c>
        label, rest = name.split("_", 1); ds, c = rest.rsplit("_c", 1)
        out[(label, ds, int(c))] = json.load(open(f))
    return out
e2e = load_e2e(a.dir)
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
    L.append("Output tokens/s of `specforge benchmark`: c = concurrent requests (100 prompts at c=1, 200 at c=8 and c=32, "
             "a smaller dataset wraps around); MT-Bench sends the first turn only. Speed-up = tok/s over the no-draft "
             "baseline on the same task and concurrency.\n")
    if a.e2e_note: L.append(a.e2e_note + "\n")
    labels = sorted({k[0] for k in e2e}, key=lambda x: (x != "baseline", x))
    e2e_ds = [(k, n) for k, n in DS if any(kk[1] == k for kk in e2e)]
    tp = lambda lab, ds, c: (e2e.get((lab, ds, c)) or {}).get("throughput_tokens_per_second")
    for ds, dname in e2e_ds:
        L.append(f"\n### {dname}\n")
        L.append("| Config | " + " | ".join(f"c={c} tok/s | c={c} speed-up" for c in (1, 8, 32)) + " | accept len (c=1) |")
        L.append("|" + "---|" * 8)
        for lab in labels:
            cells = []
            for c in (1, 8, 32):
                r, b = tp(lab, ds, c), tp("baseline", ds, c)
                cells += [f"{r:.1f}" if r else "-", f"{r / b:.2f}x" if r and b else "-"]
            al = (e2e.get((lab, ds, 1)) or {}).get("average_acceptance_length")
            L.append(f"| {lab} | " + " | ".join(cells) + (f" | {al:.2f} |" if al else " | - |"))
    spec = [lab for lab in labels if lab != "baseline"]
    if len(e2e_ds) > 1 and spec:
        names = ", ".join(n for _, n in e2e_ds)
        L.append(f"\n### Mean over the {len(e2e_ds)} tasks ({names}), equal weight\n")
        L.append("| Config | c=1 speed-up | c=8 speed-up | c=32 speed-up | accept len (c=1) |")
        L.append("|---|---|---|---|---|")
        for lab in spec:
            cells = []
            for c in (1, 8, 32):
                sp = [tp(lab, d, c) / tp("baseline", d, c) for d, _ in e2e_ds if tp(lab, d, c) and tp("baseline", d, c)]
                cells.append((f"{sum(sp) / len(sp):.2f}x" + ("" if len(sp) == len(e2e_ds) else f" ({len(sp)} tasks)")) if sp else "-")
            als = [x for x in ((e2e.get((lab, d, 1)) or {}).get("average_acceptance_length") for d, _ in e2e_ds) if x]
            L.append(f"| {lab} | " + " | ".join(cells) + (f" | {sum(als) / len(als):.2f} |" if als else " | - |"))
    pairs = [(f"V{w}", f"moe-v{w}", f"dense18-v{w}") for w in (16, 8) if f"moe-v{w}" in labels and f"dense18-v{w}" in labels]
    if pairs:
        L.append("\n### MoE / dense-1/8 throughput ratio (same width, task and concurrency)\n")
        L.append("| Task | " + " | ".join(f"{v} c={c}" for v, _, _ in pairs for c in (1, 8, 32)) + " |")
        L.append("|" + "---|" * (1 + 3 * len(pairs)))
        rows = e2e_ds + ([("__mean__", "**Mean**")] if len(e2e_ds) > 1 else [])
        for ds, dname in rows:
            cells = []
            for v, m, d in pairs:
                for c in (1, 8, 32):
                    if ds == "__mean__":
                        rs = [tp(m, x, c) / tp(d, x, c) for x, _ in e2e_ds if tp(m, x, c) and tp(d, x, c)]
                        cells.append(f"{sum(rs) / len(rs):.3f}" if rs else "-")
                    else:
                        cells.append(f"{tp(m, ds, c) / tp(d, ds, c):.3f}" if tp(m, ds, c) and tp(d, ds, c) else "-")
            L.append(f"| {dname} | " + " | ".join(cells) + " |")
if e2e and a.compare_e2e:
    old = load_e2e(a.compare_e2e)
    both = sorted({(k[0], k[1]) for k in old if (k[0], k[1], k[2]) in e2e},
                  key=lambda x: (x[0] != "baseline", x[0], [k for k, _ in DS].index(x[1]) if x[1] in dict(DS) else 99))
    if both:
        L.append(f"\n### Cross-check against {a.compare_name}\n")
        L.append("tok/s of this run, with the ratio this run / earlier run in brackets.\n")
        L.append("| Config | Task | c=1 | c=8 | c=32 |"); L.append("|---|---|---|---|---|")
        rs = []
        for lab, ds in both:
            cells = []
            for c in (1, 8, 32):
                n_, o_ = (e2e.get((lab, ds, c)) or {}).get("throughput_tokens_per_second"), (old.get((lab, ds, c)) or {}).get("throughput_tokens_per_second")
                if n_ and o_: rs.append(n_ / o_); cells.append(f"{n_:.1f} ({n_ / o_:.3f})")
                else: cells.append("-")
            L.append(f"| {lab} | {dict(DS).get(ds, ds)} | " + " | ".join(cells) + " |")
        if rs: L.append(f"\nRatio range {min(rs):.3f} to {max(rs):.3f}, mean {sum(rs) / len(rs):.3f} over {len(rs)} cells.\n")
text = "\n".join(L) + "\n"
print(text)
if a.md: open(a.md, "w").write(text)
if a.json:
    json.dump({"accept": {f"{m}/V{w}": r for (m, w), r in acc.items()},
               "e2e": {f"{l}/{d}/c{c}": r for (l, d, c), r in e2e.items()}}, open(a.json, "w"), indent=2)
