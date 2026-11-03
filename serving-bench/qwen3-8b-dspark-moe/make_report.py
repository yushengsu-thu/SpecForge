#!/usr/bin/env python3
"""Produce the agreed deliverables from serving-bench result dirs, in the agreed format:
  1. accept-length table   | | Avg (all requests) | Avg (per task) | gsm8k | math500 | humaneval | mbpp | mt-bench |  rows dense / MoE / diff / diff %
  2. five per-task tables  | c | dense tok/s | MoE tok/s | diff % | dense step ms | MoE step ms |   cells = mean ± sample std over the groups
  3. mixed-stream table, same columns; plus the extended one with the MoE default config and both deltas when --moe-default is given
  4. the line chart (dense blue circles, MoE default gray triangles, MoE tuned orange squares; x = 1,8,16,32,64,128,256 equally spaced)
A group is one repeat of the whole protocol; give each group as one or more dirs (comma-separated) when it was split into units.
Dirs are the serving-bench layout: <root>/tasks/<name>/<task>_c<c>.json and <root>/mixed/<name>/mixed_c<c>.json.

usage: make_report.py --dense ROOT:g1,g2,... --moe ROOT:g1,... [--moe-default ROOT:g1,...] --out DIR [--title ...]
   e.g. --dense /scratch/ckpt-sweep/bench/logs:dense-g1,dense-g2,dense-g3,dense-g4-big+dense-g4-small,dense-g5-big+dense-g5-small
        (a '+' joins unit dirs of one group; mixed units are found by the same names with -m1/-mrest or -big/-small swapped automatically)"""
import argparse, json, os, statistics, sys
TASKS = [("gsm8k", 1319), ("math500", 500), ("humaneval", 164), ("mbpp", 257), ("mt-bench", 160)]; CONCS = [1, 8, 16, 32, 64, 128, 256]
pct = lambda a, b: 100.0 * (a - b) / b
def parse(spec):
    root, names = spec.split(":", 1)
    return [[os.path.join(root, u) for u in g.split("+")] for g in names.split(",")]
def variants(dirs, kind):   # tasks units: -big/-small ; mixed units: -m1/-mrest ; plain names serve both
    out = []
    for d in dirs:
        base = os.path.basename(d); parent = os.path.dirname(d)
        cands = [base] + ([base.replace("-m1", "-big").replace("-mrest", "-small")] if kind == "tasks" else [base.replace("-big", "-m1").replace("-small", "-mrest")])
        out += [os.path.join(parent, kind, c) for c in cands]
    return out
def find(dirs, kind, fname):
    hits = [os.path.join(d, fname) for d in variants(dirs, kind) if os.path.exists(os.path.join(d, fname))]
    return json.load(open(hits[0])) if hits else None
def task_point(g, ds, c):
    r = find(g, "tasks", "%s_c%d.json" % (ds, c))
    return (r["throughput_tokens_per_second"], 1000.0 * r["latency_seconds"] * min(c, r["samples"]) / r["spec_verify_count"], r["average_acceptance_length"]) if r else None
def mixed_point(g, c):
    r = find(g, "mixed", "mixed_c%d.json" % c)
    return (r["throughput_tokens_per_second"], r["per_request_step_ms_mean"], r["average_acceptance_length"]) if r else None
def stats(v): return (statistics.fmean(v), statistics.stdev(v) if len(v) > 1 else 0.0, len(v))
EXCL = []
def collect(groups, getter, label):
    pts = [(g, getter(g)) for g in groups]; pts = [(g, p) for g, p in pts if p]
    if not pts: return None
    med = statistics.median(p[0] for _, p in pts); keep = [(g, p) for g, p in pts if not (len(pts) >= 3 and abs(pct(p[0], med)) > 25)]
    EXCL.extend("%s: %s %.0f tok/s vs median %.0f excluded" % (label, os.path.basename(g[0]), p[0], med) for g, p in pts if (g, p) not in keep)
    return stats([p[0] for _, p in keep]), stats([p[1] for _, p in keep]), stats([p[2] for _, p in keep])
def al_table(dense, moe):
    def al_set(groups):
        r = {}
        for ds, n in TASKS:
            j = find(groups[0], "tasks", "%s_c1.json" % ds)
            if not j: return None
            r[ds] = j["average_acceptance_length"]
        r["all"] = sum(r[ds] * n for ds, n in TASKS) / sum(n for _, n in TASKS); r["per_task"] = statistics.fmean(r[ds] for ds, _ in TASKS); return r
    d, m = al_set(dense), al_set(moe)
    if not d or not m: return "(accept length: c=1 task files not found)"
    cols = ["all", "per_task"] + [ds for ds, _ in TASKS]
    t = ["| | Avg (all requests) | Avg (per task) | gsm8k | math500 | humaneval | mbpp | mt-bench |", "|---|---|---|---|---|---|---|---|",
         "| dense | " + " | ".join("%.3f" % d[k] for k in cols) + " |", "| MoE | " + " | ".join("%.3f" % m[k] for k in cols) + " |",
         "| diff | " + " | ".join("%+.3f" % (m[k] - d[k]) for k in cols) + " |", "| diff % | " + " | ".join("%+.1f%%" % pct(m[k], d[k]) for k in cols) + " |"]
    return "\n".join(t)
def table(dense, moe, getter, label):
    rows = []
    for c in CONCS:
        d = collect(dense, lambda g: getter(g, c), "dense %s c=%d" % (label, c)); m = collect(moe, lambda g: getter(g, c), "MoE %s c=%d" % (label, c))
        if d and m: rows.append((c, d, m))
    t = ["| c | dense tok/s | MoE tok/s | diff % | dense step ms | MoE step ms |", "|---|---|---|---|---|---|"]
    t += ["| %d | %.0f ± %.0f | %.0f ± %.0f | %+.1f%% | %.1f ± %.1f | %.1f ± %.1f |" % (c, d[0][0], d[0][1], m[0][0], m[0][1], pct(m[0][0], d[0][0]), d[1][0], d[1][1], m[1][0], m[1][1]) for c, d, m in rows]
    return "\n".join(t), rows
def extended(dense, dflt, moe):
    t = ["| c | dense tok/s | MoE default | MoE tuned | tuned vs default (%) | tuned vs dense (diff %) | dense step ms | default step | tuned step |", "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for c in CONCS:
        d = collect(dense, lambda g: mixed_point(g, c), "dense mixed c=%d" % c); f = collect(dflt, lambda g: mixed_point(g, c), "default mixed c=%d" % c); m = collect(moe, lambda g: mixed_point(g, c), "tuned mixed c=%d" % c)
        if d and f and m:
            rows.append((c, d, f, m)); t.append("| %d | %.0f ± %.0f | %.0f ± %.0f | %.0f ± %.0f | %+.1f%% | %+.1f%% | %.1f ± %.1f | %.1f ± %.1f | %.1f ± %.1f |" % (
                c, d[0][0], d[0][1], f[0][0], f[0][1], m[0][0], m[0][1], pct(m[0][0], f[0][0]), pct(m[0][0], d[0][0]), d[1][0], d[1][1], f[1][0], f[1][1], m[1][0], m[1][1]))
    return "\n".join(t), rows
def chart(path, rows_mixed, rows_ext, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cs = [r[0] for r in rows_mixed]; x = list(range(len(cs)))
    dense = [r[1][0][0] for r in rows_mixed]; moe = [r[2][0][0] for r in rows_mixed]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=180)
    ax.plot(x, dense, color="#2b7de9", marker="o", linewidth=2, markersize=7, label="dense draft")
    if rows_ext: ax.plot(x, [r[2][0][0] for r in rows_ext], color="#8c8c8c", marker="^", linewidth=2, markersize=7, label="MoE draft (default fused-MoE config)")
    ax.plot(x, moe, color="#ef6a2f", marker="s", linewidth=2, markersize=7, label="MoE draft (tuned fused-MoE config)" if rows_ext else "MoE draft")
    ax.set_title(title, fontsize=10.5); ax.set_ylabel("output throughput (tok/s)"); ax.set_xlabel("client concurrency")
    ax.set_xticks(x); ax.set_xticklabels([str(c) for c in cs]); ax.set_ylim(0, max(dense + moe) * 1.12)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8); ax.set_axisbelow(True); ax.legend(loc="lower right")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path)
ap = argparse.ArgumentParser(); ap.add_argument("--dense", required=True); ap.add_argument("--moe", required=True); ap.add_argument("--moe-default"); ap.add_argument("--out", required=True)
ap.add_argument("--title", default="Qwen3-8B DSpark, block size 7: e2e throughput, dense vs MoE draft (mixed workload, 5 datasets)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
dense, moe = parse(a.dense), parse(a.moe); dflt = parse(a.moe_default) if a.moe_default else None
md = ["## 1. Accept length (c=1)", "", al_table(dense, moe), "", "## 2. e2e throughput per task (mean ± std over the groups)", ""]
for ds, _ in TASKS:
    t, _ = table(dense, moe, lambda g, c, ds=ds: task_point(g, ds, c), ds); md += ["### " + ds, "", t, ""]
tm, rows_mixed = table(dense, moe, mixed_point, "mixed"); md += ["## 3. Mixed workload (five datasets in one 2,400-request stream)", "", tm, ""]
rows_ext = None
if dflt:
    te, rows_ext = extended(dense, dflt, moe); md += ["### Mixed workload: dense / MoE default / MoE tuned", "", te, ""]
chart(os.path.join(a.out, "mixed_throughput.png"), rows_mixed, rows_ext, a.title); md += ["## 4. Chart", "", "`mixed_throughput.png`", ""]
if EXCL: md += ["Excluded repeats (> 25% off the cell median): " + "; ".join(EXCL), ""]
open(os.path.join(a.out, "report.md"), "w").write("\n".join(md)); print("\n".join(md))
