#!/usr/bin/env python3
"""Tables from logs/sweep_<name>.txt (tp_sweep.sh + bench_accept.py): dense vs MoE per concurrency, means over repeats."""
import os, glob, os, re, sys, statistics as st
L = os.environ.get("BENCH_DIR", "/scratch/moe8b16/serving-bench") + "/logs"
GPUS = {"tp1": 1, "tp2": 2, "tp4": 4, "tp8": 8, "dp2": 2, "dp8": 8}

def parse(path):
    rows = {}
    c = None
    for line in open(path):
        m = re.match(r"== \S+ concurrency=(\d+)", line)
        if m:
            c = int(m.group(1)); rows.setdefault(c, {"tps": [], "al": [], "step": [], "prq": []}); continue
        if c is None:
            continue
        m = re.search(r"output_throughput=([0-9.]+) tok/s", line)
        if m: rows[c]["tps"].append(float(m.group(1)))
        m = re.search(r"spec_accept_length: mean=([0-9.]+)", line)
        if m: rows[c]["al"].append(float(m.group(1)))
        m = re.search(r"per_verify_step_ms: mean=([0-9.]+).*per_request_throughput=([0-9.]+)", line)
        if m: rows[c]["step"].append(float(m.group(1))); rows[c]["prq"].append(float(m.group(2)))
    return {c: {k: (st.mean(v) if v else float("nan")) for k, v in d.items()} for c, d in rows.items() if d["tps"]}

def load(name):
    p = os.path.join(L, "sweep_%s.txt" % name)
    return parse(p) if os.path.exists(p) else None

def pct(a, b):
    return "%+.1f%%" % (100 * (a - b) / b) if b else "n/a"

configs = sys.argv[1:] or ["tp1", "tp2", "tp4", "tp8", "dp2", "dp8"]
for cfg in configs:
    d, m = load("dense-" + cfg), load("moe-" + cfg)
    if not d and not m:
        continue
    print("\n### %s (%d GPU%s)" % (cfg, GPUS.get(cfg, 1), "s" if GPUS.get(cfg, 1) > 1 else ""))
    print("| c | dense tok/s | MoE tok/s | diff | dense per-req tok/s | MoE per-req tok/s | dense step ms | MoE step ms | dense AL | MoE AL |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in sorted(set((d or {}).keys()) | set((m or {}).keys())):
        x = (d or {}).get(c); y = (m or {}).get(c)
        f = lambda r, k, fmt: (fmt % r[k]) if r else "-"
        diff = pct(y["tps"], x["tps"]) if x and y else "-"
        print("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (c, f(x, "tps", "%.0f"), f(y, "tps", "%.0f"), diff, f(x, "prq", "%.1f"), f(y, "prq", "%.1f"),
              f(x, "step", "%.1f"), f(y, "step", "%.1f"), f(x, "al", "%.2f"), f(y, "al", "%.2f")))
# GPU-count summary: total tok/s and tok/s per GPU at each concurrency, per config
print("\n### GPU-count summary (total tok/s | tok/s per GPU)")
cfgs = [c for c in ["tp1", "tp2", "tp4", "tp8", "dp2", "dp8"] if load("dense-" + c) or load("moe-" + c)]
if cfgs:
    print("| draft | c | " + " | ".join("%s (%dg)" % (c, GPUS.get(c, 1)) for c in cfgs) + " |")
    print("|---|---|" + "---|" * len(cfgs))
    for draft in ["dense", "moe"]:
        data = {c: load("%s-%s" % (draft, c)) for c in cfgs}
        concs = sorted(set().union(*[set(v.keys()) for v in data.values() if v]))
        for c in concs:
            cells = []
            for cfg in cfgs:
                r = (data[cfg] or {}).get(c)
                cells.append("%.0f | %.0f" % (r["tps"], r["tps"] / GPUS.get(cfg, 1)) if r else "- | -")
            print("| %s | %d | %s |" % (draft, c, " | ".join(cells)))
