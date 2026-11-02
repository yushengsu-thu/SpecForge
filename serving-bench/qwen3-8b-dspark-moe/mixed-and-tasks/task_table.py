#!/usr/bin/env python3
"""Per-task tables (c | dense tok/s | MoE tok/s | diff | dense step ms | MoE step ms) from logs/tasks/<name>/<task>_c<C>.json.
step ms = per-request verify-step time ~ wall x min(c, requests) / total verify steps (exact at c=1)."""
import os, json, os
L = os.environ.get("BENCH_DIR", "/scratch/moe8b16/serving-bench") + "/logs/tasks"
TASKS = ["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"]
CONCS = [1, 8, 16, 32, 64, 128, 256]
GROUPS = ["", "-g2", "-g3", "-g4", "-g5"]   # repeated runs of the whole experiment; tables show the mean over the groups present
def load(name, ds, c):
    rs = []
    for g in GROUPS:
        p = os.path.join(L, name + g, "%s_c%d.json" % (ds, c))
        if os.path.exists(p): rs.append(json.load(open(p)))
    if not rs: return None
    tps = [r["throughput_tokens_per_second"] for r in rs]
    steps = [1000.0 * r["latency_seconds"] * min(c, r["samples"]) / r["spec_verify_count"] for r in rs]
    return {"tps": sum(tps) / len(tps), "step": sum(steps) / len(steps), "al": sum(r["average_acceptance_length"] for r in rs) / len(rs),
            "n": rs[0]["samples"], "groups": len(rs), "tps_min": min(tps), "tps_max": max(tps)}
for ds in TASKS:
    rows = [(c, load("dense", ds, c), load("moe", ds, c)) for c in CONCS]
    rows = [(c, d, m) for c, d, m in rows if d or m]
    if not rows: continue
    print("\n### %s (%d requests)" % (ds, (rows[0][1] or rows[0][2])["n"]))
    print("| c | dense tok/s | MoE tok/s | diff | dense step ms | MoE step ms |")
    print("|---|---|---|---|---|---|")
    for c, d, m in rows:
        f = lambda r, k, fmt: (fmt % r[k]) if r else "-"
        diff = ("%+.1f%%" % (100 * (m["tps"] - d["tps"]) / d["tps"])) if d and m else "-"
        print("| %d | %s | %s | %s | %s | %s |" % (c, f(d, "tps", "%.0f"), f(m, "tps", "%.0f"), diff, f(d, "step", "%.1f"), f(m, "step", "%.1f")))

# Final throughput-only tables: per-task tok/s side by side, and pooled over the tasks run at that concurrency
# (sum of output tokens / sum of wall time; c=128/256 pool gsm8k+math500 only).
def raw(name, ds, c):
    p = os.path.join(L, name, "%s_c%d.json" % (ds, c))
    return json.load(open(p)) if os.path.exists(p) else None
print("\n### Throughput only (tok/s), per task")
print("| c | " + " | ".join("%s dense | %s MoE | diff" % (t, t) for t in TASKS) + " |")
print("|---|" + "---|" * (3 * len(TASKS)))
for c in CONCS:
    cells = []
    for ds in TASKS:
        d, m = raw("dense", ds, c), raw("moe", ds, c)
        if d and m:
            cells.append("%.0f | %.0f | %+.1f%%" % (d["throughput_tokens_per_second"], m["throughput_tokens_per_second"], 100 * (m["throughput_tokens_per_second"] - d["throughput_tokens_per_second"]) / d["throughput_tokens_per_second"]))
        else:
            cells.append("- | - | -")
    if any(x != "- | - | -" for x in cells):
        print("| %d | %s |" % (c, " | ".join(cells)))
print("\n### Throughput only, pooled over the tasks run at that concurrency")
print("| c | tasks | dense tok/s | MoE tok/s | diff |")
print("|---|---|---|---|---|")
for c in CONCS:
    dt = dl = mt = ml = 0.0; names = []
    for ds in TASKS:
        d, m = raw("dense", ds, c), raw("moe", ds, c)
        if d and m:
            dt += d["output_tokens"]; dl += d["latency_seconds"]; mt += m["output_tokens"]; ml += m["latency_seconds"]; names.append(ds)
    if names:
        print("| %d | %s | %.0f | %.0f | %+.1f%% |" % (c, "+".join(names) if len(names) < 5 else "all 5", dt / dl, mt / ml, 100 * (mt / ml - dt / dl) / (dt / dl)))

# Final table: equal-weight AVERAGE over the tasks measured at that concurrency (all 5 up to c=64; gsm8k+math500 at 128/256).
print("\n### Average over tasks")
print("| c | tasks | dense tok/s | MoE tok/s | diff | dense step ms | MoE step ms |")
print("|---|---|---|---|---|---|---|")
for c in CONCS:
    pairs = [(load("dense", ds, c), load("moe", ds, c)) for ds in TASKS]
    names = [ds for ds, (d, m) in zip(TASKS, pairs) if d and m]
    pairs = [(d, m) for d, m in pairs if d and m]
    if not pairs: continue
    n = len(pairs); dt = sum(d["tps"] for d, _ in pairs) / n; mt = sum(m["tps"] for _, m in pairs) / n
    dsm = sum(d["step"] for d, _ in pairs) / n; msm = sum(m["step"] for _, m in pairs) / n
    print("| %d | %s | %.0f | %.0f | %+.1f%% | %.1f | %.1f |" % (c, "all 5" if n == 5 else "%d (%s)" % (n, ", ".join(names)), dt, mt, 100 * (mt - dt) / dt, dsm, msm))
