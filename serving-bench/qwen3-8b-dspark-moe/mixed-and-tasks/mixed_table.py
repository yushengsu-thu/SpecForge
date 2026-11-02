#!/usr/bin/env python3
"""Mixed-workload table: c | dense tok/s | MoE tok/s | diff | dense step ms | MoE step ms, mean over groups m1..m3
(step ms = mean per-request e2e_latency / verify steps, exact)."""
import os, json, os, statistics
L = os.environ.get("BENCH_DIR", "/scratch/moe8b16/serving-bench") + "/logs/mixed"
GROUPS = ["-m1", "-m2", "-m3"]; CONCS = [1, 8, 16, 32, 64, 128, 256]
def load(name, c):
    rs = [json.load(open(p)) for g in GROUPS for p in [os.path.join(L, name + g, "mixed_c%d.json" % c)] if os.path.exists(p)]
    if not rs: return None
    return {"tps": statistics.fmean(r["throughput_tokens_per_second"] for r in rs), "step": statistics.fmean(r["per_request_step_ms_mean"] for r in rs),
            "al": statistics.fmean(r["average_acceptance_length"] for r in rs), "n": len(rs), "tps_all": [r["throughput_tokens_per_second"] for r in rs]}
print("| c | dense tok/s | MoE tok/s | diff | dense step ms | MoE step ms | groups |")
print("|---|---|---|---|---|---|---|")
for c in CONCS:
    d, m = load("dense", c), load("moe", c)
    if not d and not m: continue
    f = lambda r, k, fmt: (fmt % r[k]) if r else "-"
    diff = ("%+.1f%%" % (100 * (m["tps"] - d["tps"]) / d["tps"])) if d and m else "-"
    print("| %d | %s | %s | %s | %s | %s | %s |" % (c, f(d, "tps", "%.0f"), f(m, "tps", "%.0f"), diff, f(d, "step", "%.1f"), f(m, "step", "%.1f"), "%d/%d" % ((d or m)["n"], (m or d)["n"])))
print("\nspread of tok/s over groups (min-max): " + "; ".join("c=%d dense %.0f-%.0f moe %.0f-%.0f" % (c, min(d["tps_all"]), max(d["tps_all"]), min(m["tps_all"]), max(m["tps_all"])) for c in CONCS for d, m in [(load("dense", c), load("moe", c))] if d and m))
