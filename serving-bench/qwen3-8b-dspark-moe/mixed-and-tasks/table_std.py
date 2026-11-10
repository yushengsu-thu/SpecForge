#!/usr/bin/env python3
"""Per-task and mixed tables as mean +- sample std over the repeat groups (tok/s and step ms), diff = diff of means."""
import json, os, statistics
B = os.environ.get("BENCH_DIR", "/scratch/moe8b16/serving-bench")
TASKS = ["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"]; CONCS = [1, 8, 16, 32, 64, 128, 256]
TASK_GROUPS = ["", "-g2", "-g3", "-g4", "-g5"]; MIXED_GROUPS = ["-m1", "-m2", "-m3"]

def ms(v):
    return (statistics.fmean(v), statistics.stdev(v) if len(v) > 1 else 0.0, len(v))

def task_vals(name, ds, c):
    tps, step = [], []
    for g in TASK_GROUPS:
        p = os.path.join(B, "logs/tasks", name + g, "%s_c%d.json" % (ds, c))
        if os.path.exists(p):
            r = json.load(open(p)); tps.append(r["throughput_tokens_per_second"]); step.append(1000.0 * r["latency_seconds"] * min(c, r["samples"]) / r["spec_verify_count"])
    return (ms(tps), ms(step)) if tps else None

def mixed_vals(name, c):
    tps, step = [], []
    for g in MIXED_GROUPS:
        p = os.path.join(B, "logs/mixed", name + g, "mixed_c%d.json" % c)
        if os.path.exists(p):
            r = json.load(open(p)); tps.append(r["throughput_tokens_per_second"]); step.append(r["per_request_step_ms_mean"])
    return (ms(tps), ms(step)) if tps else None

def table(title, getter):
    print("\n### %s" % title)
    print("| c | dense tok/s (mean ± std) | MoE tok/s (mean ± std) | diff | dense step ms (mean ± std) | MoE step ms (mean ± std) | n |")
    print("|---|---|---|---|---|---|---|")
    for c in CONCS:
        d, m = getter("dense", c), getter("moe", c)
        if not d or not m: continue
        (dt, dts, dn), (dsm, dss, _) = d; (mt, mts, mn), (msm, mss, _) = m
        print("| %d | %.0f ± %.0f | %.0f ± %.0f | %+.1f%% | %.1f ± %.1f | %.1f ± %.1f | %d/%d |" % (c, dt, dts, mt, mts, 100 * (mt - dt) / dt, dsm, dss, msm, mss, dn, mn))

for ds in TASKS:
    table("%s" % ds, lambda name, c, ds=ds: task_vals(name, ds, c))
table("mixed (five datasets in one stream, 2,400 requests)", mixed_vals)
