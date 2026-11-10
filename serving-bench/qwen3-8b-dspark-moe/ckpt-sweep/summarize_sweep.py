#!/usr/bin/env python3
"""Rank the swept checkpoints by pooled request-macro acceptance length over jiapingW's five tasks."""
import json, os, sys
W = os.environ.get("EVAL_DIR", "/scratch/moe8b16/eval") + "/serve"
SWEEP = sys.argv[1] if len(sys.argv) > 1 else "sweep"   # sweep (run1) or sweep-run2
TASKS = ["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"]

def read(d):
    r = {}
    for t in TASKS:
        p = os.path.join(d, "%s_c1.json" % t)
        if not os.path.exists(p):
            return None
        j = json.load(open(p)); r[t] = (j["average_acceptance_length"], j["output_tokens"], j["spec_verify_count"], j["samples"])
    vals = [r[t] for t in TASKS]
    n = sum(v[3] for v in vals)
    r["pooled"] = sum(v[0] * v[3] for v in vals) / n
    r["micro"] = sum(v[1] for v in vals) / sum(v[2] for v in vals)
    r["eq"] = sum(v[0] for v in vals) / len(TASKS)
    return r

def step_of(name):
    if name.startswith("ep"):
        k = int(name[2:])
        return 64584 if k == 6 else (k - 6) * 10764   # run1 ep6 = last step; run2 epN = (N-6) epochs of run2
    return int(name.split("-")[1])

rows = []
for name in sorted(os.listdir(os.path.join(W, SWEEP))):
    d = os.path.join(W, SWEEP, name)
    if not os.path.isdir(d):
        continue
    r = read(d)
    if r:
        rows.append((step_of(name), name, r))
rows.sort(key=lambda x: x[0])

dense = read(os.path.join(W, "dense"))
if dense is None:
    # the four non-gsm8k tasks of the dense reference live in dense-tasks/
    merged = {}
    for t in TASKS:
        p = os.path.join(W, "dense" if t == "gsm8k" else "dense-tasks", "%s_c1.json" % t)
        if os.path.exists(p):
            j = json.load(open(p)); merged[t] = (j["average_acceptance_length"], j["output_tokens"], j["spec_verify_count"], j["samples"])
    if len(merged) == 5:
        vals = [merged[t] for t in TASKS]
        n = sum(v[3] for v in vals)
        merged["pooled"] = sum(v[0] * v[3] for v in vals) / n
        merged["micro"] = sum(v[1] for v in vals) / sum(v[2] for v in vals)
        merged["eq"] = sum(v[0] for v in vals) / 5
        dense = merged

refs = [("jw dense step-39000 (our measurement)", dense)]
if SWEEP != "sweep":
    # run1 reference rows: the warm-start point (ep6) and run1's best checkpoint
    for n in ("ep6", "step-61000"):
        r = read(os.path.join(W, "sweep", n))
        if r:
            refs.append(("run1 %s" % n, r))

def fmt(label, r):
    return "| %s | %s | %.3f | %.3f | %.3f |" % (label, " | ".join("%.3f" % r[t][0] for t in TASKS), r["pooled"], r["micro"], r["eq"])

print("%d checkpoints evaluated in %s" % (len(rows), SWEEP))
print("| ckpt | gsm8k | math500 | humaneval | mbpp | mt-bench | pooled macro | pooled micro | equal-weight |")
print("|---|---|---|---|---|---|---|---|---|")
for label, r in refs:
    if r:
        print(fmt(label, r))
for step, name, r in rows:
    print(fmt(name, r))
best = sorted(rows, key=lambda x: -x[2]["pooled"])[:5]
print("\nTop 5 by pooled macro AL:")
for step, name, r in best:
    extra = " (%+.2f%% vs dense)" % (100 * (r["pooled"] - dense["pooled"]) / dense["pooled"]) if dense else ""
    print("  %s: pooled %.3f, micro %.3f, equal-weight %.3f%s" % (name, r["pooled"], r["micro"], r["eq"], extra))
json.dump({name: {k: (v if not isinstance(v, tuple) else v[0]) for k, v in r.items()} for _, name, r in rows},
          open(os.path.join(W, SWEEP, "summary.json"), "w"), indent=1)
