#!/usr/bin/env python3
"""Tables for the same-GPU e2e comparison of the 2-epoch run (2ep) vs the 5-epoch run (ep5):  summarize_e2e.py <results dir>."""
import glob, json, os, sys

R = sys.argv[1]
TASKS = ["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"]
TASK_NAMES = {"gsm8k": "GSM8K", "math500": "MATH-500", "humaneval": "HumanEval", "mbpp": "MBPP", "mt-bench": "MT-Bench"}
CS = [1, 16, 64, 128]
PAIRS = {8: [(2, "2ep first"), (3, "5-epoch first"), (6, "2ep first")], 16: [(4, "2ep first"), (5, "5-epoch first"), (7, "2ep first")]}
DENSE, MOE = "dense2ep", "dense5ep"
DENSE_NAME, MOE_NAME = "dense 2ep (2-epoch run)", "dense 5-epoch run ep5"
TP, ACC = "throughput_tokens_per_second", "average_acceptance_length"
data = {}
for f in glob.glob(os.path.join(R, "e2e_*.json")):
    label, task, c = os.path.basename(f)[4:-5].rsplit("_", 2)
    data[(label, task, int(c[1:]))] = json.load(open(f))


def get(label, task, c, key):
    return data.get((label, task, c), {}).get(key)


def mean5(label, c, key):
    vals = [get(label, t, c, key) for t in TASKS]
    return None if None in vals else sum(vals) / len(vals)


def pct(x):
    if x is None:
        return "-"
    p = (x - 1) * 100
    return ("+" if p >= 0.05 else ("−" if p <= -0.05 else "±")) + f"{abs(p):.1f}%"


def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x):
    return f"{x:,.0f}" if x else "-"


print(f"Same-GPU comparison: {DENSE_NAME} and {MOE_NAME} served one after the other on the same GPU (three GPUs per width, "
      "one with the order reversed). Temperature 0, 512 new tokens, prompts per task = max(200, 4 x concurrency), CUDA graphs on, "
      "servers allow 128 running requests. Unless a task is named, values are means of the five tasks.")
summary = {}
for w in (8, 16):
    print(f"\n## Verify width {w}\n")
    print("| Quantity | GPU | 1 concurrent request | 16 | 64 | 128 |")
    print("|---|---|---|---|---|---|")
    ratios = {c: [] for c in CS}; step = {c: [] for c in CS}; alr = {c: [] for c in CS}
    for g, order in PAIRS[w]:
        d, m = f"{DENSE}-v{w}-g{g}", f"{MOE}-v{w}-g{g}"
        row = []
        for c in CS:
            dt, mt, da, ma = mean5(d, c, TP), mean5(m, c, TP), mean5(d, c, ACC), mean5(m, c, ACC)
            r = mt / dt if dt and mt else None
            s = (ma / da) / r if r and da and ma else None
            ratios[c].append(r); step[c].append(s); alr[c].append(ma / da if da and ma else None)
            row.append(pct(r))
        print(f"| 5-epoch run throughput vs 2ep | {g} ({order}) | " + " | ".join(row) + " |")
    print("| **5-epoch run throughput vs 2ep, mean of the three GPUs** | | " + " | ".join(pct(avg(ratios[c])) for c in CS) + " |")
    print("| **5-epoch run time per verify step vs 2ep, mean of the three GPUs** | | " + " | ".join(pct(avg(step[c])) for c in CS) + " |")
    print("| 5-epoch run accept length vs 2ep, mean | | " + " | ".join(pct(avg(alr[c])) for c in CS) + " |")
    dm = {c: avg([mean5(f"{DENSE}-v{w}-g{g}", c, TP) for g, _ in PAIRS[w]]) for c in CS}
    mm = {c: avg([mean5(f"{MOE}-v{w}-g{g}", c, TP) for g, _ in PAIRS[w]]) for c in CS}
    bm = {c: mean5("baseline", c, TP) for c in CS}
    print(f"\n**Absolute throughput (tokens/s), verify width {w}, mean of the five tasks and of the three GPUs**\n")
    print("| Model | 1 concurrent request | 16 | 64 | 128 |")
    print("|---|---|---|---|---|")
    print("| Base model, no draft | " + " | ".join(fmt(bm[c]) for c in CS) + " |")
    print(f"| {DENSE_NAME} | " + " | ".join(f"{fmt(dm[c])} ({pct(dm[c] / bm[c])} vs no draft)" if dm[c] and bm[c] else "-" for c in CS) + " |")
    print(f"| {MOE_NAME} | " + " | ".join(f"{fmt(mm[c])} ({pct(mm[c] / dm[c])} vs 2ep)" if dm[c] and mm[c] else "-" for c in CS) + " |")
    print("| 2ep, spread between GPUs | " + " | ".join(
        pct(max(v) / min(v)) if (v := [x for x in (mean5(f"{DENSE}-v{w}-g{g}", c, TP) for g, _ in PAIRS[w]) if x]) else "-" for c in CS) + " |")
    # per task, means over the three GPUs
    print(f"\n**Per task throughput (tokens/s), verify width {w}, mean of the three GPUs; 5-epoch cell = 5-epoch run (difference vs 2ep)**\n")
    print("| Task | Model | 1 concurrent request | 16 | 64 | 128 |")
    print("|---|---|---|---|---|---|")
    per_task = {}
    for t in TASKS:
        bt = {c: get("baseline", t, c, TP) for c in CS}
        dt = {c: avg([get(f"{DENSE}-v{w}-g{g}", t, c, TP) for g, _ in PAIRS[w]]) for c in CS}
        mt = {c: avg([get(f"{MOE}-v{w}-g{g}", t, c, TP) for g, _ in PAIRS[w]]) for c in CS}
        print(f"| {TASK_NAMES[t]} | no draft | " + " | ".join(fmt(bt[c]) for c in CS) + " |")
        print(f"| {TASK_NAMES[t]} | {DENSE_NAME} | " + " | ".join(f"{fmt(dt[c])} ({pct(dt[c] / bt[c])} vs no draft)" if dt[c] and bt[c] else "-" for c in CS) + " |")
        print(f"| {TASK_NAMES[t]} | {MOE_NAME} | " + " | ".join(f"{fmt(mt[c])} ({pct(mt[c] / dt[c])} vs 2ep)" if dt[c] and mt[c] else "-" for c in CS) + " |")
        per_task[t] = {"baseline": bt, "dense": dt, "moe": mt}
    summary[w] = {"baseline": bm, "dense": dm, "moe": mm, "throughput_ratio": {c: avg(ratios[c]) for c in CS},
                  "step_time_ratio": {c: avg(step[c]) for c in CS}, "accept_ratio": {c: avg(alr[c]) for c in CS}, "per_task": per_task}
expected = [(l, t, c) for w in (8, 16) for g, _ in PAIRS[w] for l in (f"{DENSE}-v{w}-g{g}", f"{MOE}-v{w}-g{g}") for t in TASKS for c in CS]
expected += [("baseline", t, c) for t in TASKS for c in CS]
missing = [k for k in expected if k not in data]
print(f"\nresult files: {len(data)}; missing: {missing if missing else 'none'}")
json.dump(summary, open(os.path.join(os.path.dirname(R.rstrip("/")), "summary_e2e.json"), "w"), indent=1)
