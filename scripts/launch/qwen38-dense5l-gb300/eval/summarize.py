#!/usr/bin/env python3
"""Print the accept-length table (eval_<name>.json) and the e2e throughput table (e2e_<label>_<ds>_c<c>.json)."""
import json, os, glob
E = "/scratch/dense5l-eval/logs"

def load(name):
    p = f"{E}/eval_{name}.json"
    return json.load(open(p))["datasets"] if os.path.exists(p) else {}

names = ["dense5l-1ep", "dense5l-2ep", "dense5l-3ep", "dense5l-4ep", "official-v1"]
R = {n: load(n) for n in names}
dss = [("gsm8k", "GSM8K (1319, 5-shot, non-thinking)"), ("math500", "MATH500 (500, thinking)"),
       ("mt_bench", "MT-Bench (80 x 2 turns, thinking)"), ("aime26", "AIME26 (30, thinking, 32k)"),
       ("traces", "training traces (512, thinking)")]
print("### Accepted drafts per verify step (aggregate accept length minus the bonus token); concurrency 8, greedy, bf16 target, one H200 per server")
print("| dataset | dense 1ep | dense 2ep | dense 3ep | dense 4ep | official v1 |")
print("|---|---|---|---|---|---|")
for k, label in dss:
    cells = []
    for n in names:
        r = R[n].get(k)
        cells.append("%.2f" % r["drafts_per_step"] if r else "(running)")
    print("| %s | %s |" % (label, " | ".join(cells)))
print()
print("### Output tokens/s at concurrency 8 during the accept-length eval")
print("| dataset | 1ep | 2ep | 3ep | 4ep | official v1 |")
print("|---|---|---|---|---|---|")
for k, label in dss:
    print("| %s | %s |" % (k, " | ".join("%.0f" % R[n][k]["tokens_per_s"] if k in R[n] else "(running)" for n in names)))
print("errors:", {n: sum(r["errors"] for r in R[n].values()) for n in names})
print()
# --- e2e ---
labels = ["baseline", "dense-4ep", "official-v1"]
PRETTY = {"baseline": "no draft (autoregressive)", "dense-4ep": "dense 4ep", "official-v1": "official v1"}
def e2e(label, ds, c):
    p = f"{E}/e2e_{label}_{ds}_c{c}.json"
    return json.load(open(p)) if os.path.exists(p) else None
print("### End-to-end serving throughput (`specforge benchmark`, 512 new tokens, greedy); cell = output tokens/s (speed-up over autoregressive) [acceptance length]")
for ds in ["gsm8k", "mt-bench"]:
    print()
    print("| %s | concurrency 1 (100 prompts) | concurrency 8 (200 prompts) | concurrency 32 (200 prompts) |" % ds)
    print("|---|---|---|---|")
    for label in labels:
        cells = []
        for c in [1, 8, 32]:
            d = e2e(label, ds, c); b = e2e("baseline", ds, c)
            if not d:
                cells.append("(missing)"); continue
            tps = d["throughput_tokens_per_second"]
            s = "%.1f" % tps
            if label != "baseline" and b:
                s += " (%.2fx)" % (tps / b["throughput_tokens_per_second"])
            if d.get("average_acceptance_length"):
                s += " [%.2f]" % d["average_acceptance_length"]
            cells.append(s)
        print("| %s | %s |" % (PRETTY.get(label, label), " | ".join(cells)))
