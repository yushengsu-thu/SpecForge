#!/usr/bin/env python3
"""Print the accept-length table (eval_<name>.json) and the e2e throughput table (e2e_<label>_<ds>_c<c>.json)."""
import json, os, glob
E = "/scratch/dense5l-eval/logs"

def load(name):
    p = f"{E}/eval_{name}.json"
    return json.load(open(p))["datasets"] if os.path.exists(p) else {}

names = (["dense5l-%dep" % k for k in range(1, 9)] + ["official-v1"] + ["moe2l-%dep" % k for k in range(1, 9)] + ["moe256-%dep" % k for k in range(1, 9)] + ["moe16x-%dep" % k for k in range(1, 9)])
def _hdr(n):
    base, _, ep = n.rpartition("-")
    return {"dense5l": "dense", "moe2l": "MoE-2L", "moe256": "MoE-256exp", "moe16x": "MoE-16x1024"}.get(base, base) + " " + ep if n != "official-v1" else "official v1"
HDR = {n: _hdr(n) for n in names}
R = {n: load(n) for n in names}
dss = [("gsm8k", "GSM8K (1319, 5-shot, non-thinking)"), ("math500", "MATH500 (500, thinking)"),
       ("mt_bench", "MT-Bench (80 x 2 turns, thinking)"), ("aime26", "AIME26 (30, thinking, 32k)"),
       ("traces", "training traces (512, thinking)")]
print("### Accepted drafts per verify step (aggregate accept length minus the bonus token); concurrency 8, greedy, bf16 target, one H200 per server")
present = [n for n in names if R[n]]
print("| dataset | " + " | ".join(HDR[n] for n in present) + " |")
print("|---|" + "---|" * len(present))
for k, label in dss:
    cells = []
    for n in present:
        r = R[n].get(k)
        cells.append("%.2f" % r["drafts_per_step"] if r else "(running)")
    print("| %s | %s |" % (label, " | ".join(cells)))
print()
print("### Output tokens/s at concurrency 8 during the accept-length eval")
print("| dataset | " + " | ".join(HDR[n] for n in present) + " |")
print("|---|" + "---|" * len(present))
for k, label in dss:
    print("| %s | %s |" % (k, " | ".join("%.0f" % R[n][k]["tokens_per_s"] if k in R[n] else "(running)" for n in present)))
print("errors:", {n: sum(r["errors"] for r in R[n].values()) for n in present})
print()
# --- e2e ---
labels = [l for l in ["baseline", "dense-4ep", "official-v1", "moe2l-4ep", "moe256-4ep", "moe16x-5ep"] if os.path.exists(f"{E}/e2e_{l}_gsm8k_c1.json")]
PRETTY = {"baseline": "no draft (autoregressive)", "dense-4ep": "dense 4ep", "official-v1": "official v1", "moe2l-4ep": "MoE-2L 4ep", "moe256-4ep": "MoE-256exp 4ep", "moe16x-5ep": "MoE-16x1024 5ep"}
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

print()
print("### Speed gate at a pinned acceptance length of 5.0 (SGLANG_SIMULATE_ACC_LEN, sglang.bench_serving, random 512 input / 512 output tokens, one server at a time on one H200); cell = output tokens/s (mean TPOT)")
print()
print("| drafter | concurrency 1 | concurrency 8 | concurrency 32 |")
print("|---|---|---|---|")
SG = {"official-v1": "official v1", "dense-4ep": "dense 4ep", "moe2l-4ep": "MoE-2L 4ep", "moe256-4ep": "MoE-256exp 4ep", "moe16x-5ep": "MoE-16x1024 5ep"}
rows = {}
for tsv in glob.glob("/scratch/dense5l-eval/moe_speed*/summary.tsv"):
    for line in open(tsv).read().splitlines()[1:]:
        f = line.split("\t")
        if len(f) < 7: continue
        rows.setdefault(f[0], {})[int(f[2])] = "%.1f (%.2f ms)" % (float(f[4]), float(f[6]))
for k in ["official-v1", "dense-4ep", "moe2l-4ep", "moe256-4ep", "moe16x-5ep"]:
    if k in rows:
        print("| %s | %s | %s | %s |" % (SG[k], rows[k].get(1, ""), rows[k].get(8, ""), rows[k].get(32, "")))
