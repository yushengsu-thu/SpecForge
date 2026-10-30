# Host side: summarize bench_<tag>_g8_<dataset>_c1.json files (specforge benchmark, width 8, c=1) as one table.
import glob, json, re, sys
from collections import defaultdict
DS = ["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"]
TRAIN_ACC = {11960: None, 38000: 0.774, 43000: 0.772, 44000: 0.778, 45000: 0.783, 46000: 0.780,
             47000: 0.789, 48000: 0.808, 49000: 0.781, 50000: 0.797, 51000: 0.793, 52000: 0.786}
acc, thr = defaultdict(dict), defaultdict(dict)
for f in glob.glob("bench_*_g8_*_c1.json"):
    tag, ds = re.match(r"bench_(.+)_g8_(.+)_c1\.json", f).groups()
    d = json.load(open(f))
    acc[tag][ds] = d["average_acceptance_length"]
    thr[tag][ds] = d["throughput_tokens_per_second"]
def step(t):
    return int(re.sub(r"\D", "", t) or 0)
def key(t):
    return (0, 0) if t == "dense" else (1, step(t))
dense = acc["dense"]
dmean = sum(dense[d] for d in DS) / len(DS)
hdr = "%-9s %-9s" % ("draft", "train/acc") + "".join("%10s" % d for d in DS) + "%8s %9s %9s" % ("mean", "vs dense", "tok/s")
print(hdr)
best = None
for t in sorted(acc, key=key):
    row = acc[t]
    vals = [row.get(d) for d in DS]
    full = all(v is not None for v in vals)
    ta = "-" if t == "dense" else ("%.3f" % TRAIN_ACC[step(t)] if TRAIN_ACC.get(step(t)) else "?")
    cells = "".join("%10s" % ("%.3f" % v if v is not None else "-") for v in vals)
    if full:
        mean = sum(vals) / len(vals)
        rel = (mean / dmean - 1) * 100
        tp = sum(thr[t][d] for d in DS) / len(DS)
        print("%-9s %-9s%s%8.3f %8.1f%% %9.0f" % (t, ta, cells, mean, rel, tp))
        if t != "dense" and (best is None or mean > best[1]):
            best = (t, mean)
    else:
        print("%-9s %-9s%s %s" % (t, ta, cells, "(partial)"))
if best:
    print("\nbest MoE by five-dataset mean accept length: %s (%.3f; dense %.3f, %.1f%%)" % (best[0], best[1], dmean, (best[1] / dmean - 1) * 100))
