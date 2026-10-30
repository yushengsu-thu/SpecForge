# Summarize logs/prof_<tag>_c<c>.json: mean GPU ms per step for the draft, the target verify and the whole step,
# at the steady-state batch size (the most frequent bs in the records). Run inside logs/.
import glob, json, re, statistics
from collections import defaultdict
def find_records(obj):
    if isinstance(obj, dict):
        rec = obj.get("dspark_info_record")
        if isinstance(rec, dict) and "records" in rec:
            return rec["records"]
        for v in obj.values():
            r = find_records(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_records(v)
            if r:
                return r
    return []
rows = []
for f in glob.glob("prof_*_c*.json"):
    tag, c = re.match(r"prof_(.+)_c(\d+)\.json", f).groups()
    rows.append((tag != "dense", tag, int(c), f))
print("%-9s %5s %5s %6s %10s %10s %10s %7s %7s" % ("draft", "c", "bs", "steps", "draft ms", "verify ms", "step ms", "draft%", "verify%"))
for _, tag, c, f in sorted(rows):
    recs = [r for r in find_records(json.load(open(f))) if r.get("draft_gpu_ms") is not None and r.get("target_verify_gpu_ms") is not None]
    if not recs:
        print("%-9s %5d  no timed records" % (tag, c)); continue
    by_bs = defaultdict(list)
    for r in recs:
        by_bs[r.get("bs", -1)].append(r)
    bs, rs = max(by_bs.items(), key=lambda kv: len(kv[1]))
    d = statistics.mean(r["draft_gpu_ms"] for r in rs)
    v = statistics.mean(r["target_verify_gpu_ms"] for r in rs)
    steps = [r["step_gpu_ms"] for r in rs if r.get("step_gpu_ms") is not None]
    s = statistics.mean(steps) if steps else float("nan")
    print("%-9s %5d %5d %6d %10.2f %10.2f %10.2f %6.0f%% %6.0f%%" % (tag, c, bs, len(rs), d, v, s, 100 * d / s, 100 * v / s))
