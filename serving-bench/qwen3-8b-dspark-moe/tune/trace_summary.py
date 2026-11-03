#!/usr/bin/env python3
"""Summarize a torch-profiler chrome trace from SGLang: GPU kernel time by category, and by launch group (each CUDA-graph replay or
plain kernel launch is a group, found through the correlation id). The group that contains the fused-MoE kernels (MoE draft) or the
smaller of the two repeating graph groups (dense draft) is the draft forward; the big one is the target verify.
usage: trace_summary.py <trace file or dir> [label]"""
import gzip, json, os, re, sys, glob
from collections import defaultdict, Counter
p = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else p
if os.path.isdir(p):
    fs = sorted(glob.glob(os.path.join(p, "**", "*.json*"), recursive=True), key=os.path.getsize); p = fs[-1]
opener = gzip.open if p.endswith(".gz") else open
with opener(p, "rt") as f: data = json.load(f)
events = data["traceEvents"] if isinstance(data, dict) else data
CATS = [("moe_gemm", r"fused_moe_kernel"), ("moe_glue", r"moe_align|topk|moe_sum|moe_fused_gate|softmax|biased_grouped|route|sigmoid_gating|gatherTopK"),
        ("ffn_act", r"act_and_mul|silu_and_mul|gelu"), ("norm", r"rms_norm|rmsnorm|layer_norm|qknorm|norm"), ("rope", r"rope|rotary"),
        ("attention", r"BatchPrefill|BatchDecode|MergeStates|attention|fmha|mla|paged"), ("gemm", r"nvjet|gemm|cutlass|xmma|cublas|Gemm|matmul|sm90|sm100|splitK"),
        ("embed", r"embedding|indexSelect|index_select|gather"), ("sampling", r"sampl|markov|top_k|top_p|argmax|multinomial|bitonic|sort|radix"), ("memcpy", r"memcpy|Memcpy|memset|Memset|copy|cat_|Cat")]
def cat(name):
    for c, rx in CATS:
        if re.search(rx, name): return c
    return "other"
kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("dur") is not None]
runtime = {e.get("args", {}).get("correlation"): e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver") and e.get("args", {}).get("correlation") is not None}
print("== %s ==  file %s  kernels %d  span %.1f ms" % (label, os.path.basename(p), len(kernels), (max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)) / 1000 if kernels else 0))
# by launch group: correlation -> launching runtime call; graph replays are cudaGraphLaunch
groups = defaultdict(list)
for k in kernels:
    corr = k.get("args", {}).get("correlation"); r = runtime.get(corr)
    groups[(r["name"] if r else "?", corr)].append(k)
graph_groups = [(key, ks) for key, ks in groups.items() if "Graph" in key[0]]
# signature = (kernel count, has fused_moe); cluster graph launches by kernel count
sig = Counter((len(ks), any("fused_moe" in k["name"] for k in ks)) for key, ks in graph_groups)
print("graph launch clusters (kernel count, has fused_moe) -> launches:", dict(sig.most_common(6)))
tot = sum(k["dur"] for k in kernels); nongraph = sum(k["dur"] for key, ks in groups.items() if "Graph" not in key[0] for k in ks)
print("GPU kernel time: total %.1f ms, inside graphs %.1f ms, outside graphs %.1f ms" % (tot / 1000, (tot - nongraph) / 1000, nongraph / 1000))
# label clusters: the cluster containing fused_moe = draft; else among the two most common clusters the smaller count = draft
clusters = defaultdict(list)
for key, ks in graph_groups: clusters[(len(ks), any("fused_moe" in k["name"] for k in ks))].append(ks)
big = [s for s in clusters if s[0] >= 50 and len(clusters[s]) >= 5]
keys = sorted(big, key=lambda s: -len(clusters[s]))[:2]
draft_key = next((s for s in keys if s[1]), None) or (min(keys, key=lambda s: s[0]) if len(keys) == 2 else None)
for s in keys:
    role = "DRAFT" if s == draft_key else "TARGET-VERIFY"; launches = clusters[s]
    per = [sum(k["dur"] for k in ks) / 1000 for ks in launches]
    print("\n-- %s graph: %d launches, %d kernels each, mean %.3f ms per replay (min %.3f, max %.3f)" % (role, len(launches), s[0], sum(per) / len(per), min(per), max(per)))
    bycat = defaultdict(float); byname = defaultdict(lambda: [0, 0.0])
    for ks in launches:
        for k in ks: bycat[cat(k["name"])] += k["dur"]; byname[k["name"]][0] += 1; byname[k["name"]][1] += k["dur"]
    n = len(launches)
    for c, d in sorted(bycat.items(), key=lambda kv: -kv[1]): print("   %-12s %7.3f ms/replay  %5.1f%%" % (c, d / n / 1000, 100 * d / sum(bycat.values())))
    print("   top kernels (per replay):")
    for name, (cnt, d) in sorted(byname.items(), key=lambda kv: -kv[1][1])[:12]: print("     %7.3f ms  x%-4d %s" % (d / n / 1000, cnt // n, name[:110]))
steps = len(clusters[draft_key]) if draft_key in clusters else 0
if steps:
    dm = sum(sum(k["dur"] for k in ks) for ks in clusters[draft_key]) / steps / 1000
    vk = [s for s in keys if s != draft_key]; vm = (sum(sum(k["dur"] for k in ks) for ks in clusters[vk[0]]) / len(clusters[vk[0]]) / 1000) if vk else 0
    print("\n== per step (GPU, from the trace): draft %.3f ms + verify %.3f ms + eager %.3f ms = %.3f ms  (%d draft replays)" % (dm, vm, nongraph / steps / 1000, dm + vm + nongraph / steps / 1000, steps))
out = defaultdict(float)
for key, ks in groups.items():
    if "Graph" in key[0]: continue
    for k in ks: out[cat(k["name"])] += k["dur"]
print("\n-- outside graphs (eager kernels: sampling, copies, prefill etc.): " + ", ".join("%s %.1f ms" % (c, d / 1000) for c, d in sorted(out.items(), key=lambda kv: -kv[1])[:8]))
