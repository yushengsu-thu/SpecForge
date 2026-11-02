#!/usr/bin/env python3
"""Mixed-workload e2e serving benchmark: the five jiapingW tasks (gsm8k 1319, math500 500, humaneval 164, mbpp 257,
mt-bench 160 = the CLI's 80 first turns x 2) shuffled into ONE request stream (seed 42, same order for every draft),
sent at a fixed client concurrency with the specforge benchmark CLI's request format and sampling defaults
(chat template, thinking off, temperature 0, top_p 1, top_k 1, max_new_tokens 2048). Reports tok/s over the whole mix,
the per-request verify-step time (e2e_latency / spec_verify_ct), accept length, and a per-task breakdown."""
import argparse, json, random, statistics, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from transformers import AutoTokenizer
from specforge.benchmarks.sglang import _load_prompts, _apply_chat_template

COUNTS = {"gsm8k": 1319, "math500": 500, "humaneval": 164, "mbpp": 257, "mt-bench": 160}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000"); ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=1); ap.add_argument("--output-json", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=2048); ap.add_argument("--timeout-seconds", type=int, default=14400)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    items = []
    for ds, n in COUNTS.items():
        rows = _load_prompts(ds, None)
        items += [(ds, rows[i % len(rows)][0]) for i in range(n)]
    random.Random(42).shuffle(items)
    prompts = [(ds, _apply_chat_template(tok, [{"role": "user", "content": p}], False)) for ds, p in items]
    params = {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "max_new_tokens": args.max_new_tokens}
    url = args.base_url.rstrip("/")
    def send(text):
        r = requests.post(url + "/generate", json={"text": text, "sampling_params": params}, timeout=args.timeout_seconds)
        r.raise_for_status(); out = r.json(); return out[0] if isinstance(out, list) else out
    try: requests.get(url + "/flush_cache", timeout=60).raise_for_status()
    except requests.RequestException: print("warning: /flush_cache failed")
    warm = [prompts[i % len(prompts)][1] for i in range(args.concurrency)]   # untimed warm-up, like the CLI
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex: list(ex.map(send, warm))
    recs = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(send, text): ds for ds, text in prompts}
        for f in as_completed(futs):
            m = f.result().get("meta_info", {}) or {}
            recs.append((futs[f], int(m.get("completion_tokens", 0)), int(m.get("spec_verify_ct", 0) or 0), float(m.get("spec_accept_length") or 0), float(m.get("e2e_latency") or 0)))
    wall = time.perf_counter() - start
    tokens = sum(r[1] for r in recs); steps = sum(r[2] for r in recs)
    step_ms = [1000 * r[4] / r[2] for r in recs if r[2] and r[4]]
    res = {"concurrency": args.concurrency, "samples": len(recs), "output_tokens": tokens, "latency_seconds": wall,
           "throughput_tokens_per_second": tokens / wall, "spec_verify_count": steps,
           "average_acceptance_length": statistics.fmean(r[3] for r in recs if r[3]),
           "micro_acceptance_length": tokens / steps if steps else None,
           "per_request_step_ms_mean": statistics.fmean(step_ms), "per_request_step_ms_median": statistics.median(step_ms),
           "per_task": {}}
    for ds in COUNTS:
        rs = [r for r in recs if r[0] == ds]
        res["per_task"][ds] = {"samples": len(rs), "output_tokens": sum(r[1] for r in rs),
                               "average_acceptance_length": statistics.fmean(r[3] for r in rs if r[3]),
                               "per_request_step_ms_mean": statistics.fmean(1000 * r[4] / r[2] for r in rs if r[2] and r[4])}
    json.dump(res, open(args.output_json, "w"), indent=1)
    print("concurrency=%d requests=%d tokens=%d wall=%.1fs tok/s=%.1f AL=%.3f step_ms=%.2f" % (
        args.concurrency, len(recs), tokens, wall, res["throughput_tokens_per_second"], res["average_acceptance_length"], res["per_request_step_ms_mean"]))

if __name__ == "__main__":
    main()
