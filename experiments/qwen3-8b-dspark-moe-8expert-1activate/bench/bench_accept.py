#!/usr/bin/env python3
"""Acceptance-length benchmark following the protocol of
https://huggingface.co/jiapingW/Qwen3-8b-dspark-SpecForge (README "Benchmark (temperature=0)").

2400 requests: GSM8K test 1319, MATH-500 500, HumanEval 164, MBPP sanitized test 257, MT-Bench 80 x 2 turns (the
second turn is asked after the served first-turn answer). Prompts use the formats of specforge.benchmarks.sglang
(SpecForge's `benchmark` CLI), the target's chat template with thinking disabled and no system prompt, greedy decoding
(temperature 0), max_new_tokens 2048. Each server runs one request at a time (start it with --max-running-requests 1);
conversations are spread over all servers given with --servers.

Per request tau = completion_tokens / spec_verify_ct (correction / bonus tokens included). Reported per dataset and
pooled: request macro (mean of per-request tau) and micro (sum tokens / sum verify rounds), plus the equal-weight
means over the five benchmarks, output tokens and length-limited requests.
"""
import argparse, json, os, queue, statistics, sys, threading, time

DATASET_ORDER = ["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"]


def load_conversations(names, limit=None):
    from datasets import load_dataset
    from specforge.benchmarks.sglang import DATASETS
    convs = []
    for name in names:
        d = DATASETS[name]
        rows = load_dataset(*d["load_args"], **d["load_kwargs"])
        n = 0
        for i, row in enumerate(rows):
            turns = list(d["format"](row)) if d.get("multi_turn") else [d["format"](row)]
            convs.append({"dataset": name, "id": i, "turns": turns})
            n += 1
            if limit and n >= limit:
                break
    return convs


def generate(url, text, max_new_tokens, timeout):
    import requests
    r = requests.post(url.rstrip("/") + "/generate", json={
        "text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens}}, timeout=timeout)
    r.raise_for_status()
    out = r.json()
    return out[0] if isinstance(out, list) else out


def summarize(records):
    def stats(rs):
        taus = [r["tau"] for r in rs if r["verify_ct"] > 0]
        tok = sum(r["completion_tokens"] for r in rs); ver = sum(r["verify_ct"] for r in rs)
        return {"requests": len(rs), "request_macro_tau": statistics.fmean(taus) if taus else None,
                "micro_tau": tok / ver if ver else None, "output_tokens": tok, "verify_count": ver,
                "length_limited_requests": sum(1 for r in rs if r["finish"] == "length")}
    per = {ds: stats([r for r in records if r["dataset"] == ds]) for ds in DATASET_ORDER if any(r["dataset"] == ds for r in records)}
    pooled = stats(records)
    pooled["benchmark_equal_weight_macro_tau"] = statistics.fmean(v["request_macro_tau"] for v in per.values())
    pooled["benchmark_equal_weight_micro_tau"] = statistics.fmean(v["micro_tau"] for v in per.values())
    return per, pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", required=True, help="comma-separated base URLs, one request at a time each")
    ap.add_argument("--tokenizer", default="/scratch/models/Qwen3-8B")
    ap.add_argument("--datasets", default=",".join(DATASET_ORDER))
    ap.add_argument("--limit", type=int, default=None, help="conversations per dataset (smoke tests)")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--label", required=True)
    ap.add_argument("--meta", default="{}", help="JSON merged into the result (model, mode, server flags)")
    ap.add_argument("--out", required=True, help="result JSON; per-request records go to <out>.requests.jsonl")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    convs = load_conversations(args.datasets.split(","), args.limit)
    q = queue.Queue()
    for c in convs:
        q.put(c)
    records, errors, lock = [], [], threading.Lock()
    rec_path = args.out + ".requests.jsonl"
    rec_fh = open(rec_path, "w")
    t0 = time.time()

    def worker(url):
        while True:
            try:
                c = q.get_nowait()
            except queue.Empty:
                return
            messages = []
            for turn, user in enumerate(c["turns"]):
                messages.append({"role": "user", "content": user})
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                try:
                    out = generate(url, text, args.max_new_tokens, args.timeout)
                except Exception as exc:  # recorded; a valid run has zero errors
                    with lock:
                        errors.append({"dataset": c["dataset"], "id": c["id"], "turn": turn, "error": repr(exc)[:300]})
                    break
                m = out.get("meta_info", {}) or {}
                ct, vc = int(m.get("completion_tokens", 0)), int(m.get("spec_verify_ct", 0) or 0)
                fr = m.get("finish_reason") or {}
                rec = {"dataset": c["dataset"], "id": c["id"], "turn": turn, "completion_tokens": ct, "verify_ct": vc,
                       "tau": ct / vc if vc else None, "finish": fr.get("type") if isinstance(fr, dict) else str(fr),
                       "server": url, "e2e_latency": m.get("e2e_latency")}
                with lock:
                    records.append(rec); rec_fh.write(json.dumps(rec) + "\n"); rec_fh.flush()
                messages.append({"role": "assistant", "content": out.get("text", "")})

    servers = [s for s in args.servers.split(",") if s]
    threads = [threading.Thread(target=worker, args=(u,), daemon=True) for u in servers]
    for t in threads:
        t.start()
    last = 0
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        if len(records) - last >= 100:
            last = len(records)
            print(f"[{time.strftime('%H:%M:%S')}] {args.label}: {len(records)} requests, {len(errors)} errors, {time.time()-t0:.0f}s", flush=True)
    rec_fh.close()
    per, pooled = summarize(records) if records else ({}, {})
    result = {"label": args.label, "elapsed_s": time.time() - t0, "servers": servers, "errors": errors,
              "protocol": {"temperature": 0, "enable_thinking": False, "system_prompt": "", "max_new_tokens": args.max_new_tokens,
                           "max_running_requests_per_server": 1, "prompt_formats": "specforge.benchmarks.sglang.DATASETS",
                           "mt_bench": "2 turns, turn 2 after the served turn-1 answer"},
              "datasets": per, "pooled": pooled}
    result.update(json.loads(args.meta))
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps({"label": args.label, "pooled": pooled, "errors": len(errors)}, indent=1))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
