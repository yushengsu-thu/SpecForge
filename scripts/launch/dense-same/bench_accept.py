"""Acceptance-length benchmark with the protocol of jiapingW/Qwen3-8b-dspark-SpecForge (bench/results.json).

2400 requests per mode: GSM8K 1319, MATH500 500, HumanEval 164, MBPP (sanitized test) 257 and MT-Bench 80
two-turn conversations (160 requests; turn 2 carries the model's own turn-1 answer). Prompts use SpecForge's
benchmark formats (specforge.benchmarks.sglang.DATASETS), the Qwen3 chat template with thinking disabled and no
system prompt, max_new_tokens 2048. Each server runs one request at a time (max_running_requests 1). Per request
tau = completion_tokens / spec_verify_ct (correction/bonus token included); request macro = mean of per-request
tau, micro = sum(tokens) / sum(verify rounds); pooled over all 2400 requests and equal-weight over the five
benchmarks. Temperature-1 requests use sampling_seed = 42 + uint32_le(SHA256("dataset:sample_id:turn")[:4]).

usage: python bench_accept.py --urls http://127.0.0.1:31000,... --tokenizer /scratch/models/Qwen3-8B \
         --temperature 0 --out-dir DIR [--max-per-dataset N] [--datasets gsm8k,math500,...]
"""
import argparse, hashlib, json, os, queue, struct, sys, threading, time

import requests
from datasets import load_dataset
from transformers import AutoTokenizer

from specforge.benchmarks.sglang import DATASETS

ORDER = ["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"]


def seed_for(dataset, sample_id, turn):
    digest = hashlib.sha256(f"{dataset}:{sample_id}:{turn}".encode()).digest()
    return 42 + struct.unpack("<I", digest[:4])[0]


def load_items(name, limit):
    d = DATASETS[name]
    rows = load_dataset(*d["load_args"], **d["load_kwargs"])
    items = []
    for idx, row in enumerate(rows):
        formatted = d["format"](row)
        turns = list(formatted) if d.get("multi_turn") else [formatted]
        items.append((name, idx, turns))
        if limit and len(items) >= limit:
            break
    return items


def render(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def generate(url, text, temperature, seed, max_new_tokens, timeout):
    params = {"temperature": temperature, "max_new_tokens": max_new_tokens}
    if temperature > 0:
        params.update({"top_p": 1.0, "top_k": -1, "sampling_seed": seed})
    else:
        params.update({"top_p": 1.0, "top_k": 1})
    for attempt in range(3):
        try:
            r = requests.post(url.rstrip("/") + "/generate", json={"text": text, "sampling_params": params}, timeout=timeout)
            r.raise_for_status()
            out = r.json()
            return out[0] if isinstance(out, list) else out
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(10)


def worker(url, tok, args, work, results, lock, errors):
    while True:
        try:
            name, idx, turns = work.get_nowait()
        except queue.Empty:
            return
        try:
            messages = []
            for turn, user_text in enumerate(turns):
                messages.append({"role": "user", "content": user_text})
                out = generate(url, render(tok, messages), args.temperature, seed_for(name, idx, turn),
                               args.max_new_tokens, args.timeout)
                meta = out.get("meta_info", {}) or {}
                verify = meta.get("spec_verify_ct")
                verify = int(verify[0] if isinstance(verify, list) else (verify or 0))
                tokens = int(meta.get("completion_tokens", 0))
                finish = (meta.get("finish_reason") or {}).get("type") if isinstance(meta.get("finish_reason"), dict) else meta.get("finish_reason")
                rec = {"dataset": name, "sample_id": idx, "turn": turn, "completion_tokens": tokens,
                       "verify_ct": verify, "tau": (tokens / verify) if verify else None,
                       "length_limited": finish == "length", "server": url}
                with lock:
                    results.append(rec)
                messages.append({"role": "assistant", "content": out.get("text", "")})
        except Exception as exc:  # keep going; the summary reports failures
            with lock:
                errors.append({"dataset": name, "sample_id": idx, "error": repr(exc)[:300]})


def summarize(records, temperature):
    by = {}
    for r in records:
        by.setdefault(r["dataset"], []).append(r)
    def stats(rs):
        ok = [r for r in rs if r["verify_ct"]]
        tok = sum(r["completion_tokens"] for r in ok); ver = sum(r["verify_ct"] for r in ok)
        return {"requests": len(rs), "output_tokens": tok, "verify_count": ver,
                "request_macro_tau": (sum(r["tau"] for r in ok) / len(ok)) if ok else None,
                "micro_tau": (tok / ver) if ver else None,
                "length_limited_requests": sum(1 for r in rs if r["length_limited"])}
    datasets = {name: stats(by[name]) for name in ORDER if name in by}
    pooled = stats(records)
    vals = [v for v in datasets.values() if v["request_macro_tau"] is not None]
    pooled["benchmark_equal_weight_macro_tau"] = sum(v["request_macro_tau"] for v in vals) / len(vals) if vals else None
    pooled["benchmark_equal_weight_micro_tau"] = sum(v["micro_tau"] for v in vals) / len(vals) if vals else None
    return {"temperature": temperature, "datasets": datasets, "pooled": pooled}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--urls", required=True); p.add_argument("--tokenizer", required=True)
    p.add_argument("--temperature", type=float, default=0.0); p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--out-dir", required=True); p.add_argument("--max-per-dataset", type=int, default=0)
    p.add_argument("--datasets", default=",".join(ORDER)); p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--label", default="")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    work = queue.Queue()
    names = [n for n in a.datasets.split(",") if n]
    items = [it for n in names for it in load_items(n, a.max_per_dataset)]
    # longest conversations first (MT-Bench two-turn) so the tail of the run is not one long item
    items.sort(key=lambda it: (-len(it[2]), ORDER.index(it[0]), it[1]))
    for it in items:
        work.put(it)
    urls = [u for u in a.urls.split(",") if u]
    results, errors, lock = [], [], threading.Lock()
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(u, tok, a, work, results, lock, errors), daemon=True) for u in urls]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        with lock:
            done = len(results)
        print(f"[{time.strftime('%H:%M:%S')}] {a.label} done {done} requests, {len(errors)} errors, {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(a.out_dir, "requests.jsonl"), "w") as fh:
        for r in sorted(results, key=lambda r: (r["dataset"], r["sample_id"], r["turn"])):
            fh.write(json.dumps(r) + "\n")
    summary = summarize(results, a.temperature)
    summary.update({"label": a.label, "errors": errors, "elapsed_s": round(time.time() - t0, 1), "servers": len(urls)})
    with open(os.path.join(a.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    pooled = summary["pooled"]
    print(f"{a.label}: pooled macro {pooled['request_macro_tau']:.4f} micro {pooled['micro_tau']:.4f} "
          f"requests {pooled['requests']} errors {len(errors)}")
    for name, s in summary["datasets"].items():
        print(f"  {name:10s} n={s['requests']:5d} macro {s['request_macro_tau']:.4f} micro {s['micro_tau']:.4f} len_limited {s['length_limited_requests']}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
