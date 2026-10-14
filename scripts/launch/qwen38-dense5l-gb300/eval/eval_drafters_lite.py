#!/usr/bin/env python3
"""Accept-length evaluation of DSpark drafters served by SGLang (Kan's protocol, lite).

Mirrors the metric of Kan's eval_drafters.py: the forward-step-weighted aggregate
accept length (total generated tokens / total verify steps, bonus token included)
and ``drafts/step = aggregate - 1``. Prompts are rendered with the target's chat
template (Qwen3.5 Jinja) with ``enable_thinking`` on/off per dataset and posted to
``/generate`` (greedy). Datasets: gsm8k (5-shot, non-thinking), math500 (thinking),
mt_bench (80 prompts x 2 turns, thinking; turn 2 is appended after the served
turn-1 answer).

  python eval_drafters_lite.py --name official --base-url http://127.0.0.1:30200 \
      --model /scratch/models/Qwen3.8-27B --datasets gsm8k,math500,mt_bench --concurrency 8 \
      --out /scratch/logs/eval_official.json
"""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from transformers import AutoTokenizer

GSM8K_FEWSHOT_N = 5


def load_gsm8k(n):
    from datasets import load_dataset

    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    shots = [(r["question"], r["answer"]) for r in train.select(range(GSM8K_FEWSHOT_N))]
    prefix = "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a in shots)
    rows = test if n is None else test.select(range(min(n, len(test))))
    return [[prefix + f"Question: {r['question']}\nAnswer:"] for r in rows]


def load_math500(n):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = ds if n is None else ds.select(range(min(n, len(ds))))
    return [[f"{r['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."] for r in rows]


def load_mt_bench(n):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    rows = ds if n is None else ds.select(range(min(n, len(ds))))
    return [list(r["prompt"]) for r in rows]


def _first_key(row, keys):
    for k in keys:
        if k in row and row[k]:
            return row[k]
    raise KeyError(f"none of {keys} in {list(row.keys())}")


def load_aime26(n):
    from datasets import load_dataset

    try:
        ds = load_dataset("math-ai/aime26", split="test")
    except Exception:
        ds = load_dataset("math-ai/aime26")
        ds = ds[list(ds.keys())[0]]
    rows = ds if n is None else ds.select(range(min(n, len(ds))))
    return [[f"{_first_key(r, ['problem', 'question', 'Problem', 'Question'])}\nPlease reason step by step, and put your final answer within \\boxed{{}}."] for r in rows]


TRACES_JSONL = "/personal/specforge-qwen38-backup/data/qwen38_regen_mixture_v1_train.jsonl"


def load_traces(n):
    """First user turn of the first n training conversations (Kan: training traces, 512, thinking)."""
    import json as _json

    n = 512 if n is None else n
    out = []
    with open(TRACES_JSONL, encoding="utf-8") as fh:
        for line in fh:
            row = _json.loads(line)
            conv = row.get("conversations") or []
            user = next((m for m in conv if m.get("role") in ("user", "human")), None)
            if user is None:
                continue
            out.append([user.get("content") or user.get("value") or ""])
            if len(out) >= n:
                break
    return out


DATASETS = {
    "gsm8k": dict(load=load_gsm8k, thinking=False, max_new_tokens=1024),
    "math500": dict(load=load_math500, thinking=True, max_new_tokens=8192),
    "mt_bench": dict(load=load_mt_bench, thinking=True, max_new_tokens=4096),
    "aime26": dict(load=load_aime26, thinking=True, max_new_tokens=32768),
    "traces": dict(load=load_traces, thinking=True, max_new_tokens=8192),
}


def render(tok, messages, thinking):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tok.apply_chat_template(messages, enable_thinking=thinking, **kwargs)
    except TypeError:
        return tok.apply_chat_template(messages, **kwargs)


def generate(base_url, prompt, max_new_tokens, timeout):
    r = requests.post(
        base_url.rstrip("/") + "/generate",
        json={"text": prompt, "sampling_params": {"temperature": 0.0, "top_k": 1, "max_new_tokens": max_new_tokens}},
        timeout=timeout,
    )
    r.raise_for_status()
    p = r.json()
    return p[0] if isinstance(p, list) else p


def run_conversation(tok, base_url, turns, thinking, max_new_tokens, timeout):
    """Serve each user turn in order, feeding the model's own answer back as context."""
    messages = []
    toks = 0
    verify = 0
    for user in turns:
        messages.append({"role": "user", "content": user})
        out = generate(base_url, render(tok, messages, thinking), max_new_tokens, timeout)
        mi = out.get("meta_info", {}) or {}
        toks += int(mi.get("completion_tokens", 0))
        verify += int(mi.get("spec_verify_ct", 0))
        messages.append({"role": "assistant", "content": out.get("text", "")})
    return toks, verify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="target model dir (tokenizer + chat template)")
    ap.add_argument("--datasets", default="gsm8k,math500,mt_bench")
    ap.add_argument("--gsm8k-n", type=int, default=None)
    ap.add_argument("--math500-n", type=int, default=None)
    ap.add_argument("--mt-bench-n", type=int, default=None)
    ap.add_argument("--aime26-n", type=int, default=None)
    ap.add_argument("--traces-n", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    results = {"name": args.name, "base_url": args.base_url, "datasets": {}}
    for ds in args.datasets.split(","):
        spec = DATASETS[ds]
        n = getattr(args, f"{ds.replace('-', '_')}_n", None)
        convs = spec["load"](n)
        try:
            requests.post(args.base_url.rstrip("/") + "/flush_cache", timeout=60)
        except requests.RequestException:
            pass
        t0 = time.perf_counter()
        total_toks = 0
        total_verify = 0
        errors = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [
                ex.submit(run_conversation, tok, args.base_url, turns, spec["thinking"], spec["max_new_tokens"], args.timeout)
                for turns in convs
            ]
            for f in as_completed(futs):
                try:
                    t, v = f.result()
                    total_toks += t
                    total_verify += v
                except Exception as e:  # noqa: BLE001
                    errors += 1
        el = time.perf_counter() - t0
        agg = total_toks / total_verify if total_verify else None
        rec = {
            "conversations": len(convs),
            "thinking": spec["thinking"],
            "max_new_tokens": spec["max_new_tokens"],
            "output_tokens": total_toks,
            "verify_steps": total_verify,
            "aggregate_accept_length": agg,
            "drafts_per_step": (agg - 1) if agg else None,
            "tokens_per_s": total_toks / max(el, 1e-9),
            "elapsed_s": el,
            "errors": errors,
        }
        results["datasets"][ds] = rec
        print(f"[{args.name}] {ds}: drafts/step={rec['drafts_per_step'] if agg is None else round(rec['drafts_per_step'],3)} "
              f"aggregate={None if agg is None else round(agg,3)} tok/s={rec['tokens_per_s']:.1f} "
              f"tokens={total_toks} steps={total_verify} errors={errors} ({el/60:.1f} min)", flush=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
    print("EVAL DONE", args.name)


if __name__ == "__main__":
    main()
