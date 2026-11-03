"""Greedy chat completions at a given client concurrency; report the per-request
spec_accept_length SGLang returns in meta_info, and the output throughput.
The server batches the in-flight requests, so --concurrency is the decode
batch size it sees (up to --max-running-requests)."""
import argparse
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Think step by step.",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? Think step by step.",
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
    "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
    "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
    "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?",
    "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
    "Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?",
    "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?",
    "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?",
    "Explain in a few paragraphs how a hash table works and when its average-case guarantees break down.",
    "Write a Python function that merges two sorted lists into one sorted list, with a short explanation.",
    "Summarize the causes of the French Revolution in about 200 words.",
    "A train leaves city A at 9:00 travelling at 60 km/h toward city B, 300 km away. Another train leaves city B at 10:00 travelling toward city A at 90 km/h. When and where do they meet? Think step by step.",
    "List five practical tips for improving sleep quality and explain the reasoning behind each.",
    "Translate the following to French and then explain the grammar: 'The children were playing in the garden when it started to rain.'",
]


QWEN3_NO_THINK = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def one(args, prompt):
    """/generate with the Qwen3 chat template (thinking off): its meta_info
    carries the per-request spec_accept_length; the OpenAI endpoint does not."""
    body = json.dumps(
        {
            "text": QWEN3_NO_THINK.format(prompt=prompt),
            "sampling_params": {"temperature": 0, "max_new_tokens": args.max_tokens},
        }
    ).encode()
    req = urllib.request.Request(
        args.url + "/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        out = json.loads(resp.read())
    meta = out["meta_info"]
    return (
        meta["completion_tokens"],
        meta.get("spec_accept_length"),
        meta.get("spec_verify_ct"),
        meta.get("e2e_latency"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--num-requests", type=int, default=0, help="total requests (cycles the prompts); 0 = len(PROMPTS) * repeat")
    args = ap.parse_args()
    prompts = PROMPTS * args.repeat
    if args.num_requests:
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.num_requests)]
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda p: one(args, p), prompts))
    elapsed = time.time() - start
    tokens = sum(r[0] for r in results)
    accepts = [float(r[1]) for r in results if r[1] is not None]
    steps = [(r[2], r[3]) for r in results if r[2] and r[3]]
    print(
        f"concurrency={args.concurrency} requests={len(prompts)} completion_tokens={tokens} "
        f"elapsed={elapsed:.1f}s output_throughput={tokens / elapsed:.1f} tok/s"
    )
    if accepts:
        print(
            f"spec_accept_length: mean={statistics.mean(accepts):.3f} "
            f"min={min(accepts):.2f} max={max(accepts):.2f} (n={len(accepts)})"
        )
    if steps:
        # e2e_latency / verify steps: the time one verify step (target + draft) takes a request at this load
        per_step = [lat / ct * 1000 for ct, lat in steps]
        per_request = tokens / len(results) / statistics.mean(lat for _, lat in steps)
        print(
            f"per_verify_step_ms: mean={statistics.mean(per_step):.2f} median={statistics.median(per_step):.2f} "
            f"| per_request_throughput={per_request:.1f} tok/s"
        )
    else:
        print("no spec_accept_length in meta_info (is speculative decoding on?)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
