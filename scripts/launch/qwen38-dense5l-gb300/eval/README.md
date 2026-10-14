# Accept-length / throughput eval of the dense 5-layer drafter (Kan's protocol, lite)

Runs on a 4x H200 devbox (`specforge-fsdp2`, image sglang 0.5.21): one SGLang server per GPU, one drafter per server,
evaluated concurrently with the same datasets and the same client. Everything lives under `/scratch/dense5l-eval/`.

| file | role |
|---|---|
| `eval_drafters_lite.py` | the client. Renders each prompt with the target's chat template (`enable_thinking` per dataset), posts to `/generate` (greedy, `temperature 0`, `top_k 1`), sums `completion_tokens` and `spec_verify_ct` from `meta_info`. Metric = aggregate accept length (`output tokens / verify steps`, bonus token included) and `drafts/step = aggregate - 1`; also reports output tok/s at the client concurrency. Datasets: `gsm8k` (openai/gsm8k test, 1319, 5-shot from train, non-thinking, 1024 new tokens), `math500` (HuggingFaceH4/MATH-500, 500, thinking, 8192), `mt_bench` (HuggingFaceH4/mt_bench_prompts, 80 x 2 turns, thinking, 4096; turn 2 is appended after the served turn-1 answer), `aime26` (math-ai/aime26, 30, thinking, 32768), `traces` (first user turn of the first 512 rows of `qwen38_regen_mixture_v1_train.jsonl`, thinking, 8192). |
| `serve_dense.sh <draft dir> <gpu> <port>` | the server: bf16 target `Qwen/Qwen3.8-27B`, `--speculative-algorithm DSPARK --speculative-dspark-block-size 7 --speculative-num-steps 1`, draft unquantized, triton attention for target and draft, `--mem-fraction-static 0.85 --max-running-requests 16 --max-mamba-cache-size 32 --context-length 40960 --disable-radix-cache`. (`--max-mamba-cache-size 160` leaves only 31k KV tokens on an H200 and makes long thinking generations OOM-retract; 32 states leave 489k tokens.) |
| `eval_one.sh <k>` | serves `drafts/dense5l-<k>ep` on GPU k-1 / port 3020k, waits for `/health`, runs the five datasets at concurrency 8, writes `logs/eval_dense5l-<k>ep.json`, stops the server. |
| `eval_generic.sh <name> <draft dir> <gpu> <port>` | same for any drafter (used for the official v1 `RadixArk/Qwen3.8-27B-DSpark@b9a5dbdf`). |
| `queue_official.sh` | waits for the epoch-1 chain to free GPU 0, then runs `eval_generic.sh official-v1 ...`. |

Setup on the box (once): `hf download RadixArk/qwen38-dspark-dense-5layer --include "<k>ep/*"` into `drafts/dense5l-<k>ep`,
`hf download RadixArk/Qwen3.8-27B-DSpark --revision b9a5dbdf` into `drafts/official-v1`, the bf16 target copied to
`models/Qwen3.8-27B`, and `HF_HOME` / `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` / `HF_DATASETS_CACHE` pointed at a
writable directory (the image sets them to a read-only `/cluster-storage`). Then:

```
for k in 1 2 3 4; do setsid nohup bash eval_one.sh $k > logs/eval_one_$k.log 2>&1 < /dev/null & sleep 20; done
setsid nohup bash queue_official.sh > logs/eval_one_official.log 2>&1 < /dev/null &
```

Each `logs/eval_<name>.json` holds, per dataset: conversations, output_tokens, verify_steps, aggregate_accept_length,
drafts_per_step, tokens_per_s, elapsed_s, errors. `errors` must be 0 for a valid row.
