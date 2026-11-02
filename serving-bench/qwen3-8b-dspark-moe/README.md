# Qwen3-8B DSpark MoE drafter: serving and benchmarking against the dense draft

Scripts and results for serving SpecForge's Qwen3-8B DSpark MoE drafts (8 experts x 1536, top-1) with SGLang and comparing them
with jiapingW's dense draft. The numbers and their discussion are in [RESULTS.md](RESULTS.md).

## What runs the draft: SGLang + sgl-project/sglang#47181

The serving code lives in SGLang, not in this repository. SGLang main's DSpark path loads dense DFlash/DSpark drafts only; the MoE
FFN support comes from [sgl-project/sglang#47181](https://github.com/sgl-project/sglang/pull/47181) (`python/sglang/srt/models/dflash.py`:
`parse_dflash_moe_spec`, `DFlashMoE`, the MoE branch of `DFlashDraftModel.load_weights`; `dspark.py`: the `Qwen3MoeDSparkModel` alias).
Until that PR is merged, an sglang checkout needs the patch here, `pr47181-sglang-c10e0e17.patch` (the PR's commit ef08ef9, applies
with `git am -3` on sglang main c10e0e17, image `lmsysorg/sglang:nightly-dev-20261029-c10e0e17`); afterwards a newer SGLang image
is enough and the patch step in `setup2.sh` becomes a no-op. The dense draft needs no patch. Nothing in SpecForge's own code
(training, export) is required; the drafts are served straight from the HF exports (`config.json` + `model.safetensors`,
`architectures=["Qwen3MoeDSparkModel"]`, `gate.selection_bias`), without `specforge export` and without SpecForge#1016's external
serving package.

## Bootstrap on an rx devbox

```bash
rx devbox acquire --gpu h200 --count 2 --image lmsysorg/sglang:nightly-dev-20261029-c10e0e17 --name specforge-moe-serving --ttl 12h
rx devbox run specforge-moe-serving -- bash -c 'mkdir -p /personal/moe-serving'
tar cf - -C serving-bench/qwen3-8b-dspark-moe . | rx devbox run specforge-moe-serving -- tar xf - -C /personal/moe-serving
rx devbox run specforge-moe-serving -- bash -c 'cp /personal/moe-serving/pr47181-sglang-c10e0e17.patch /personal/moe-serving/pr47181.patch; bash /personal/moe-serving/setup2.sh; bash /personal/moe-serving/fix.sh'
```

`setup2.sh` applies the patch to `/sgl-workspace/sglang`, installs SpecForge's CLI with `pip install --no-deps -e` (its pins would
replace the image's torch/transformers/flashinfer), and `fix.sh` points the HF cache at node-local `/scratch` (the box's default
cache is read-only) and fetches Qwen3-8B and the dense draft. Then, per draft:

```bash
rx devbox run specforge-moe-serving -- bash -c '(cd /personal/moe-serving && setsid nohup bash bench_all.sh 1 /personal/moe-serving/export2/step-56000 30001 moe56000 > logs/bench_all_moe56000.log 2>&1 < /dev/null &)'
```

## Files

| file | role |
|---|---|
| `serve.sh` | the server: `python -m sglang.launch_server --model-path Qwen/Qwen3-8B --tp-size 1 --dtype bfloat16 --attention-backend flashinfer --mem-fraction-static 0.8 --speculative-algorithm DSPARK --speculative-draft-model-path $DRAFT` + extra flags (`--speculative-dspark-block-size 8` everywhere; sweeps/profiles add `--max-running-requests 256 --cuda-graph-max-bs-decode 256`). |
| `bench_all.sh <gpu> <draft> <port> <tag>` | one draft over gsm8k/math500/humaneval/mbpp/mt-bench with `specforge benchmark` (c=1, greedy, 2048 max tokens): the five-dataset table. |
| `screen_queue.sh <gpu> <port> <step>...` | downloads and runs `bench_all.sh` for several checkpoints in sequence on one GPU. |
| `dl_steps.sh <step>...` | fetches MoE-run exports into `export2/step-N`. |
| `table.py` | builds the accept-length table from `logs/bench_<tag>_g8_<dataset>_c1.json`. |
| `bench_accept.py` | concurrency-sweep client: 16 fixed prompts, `/generate`, Qwen3 no-think template, greedy, 256 tokens, `--concurrency`; reports accept length, per-verify-step ms, per-request and total tok/s. |
| `sweep.sh`, `sweep_moe43000.sh` | the width-8 concurrency sweep c = 1..256 (written for the earlier docker-based box; adapt the `docker exec` lines). |
| `bench_datasets.sh` | the earlier four-dataset driver (docker-based box). |
| `profile_segments.sh <gpu> <draft> <port> <tag>` | serves with `SGLANG_DSPARK_DEBUG_DUMP=core,step_gpu_time,draft_gpu_time,target_verify_gpu_time`, drives c = 1/8/32/128, saves `GET /server_info` records per concurrency. |
| `summarize_prof.py` | means of draft / target-verify / step GPU ms at the steady-state batch size from those records. |
| `setup2.sh`, `fix.sh` | devbox bootstrap (see above). |
| `pr47181-sglang-c10e0e17.patch` | sgl-project/sglang#47181 as a patch for sglang main c10e0e17. |

## `mixed-and-tasks/`: per-task and mixed-workload e2e throughput (2026-11-01)

A second set of scripts, written for an rx devbox where SGLang c10e0e17 + PR #47181 is installed from source in its own venv
(`BENCH_DIR`, default `/scratch/moe8b16/serving-bench`) and the `specforge benchmark` client runs from a separate checkout
(`SPECFORGE_SRC`, `CLIENT_PY`). Every server start is guarded: the script refuses to start unless its GPU has no compute
process and under 1 GB in use, and logs the check. One server per GPU, `--tp-size 1`, bf16 target, flashinfer,
`--mem-fraction-static 0.8 --max-running-requests 256 --cuda-graph-max-bs-decode 256 --speculative-dspark-block-size 16`
(at c10e0e17 the decode flag is `--cuda-graph-max-bs-decode` and verify graphs capture up to batch 256 at block 16);
`task_sweep.sh`/`mixed_sweep.sh` add jiapingW's `--context-length 16384 --disable-radix-cache`. The MoE draft runs the fused-MoE
Triton kernel with its default config unless the tuned `E=8,N=1536` file is present.

| file | role |
|---|---|
| `tp_sweep.sh <name> <draft> [server args]` | `bench_accept.py` concurrency sweep (16 prompts, 256 tokens) with `TP=`/`DP=`/`GPUS=`/`REPEAT=` env; `tp_table.py` tabulates. |
| `task_sweep.sh <name> <draft>` | one draft over the five tasks with full prompt sets (gsm8k 1319, math500 500, humaneval 164, mbpp 257, mt-bench 160) via `specforge benchmark` at c = 1/8/16/32/64 (+128/256 for the two large sets); `run_tasks_all.sh` (dense then MoE, one GPU), `run_tasks_groups_par.sh` / `run_tasks_groups45.sh` (repeat groups on idle GPUs); `task_table.py` prints one table per task, mean over groups. step ms there is wall x min(c, requests) / verify steps (exact at c=1). |
| `mixed_bench.py --model <tokenizer> --concurrency c --output-json f` | the five tasks shuffled into ONE request stream (2,400 requests, seed 42, same order for every draft), each prompt built with the CLI's own `_load_prompts` formatter and chat template, CLI sampling defaults; reports tok/s, per-request verify-step ms (e2e_latency / spec_verify_ct), accept length and a per-task breakdown. At c=1 the per-task accept lengths inside the mix equal the standalone runs to three decimals. |
| `mixed_sweep.sh <name> <draft>` / `run_mixed_all.sh` | the mixed sweep at c = 1..256 for one draft / three groups x {dense, MoE} on six empty GPUs; `mixed_table.py` prints the final table (mean over groups). |
