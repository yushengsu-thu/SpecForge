# Fused-MoE kernel config tuning for a DSpark MoE draft (TMA-aware, up/down separate)

Tunes the Triton `fused_moe_kernel` launch parameters (BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages, USE_TMA) that SGLang uses
for the draft's MoE FFN, separately for the up (gate_up) and down projections, per token-count bucket, and installs them as the two files
SGLang reads: `E=<E>,N=<N>,device_name=<GPU>.json` and `..._down.json`. Measured on the Qwen3-8B 2x12288 top-1 draft (H200, block size 7):
mixed-stream throughput vs the dense draft went from -6..-12% (Triton default config) to -1..-5% (this tune); accept length is unchanged
(the config only changes tiling, not the math).

## What you need

- An SGLang build whose Triton fused-MoE kernel has the TMA path and the separate down config (SGLang main `c10e0e17` + sgl-project/sglang#47181,
  or newer), in a venv: `VENV` (default `/scratch/moe8b16/serving-bench/venv`). The tuner module imports `ray`; it is not in that venv, so a
  private copy is used via `PYTHONPATH=/scratch/ckpt-sweep/tune/pylib` (`pip install --no-deps --target <dir> ray`).
- The same GPU model and Triton version you will serve on: the config file name carries `device_name`, the directory carries the Triton version,
  and the best tiles differ between GPUs. Re-tune when either changes.
- `SHAPE_CFG`: a directory with a stand-in `config.json` of architecture `Qwen3MoeForCausalLM` describing the draft's MoE shape (the tuner reads
  `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `hidden_size`, `dtype`, `num_hidden_layers`). Example for 2x12288:
  `{"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "hidden_size": 4096, "moe_intermediate_size": 12288,
  "num_experts": 2, "num_experts_per_tok": 1, "num_hidden_layers": 5, "dtype": "bfloat16", ...}`
- For the verification step: the target model (`TARGET`), the draft (`DRAFT`, HF layout, `architectures: ["Qwen3MoeDSparkModel"]`),
  `bench_accept.py` (in this folder). For `E2E=1`: the serving-bench scripts (`task_sweep_sel.sh`, `mixed_sweep.sh`, `mixed_bench.py`),
  the specforge client package and the dataset cache at the paths `run_e2e_tma2.sh` expects.

## Buckets: which token counts to tune

The kernel sees M = (draft tokens per request) x (requests in the batch). A DSpark draft at `--speculative-dspark-block-size 7` runs 7 query
tokens per request per step, so M = 7 x c. Tune the concurrencies you serve at, plus one large bucket for prefill batches:

| c | 1 | 8 | 16 | 32 | 64 | 128 | 256 | (prefill) |
|---|---|---|---|---|---|---|---|---|
| M | 7 | 56 | 112 | 224 | 448 | 896 | 1792 | 3584 |

At run time the loader picks the bucket nearest to the actual M, so buckets only need to bracket the range. With block size 15 (V16) use 15 x c.

## Run

```bash
# defaults = the Qwen3-8B 2x12288 draft, buckets 7 56 112 224 448 896 1792 3584, 8 GPUs, pruned 648-config search (~15 min per bucket)
bash tune_moe_draft_e2e.sh

BUCKETS="7 56 224 1792" GPUS="0 1 2 3" bash tune_moe_draft_e2e.sh     # only the concurrencies you serve at
FULL_SPACE=1 bash tune_moe_draft_e2e.sh                               # SGLang's full 1,920-config grid (~1 h per bucket per GPU)
BASE=/path/to/old/E=2,N=12288,device_name=NVIDIA_H200.json bash ...   # keep an earlier config's entries for buckets not re-tuned
E2E=1 bash tune_moe_draft_e2e.sh                                      # afterwards run the 5-group per-task + mixed protocol with the new config
SHAPE_CFG=/path/cfg DRAFT=/path/draft OUT=/path/out bash ...          # another draft / another output dir
```

Steps the script performs (each GPU is checked to be empty before anything starts on it):

1. Writes routing dumps for the tuner (`topk_ids_layer<L>_idx<K>.pt`, balanced top-1 over the draft's experts). Replace them with dumps from a
   real serving run if the router is unbalanced.
2. Runs `tune_tma_one.py` for one bucket per GPU (queues when there are more buckets than GPUs). For every candidate config it measures the
   down GEMM without and with TMA (round 1), then the up GEMM without and with TMA (round 2), and keeps the BLOCK_SIZE_M whose up+down total
   is fastest. Output: `<WORK>/tma-tune/M<M>.json` with the up and down configs (each with its `USE_TMA` decision) and their kernel times.
3. `assemble_tma2.py` merges the buckets into `<OUT>/configs/triton_<ver>/E=<E>,N=<N>,device_name=<GPU>.json` (+ `_down.json`).
4. Starts one server with `SGLANG_MOE_CONFIG_DIR=<OUT>` and checks the log says `Using MoE kernel config from ...json`, `... _down.json`,
   `Up MoE TMA is enabled`, `Down MoE TMA is enabled`; then measures accept length and draft / verify / step time at c = 1, 8, 32, 128
   (`profile_segments3.sh`, DSpark's own CUDA-event timers via `/server_info`). Accept length must be identical to the untuned run.
5. With `E2E=1`, runs the full protocol (`run_e2e_tma2.sh`: 5 groups, units spread over the GPUs).

## Using the result

Point the server at the config directory (nothing in the sglang tree is modified):

```bash
SGLANG_MOE_CONFIG_DIR=/scratch/ckpt-sweep/moe-config-auto python -m sglang.launch_server ... --speculative-algorithm DSPARK \
  --speculative-draft-model-path <draft> --speculative-dspark-block-size 7
```

or copy the two JSON files into the package directory `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_<ver>/`.

## Files

| file | role |
|---|---|
| `tune_moe_draft_e2e.sh` | orchestrator: routing dumps -> per-bucket tunes -> assemble -> install -> verify (-> e2e) |
| `tune_tma_one.py` | one bucket on one GPU; wraps `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` (`BenchmarkWorker.tune`, `enable_up_tma=True`) |
| `assemble_tma2.py` | merges `M*.json` into the up / down config files named after the shape and the GPU |
| `run_e2e_tma2.sh` | the 5-group per-task + mixed protocol for the MoE draft, units spread over 8 GPUs (uses `task_sweep_sel.sh` / `mixed_sweep.sh`) |
| `profile_segments3.sh`, `bench_accept.py`, `summarize_prof.py` | verification: server + DSpark timers at c = 1 / 8 / 32 / 128 |
| `make_report.py` | the agreed deliverables from result dirs: accept-length table, five per-task tables, mixed table, chart |
| `run_tma_tune.sh` | the manual 8-GPU launch used for the 2x12288 tune (superseded by the orchestrator) |
| `E=2,N=12288,device_name=NVIDIA_H200.json`, `..._down.json`, `M*.json` | the tuned result for the 2x12288 draft on H200 / triton 3.8.0 |

## Notes and caveats

- Switching to a draft with a different expert shape (E or N) invalidates the result: the file name changes and the tiles differ. Another checkpoint
  of the same architecture can reuse the files.
- The pruned search space (BLOCK_M 16-128, BLOCK_N 64-256, BLOCK_K 64-256, GROUP 1/16/32, stages 3-5, warps 4/8) was chosen around the winners
  of a full-grid tune of this shape; for a very different shape run once with `FULL_SPACE=1`.
- The tuner's kernel times are isolated micro-benchmarks; always confirm with the server-side step times (step 4). A bucket that measures slower
  in the server than the previous config can be reverted by hand in the JSON.
- `--moe-runner-backend` / `--speculative-moe-runner-backend` do not reach a DSpark draft in SGLang `c10e0e17` + #47181 (the draft is built before
  `initialize_moe_config()` runs); the Triton path tuned here needs no change, but to try other backends apply `sglang-dflash-init-moe-config.patch` (against sglang `c10e0e17` + #47181).
- Only empty GPUs are used; a non-empty GPU is skipped or the step aborts.

## Reporting

```bash
python make_report.py \
  --dense /scratch/ckpt-sweep/bench/logs:dense-g1,dense-g2,dense-g3,dense-g4-big+dense-g4-small,dense-g5-big+dense-g5-small \
  --moe   /scratch/ckpt-sweep/bench/logs:moe-tma2-g1-big+moe-tma2-g1-small,...,moe-tma2-g5-big+moe-tma2-g5-small \
  --out report/
```

Writes `report/report.md` with (1) `| | Avg (all requests) | Avg (per task) | gsm8k | math500 | humaneval | mbpp | mt-bench |` rows dense / MoE /
diff / diff %, (2) one `| c | dense tok/s | MoE tok/s | diff % | dense step ms | MoE step ms |` table per task (c = 1 … 256; cells mean ± sample
std over the groups, diff = diff of the means), (3) the same table for the mixed stream, and (4) `report/mixed_throughput.png` (x = 1, 8, …, 256
equally spaced, y from 0, dense blue circles, MoE orange squares). The MoE side is always the tuned config: tune first, then run the e2e.
A group split into unit dirs is given as `dirA+dirB`; a repeat more than 25% off its cell median is dropped and listed.
