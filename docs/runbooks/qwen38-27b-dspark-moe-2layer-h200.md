# Qwen3.8-27B DSpark MoE drafter: 2-layer ablation on one 8x H200 node

Status as of 2026-10-08 08:00 UTC: phase A (one epoch) is running on rx devbox `specforge-qwen-h200`
(node gpu-10-220-71-44); phase B (two more epochs) is chained to start automatically when A finishes.
Live metrics: https://wandb.ai/miles_training/specforge-2%20layer%20moe

## 1. What is being tested

Kan's Qwen3.8-27B DSpark MoE drafter (`configs/qwen3.8-27b-dspark-moe-v2.json`, results in
`docs/reports/qwen38-dspark-moe-3ep-report.md`) with the draft reduced from **5 to 2 decoder layers** and nothing
else changed: 512 routed experts of width 512, top-10 softmax routing with renormalisation, one sigmoid-gated shared
expert of width 2048, `moe_router_center: sample` (momentum 0.99), aux loss 0.001, grouped-GEMM dispatch, block size 7,
Markov head rank 256, confidence head, target capture layers 5/19/33/47/61.

`target_layer_ids` stays at five layers on purpose: `DFlashDraftModel` concatenates the captured target features,
projects them once with `fc` (25600 -> 5120) and feeds the same tensor to every draft layer, so target depth is
independent of draft depth. Parameter count 8.48B total (experts 8.12B) versus 20.82B for the 5-layer drafter.

Question: does a drafter with 40% of the depth (and roughly half the per-step draft cost) keep the accept length?

## 2. Code

- This branch = upstream `kan/moe-3-qwen38` @ `abf003e9` plus the files listed in section 3. No source changes.
  (`abf003e9` is four commits after the `c766217` HEAD Kan trained with: the report, continuation/restart recipes and
  the opt-in `SPECFORGE_FSDP_SHARDED_GRAD_ACCUM` toggle, which is off.)
- SGLang 0.5.18 from `lmsysorg/sglang:v0.5.18-cu130` (`/sgl-workspace/sglang`, release tree `71de97b`) with Kan's
  `patches/sglang/v0.5.18/spec-capture.patch` applied via `scripts/apply_sglang_spec_capture_patch.sh --target v0.5.18`
  (use the `main` version of the apply script if the branch's copy silently no-ops on your image), plus
  `patches/sglang/v0.5.18/devbox-stop-file.patch` (section 6).
- Extra pip packages on top of the image: yunchang 0.6.4, tensorboard, accelerate, wandb.

## 3. Files added by this branch

| file | purpose |
|---|---|
| `configs/qwen3.8-27b-dspark-moe-v2-2layer.json` | Kan's v2 draft config with `num_hidden_layers 2`, `layer_types` x2, `max_window_layers 2` |
| `examples/.../managed-local/qwen3.8-27b-dspark-moe-2layer-regen-mixture-v1-1ep-h200.yaml` | phase A: one epoch from scratch, 4 TP1 bf16 capture servers (GPUs 0-3) + 4 FSDP ranks (GPUs 4-7) |
| `examples/.../external/qwen3.8-27b-dspark-moe-2layer-regen-mixture-v1-cont2ep-from-1ep-h200.yaml` | phase B: weights-only warm start from A's step-4958 checkpoint, `num_epochs 2`, external mode reusing A's servers |
| `scripts/launch/launch-moe-2layer-h200.sh` | guarded launcher (refuses on busy GPUs / existing train process / existing control dir) |
| `scripts/launch/chain-moe-2layer-cont2ep-h200.sh` | waits for A's final checkpoint and trainer exit, checks servers and GPUs, launches B |
| `scripts/launch/wandb_mirror_launch_log.py` | mirrors `step N: {...}` lines of `launch.log` into W&B every 5 min (A, then B) |
| `patches/sglang/v0.5.18/devbox-stop-file.patch` | lets SGLang servers exit when `$SGLANG_STOP_FILE` appears (devboxes block `kill`) |

## 4. Recipe (identical to Kan's `-1ep-v3` except hardware)

- Data: `RadixArk/Qwen3.8-27B-Regen-Mixture-v1`, 1,269,290 conversations after the trainable-token filter,
  chat template `qwen3.5`, max_length 8192; one epoch = 4,958 steps at global batch 256.
- Optimiser: `batch_size 4 x accumulation_steps 16 x 4 ranks = 256`, AdamW CPU-offloaded, FULL_SHARD, lr 5e-4
  constant after 4% warm-up, grad clip 1.0, bf16, seed 42 / prompt_seed 42, checkpoint every 500 steps (3 kept).
- DSpark objective: CE 0.1 + L1 0.9 + confidence 1.0, 512 anchors, loss decay gamma 4, 128 objective chunk blocks,
  flex attention.
- Differences from Kan's B200 run: target served in **bf16** (`Qwen/Qwen3.8-27B`; the NVFP4 build needs Blackwell)
  on 8x H200 141 GB; capture is the bottleneck here (about 8.5 samples/s, Kan had 11.6), giving 28-31 s per step
  (train compute ~20 s + data wait 7-11 s), peak 73 GB allocated / 103 GB reserved per trainer rank.

Launch:

```sh
bash scripts/launch/launch-moe-2layer-h200.sh   # phase A (edit the /scratch paths for your node)
setsid nohup bash scripts/launch/chain-moe-2layer-cont2ep-h200.sh &   # waits for A, launches B
```

Phase B mirrors Kan's phase B (`-cont2ep-from-1ep-v3`): `model.draft_checkpoint_path` is a weights-only warm start,
so AdamW moments, the lr warm-up and the data position restart (prompt_seed 42 kept, as in his recipe). The only
structural difference is deployment: B runs in external disaggregated mode against the capture servers (:30000-30003)
and Mooncake master (127.0.0.1:35551 / :35880) left behind by A, because seccomp on these devboxes blocks `kill()`
so managed_local cannot tear them down and restart them. The trainer is launched with `CUDA_VISIBLE_DEVICES=4,5,6,7`.

## 5. Phase A progress (log every 10 steps; W&B run `qwen3-8-27b-dspark-moe-2layer-regen-mixture-v1-1ep-h200`)

| step | loss | ce | l1 | acc | experts unused | load_max | grad norm |
|---|---|---|---|---|---|---|---|
| 50 | 2.69 | 7.7 | 1.91 | 0.055 | 1% | 10.9 | 2.2 |
| 210 (end of warm-up) | 2.61 | 6.4 | 1.74 | 0.118 | 9% | 11.6 | 1.6 |
| 400 | 2.11 | 4.3 | 1.29 | 0.307 | 0% | 6.3 | 2.0 |
| 580 | 1.88 | 3.4 | 1.13 | 0.376 | 0% | 5.5 | 3.0 |
| 590 | 2.94 | 4.3 | 1.32 | 0.296 | 0% | 13.3 | 23.1 (single spike, recovered by step 640) |
| 850 | 1.73 | 2.8 | 0.95 | 0.447 | 0% | 3.5 | 0.3 |
| 1100 | 1.57 | 2.5 | 0.88 | 0.475 | 0% | 14.1 | 0.5 |
| 1350 | 1.49 | 2.3 | 0.82 | 0.501 | 0% | 4.2 | 0.2 |
| 1600 | 1.47 | 2.3 | 0.81 | 0.506 | 0% | 5.6 | 0.1 |

Routing never collapsed (experts_unused 0-3% after step 300, load_max oscillating 4-14 with ceiling 51.2). The acc
slope flattened after step 1300 (+0.005 over 300 steps). Reference points from Kan's 5-layer phase A: train/acc
0.55-0.57 around step 2400, 0.594 / loss 1.22 at step 4958.

## 6. Evaluation plan (after each phase)

Export with `specforge export --to hf --draft-config configs/qwen3.8-27b-dspark-moe-v2-2layer.json`, normalise with
`scripts/gates/normalize_dflash_export.py --block-size 7` (architectures `Qwen3MoeDSparkModel`, centering folded into
`gate.bias`), copy the target tokenizer, serve with SGLang DSPARK (gamma 7, bf16 target) and measure accepted
drafts per verify step on GSM8K (1319, 5-shot, non-thinking), MATH500 and MT-Bench (thinking), concurrency 8.
Baselines measured on the same bf16 H200 setup on 2026-10-07: official v1 dense 4.59 / 3.63 / 2.60; Kan's 5-layer
MoE 3-ep export 4.83 / 3.62 / 2.68. Kan's own numbers (NVFP4 target): dense 1-ep 4.01, MoE 1-ep 4.18, MoE 3-ep 4.56.

## 7. Devbox notes

- seccomp blocks the `kill` syscall: processes cannot be stopped. Apply `devbox-stop-file.patch` and set
  `SGLANG_STOP_FILE` so capture servers can be asked to exit; trainers end on their own.
- The rx ssh relay resets sessions longer than about 60 s: run long work detached (`setsid nohup`) and poll.
- Each checkpoint of the 2-layer drafter is 111 GB (training_state.pt + 4 optimizer shards); 3 are kept.
