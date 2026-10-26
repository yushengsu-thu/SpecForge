# Qwen3.8-27B DSpark MoE drafter (16 x 1088, top-1), 6 epochs on two 8x H200 devboxes

Box-side scripts for the run described in
`examples/configs/online/disaggregated/external/qwen3.8-27b-moe-dspark-8server-dp8-2node-h200.yaml`
(draft config `configs/qwen3.8-27b-moe-dspark.json`: the RadixArk/Qwen3.8-27B-DSpark v2 architecture with the dense
17408-wide FFN split into 16 experts of 1088, one per token; training block 16). Adapted from
`scripts/launch/qwen38-dense5l-gb300/` (branch `yusheng/dense-5layer-3ep-gb300`).

| what | where |
|---|---|
| devboxes | two 8x H200 boxes on `h200-sci-k8s-1` (`rx devbox acquire --gpu h200 --count 8 ...`), or one `--count 16` devbox (rank 0 / rank 1) |
| secrets | `/scratch/secrets/hf_token`, `/scratch/secrets/wandb_key` (mode 600; never committed), trainer node only |
| setup (code, deps, spec-capture patch, BF16 teacher, corpus jsonl) | both nodes: `setsid nohup bash setup_box.sh > /scratch/moe16x1088/logs/setup_box.log 2>&1 < /dev/null &` (capture node with `SKIP_DATA=1`) |
| capture node | `bash capture_node.sh <CAPTURE_IP> <TRAINER_IP> 8 rdma` (8 TP1 BF16 capture servers, ports 30000-30007, 100 GiB Mooncake segment each) |
| trainer node, smoke (3 optimizer steps, no W&B, no checkpoints) | `bash trainer_node.sh <CAPTURE_IP> <TRAINER_IP> smoke rdma` |
| trainer node, full run (6 epochs, 14,874 steps) | `bash trainer_node.sh <CAPTURE_IP> <TRAINER_IP> full rdma` (W&B entity `eigent_radixark_training` by default, `WANDB_ENTITY=` overrides; project `specforge-qwen38-moe-16x1088`) |
| per-epoch export + upload | trainer node: `setsid nohup bash watch_epoch_exports.sh > /scratch/moe16x1088/logs/watch_epoch_exports.log 2>&1 < /dev/null &` |
| outputs | `/scratch/moe16x1088/outputs/<run_id>/launch.log` (`step N: {...}` lines every 10 steps), checkpoints `<run_id>-step<N>` |
| HF | `RadixArk/qwen38-dspark-moe-16x1088` (private; folders `1ep/` … `6ep/`; `HF_REPO=` overrides) |

All scripts activate `$VENV` (default `/scratch/moe2ep/venv`: the boxes' sglang 0.5.18 with the spec-capture patch and
mooncake) and import specforge from `/scratch/SpecForge` through `PYTHONPATH`; `REPO=` may point `setup_box.sh` at a
local clone instead of GitHub. Order: `setup_box.sh` on both nodes; `trainer_node.sh` first (it starts the mooncake master the capture servers register
with, then waits for them), `capture_node.sh` second. `prep_regen_mixture.py` turns the
`RadixArk/Qwen3.8-27B-Regen-Mixture-v1` shards into the slim `{id, conversations}` JSONL the v2 recipe trains on.
`make_readme.py` refreshes the HF model card from `launch.log` after each epoch. The exported drafts carry
`architectures: ["Qwen3MoeDSparkModel"]`, which SGLang needs to pick the MoE draft class
(`SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang_models`, branch `dev-moe-serving`); with the stock
`DSparkDraftModel` name it would load the dense class and ignore the experts.

Devbox gotchas (from the earlier runs): SGLang 0.5.18's `get_free_port()` can hand two capture servers the same NCCL
port, so the servers pin `--nccl-port`; the rx ssh relay drops connections after ~60 s, so everything long runs under
`setsid nohup`; on H200 the first BF16 server start is quick, but an FP8 target spends 20+ minutes in DeepGEMM JIT
warm-up unless `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`; a captured 8k sample is up to ~0.42 GiB, so the trainer node's
feature store is capped at 640 GiB (768 samples in flight) in the recipe.

## One-node variant (one 8x H200 devbox)

Recipe `examples/configs/online/disaggregated/managed-local/qwen3.8-27b-moe-dspark-4server-dp4-h200.yaml` (4 capture
servers on GPUs 0-3 + DP4 on GPUs 4-7, batch 2 x 64 x 4 = 512, same protocol): after `setup_box.sh`,
`bash launch_train.sh smoke|full`, then `RUN_ID=qwen3.8-27b-moe-dspark-4server-dp4-h200 bash watch_epoch_exports.sh`.
