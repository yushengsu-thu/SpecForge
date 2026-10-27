# Qwen3-8B DSpark MoE drafter (16 x 768, top-1), jiapingW's protocol, one 8x H200 devbox

Box-side wrappers for `examples/configs/online/disaggregated/managed-local/qwen3-8b-moe16x768-16A1-dspark-2server-dp6-h200.yaml` (draft
`configs/qwen3-8b-moe16x768-16A1-dspark.json`): two capture servers on GPUs 0-1, a six-rank DDP trainer on GPUs 2-7,
all owned by `specforge train`. The box holds the assets under `/scratch/moe2ep/` (`models/Qwen3-8B`,
`data/perfectblend_train_regen_temperature0_no_think.jsonl`, `cache/qwen3-8b-dspark-perfectblend-8192`) and the venv
`/scratch/moe2ep/venv`; secrets in `/scratch/secrets/` (`hf_token` = the account that owns the HF repo, `wandb_key`).

- `bash run_1node.sh`: 3-step smoke, then the full run (W&B project `specforge-qwen3-8b-moe16x768`, entity
  `eigent_radixark_training` by default) and the export watcher.
- `bash launch_1node.sh smoke|full` and `bash watch_epoch_exports.sh` run the pieces separately.
- Every epoch end (step 10764 x k) is exported (block 16, `Qwen3MoeDSparkModel`, tokenizer) and uploaded to
  `yushengsu/qwen3-8b-moe16x768-16A1-dspark/<k>ep/` (private) with a refreshed README; outputs under
  `/scratch/moe8b16/outputs/qwen3-8b-moe16x768-16A1-dspark-2server-dp6-h200/`.
