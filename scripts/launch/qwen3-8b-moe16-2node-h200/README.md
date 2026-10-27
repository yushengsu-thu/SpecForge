# Qwen3-8B DSpark MoE drafter (16 x 768, top-1) with jiapingW's protocol, two 8x H200 devboxes

Box-side wrappers for `examples/configs/online/disaggregated/external/qwen3-8b-moe16x768-dspark-8server-dp8-2node-h200.yaml`
(draft `configs/qwen3-8b-moe16x768-dspark.json`), adapted from `scripts/launch/qwen38-moe-2node-h200/`. The boxes already
hold the assets under `/scratch/moe2ep/` (`models/Qwen3-8B`, `data/perfectblend_train_regen_temperature0_no_think.jsonl`,
`cache/qwen3-8b-dspark-perfectblend-8192`) and the venv `/scratch/moe2ep/venv`; secrets in `/scratch/secrets/`.

1. trainer node: `bash trainer_node.sh <CAPTURE_IP> <TRAINER_IP> smoke|full rdma` (starts the Mooncake master, waits
   for the eight servers, launches); capture node: `bash capture_node.sh <CAPTURE_IP> <TRAINER_IP> 8 rdma`.
2. `run_8b.sh <CAPTURE_IP> <TRAINER_IP>` on the trainer node runs the smoke and, when it exits cleanly, the full run.

Outputs: `/scratch/moe8b16/outputs/<run_id>/launch.log`; W&B project `specforge-qwen3-8b-moe16x768`
(entity `eigent_radixark_training` by default). Checkpoints every 3,000 steps as in jiapingW's run (one kept).
