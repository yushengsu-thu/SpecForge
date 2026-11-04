# pr1031-qwen3-8b-moe16x768-16A1-dspark-run1

A 6-epoch Qwen3-8B DSpark draft whose FFN is the MoE of PR [#1031](https://github.com/sgl-project/SpecForge/pull/1031),
with 16 experts of 768 (the dense FFN's 12288 split 16 ways) instead of the 8 x 1536 of
`exp/pr1031-qwen3-8b-moe8x1536-8A1-dspark-6ep` (this branch = that branch plus this folder,
`configs/qwen3-8b-moe16x768-16A1-dspark.json`, and `training.save_at_epoch_end`, the trainer option that checkpoints at
every epoch end on top of `save_interval`).

- Draft: 5 layers, GQA 32 heads / 8 KV heads (head_dim 128), 16 experts x 768 per layer, top-1, batch-wise load-balance
  loss 0.005, expert bias 0.001 per step, DSpark projector over target layers 1/9/17/25/33, block size 16.
- Recipe (as jiapingW/Qwen3-8b-dspark-SpecForge, FFN swapped for the MoE): PerfectBlend regen (temperature 0, thinking
  off), `qwen3-instruct`, max length 8192, global batch 132 (2 x 11 x 6 ranks), lr 6e-4 cosine over 6 epochs with 4%
  warmup, 0.1 CE + 0.9 L1, 512 anchors, loss_decay_gamma 7.0, `use_liger_kernel: true` (Liger RMSNorm; the experts stay on
  grouped_mm), seed 42.
- Node: specforge-exp-branch (8x H200): 2 SGLang capture servers (GPU 0-1) + 6 DDP trainer ranks (GPU 2-7). The target
  model and the data are the copies already on that box (`/scratch/moe2ep/models/Qwen3-8B`,
  `/scratch/moe2ep/data/perfectblend_train_regen_temperature0_no_think.jsonl`); everything else lives under `/scratch/pr1031`.
- Checkpoints: every 1,000 optimizer steps plus every epoch end (`save_at_epoch_end`; the epoch ends come from the
  producer's schedule record, e.g. steps 10765 / 21529 / 32294 / 43058 / 53823 / 64586 for 1,420,903 prompts). The trainer
  keeps all of them (`max_checkpoints: 0`); `watch_ckpts.sh` exports each one to the servable HF layout, normalizes it for
  SGLang 0.5.18, uploads it to the public repo https://huggingface.co/yushengsu/qwen3-3b-16E1A as `step-N/` (epoch ends
  also as `ep1/` .. `ep6/`), verifies size and sha256, then deletes the export and every older raw checkpoint (the newest
  raw checkpoint stays on the box).
- W&B: entity `eigent_radixark_training`, project `qwen3-3b-16E1A`, run `pr1031-qwen3-8b-moe16x768-16A1-dspark-run1`.

Files: `train.yaml` (recipe, devbox paths), `setup.sh` (venv), `launch.sh` (trainer + watcher), `watch_ckpts.sh`,
`upload_ckpt.py`. Secrets are read from `/scratch/secrets/{wandb_key,hf_token}` on the box only.

Run on the box:

```bash
bash /scratch/pr1031/SpecForge/experiments/pr1031-qwen3-8b-moe16x768-16A1-dspark-run1/setup.sh
bash /scratch/pr1031/SpecForge/experiments/pr1031-qwen3-8b-moe16x768-16A1-dspark-run1/launch.sh
```
