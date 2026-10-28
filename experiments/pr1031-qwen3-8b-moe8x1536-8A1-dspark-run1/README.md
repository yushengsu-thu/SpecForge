# pr1031-qwen3-8b-moe8x1536-8A1-dspark-run1

A 6-epoch Qwen3-8B DSpark draft whose FFN is the MoE of PR [#1031](https://github.com/sgl-project/SpecForge/pull/1031)
(this branch = `dev-moe-ffn` @ 3543fa2a plus this folder and `configs/qwen3-8b-moe8x1536-8A1-dspark.json`, a copy of
the PR's `configs/qwen3-8b-moe-dspark.json`). No code change.

- Draft: 2 layers, GQA 8 heads / 2 KV heads, 8 experts x 1536 per layer, top-1, batch-wise load-balance loss 0.005,
  expert bias 0.001 per step, DSpark projector over target layers 1/9/17/25/33, block size 16.
- Recipe (as jiapingW/Qwen3-8b-dspark-SpecForge, FFN swapped for the MoE): PerfectBlend regen (temperature 0, thinking
  off), `qwen3-instruct`, max length 8192, global batch 132 (2 x 11 x 6 ranks), lr 6e-4 cosine over 6 epochs with 4%
  warmup, 0.1 CE + 0.9 L1, 512 anchors, loss_decay_gamma 7.0, `use_liger_kernel: true` (Liger RMSNorm; the experts
  stay on grouped_mm), seed 42.
- Node: specforge-qwen-h200 (8x H200): 2 SGLang capture servers (GPU 0-1) + 6 DDP trainer ranks (GPU 2-7).
- Checkpoints: every 1,196 optimizer steps (= 1/9 epoch; 10,764 steps per epoch, 64,584 in total), so the 9th, 18th, ...
  checkpoints are exact epoch ends. The trainer keeps all of them (`max_checkpoints: 0`); `watch_ckpts.sh` exports each
  one to the servable HF layout, normalizes it for SGLang 0.5.18, uploads it to
  https://huggingface.co/yushengsu/new-repo-qwen3-8b-moe16x768-16A1-dspark/tree/main/pr1031-qwen3-8b-moe8x1536-8A1-dspark-run1
  as `step-N/` (epoch ends also as `ep1/` .. `ep6/`), verifies size and sha256, then deletes the export and every older
  raw checkpoint (the newest raw checkpoint stays on the box).
- W&B: entity `eigent_radixark_training`, project `specforge-qwen3-8b-moe16x768`, run `pr1031-qwen3-8b-moe8x1536-8A1-dspark-run1`.

Files: `train.yaml` (recipe, devbox paths), `setup.sh` (venv), `launch.sh` (trainer + watcher), `watch_ckpts.sh`,
`upload_ckpt.py`. Secrets are read from `/scratch/secrets/{wandb_key,hf_token}` on the box only.

Run on the box:

```bash
bash /scratch/pr1031/SpecForge/experiments/pr1031-qwen3-8b-moe8x1536-8A1-dspark-run1/setup.sh
bash /scratch/pr1031/SpecForge/experiments/pr1031-qwen3-8b-moe8x1536-8A1-dspark-run1/launch.sh
```
