# pr1031-qwen3-8b-moe16x768-16A2-dspark-run1

The 16x768 MoE DSpark draft of `exp/pr1031-qwen3-8b-moe16x768-16A1-dspark-6ep` with **two active experts per token** (16E2A)
instead of one: this branch = that branch plus this folder and `configs/qwen3-8b-moe16x768-16A2-dspark.json`
(`num_experts_per_tok` 1 → 2, `norm_topk_prob` false → true so the two routing weights sum to one, as in Qwen3-MoE; everything
else as the 16A1 config). Same recipe, node and checkpoint handling as the 16A1 run.

- Draft: 5 layers, GQA 32 heads / 8 KV heads (head_dim 128), 16 experts x 768 per layer, top-2 (renormalized), batch-wise
  load-balance loss 0.005, expert bias 0.001 per step, DSpark projector over target layers 1/9/17/25/33, block size 16.
- Recipe (as jiapingW/Qwen3-8b-dspark-SpecForge, FFN swapped for the MoE): PerfectBlend regen (temperature 0, thinking
  off), `qwen3-instruct`, max length 8192, global batch 132 (2 x 11 x 6 ranks), lr 6e-4 cosine over 6 epochs with 4%
  warmup, 0.1 CE + 0.9 L1, 512 anchors, loss_decay_gamma 7.0, `use_liger_kernel: true`, seed 42.
- Node: specforge-exp-branch (8x H200): 2 SGLang capture servers (GPU 0-1) + 6 DDP trainer ranks (GPU 2-7); target model and
  data from `/scratch/moe2ep`, everything else under `/scratch/pr1031` (shared venv with the 16A1 run).
- Checkpoints: every 1,000 optimizer steps plus every epoch end (`save_at_epoch_end`); `watch_ckpts.sh` exports each one to
  the servable HF layout, normalizes it for SGLang 0.5.18, uploads it to the public repo
  https://huggingface.co/yushengsu/Qwen3-8B-5layer-MoE-16E2A-v1 as `step-N/` (epoch ends also as `ep1/` .. `ep6/`), verifies
  size and sha256, then deletes the export and every older raw checkpoint.
- W&B: entity `eigent_radixark_training`, project `Qwen3-8B-5layer-MoE-16E2A-v1`, run `pr1031-qwen3-8b-moe16x768-16A2-dspark-run1`.

Files: `train.yaml`, `setup.sh`, `launch.sh`, `watch_ckpts.sh`, `upload_ckpt.py`, `status.sh`. Secrets are read from
`/scratch/secrets/{wandb_key,hf_token}` on the box only.

Run on the box:

```bash
bash /scratch/pr1031/SpecForge/experiments/pr1031-qwen3-8b-moe16x768-16A2-dspark-run1/setup.sh
bash /scratch/pr1031/SpecForge/experiments/pr1031-qwen3-8b-moe16x768-16A2-dspark-run1/launch.sh
```
