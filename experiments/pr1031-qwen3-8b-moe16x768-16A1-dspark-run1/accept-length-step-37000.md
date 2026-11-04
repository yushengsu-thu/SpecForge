# Serving accept length at step 37,000 (epoch 4 of 6): MoE 16x768 vs MoE 8x1536 run1

Measured 2026-11-04 on specforge-exp-branch (GPU 0 / GPU 1, next to the training run's capture servers) with the handoff
protocol: SGLang 0.5.18 (`/scratch/moe2ep/venv`), Qwen3-8B BF16 target, DSpark block 16, flashinfer, context 16384, radix
cache off, Triton's default fused-MoE config, `specforge benchmark` at concurrency 1, thinking off, temperature 0,
max_new_tokens 2048, full prompt sets (gsm8k 1319, math500 500, humaneval 164, mbpp 257, mt-bench 160). The 16x768 export
is served by the serving package of PR 1016 at `dev-moe-serving` cdd23545 (`gate.expert_bias`, config normalized to
`Qwen3MoeDSparkModel`); the 8x1536 run1 export by the same package at 7ff4a7dc (`gate.selection_bias`), as in every earlier
measurement. Dense row = the handoff's jiapingW dense draft (step-39000). Per-task AL = request-macro average acceptance
length; pooled macro = request-weighted over the 2,400 requests; pooled micro = output tokens / verify steps.

| ckpt | gsm8k | math500 | humaneval | mbpp | mt-bench | pooled macro | pooled micro | equal-weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense (jw, step-39000) | 9.971 | 9.600 | 7.551 | 7.059 | 5.239 | 9.101 | 7.926 | 7.884 |
| MoE 16x768 top-1, step-37000 (`yushengsu/qwen3-3b-16E1A/step-37000`) | 9.496 | 9.186 | 7.120 | 6.571 | 4.916 | 8.651 | 7.542 | 7.458 |
| MoE 8x1536 top-1, run1 step-37000 | 9.531 | 9.282 | 7.161 | 6.685 | 5.011 | 8.711 | 7.604 | 7.534 |
| 16x768 minus 8x1536 | -0.035 | -0.096 | -0.042 | -0.114 | -0.095 | -0.061 (-0.7%) | -0.062 | -0.076 (-1.0%) |

Both MoE drafts are at 57% of their schedule (lr 2.3e-4, still decaying); the 8x1536 reached pooled 8.925 at its best
checkpoint (run2 ep10). Raw JSONs: `/scratch/pr1031/eval/serve/{moe16,moe8-run1}-step-37000/<task>_c1.json` on the box;
runner `/scratch/pr1031/eval/run_eval.sh`, summary `/scratch/pr1031/eval/summarize.py`.
