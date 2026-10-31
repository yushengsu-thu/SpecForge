# dense-1-8-only-5-epoch

The dense-1/8 draft (2 layers, GQA 8/2, one 1536-wide MLP; `configs/qwen3-8b-dspark-dense-1-8-2layer-causal-noconf-b16.json`
on this branch, byte-identical to the copy used) retrained for a 5-epoch cosine schedule with the code of SpecForge
PR [#1003](https://github.com/sgl-project/SpecForge/pull/1003) at commit
[cde2f059](https://github.com/yushengsu-thu/SpecForge/commit/cde2f059) (branch `dev-moe-draft` of this fork). Everything else is the
`yushengsu/dense-1-8-only-2-epoch` recipe: global batch 132, lr 6e-4 cosine, warmup 4%, 0.1 CE + 0.9 L1, 512 anchors,
no system prompt, seed 42, Liger off (PR #1003's DSpark rejects it), one checkpoint per epoch (`save_interval 10764`).

| | |
| --- | --- |
| node | `specforge-qwen-h200` (8x H200), work dir `/scratch/pr1003`, 2 SGLang capture servers (GPU 0-1) + 6 DDP ranks (GPU 2-7) |
| started | 2026-10-31 23:53 UTC; about 7.2 h per epoch (2.41 s/step), 53,820 steps total |
| W&B | https://wandb.ai/eigent_radixark_training/specforge-qwen3-8b-dspark-dense/runs/w7a1a4et (run name `dense-1-8-only-5-epoch`) |
| checkpoints | https://huggingface.co/yushengsu/dense-1-8-only-5-epoch (private): `1ep/` ... `5ep/`, each a servable SGLang 0.5.18 export; the repo was renamed from `dense-1-8-only-2-epoch` and its 1ep/2ep folders removed on 2026-10-31 (still in the repo history) |

Files: `setup_pr1003.sh` (checkout + venv), `train_dense-1-8-only-5-epoch.yaml` (recipe), `launch_5ep.sh` (trainer + watcher),
`watch_epochs_5ep.sh` / `upload_epoch_5ep.py` (per-epoch export, SGLang normalization, upload, model card).
