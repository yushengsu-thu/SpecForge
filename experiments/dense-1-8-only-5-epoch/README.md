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
`watch_epochs_5ep.sh` / `upload_epoch_5ep.py` (per-epoch export, SGLang normalization, upload, model card),
`finish_5ep.sh` (end-of-run hang cleanup, restricted to this run's process tree), `end_pyspy.txt` (py-spy dump of the hung producer).

## Results

Training finished 2026-11-02 12:35 UTC: 53,820 optimizer steps at 2.45 s/step (7.3 h per epoch, 36.7 h in all), 0 tracebacks,
data wait 0.002 s. As in every dense-1/8 segment, the run hung at shutdown: the trainer ranks exited, but the managed producer
kept spinning in `streaming_ref_channel.poll` (`end_pyspy.txt`; same bug under the PR #1003 code) and the launcher, capture servers
and mooncake stayed up until `finish_5ep.sh` killed the tree 10 min later (GPU memory back to 0 on all 8 cards). Every epoch is on the Hub (`1ep/` ... `5ep/`, 440.8 MB each, size and LFS sha256 verified after upload).
Training metrics from the trainer log (`train/acc` = mean hard-label accuracy over the 16 drafted positions, `loss` = 0.1 CE + 0.9 L1);
"last" is the epoch's last logged step, "mean" the average over the epoch's last 1,000 optimizer steps.

| epoch | step | acc (last) | ce (last) | loss (last) | acc (mean) | ce (mean) | loss (mean) | lr at epoch end |
|---|---|---|---|---|---|---|---|---|
| 1 | 10,764 | 0.6032 | 1.7603 | 0.6947 | 0.6117 | 1.7242 | 0.7869 | 5.6e-4 |
| 2 | 21,528 | 0.6441 | 1.5688 | 0.6937 | 0.6420 | 1.5479 | 0.6969 | 4.1e-4 |
| 3 | 32,292 | 0.6582 | 1.4500 | 0.7237 | 0.6564 | 1.4711 | 0.6550 | 2.2e-4 |
| 4 | 43,056 | 0.6714 | 1.3807 | 0.5948 | 0.6610 | 1.4552 | 0.6441 | 6.2e-5 |
| 5 | 53,820 | 0.6624 | 1.4405 | 0.5785 | 0.6696 | 1.4120 | 0.6401 | 0 |

Final step: position-wise acc 0.857 (p0) ... 0.524 (p15), `tau_probabilistic` 6.62, hard-label walk accepted length 4.57,
teacher agreement 0.665, draft top-1 probability 0.781. Reference points: `dense-1-8-only-2-epoch` (same recipe on a 2-epoch
cosine) ended at acc 0.6601 / loss 0.537; the 32-epoch `dense-1-8` run (lr 2e-4 segments, pre-PR-1003 code) logged acc 0.6494
at its epoch 5 and 0.6771 at epoch 12. No acceptance-length benchmark has been run on these checkpoints.
