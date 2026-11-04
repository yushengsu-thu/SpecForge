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
| checkpoints | https://huggingface.co/yushengsu/Qwen3-8B-2layer-Dense-v2 (private): `1ep/` ... `5ep/`, each a servable SGLang 0.5.18 export. Renamed on 2026-11-04 from `yushengsu/dense-1-8-only-5-epoch` (naming scheme `<base model>-<layers>layer-<Dense|MoE>[-<E>E<A>A]`, `-vN` = N-th run of that architecture by repo creation date; `-v1` is the 32-epoch `dense-1-8` run, now `yushengsu/Qwen3-8B-2layer-Dense-v1`; old ids redirect); before that the repo was `dense-1-8-only-2-epoch`, whose 1ep/2ep folders were removed on 2026-10-31 (still in the repo history) |

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

## Acceptance length (SGLang serving, 2026-11-02)

Measured on `specforge-qwen-h200` with the protocol of jiapingW/Qwen3-8b-dspark-SpecForge, exactly as the earlier
`/scratch/dense-eval` (2026-10-27) and `/scratch/ep6-serving` (2026-10-31) runs on that node: SGLang 0.5.18, bf16 Qwen3-8B,
TP 1, FlashInfer, context 16384, radix cache and CUDA graphs off, one request at a time; 2,400 requests = GSM8K test 1319 +
MATH-500 500 + HumanEval 164 + MBPP sanitized 257 + MT-Bench 80 x 2 turns (turn 2 after the served turn-1 answer), Qwen3
non-thinking chat template, no system prompt, temperature 0, max 2048 new tokens. Per request tau = completion tokens /
verify steps (bonus and correction tokens included); "request average" is the mean of per-request tau, "tokens / verify steps"
the pooled micro average. Verify width V = `--speculative-dspark-block-size V-1`; the draft always computes its native 16-token
block. Four servers per model (eight for ep5), ~18 min per round at width 8 and ~15 min at width 16. The reference columns are
earlier measurements on the same node with the same client (`dense 2ep` = `yushengsu/dense-1-8-only-2-epoch` 2ep;
`dense-1/8 ep12` = epoch 12 of the 32-epoch run; the 5-layer MoE is PR #1031's `moe8x1536-8A1` run at epoch 6).
Scripts and raw per-model JSON: [`eval/`](eval/) (`run_eval5ep.sh`, `summarize_accept.py`, `results/`).

### Accept length, temperature 0, verify width 8 (request average)

| Task | 5-epoch run ep1 | 5-epoch run ep2 | 5-epoch run ep3 | 5-epoch run ep4 | 5-epoch run ep5 | dense 2ep (2-epoch run) | dense-1/8 ep12 (32-epoch run) | 5-layer MoE 8x1536 ep6 (PR #1031) |
|---|---|---|---|---|---|---|---|---|
| GSM8K | 5.479 | 5.639 | 5.705 | 5.759 | 5.772 | 5.704 | 5.759 | 6.382 |
| MATH-500 | 5.286 | 5.461 | 5.534 | 5.588 | 5.604 | 5.521 | 5.563 | 6.318 |
| HumanEval | 4.264 | 4.448 | 4.567 | 4.619 | 4.631 | 4.483 | 4.611 | 5.591 |
| MBPP | 4.068 | 4.251 | 4.354 | 4.417 | 4.441 | 4.295 | 4.473 | 5.392 |
| MT-Bench | 3.195 | 3.323 | 3.420 | 3.458 | 3.467 | 3.399 | 3.491 | 4.299 |
| Pooled (request average) | 5.052 | 5.218 | 5.295 | 5.349 | 5.363 | 5.278 | 5.351 | 6.070 |
| Equal-weight mean of 5 tasks | 4.458 | 4.625 | 4.716 | 4.768 | 4.783 | 4.680 | 4.780 | 5.596 |
| Pooled (tokens / verify steps) | 4.660 | 4.830 | 4.906 | 4.956 | 4.970 | 4.872 | 4.945 | 5.754 |

<details><summary>Verify width 16 (measured in the same run; width 8 is the configuration of interest)</summary>

| Task | 5-epoch run ep1 | 5-epoch run ep2 | 5-epoch run ep3 | 5-epoch run ep4 | 5-epoch run ep5 | dense 2ep (2-epoch run) | dense-1/8 ep12 (32-epoch run) | 5-layer MoE 8x1536 ep6 (PR #1031) |
|---|---|---|---|---|---|---|---|---|
| GSM8K | 6.906 | 7.263 | 7.454 | 7.545 | 7.569 | 7.389 | 7.590 | 9.463 |
| MATH-500 | 6.429 | 6.789 | 6.970 | 7.061 | 7.089 | 6.931 | 7.015 | 9.196 |
| HumanEval | 4.645 | 4.926 | 5.133 | 5.204 | 5.222 | 4.996 | 5.211 | 7.321 |
| MBPP | 4.448 | 4.722 | 4.899 | 4.985 | 5.020 | 4.780 | 5.093 | 6.838 |
| MT-Bench | 3.456 | 3.676 | 3.791 | 3.868 | 3.879 | 3.736 | 3.853 | 5.215 |
| Pooled (request average) | 6.159 | 6.493 | 6.677 | 6.765 | 6.790 | 6.607 | 6.791 | 8.696 |
| Equal-weight mean of 5 tasks | 5.177 | 5.475 | 5.649 | 5.733 | 5.756 | 5.566 | 5.753 | 7.606 |
| Pooled (tokens / verify steps) | 5.343 | 5.631 | 5.783 | 5.865 | 5.886 | 5.702 | 5.851 | 7.680 |

</details>

Every epoch improves acceptance, with shrinking steps (pooled request average at width 8: +0.17, +0.08, +0.05, +0.01).
ep5 is 1.6% above the 2-epoch run's 2ep (5.363 vs 5.278) and level with epoch 12 of the 32-epoch run (5.351);
the 5-layer 8-expert MoE draft stays 13% higher (6.070). Width 16 (collapsed above) tells the same story: 6.790 vs 6.607 / 6.791 / 8.696.

## End-to-end throughput (same-GPU comparison, 2026-11-02)

Same-GPU comparison: dense 2ep (2-epoch run) and dense 5-epoch run ep5 served one after the other on the same GPU (three GPUs per width, one with the order reversed). Temperature 0, 512 new tokens, prompts per task = max(200, 4 x concurrency), CUDA graphs on, servers allow 128 running requests. Unless a task is named, values are means of the five tasks. Run right after the acceptance measurement with the same server stack
(`/scratch/conc-sweep-epoch12/serve.sh`, e2e mode: FlashInfer, context 16384, radix cache off, `--max-running-requests 128`,
CUDA graphs up to batch 128) and `specforge benchmark` as the client; the 5-epoch model is ep5, the epoch with the highest pooled
acceptance at width 16. Raw per-(model, task, concurrency) JSON: [`eval/results_e2e/`](eval/results_e2e/) (260 files, none missing).

### Verify width 8

| Quantity | GPU | 1 concurrent request | 16 | 64 | 128 |
|---|---|---|---|---|---|
| 5-epoch run throughput vs 2ep | 2 (2ep first) | +2.7% | +3.4% | +2.1% | +1.8% |
| 5-epoch run throughput vs 2ep | 3 (5-epoch first) | +2.3% | +1.8% | +2.1% | +2.1% |
| 5-epoch run throughput vs 2ep | 6 (2ep first) | +2.4% | +2.5% | +2.3% | +1.5% |
| **5-epoch run throughput vs 2ep, mean of the three GPUs** | | +2.5% | +2.6% | +2.2% | +1.8% |
| **5-epoch run time per verify step vs 2ep, mean of the three GPUs** | | −0.4% | −0.4% | −0.1% | +0.3% |
| 5-epoch run accept length vs 2ep, mean | | +2.1% | +2.2% | +2.0% | +2.1% |

**Absolute throughput (tokens/s), verify width 8, mean of the five tasks and of the three GPUs**

| Model | 1 concurrent request | 16 | 64 | 128 |
|---|---|---|---|---|
| Base model, no draft | 194 | 2,590 | 7,402 | 11,710 |
| dense 2ep (2-epoch run) | 588 (+203.0% vs no draft) | 6,344 (+145.0% vs no draft) | 11,030 (+49.0% vs no draft) | 13,078 (+11.7% vs no draft) |
| dense 5-epoch run ep5 | 602 (+2.5% vs 2ep) | 6,508 (+2.6% vs 2ep) | 11,268 (+2.2% vs 2ep) | 13,313 (+1.8% vs 2ep) |
| 2ep, spread between GPUs | +2.4% | +0.9% | +0.7% | +1.5% |

**Per task throughput (tokens/s), verify width 8, mean of the three GPUs; 5-epoch cell = 5-epoch run (difference vs 2ep)**

| Task | Model | 1 concurrent request | 16 | 64 | 128 |
|---|---|---|---|---|---|
| GSM8K | no draft | 194 | 2,570 | 7,077 | 11,201 |
| GSM8K | dense 2ep (2-epoch run) | 722 (+272.2% vs no draft) | 7,338 (+185.6% vs no draft) | 13,001 (+83.7% vs no draft) | 15,706 (+40.2% vs no draft) |
| GSM8K | dense 5-epoch run ep5 | 729 (+1.0% vs 2ep) | 7,423 (+1.2% vs 2ep) | 13,054 (+0.4% vs 2ep) | 15,831 (+0.8% vs 2ep) |
| MATH-500 | no draft | 194 | 2,641 | 7,712 | 12,179 |
| MATH-500 | dense 2ep (2-epoch run) | 699 (+260.0% vs no draft) | 7,474 (+183.0% vs no draft) | 12,882 (+67.0% vs no draft) | 15,214 (+24.9% vs no draft) |
| MATH-500 | dense 5-epoch run ep5 | 711 (+1.8% vs 2ep) | 7,728 (+3.4% vs 2ep) | 13,275 (+3.0% vs 2ep) | 15,409 (+1.3% vs 2ep) |
| HumanEval | no draft | 194 | 2,589 | 7,542 | 11,643 |
| HumanEval | dense 2ep (2-epoch run) | 587 (+203.1% vs no draft) | 6,384 (+146.5% vs no draft) | 11,168 (+48.1% vs no draft) | 12,646 (+8.6% vs no draft) |
| HumanEval | dense 5-epoch run ep5 | 607 (+3.3% vs 2ep) | 6,552 (+2.6% vs 2ep) | 11,492 (+2.9% vs 2ep) | 13,002 (+2.8% vs 2ep) |
| MBPP | no draft | 194 | 2,576 | 7,254 | 11,753 |
| MBPP | dense 2ep (2-epoch run) | 551 (+183.5% vs no draft) | 5,983 (+132.3% vs no draft) | 10,646 (+46.8% vs no draft) | 13,049 (+11.0% vs no draft) |
| MBPP | dense 5-epoch run ep5 | 568 (+3.2% vs 2ep) | 6,161 (+3.0% vs 2ep) | 10,840 (+1.8% vs 2ep) | 13,299 (+1.9% vs 2ep) |
| MT-Bench | no draft | 194 | 2,572 | 7,425 | 11,774 |
| MT-Bench | dense 2ep (2-epoch run) | 382 (+96.4% vs no draft) | 4,541 (+76.6% vs no draft) | 7,453 (+0.4% vs no draft) | 8,777 (−25.5% vs no draft) |
| MT-Bench | dense 5-epoch run ep5 | 397 (+4.0% vs 2ep) | 4,674 (+2.9% vs 2ep) | 7,680 (+3.0% vs 2ep) | 9,024 (+2.8% vs 2ep) |

<details><summary>Verify width 16 (measured in the same run; width 8 is the configuration of interest)</summary>

| Quantity | GPU | 1 concurrent request | 16 | 64 | 128 |
|---|---|---|---|---|---|
| 5-epoch run throughput vs 2ep | 4 (2ep first) | +3.2% | +2.7% | +2.4% | +3.2% |
| 5-epoch run throughput vs 2ep | 5 (5-epoch first) | +3.8% | +3.3% | +3.0% | +3.7% |
| 5-epoch run throughput vs 2ep | 7 (2ep first) | +3.4% | +3.6% | +2.3% | +3.2% |
| **5-epoch run throughput vs 2ep, mean of the three GPUs** | | +3.5% | +3.2% | +2.6% | +3.4% |
| **5-epoch run time per verify step vs 2ep, mean of the three GPUs** | | −0.2% | −0.1% | +0.6% | −0.1% |
| 5-epoch run accept length vs 2ep, mean | | +3.2% | +3.1% | +3.2% | +3.2% |

**Absolute throughput (tokens/s), verify width 16, mean of the five tasks and of the three GPUs**

| Model | 1 concurrent request | 16 | 64 | 128 |
|---|---|---|---|---|
| Base model, no draft | 194 | 2,590 | 7,402 | 11,710 |
| dense 2ep (2-epoch run) | 650 (+234.8% vs no draft) | 5,657 (+118.5% vs no draft) | 7,883 (+6.5% vs no draft) | 8,659 (−26.1% vs no draft) |
| dense 5-epoch run ep5 | 672 (+3.5% vs 2ep) | 5,836 (+3.2% vs 2ep) | 8,086 (+2.6% vs 2ep) | 8,950 (+3.4% vs 2ep) |
| 2ep, spread between GPUs | +0.7% | +1.6% | +3.9% | +4.5% |

**Per task throughput (tokens/s), verify width 16, mean of the three GPUs; 5-epoch cell = 5-epoch run (difference vs 2ep)**

| Task | Model | 1 concurrent request | 16 | 64 | 128 |
|---|---|---|---|---|---|
| GSM8K | no draft | 194 | 2,570 | 7,077 | 11,201 |
| GSM8K | dense 2ep (2-epoch run) | 868 (+347.5% vs no draft) | 7,032 (+173.6% vs no draft) | 10,247 (+44.8% vs no draft) | 11,407 (+1.8% vs no draft) |
| GSM8K | dense 5-epoch run ep5 | 880 (+1.4% vs 2ep) | 7,171 (+2.0% vs 2ep) | 10,345 (+1.0% vs 2ep) | 11,728 (+2.8% vs 2ep) |
| MATH-500 | no draft | 194 | 2,641 | 7,712 | 12,179 |
| MATH-500 | dense 2ep (2-epoch run) | 799 (+311.7% vs no draft) | 6,965 (+163.7% vs no draft) | 9,557 (+23.9% vs no draft) | 10,440 (−14.3% vs no draft) |
| MATH-500 | dense 5-epoch run ep5 | 827 (+3.5% vs 2ep) | 7,177 (+3.0% vs 2ep) | 9,737 (+1.9% vs 2ep) | 10,712 (+2.6% vs 2ep) |
| HumanEval | no draft | 194 | 2,589 | 7,542 | 11,643 |
| HumanEval | dense 2ep (2-epoch run) | 623 (+221.9% vs no draft) | 5,513 (+112.9% vs no draft) | 7,717 (+2.3% vs no draft) | 8,210 (−29.5% vs no draft) |
| HumanEval | dense 5-epoch run ep5 | 652 (+4.6% vs 2ep) | 5,736 (+4.0% vs 2ep) | 8,020 (+3.9% vs 2ep) | 8,514 (+3.7% vs 2ep) |
| MBPP | no draft | 194 | 2,576 | 7,254 | 11,753 |
| MBPP | dense 2ep (2-epoch run) | 581 (+199.1% vs no draft) | 5,089 (+97.6% vs no draft) | 7,181 (−1.0% vs no draft) | 8,025 (−31.7% vs no draft) |
| MBPP | dense 5-epoch run ep5 | 607 (+4.4% vs 2ep) | 5,293 (+4.0% vs 2ep) | 7,427 (+3.4% vs 2ep) | 8,380 (+4.4% vs 2ep) |
| MT-Bench | no draft | 194 | 2,572 | 7,425 | 11,774 |
| MT-Bench | dense 2ep (2-epoch run) | 377 (+94.0% vs no draft) | 3,685 (+43.3% vs no draft) | 4,714 (−36.5% vs no draft) | 5,213 (−55.7% vs no draft) |
| MT-Bench | dense 5-epoch run ep5 | 395 (+4.8% vs 2ep) | 3,805 (+3.2% vs 2ep) | 4,900 (+3.9% vs 2ep) | 5,416 (+3.9% vs 2ep) |

</details>

At width 8 the ep5 draft delivers +2.5% / +2.6% / +2.2% / +1.8% more tokens per second than the 2-epoch run's draft at
concurrency 1 / 16 / 64 / 128 (602 / 6,508 / 11,268 / 13,313 tokens/s), which is the +2.1% acceptance-length gain at an unchanged
time per verify step (same architecture, same weights count); against the base model without a draft it is +210% at concurrency 1
and still +14% at 128.
