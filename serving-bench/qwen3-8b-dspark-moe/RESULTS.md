# Qwen3-8B DSpark MoE drafter (8 experts x 1536, top-1) vs jiapingW dense: SGLang serving results (H200, 2026-10-29)

MoE run: `yushengsu/qwen3-8b-moe16x768-16A1-dspark`, folder `qwen3-8b-moe8x1536-8A1-dspark-run1` (recipe of
jiapingW/Qwen3-8b-dspark-SpecForge with the FFN replaced by a Qwen3.5-MoE FFN, 8 x 1536, one expert per token,
6 epochs = 64584 steps; checkpoints up to step-52000 were on HF when this was measured).
Dense reference: `jiapingW/Qwen3-8b-dspark-SpecForge` (its published checkpoint, FFN 12288).

Serving: one H200 per server, `lmsysorg/sglang:dev` (main c10e0e17) with sgl-project/sglang#47181 cherry-picked
(the MoE draft FFN is built from the export's HF MoE config fields; `Qwen3MoeDSparkModel` and `gate.selection_bias`
are accepted as aliases), Qwen3-8B bf16 target, TP 1, flashinfer attention, mem-fraction 0.8,
`--speculative-algorithm DSPARK --speculative-dspark-block-size 8` (width 8) for every draft.
Accept lengths: `specforge benchmark --model Qwen/Qwen3-8B --dataset <ds> --num-prompts N --concurrency 1`
(thinking off), N = gsm8k 100, math500 100, humaneval 164, mbpp 200, mt-bench 80; identical settings for all rows.

## Every checkpoint from step-43000 to step-57000, five tasks, width 8, concurrency 1 (accept length)

train/acc is the README's training accuracy (mean over the block positions, 100-step average before the checkpoint).

```
draft           train/acc   gsm8k  math500  humaneval   mbpp  mt-bench   mean  vs dense  tok/s(mean)
dense (jiapingW)     -      7.039    6.854      5.995  5.761     4.333  5.996      0.0%      740
moe step-43000   0.772      6.929    6.788      5.919  5.631     4.357  5.925     -1.2%      739
moe step-44000   0.778      6.940    6.791      5.899  5.617     4.325  5.914     -1.4%      735
moe step-45000   0.783      6.920    6.795      5.906  5.644     4.347  5.923     -1.2%      741
moe step-46000   0.780      6.892    6.812      5.919  5.663     4.365  5.930     -1.1%      742
moe step-47000   0.789      6.957    6.798      5.947  5.640     4.371  5.943     -0.9%      743
moe step-48000   0.808      6.900    6.812      5.935  5.674     4.356  5.935     -1.0%      739
moe step-49000   0.781      6.946    6.802      5.940  5.654     4.381  5.945     -0.9%      742
moe step-50000   0.797      6.968    6.816      5.948  5.672     4.351  5.951     -0.8%      743
moe step-51000   0.793      6.937    6.801      5.939  5.666     4.364  5.941     -0.9%      744
moe step-52000   0.786      6.959    6.811      5.947  5.671     4.371  5.952     -0.7%      744
moe step-53000   (n/a)      6.990    6.813      5.941  5.676     4.356  5.955     -0.7%      742
moe step-54000   (n/a)      6.962    6.817      5.943  5.670     4.378  5.954     -0.7%      741
moe step-55000   (n/a)      6.979    6.818      5.956  5.671     4.381  5.961     -0.6%      742
moe step-56000   (n/a)      6.962    6.826      5.958  5.674     4.385  5.961     -0.6%      741
moe step-57000   (n/a)      6.957    6.810      5.957  5.670     4.402  5.959     -0.6%      741
```
(steps 53000-57000 were measured on the rx devbox `specforge-moe-serving`, see below; their train/acc was not yet in the README.)

Output throughput per task (tok/s, concurrency 1):

```
draft             gsm8k  math500  humaneval   mbpp  mt-bench
dense               868      860        776    720       475
moe step-43000      861      861        775    715       485
moe step-47000      867      864        781    715       488
moe step-48000      858      862        775    713       487
moe step-50000      867      864        779    719       487
moe step-51000      867      867        779    718       488
moe step-52000      868      866        779    719       489
```

Findings
- Best MoE checkpoints by the five-task mean: step-55000 and step-56000 (both 5.961, -0.6% vs dense), then
  57000 (5.959), 53000/54000 (5.955/5.954) and 52000 (5.952). Everything from step-47000 on lies within 0.3% of each
  other, which is noise level; the trend is a slow rise with training (+0.8% from 44000 to 56000, flat since), so the
  remaining checkpoints (58000 .. 64584, cosine tail) should be screened the same way as they land. No MoE checkpoint
  beats dense on the mean yet; mt-bench is the one task where every checkpoint from 46000 on is above dense (+0.5-1.6%).
- train/acc does not predict accept length: step-48000 has the highest train/acc (0.808, next best 0.797) but a
  mid-pack mean (5.935). Pick checkpoints with the serving benchmark, not with train/acc.
- Per task the MoE draft is 0.5-2% below dense on gsm8k/math500/humaneval/mbpp and 0.5-1% above it on mt-bench.
- Throughput at concurrency 1 is the same within 1-3% (the draft FFN is memory-bound there; MoE reads 1536 of
  12288 FFN rows per token and the difference is hidden behind the 8B target verify).

## Width-8 concurrency sweep: dense vs MoE step-43000 (same server settings, same prompt set per concurrency)

`bench_accept.py` against `/generate` (Qwen3 no-think template); step_ms = per verify step, per_req = tok/s per request.

```
      ---------------- dense ----------------   ------------- moe step-43000 -------------
conc  accept  step_ms   tok/s  per_req  reqs    accept  step_ms   tok/s  per_req  reqs
   1   6.660     8.22     693    695.7    32     6.583     8.12     694    696.8    32
   2   6.683     9.29    1199    623.5    32     6.594     9.25    1187    617.4    32
   4   6.616    10.06    2132    569.6    32     6.571    10.13    2066    558.6    32
   8   6.636    10.97    3564    520.0    32     6.620    11.18    3472    516.9    32
  16   6.610    14.34    5355    399.3    64     6.596    14.66    5183    387.4    64
  24   6.568    16.32    6964    345.4    96     6.597    16.12    7029    347.5    96
  32   6.601    18.13    8389    311.5   128     6.586    17.97    8418    314.2   128
  48   6.599    24.28    9799    235.6   192     6.602    22.45   10493    252.9   192
  64   6.569    27.54   11607    205.4   256     6.539    26.80   11669    208.6   256
  96   6.598    37.72   12924    151.4   384     6.547    35.93   13579    158.0   384
 128   6.600    45.28   14349    126.0   512     6.547    43.86   14834    129.7   512
 192   6.604    64.59   15343     88.6   768     6.537    63.02   15655     90.3   768
 256   6.614    81.64   16207     70.0  1024     6.541    78.91   16730     72.1  1024
```

- Per-step time and total throughput match within 3% at every concurrency; from 48 up the MoE server is 2-5% faster.
- Per-request throughput of the dense server drops 25% by concurrency 8, halves by 32, and total throughput
  saturates around 128-256 (about 16k tok/s); the MoE draft follows the same curve.

## Reproduced on an rx devbox (2026-10-29)

The dense and MoE step-52000 rows were re-run on the devbox `specforge-moe-serving` (2x H200 on `h200-sci-k8s-1`, same image
`lmsysorg/sglang:nightly-dev-20261029-c10e0e17` + PR #47181, same settings) and came out identical to three decimals on all five
tasks (dense 7.039 / 6.854 / 5.995 / 5.761 / 4.333, MoE step-52000 6.959 / 6.811 / 5.947 / 5.671 / 4.371, mean 5.996 vs 5.952),
with the same speculative verify counts: at temperature 0 the benchmark is deterministic across machines. The devbox keeps the
scripts, the PR patch, the dense draft and the step-52000 export under `/personal/moe-serving/` (survives release); setup of a
fresh box is `rx devbox acquire --gpu h200 --count 2 --image lmsysorg/sglang:nightly-dev-20261029-c10e0e17 --init setup2.sh`
(install SpecForge with `--no-deps`, point `HF_HUB_CACHE` at `/scratch`, apply the PR with `git am`), then
`bench_all.sh <gpu> <draft dir> <port> <tag>` per checkpoint and `python3 ../table.py` inside `logs/`.

## Draft vs target-verify time inside one step (measured, rx devbox, 2026-10-30)

SGLang's DSpark worker times each step's segments with CUDA events when `SGLANG_DSPARK_DEBUG_DUMP=core,step_gpu_time,draft_gpu_time,target_verify_gpu_time`
is set; the per-step records come back in `GET /server_info` (`dspark_info_record.records`) and are cleared with
`POST /set_internal_state {"server_args": {"dspark_clear_info_records": 1}}`. Same servers as the sweep (width 8, graphs and running
requests up to 256), `bench_accept.py` prompts x 256 tokens at each concurrency, mean over the steps at the steady-state batch size
(`profile_segments.sh` + `summarize_prof.py`). GPU ms per verify step:

```
draft           c    bs  steps   draft ms  verify ms   step ms   draft%  verify%
dense           1     1   1380       1.85       5.32      7.41      25%      72%
dense           8     8    110       2.07       5.92      8.27      25%      72%
dense          32    32     99       2.53       9.31     12.24      21%      76%
dense         128   128     94       6.35      29.39     36.46      17%      81%
moe step-56000  1     1   1364       1.78       5.32      7.34      24%      73%
moe step-56000  8     8    104       2.31       5.93      8.51      27%      70%
moe step-56000 32    32     96       2.67       9.36     12.43      22%      75%
moe step-56000 128  128     82       5.45      29.36     35.52      15%      83%
```

- The draft (5 layers, one forward for the whole 8-token block) is about 3x faster than the target verify (36 layers, 9 tokens per
  request) at c=1 and takes a quarter of the GPU step; by c=128 it is 15-17%. The remaining ~3% of the step is the accept/sampling glue.
  The client-side step time (8.4 ms at c=1) exceeds the GPU step (7.4 ms) by the CPU scheduling and HTTP overhead between steps.
- The MoE draft costs the same as the dense draft at c=1 (1.78 vs 1.85 ms: with top-1 routing, the 8 tokens of a block still touch
  most of the 8 experts, so the weight traffic is not smaller), 0.1-0.25 ms more at c=8-32 (Triton MoE without a tuned config for
  E=8, N=1536, plus the torch routing), and 14% less at c=128 where compute dominates. The verify time is identical, as it must be.

## Where the numbers live

Box `gmi-h200-n01-nda`, `~/claude-dflash-moe/`: `logs/bench_<tag>_g8_<dataset>_c1.json` (specforge benchmark output),
`logs/all_g8_<tag>.txt`, `logs/sweep_g8_{dense,moe43000}.txt`; `python3 ../table.py` inside `logs/` rebuilds the table;
`bench_all.sh <container> <draft dir> <port> <tag>` runs one draft over the five tasks; image `claude-dflash-moe:pr47181`
holds the sglang build with the PR applied for screening further checkpoints.

## Per-task and mixed-workload e2e throughput, run2 ep10 vs dense (rx devbox specforge-exp-branch, H200, 2026-11-01 PST)

MoE draft: `qwen3-8b-moe8x1536-8A1-dspark-run2/ep10` (the best checkpoint of the run2 accept-length sweep: 8 x 1536 top-1,
block 16; pooled request-macro AL 8.925 vs dense 9.101 on the five tasks at c=1). Stack: SGLang main c10e0e17 + PR #47181 applied
with `git apply` (venv from source: torch 2.14.1, flashinfer 0.7.2, triton 3.8.0), scripts in `mixed-and-tasks/`. Server per draft:
`--tp-size 1` on ONE H200 (draft on the same card), bf16 target, flashinfer, `--mem-fraction-static 0.8 --max-running-requests 256
--cuda-graph-max-bs-decode 256 --speculative-dspark-block-size 16 --context-length 16384 --disable-radix-cache`; the MoE server used the
fused-MoE Triton kernel's DEFAULT config (the tuned `E=8,N=1536` file was disabled; log line "Using default MoE kernel config").
Every server was started on a GPU verified empty (0 compute processes, 0 MiB; the scripts refuse otherwise); repeat groups ran one
server per GPU. Client: `specforge benchmark` with jiapingW's settings (thinking off, temperature 0, top_p 1, top_k 1,
max_new_tokens 2048), full prompt sets (gsm8k 1319, math500 500, humaneval 164, mbpp 257, mt-bench 160); the three small sets stop at
c=64 (above that they cannot fill the batch). tok/s = output tokens / wall time; per-task step ms = wall x min(c, requests) / verify
steps (exact at c=1); mixed step ms = mean per-request e2e_latency / spec_verify_ct. Mean +- sample std over 5 groups (per task) and
3 groups (mixed); diff = difference of the means.


### gsm8k
| c | dense tok/s (mean ± std) | MoE tok/s (mean ± std) | diff | dense step ms (mean ± std) | MoE step ms (mean ± std) | n |
|---|---|---|---|---|---|---|
| 1 | 1108 ± 2 | 1059 ± 3 | -4.4% | 8.5 ± 0.0 | 8.6 ± 0.0 | 5/5 |
| 8 | 6531 ± 31 | 6225 ± 30 | -4.7% | 11.5 ± 0.1 | 11.8 ± 0.0 | 5/5 |
| 16 | 9009 ± 31 | 8690 ± 58 | -3.5% | 16.7 ± 0.1 | 16.8 ± 0.1 | 5/5 |
| 32 | 11354 ± 26 | 11202 ± 91 | -1.3% | 26.5 ± 0.1 | 26.1 ± 0.2 | 5/5 |
| 64 | 12586 ± 55 | 12431 ± 68 | -1.2% | 48.0 ± 0.2 | 47.2 ± 0.3 | 5/5 |
| 128 | 13160 ± 113 | 13252 ± 150 | +0.7% | 91.2 ± 0.9 | 88.4 ± 0.8 | 5/5 |
| 256 | 13861 ± 72 | 13830 ± 125 | -0.2% | 173.6 ± 1.1 | 169.3 ± 1.4 | 5/5 |

### math500
| c | dense tok/s (mean ± std) | MoE tok/s (mean ± std) | diff | dense step ms (mean ± std) | MoE step ms (mean ± std) | n |
|---|---|---|---|---|---|---|
| 1 | 1071 ± 3 | 1038 ± 3 | -3.1% | 8.2 ± 0.0 | 8.3 ± 0.0 | 5/5 |
| 8 | 6549 ± 25 | 6410 ± 34 | -2.1% | 10.6 ± 0.0 | 10.9 ± 0.0 | 5/5 |
| 16 | 8873 ± 57 | 8747 ± 94 | -1.4% | 15.7 ± 0.0 | 15.9 ± 0.2 | 5/5 |
| 32 | 10649 ± 87 | 10633 ± 101 | -0.2% | 26.1 ± 0.2 | 26.0 ± 0.2 | 5/5 |
| 64 | 11778 ± 45 | 11977 ± 141 | +1.7% | 47.1 ± 0.3 | 46.2 ± 0.5 | 5/5 |
| 128 | 12305 ± 38 | 12552 ± 83 | +2.0% | 89.8 ± 0.4 | 87.9 ± 0.5 | 5/5 |
| 256 | 13047 ± 76 | 13289 ± 104 | +1.9% | 169.7 ± 0.7 | 166.2 ± 1.1 | 5/5 |

### humaneval
| c | dense tok/s (mean ± std) | MoE tok/s (mean ± std) | diff | dense step ms (mean ± std) | MoE step ms (mean ± std) | n |
|---|---|---|---|---|---|---|
| 1 | 890 ± 3 | 875 ± 2 | -1.6% | 8.3 ± 0.0 | 8.4 ± 0.0 | 5/5 |
| 8 | 5362 ± 38 | 5221 ± 20 | -2.6% | 11.0 ± 0.0 | 11.2 ± 0.0 | 5/5 |
| 16 | 7479 ± 34 | 7320 ± 71 | -2.1% | 16.1 ± 0.1 | 16.2 ± 0.1 | 5/5 |
| 32 | 9119 ± 71 | 9195 ± 49 | +0.8% | 25.7 ± 0.2 | 25.6 ± 0.1 | 5/5 |
| 64 | 9485 ± 80 | 9693 ± 92 | +2.2% | 49.4 ± 0.5 | 48.2 ± 0.5 | 5/5 |

### mbpp
| c | dense tok/s (mean ± std) | MoE tok/s (mean ± std) | diff | dense step ms (mean ± std) | MoE step ms (mean ± std) | n |
|---|---|---|---|---|---|---|
| 1 | 796 ± 2 | 784 ± 3 | -1.6% | 8.3 ± 0.0 | 8.4 ± 0.0 | 5/5 |
| 8 | 4810 ± 15 | 4701 ± 5 | -2.3% | 11.0 ± 0.0 | 11.1 ± 0.0 | 5/5 |
| 16 | 6611 ± 93 | 6488 ± 87 | -1.9% | 15.9 ± 0.2 | 16.0 ± 0.2 | 5/5 |
| 32 | 8162 ± 107 | 8192 ± 179 | +0.4% | 25.6 ± 0.4 | 25.4 ± 0.5 | 5/5 |
| 64 | 9228 ± 97 | 9247 ± 140 | +0.2% | 45.4 ± 0.2 | 45.0 ± 0.7 | 5/5 |

### mt-bench
| c | dense tok/s (mean ± std) | MoE tok/s (mean ± std) | diff | dense step ms (mean ± std) | MoE step ms (mean ± std) | n |
|---|---|---|---|---|---|---|
| 1 | 478 ± 1 | 484 ± 2 | +1.2% | 8.1 ± 0.0 | 8.3 ± 0.0 | 5/5 |
| 8 | 2927 ± 17 | 2975 ± 31 | +1.6% | 10.6 ± 0.0 | 10.8 ± 0.1 | 5/5 |
| 16 | 3849 ± 15 | 3923 ± 28 | +1.9% | 16.1 ± 0.1 | 16.2 ± 0.1 | 5/5 |
| 32 | 4709 ± 33 | 4915 ± 83 | +4.4% | 26.3 ± 0.1 | 25.9 ± 0.4 | 5/5 |
| 64 | 5164 ± 37 | 5437 ± 44 | +5.3% | 48.4 ± 0.3 | 46.3 ± 0.3 | 5/5 |

### mixed: the five datasets shuffled into one request stream (2,400 requests; the FINAL comparison)
| c | dense tok/s (mean ± std) | MoE tok/s (mean ± std) | diff | dense step ms (mean ± std) | MoE step ms (mean ± std) | n |
|---|---|---|---|---|---|---|
| 1 | 957 ± 1 | 931 ± 2 | -2.8% | 8.4 ± 0.0 | 8.6 ± 0.0 | 3/3 |
| 8 | 5809 ± 23 | 5660 ± 20 | -2.6% | 11.1 ± 0.0 | 11.3 ± 0.0 | 3/3 |
| 16 | 7994 ± 34 | 7997 ± 49 | +0.0% | 16.1 ± 0.1 | 15.9 ± 0.1 | 3/3 |
| 32 | 9590 ± 17 | 9826 ± 73 | +2.5% | 26.4 ± 0.1 | 25.6 ± 0.3 | 3/3 |
| 64 | 10574 ± 72 | 10912 ± 70 | +3.2% | 47.6 ± 0.2 | 45.8 ± 0.2 | 3/3 |
| 128 | 11273 ± 79 | 11733 ± 106 | +4.1% | 88.5 ± 0.5 | 84.6 ± 0.5 | 3/3 |
| 256 | 11790 ± 70 | 12313 ± 54 | +4.4% | 166.2 ± 0.7 | 158.0 ± 0.6 | 3/3 |

### bench_accept.py sweep (16 prompts, 256 tokens, radix cache on), one run of two repeats each, same servers otherwise
| c | dense tok/s | MoE tok/s | diff | dense step ms | MoE step ms | dense AL | MoE AL |
|---|---|---|---|---|---|---|---|
| 1 | 777 | 784 | +0.9% | 9.3 | 9.5 | 9.99 | 10.23 |
| 8 | 3207 | 3054 | -4.8% | 13.7 | 14.4 | 10.21 | 9.80 |
| 16 | 4320 | 4307 | -0.3% | 19.5 | 19.7 | 9.96 | 10.15 |
| 32 | 5786 | 5717 | -1.2% | 28.3 | 28.6 | 9.80 | 10.11 |
| 64 | 6926 | 7166 | +3.5% | 46.1 | 44.5 | 9.75 | 10.12 |
| 128 | 7567 | 7831 | +3.5% | 82.7 | 83.0 | 9.58 | 10.14 |
| 256 | 7409 | 7783 | +5.0% | 152.3 | 147.0 | 9.61 | 10.11 |

Findings
- Inside the mixed stream the per-task accept lengths equal the standalone runs to three decimals (dense c=1: gsm8k 9.971, math500
  9.600, humaneval 7.551, mbpp 7.059, mt-bench 5.239), i.e. mixing changes the batch composition only, not the prompts.
- Group-to-group spread is under 1% (max +-2% on mbpp at c=32), so differences of 2% and more in the tables are real.
- Mixed workload: the MoE draft is 2.6-2.8% slower at c = 1-8 (its 1.9% lower accept length plus ~0.2 ms of routing per step),
  equal at c=16, and 2.5-4.4% faster from c=32 up because its per-step time is 3-5% lower there (the draft's FFN FLOPs are 1/8 of
  dense and matter once the step is compute-bound); the same pattern holds per task, strongest on mt-bench where its AL is higher.
- At c=1 the MoE saves no memory traffic: a 16-token block routed top-1 touches ~7 of the 8 experts, so the FFN bytes read per step
  equal the dense FFN's.
