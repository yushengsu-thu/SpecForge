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
