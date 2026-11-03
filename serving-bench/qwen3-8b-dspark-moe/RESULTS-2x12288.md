# Qwen3-8B DSpark: MoE draft 2 x 12288 top-1 (step 58000) vs jiapingW dense draft, block size 7 (V8)

Generated with `make_report.py` from the serving-bench result dirs (5 repeat groups per draft, mean ± sample std, diff = diff of the means).

## Setup

- Drafts: dense = `jiapingW/Qwen3-8b-dspark-SpecForge` (step 39000, FFN 12288); MoE = `yushengsu/qwen3-8b-moe2x12288-2A1-dspark` `step-58000`
  (2 experts x 12288, top-1, selection bias; `Qwen3MoeDSparkModel`), the best of six candidate checkpoints by pooled accept length at V8
  (55000 / 58000 / 59000 / 61000 / 62000 / 64586 all within 0.1%). Target `Qwen/Qwen3-8B` bf16.
- Serving: SGLang main `c10e0e17` + sgl-project/sglang#47181, TP=1, one server + one client per empty H200, `--speculative-dspark-block-size 7`
  (7 drafted tokens, verify 8), `--cuda-graph-max-bs-decode 256`, `--context-length 16384 --disable-radix-cache`, mem-fraction 0.8.
- MoE kernel: Triton fused-MoE with the tuned config in `configs/` (TMA-aware, separate up / down files, see `tune/README.md`), served via
  `SGLANG_MOE_CONFIG_DIR`. Accept length does not depend on the kernel config.
- Clients: `specforge benchmark` per task (gsm8k 1319, math500 500, humaneval 164, mbpp 257, mt-bench 160; temperature 0, max_new_tokens 2048);
  `mixed_bench.py` for the 2,400-request mixed stream (seed 42). Per-task step ms = latency x min(c, requests) / verify steps; mixed step ms = mean
  per-request e2e latency / verify steps.
- Runs: dense 5 groups and the accept-length eval on specforge-exp-branch (2026-11-02 01:17-03:08 PST); MoE 5 groups with the tuned config on
  specforge-qwen-h200 (2026-11-02 18:54-20:01 PST). A dense gsm8k sweep on qwen-h200 differs from the exp-branch dense groups by
  -0.6 / -0.4 / +0.6 / +1.4 / +1.2 / +1.0 / +1.0 % at c = 1 ... 256, so the MoE-vs-dense gaps at c >= 32 are about one point optimistic.
- One dense repeat (humaneval c=64, an 11.6 s run instead of 5.9 s) is excluded by the > 25%-off-median rule; that cell has 4 groups.

## 1. Accept length (c=1)

| | Avg (all requests) | Avg (per task) | gsm8k | math500 | humaneval | mbpp | mt-bench |
|---|---|---|---|---|---|---|---|
| dense | 6.114 | 5.609 | 6.451 | 6.354 | 5.634 | 5.423 | 4.184 |
| MoE | 6.151 | 5.688 | 6.445 | 6.417 | 5.771 | 5.542 | 4.262 |
| diff | +0.037 | +0.078 | -0.006 | +0.063 | +0.138 | +0.119 | +0.078 |
| diff % | +0.6% | +1.4% | -0.1% | +1.0% | +2.4% | +2.2% | +1.9% |

## 2. e2e throughput per task (mean ± std over the groups)

### gsm8k

| c | dense tok/s | MoE tok/s | diff % | dense step ms | MoE step ms |
|---|---|---|---|---|---|
| 1 | 792 ± 8 | 767 ± 9 | -3.2% | 8.0 ± 0.1 | 8.3 ± 0.1 |
| 8 | 5229 ± 10 | 4969 ± 14 | -5.0% | 9.7 ± 0.0 | 10.2 ± 0.0 |
| 16 | 8709 ± 29 | 8248 ± 25 | -5.3% | 11.6 ± 0.0 | 12.3 ± 0.0 |
| 32 | 12253 ± 173 | 11842 ± 95 | -3.4% | 16.6 ± 0.2 | 17.1 ± 0.1 |
| 64 | 14615 ± 221 | 14364 ± 150 | -1.7% | 27.7 ± 0.4 | 28.1 ± 0.3 |
| 128 | 16464 ± 291 | 16254 ± 239 | -1.3% | 49.2 ± 0.9 | 49.7 ± 0.7 |
| 256 | 17555 ± 310 | 17209 ± 282 | -2.0% | 92.3 ± 1.7 | 93.9 ± 1.6 |

### math500

| c | dense tok/s | MoE tok/s | diff % | dense step ms | MoE step ms |
|---|---|---|---|---|---|
| 1 | 788 ± 8 | 774 ± 9 | -1.8% | 7.8 ± 0.1 | 8.0 ± 0.1 |
| 8 | 5390 ± 9 | 5181 ± 23 | -3.9% | 9.2 ± 0.0 | 9.6 ± 0.0 |
| 16 | 8866 ± 76 | 8425 ± 52 | -5.0% | 11.1 ± 0.1 | 11.8 ± 0.1 |
| 32 | 11981 ± 88 | 11501 ± 113 | -4.0% | 16.4 ± 0.1 | 17.3 ± 0.2 |
| 64 | 14455 ± 127 | 14122 ± 158 | -2.3% | 27.2 ± 0.2 | 28.1 ± 0.3 |
| 128 | 15742 ± 171 | 15492 ± 160 | -1.6% | 50.0 ± 0.6 | 51.3 ± 0.5 |
| 256 | 17052 ± 170 | 16853 ± 294 | -1.2% | 92.3 ± 0.9 | 94.4 ± 1.6 |

### humaneval

| c | dense tok/s | MoE tok/s | diff % | dense step ms | MoE step ms |
|---|---|---|---|---|---|
| 1 | 710 ± 9 | 703 ± 1 | -0.9% | 7.9 ± 0.1 | 8.1 ± 0.0 |
| 8 | 4748 ± 33 | 4631 ± 23 | -2.5% | 9.4 ± 0.0 | 9.9 ± 0.1 |
| 16 | 7806 ± 143 | 7584 ± 46 | -2.8% | 11.5 ± 0.2 | 12.1 ± 0.1 |
| 32 | 10507 ± 77 | 10897 ± 184 | +3.7% | 17.0 ± 0.1 | 17.0 ± 0.3 |
| 64 | 12757 ± 203 | 12406 ± 182 | -2.8% | 28.0 ± 0.4 | 29.4 ± 0.5 |

### mbpp

| c | dense tok/s | MoE tok/s | diff % | dense step ms | MoE step ms |
|---|---|---|---|---|---|
| 1 | 661 ± 8 | 652 ± 2 | -1.5% | 7.9 ± 0.1 | 8.2 ± 0.0 |
| 8 | 4402 ± 15 | 4299 ± 17 | -2.3% | 9.5 ± 0.0 | 10.0 ± 0.0 |
| 16 | 7216 ± 99 | 6993 ± 59 | -3.1% | 11.6 ± 0.2 | 12.2 ± 0.1 |
| 32 | 10293 ± 222 | 10019 ± 80 | -2.7% | 16.3 ± 0.3 | 17.1 ± 0.1 |
| 64 | 12710 ± 310 | 12588 ± 112 | -1.0% | 26.4 ± 0.7 | 27.2 ± 0.2 |

### mt-bench

| c | dense tok/s | MoE tok/s | diff % | dense step ms | MoE step ms |
|---|---|---|---|---|---|
| 1 | 458 ± 5 | 462 ± 2 | +1.0% | 7.8 ± 0.1 | 8.0 ± 0.0 |
| 8 | 3084 ± 33 | 3059 ± 50 | -0.8% | 9.3 ± 0.1 | 9.8 ± 0.1 |
| 16 | 4764 ± 59 | 4723 ± 75 | -0.9% | 12.0 ± 0.1 | 12.7 ± 0.1 |
| 32 | 6740 ± 102 | 6469 ± 177 | -4.0% | 17.1 ± 0.1 | 18.4 ± 0.5 |
| 64 | 8003 ± 76 | 8011 ± 239 | +0.1% | 28.8 ± 0.2 | 29.8 ± 0.9 |

## 3. Mixed workload (five datasets in one 2,400-request stream)

| c | dense tok/s | MoE tok/s | diff % | dense step ms | MoE step ms |
|---|---|---|---|---|---|
| 1 | 733 ± 7 | 717 ± 1 | -2.2% | 8.0 ± 0.1 | 8.3 ± 0.0 |
| 8 | 4976 ± 10 | 4782 ± 20 | -3.9% | 9.4 ± 0.0 | 9.9 ± 0.0 |
| 16 | 8392 ± 35 | 8010 ± 28 | -4.6% | 11.1 ± 0.0 | 11.8 ± 0.0 |
| 32 | 11371 ± 65 | 10811 ± 108 | -4.9% | 16.2 ± 0.1 | 17.2 ± 0.1 |
| 64 | 13645 ± 107 | 13441 ± 110 | -1.5% | 26.7 ± 0.2 | 27.4 ± 0.2 |
| 128 | 15376 ± 99 | 15169 ± 158 | -1.3% | 47.0 ± 0.3 | 48.0 ± 0.5 |
| 256 | 16588 ± 170 | 16427 ± 212 | -1.0% | 85.6 ± 0.9 | 87.2 ± 1.2 |

## 4. Chart

`results-2x12288-mixed-throughput.png`

Excluded repeats (> 25% off the cell median): dense humaneval c=64: dense-g1 6483 tok/s vs median 12673 excluded
