## Acceptance length (temperature 0, jiapingW protocol)

tau = completion_tokens / verify rounds per request (bonus token included); macro = mean over requests, micro = total tokens / total rounds.


### D16/V16

| Benchmark | Requests | dense-1/8 macro tau | dense-1/8 micro tau | MoE macro tau | MoE micro tau | MoE vs dense (macro) |
|---|---|---|---|---|---|---|
| GSM8K | 1319 | 7.590 | 7.120 | 7.613 | 7.145 | +0.3% |
| MATH500 | 500 | 7.015 | 6.291 | 7.228 | 6.491 | +3.0% |
| HumanEval | 164 | 5.211 | 5.143 | 5.422 | 5.336 | +4.0% |
| MBPP | 257 | 5.093 | 4.754 | 5.401 | 4.995 | +6.0% |
| MT-Bench | 160 | 3.853 | 3.166 | 3.967 | 3.290 | +2.9% |
| **All requests, pooled** | 2400 | 6.791 | 5.851 | 6.903 | 6.002 | +1.7% |

dense-1/8: equal-weight macro 5.753, micro 5.295, length-limited 56, errors 0; MoE: equal-weight macro 5.926, micro 5.451, length-limited 56, errors 0


### D16/V8

| Benchmark | Requests | dense-1/8 macro tau | dense-1/8 micro tau | MoE macro tau | MoE micro tau | MoE vs dense (macro) |
|---|---|---|---|---|---|---|
| GSM8K | 1319 | 5.759 | 5.603 | 5.779 | 5.625 | +0.3% |
| MATH500 | 500 | 5.563 | 5.281 | 5.645 | 5.380 | +1.5% |
| HumanEval | 164 | 4.611 | 4.574 | 4.739 | 4.707 | +2.8% |
| MBPP | 257 | 4.473 | 4.285 | 4.649 | 4.433 | +3.9% |
| MT-Bench | 160 | 3.491 | 3.020 | 3.557 | 3.119 | +1.9% |
| **All requests, pooled** | 2400 | 5.351 | 4.945 | 5.411 | 5.032 | +1.1% |

dense-1/8: equal-weight macro 4.780, micro 4.553, length-limited 51, errors 0; MoE: equal-weight macro 4.874, micro 4.653, length-limited 51, errors 0


## End-to-end serving throughput (SGLang, CUDA graphs on, 512 new tokens, greedy)

Output tokens/s of `specforge benchmark`: c = concurrent requests (100 prompts at c=1, 200 at c=8 and c=32, a smaller dataset wraps around); MT-Bench sends the first turn only. Speed-up = tok/s over the no-draft baseline on the same task and concurrency.

Measured 2026-10-19T23:05:32Z to 2026-10-19T23:27:47Z on specforge-moe-8xh200 (gpu-10-220-71-54, NVIDIA H200 x8, driver 595.71.05, sglang 0.5.18 torch 2.13.0+cu130 triton 3.7.1), one server per dedicated GPU (GPUs 0-4), because the training box's GPUs are running epochs 13-18. Same drafts (md5-identical), same target revision (Qwen/Qwen3-8B b968826d), same flags and the same borrowed fused-MoE tile config as the training-box run. That earlier GSM8K / MT-Bench run is kept in e2e_training_node_2tasks/ and compared at the end.


### GSM8K

| Config | c=1 tok/s | c=1 speed-up | c=8 tok/s | c=8 speed-up | c=32 tok/s | c=32 speed-up | accept len (c=1) |
|---|---|---|---|---|---|---|---|
| baseline | 195.6 | 1.00x | 1363.9 | 1.00x | 4545.3 | 1.00x | - |
| dense18-v16 | 897.9 | 4.59x | 4961.3 | 3.64x | 9617.6 | 2.12x | 7.65 |
| dense18-v8 | 749.7 | 3.83x | 4560.8 | 3.34x | 11865.4 | 2.61x | 5.77 |
| moe-v16 | 873.7 | 4.47x | 4875.2 | 3.57x | 9536.0 | 2.10x | 7.59 |
| moe-v8 | 717.1 | 3.67x | 4512.5 | 3.31x | 11507.2 | 2.53x | 5.79 |

### MATH500

| Config | c=1 tok/s | c=1 speed-up | c=8 tok/s | c=8 speed-up | c=32 tok/s | c=32 speed-up | accept len (c=1) |
|---|---|---|---|---|---|---|---|
| baseline | 195.8 | 1.00x | 1397.7 | 1.00x | 4669.0 | 1.00x | - |
| dense18-v16 | 810.4 | 4.14x | 4860.4 | 3.48x | 9183.9 | 1.97x | 6.92 |
| dense18-v8 | 720.4 | 3.68x | 4646.1 | 3.32x | 11627.2 | 2.49x | 5.51 |
| moe-v16 | 806.1 | 4.12x | 4823.6 | 3.45x | 9243.4 | 1.98x | 7.06 |
| moe-v8 | 695.3 | 3.55x | 4618.9 | 3.30x | 11628.3 | 2.49x | 5.56 |

### HumanEval

| Config | c=1 tok/s | c=1 speed-up | c=8 tok/s | c=8 speed-up | c=32 tok/s | c=32 speed-up | accept len (c=1) |
|---|---|---|---|---|---|---|---|
| baseline | 195.6 | 1.00x | 1385.8 | 1.00x | 4568.7 | 1.00x | - |
| dense18-v16 | 670.5 | 3.43x | 3988.8 | 2.88x | 7497.8 | 1.64x | 5.27 |
| dense18-v8 | 628.8 | 3.21x | 4040.8 | 2.92x | 9952.2 | 2.18x | 4.65 |
| moe-v16 | 674.6 | 3.45x | 3988.0 | 2.88x | 7570.6 | 1.66x | 5.46 |
| moe-v8 | 613.4 | 3.14x | 4035.7 | 2.91x | 9958.6 | 2.18x | 4.75 |

### MBPP

| Config | c=1 tok/s | c=1 speed-up | c=8 tok/s | c=8 speed-up | c=32 tok/s | c=32 speed-up | accept len (c=1) |
|---|---|---|---|---|---|---|---|
| baseline | 196.1 | 1.00x | 1364.0 | 1.00x | 4470.8 | 1.00x | - |
| dense18-v16 | 615.7 | 3.14x | 3603.9 | 2.64x | 7101.7 | 1.59x | 5.04 |
| dense18-v8 | 587.2 | 2.99x | 3681.0 | 2.70x | 9755.0 | 2.18x | 4.46 |
| moe-v16 | 629.0 | 3.21x | 3679.8 | 2.70x | 7171.3 | 1.60x | 5.32 |
| moe-v8 | 581.8 | 2.97x | 3728.8 | 2.73x | 9570.1 | 2.14x | 4.66 |

### MT-Bench

| Config | c=1 tok/s | c=1 speed-up | c=8 tok/s | c=8 speed-up | c=32 tok/s | c=32 speed-up | accept len (c=1) |
|---|---|---|---|---|---|---|---|
| baseline | 195.9 | 1.00x | 1382.1 | 1.00x | 4509.7 | 1.00x | - |
| dense18-v16 | 377.3 | 1.93x | 2543.1 | 1.84x | 4803.7 | 1.07x | 3.58 |
| dense18-v8 | 387.0 | 1.98x | 2773.6 | 2.01x | 6920.1 | 1.53x | 3.26 |
| moe-v16 | 383.2 | 1.96x | 2560.9 | 1.85x | 4859.2 | 1.08x | 3.68 |
| moe-v8 | 378.8 | 1.93x | 2738.1 | 1.98x | 6876.7 | 1.52x | 3.30 |

### Mean over the 5 tasks (GSM8K, MATH500, HumanEval, MBPP, MT-Bench), equal weight

| Config | c=1 speed-up | c=8 speed-up | c=32 speed-up | accept len (c=1) |
|---|---|---|---|---|
| dense18-v16 | 3.44x | 2.90x | 1.68x | 5.69 |
| dense18-v8 | 3.14x | 2.86x | 2.20x | 4.73 |
| moe-v16 | 3.44x | 2.89x | 1.68x | 5.82 |
| moe-v8 | 3.05x | 2.85x | 2.17x | 4.81 |

### MoE / dense-1/8 throughput ratio (same width, task and concurrency)

| Task | V16 c=1 | V16 c=8 | V16 c=32 | V8 c=1 | V8 c=8 | V8 c=32 |
|---|---|---|---|---|---|---|
| GSM8K | 0.973 | 0.983 | 0.992 | 0.956 | 0.989 | 0.970 |
| MATH500 | 0.995 | 0.992 | 1.006 | 0.965 | 0.994 | 1.000 |
| HumanEval | 1.006 | 1.000 | 1.010 | 0.975 | 0.999 | 1.001 |
| MBPP | 1.022 | 1.021 | 1.010 | 0.991 | 1.013 | 0.981 |
| MT-Bench | 1.016 | 1.007 | 1.012 | 0.979 | 0.987 | 0.994 |
| **Mean** | 1.002 | 1.001 | 1.006 | 0.973 | 0.997 | 0.989 |

### Cross-check against the training-box run (GSM8K and MT-Bench, specforge-qwen-h200)

tok/s of this run, with the ratio this run / earlier run in brackets.

| Config | Task | c=1 | c=8 | c=32 |
|---|---|---|---|---|
| baseline | GSM8K | 195.6 (1.012) | 1363.9 (1.008) | 4545.3 (1.016) |
| baseline | MT-Bench | 195.9 (1.011) | 1382.1 (1.010) | 4509.7 (1.010) |
| dense18-v16 | GSM8K | 897.9 (1.013) | 4961.3 (1.018) | 9617.6 (1.029) |
| dense18-v16 | MT-Bench | 377.3 (1.014) | 2543.1 (1.017) | 4803.7 (1.032) |
| dense18-v8 | GSM8K | 749.7 (1.034) | 4560.8 (1.016) | 11865.4 (1.038) |
| dense18-v8 | MT-Bench | 387.0 (1.035) | 2773.6 (1.007) | 6920.1 (1.033) |
| moe-v16 | GSM8K | 873.7 (1.009) | 4875.2 (1.005) | 9536.0 (1.006) |
| moe-v16 | MT-Bench | 383.2 (1.009) | 2560.9 (1.007) | 4859.2 (1.004) |
| moe-v8 | GSM8K | 717.1 (1.013) | 4512.5 (1.019) | 11507.2 (1.013) |
| moe-v8 | MT-Bench | 378.8 (1.012) | 2738.1 (1.011) | 6876.7 (1.014) |

Ratio range 1.004 to 1.038, mean 1.016 over 30 cells.

