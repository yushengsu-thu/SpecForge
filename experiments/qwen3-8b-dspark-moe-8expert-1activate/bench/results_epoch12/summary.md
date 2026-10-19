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


### gsm8k

| Config | c=1 tok/s | c=1 speed-up | c=8 tok/s | c=8 speed-up | c=32 tok/s | c=32 speed-up | accept len (c=1) |
|---|---|---|---|---|---|---|---|
| baseline | 193.4 | 1.00x | 1352.8 | 1.00x | 4472.3 | 1.00x | - |
| dense18-v16 | 886.4 | 4.58x | 4874.2 | 3.60x | 9342.0 | 2.09x | 7.65 |
| dense18-v8 | 725.1 | 3.75x | 4488.2 | 3.32x | 11430.8 | 2.56x | 5.77 |
| moe-v16 | 865.8 | 4.48x | 4851.0 | 3.59x | 9480.1 | 2.12x | 7.59 |
| moe-v8 | 707.7 | 3.66x | 4428.9 | 3.27x | 11359.3 | 2.54x | 5.79 |

### mt-bench

| Config | c=1 tok/s | c=1 speed-up | c=8 tok/s | c=8 speed-up | c=32 tok/s | c=32 speed-up | accept len (c=1) |
|---|---|---|---|---|---|---|---|
| baseline | 193.7 | 1.00x | 1368.1 | 1.00x | 4466.9 | 1.00x | - |
| dense18-v16 | 372.0 | 1.92x | 2500.4 | 1.83x | 4656.6 | 1.04x | 3.58 |
| dense18-v8 | 374.0 | 1.93x | 2753.5 | 2.01x | 6701.6 | 1.50x | 3.26 |
| moe-v16 | 379.9 | 1.96x | 2542.3 | 1.86x | 4839.5 | 1.08x | 3.68 |
| moe-v8 | 374.3 | 1.93x | 2707.4 | 1.98x | 6783.5 | 1.52x | 3.30 |
