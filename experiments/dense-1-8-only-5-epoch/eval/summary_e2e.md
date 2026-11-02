Same-GPU comparison: dense 2ep (2-epoch run) and dense 5-epoch run ep5 served one after the other on the same GPU (three GPUs per width, one with the order reversed). Temperature 0, 512 new tokens, prompts per task = max(200, 4 x concurrency), CUDA graphs on, servers allow 128 running requests. Unless a task is named, values are means of the five tasks.

## Verify width 8

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

## Verify width 16

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

result files: 260; missing: none
