
## Accept length, temperature 0, verify width 8 (request average)

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

## Accept length, temperature 0, verify width 16 (request average)

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
