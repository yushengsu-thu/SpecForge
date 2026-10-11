# fsdp1 vs fsdp2 on Qwen/Qwen3-8B drafts, 1 epoch each (13,068 regenerated ShareGPT conversations)

## Final acceptance length (SGLang 0.5.21, thinking off, concurrency 8; GSM8K 200 prompts, MT-Bench 80 prompts)

| mode | draft | fsdp1 gsm8k | fsdp2 gsm8k | diff | fsdp1 mt-bench | fsdp2 mt-bench | diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| offline | dspark | 2.081 | 2.049 | -0.031 | 1.725 | 1.713 | -0.012 |
| offline | eagle3 | 2.115 | 2.112 | -0.003 | 1.810 | 1.818 | +0.008 |
| offline | dflash | 1.792 | 1.810 | +0.018 | 1.566 | 1.591 | +0.025 |
| offline | dflash2 | 2.091 | 2.058 | -0.033 | 1.736 | 1.729 | -0.007 |
| offline | domino | 2.056 | 2.091 | +0.035 | 1.706 | 1.724 | +0.017 |
| online | dspark | 2.100 | 2.088 | -0.012 | 1.724 | 1.718 | -0.005 |
| online | eagle3 | 2.185 | 2.170 | -0.015 | 1.846 | 1.850 | +0.004 |
| online | dflash | 1.853 | 1.813 | -0.040 | 1.602 | 1.575 | -0.027 |
| online | dflash2 | 2.037 | 2.015 | -0.022 | 1.688 | 1.698 | +0.010 |
| online | domino | 2.149 | 2.130 | -0.019 | 1.742 | 1.744 | +0.003 |
| online | dpace | 2.078 | 2.073 | -0.005 | 1.750 | 1.759 | +0.009 |

## Training curves, last 50 logged points of the epoch (loss and train-time acceptance), and wall time

| mode | draft | steps | fsdp1 loss | fsdp2 loss | fsdp1 acc | fsdp2 acc | fsdp1 time | fsdp2 time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offline | dspark | 3230 | 2.160 | 2.177 | 0.3424 | 0.3363 | 9m44s | 9m39s |
| offline | eagle3 | 3260 | 5.666 | 5.669 | 0.4097 | 0.4098 | 14m18s | 11m49s |
| offline | dflash | 800 | 5.609 | 5.588 | 0.1147 | 0.1147 | 10m55s | 10m50s |
| offline | dflash2 | 800 | 6.146 | 6.199 | 0.1268 | 0.1249 | 10m22s | 10m25s |
| offline | domino | 1610 | 4.304 | 4.265 | 0.3065 | 0.3094 | 6m59s | 6m34s |
| online | dspark | 4320 | 2.135 | 2.136 | 0.3471 | 0.3465 | 14m28s | 14m20s |
| online | eagle3 | 4340 | 10.697 | 10.723 | 0.4233 | 0.4225 | 17m36s | 15m23s |
| online | dflash | 1080 | 5.299 | 5.360 | 0.1267 | 0.1240 | 16m05s | 15m09s |
| online | dflash2 | 1080 | 6.103 | 6.130 | 0.1298 | 0.1277 | 15m21s | 14m25s |
| online | domino | 2160 | 4.083 | 4.073 | 0.3200 | 0.3202 | 15m02s | 13m53s |
| online | peagle | 4340 | 3.595 | 3.591 | 0.3893 | 0.3908 | 21m19s | 19m44s |
| online | dpace | 1080 | 11.106 | 11.084 | 0.1329 | 0.1316 | 14m57s | 15m04s |
