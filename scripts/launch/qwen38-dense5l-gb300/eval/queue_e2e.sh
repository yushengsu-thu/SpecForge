#!/bin/bash
# after the accept-length evals of 2ep/3ep/4ep free GPUs 1-3: no-draft baseline on GPU 1, dense 4ep on GPU 2, official v1 on GPU 3
E=/scratch/dense5l-eval
for k in 2 3 4; do until grep -q "EVAL_ONE_DONE" $E/logs/eval_one_$k.log 2>/dev/null; do sleep 60; done; done
sleep 30
for g in 1 2 3; do until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 2000 ]; do sleep 20; done; done
setsid nohup bash $E/bench_e2e_one.sh baseline none 1 30211 > $E/logs/e2e_baseline.log 2>&1 < /dev/null &
sleep 30; setsid nohup bash $E/bench_e2e_one.sh dense-4ep $E/drafts/dense5l-4ep 2 30212 > $E/logs/e2e_dense-4ep.log 2>&1 < /dev/null &
sleep 30; setsid nohup bash $E/bench_e2e_one.sh official-v1 $E/drafts/official-v1 3 30213 > $E/logs/e2e_official-v1.log 2>&1 < /dev/null &
echo "E2E chains launched $(date -u +%H:%M:%S)"
