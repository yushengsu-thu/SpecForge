#!/bin/bash
# five-dataset e2e for dense 4ep + 16x1024 (GPU 2), official v1 + 256exp (GPU 3), no-draft baseline (GPU 0 once the
# 256exp throughput job releases it)
E=/scratch/dense5l-eval; log() { echo "[$(date -u +%H:%M:%S)] $*"; }
gpu_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)" -lt 2000 ]; }
run() { until gpu_free $3; do sleep 20; done; log "start $1 on gpu $3"; bash $E/bench_e2e_all.sh $1 $2 $3 $4 > $E/logs/e2e2_$1.log 2>&1; log "end $1: $(tail -1 $E/logs/e2e2_$1.log)"; }
( run dense-4ep $E/drafts/dense5l-4ep 2 30601; run moe16x-5ep $E/drafts/moe16x-5ep 2 30602 ) &
( run official-v1 $E/drafts/official-v1 3 30603; run moe256-4ep $E/drafts/moe256-4ep 3 30604 ) &
( until grep -q -E "TPUT_DONE|SERVER FAILED|NOT HEALTHY" $E/logs/tput_one_moe256-4ep.log 2>/dev/null; do sleep 60; done; sleep 30; run baseline none 0 30605 ) &
wait
log "E2E_ALL_DONE"
