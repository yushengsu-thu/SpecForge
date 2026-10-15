#!/bin/bash
# c=1/8/32 throughput on the five datasets for dense 4ep, 256exp 4ep, official v1 and the no-draft baseline, one GPU each,
# each started as soon as its GPU is released by the family pipeline / speed-gate re-runs
E=/scratch/dense5l-eval; log() { echo "[$(date -u +%H:%M:%S)] $*"; }
gpu_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)" -lt 2000 ]; }
launch() { until gpu_free $3; do sleep 20; done; setsid nohup bash $E/tput_one.sh $1 $2 $3 $4 > $E/logs/tput_one_$1.log 2>&1 < /dev/null & log "launched $1 on gpu $3 port $4"; }
( until grep -q POOL_DONE $E/logs/pool_16x_b.log 2>/dev/null; do sleep 60; done; sleep 30; launch dense5l-4ep $E/drafts/dense5l-4ep 1 30501 ) &
( until grep -q SPEED_256_DONE $E/logs/speed_gates_fix.log 2>/dev/null; do sleep 60; done; sleep 30; launch moe256-4ep $E/drafts/moe256-4ep 0 30500 ) &
( until grep -q FAMILIES_PIPELINE_DONE $E/logs/queue_families.log 2>/dev/null; do sleep 60; done; sleep 30; launch official-v1 $E/drafts/official-v1 2 30502 ) &
( until grep -q SPEED_GATES_DONE $E/logs/speed_gates_fix.log 2>/dev/null; do sleep 60; done; sleep 30; launch none none 3 30503 ) &
wait
for l in dense5l-4ep moe256-4ep official-v1 none; do until grep -q -E "TPUT_DONE|SERVER FAILED|NOT HEALTHY" $E/logs/tput_one_$l.log 2>/dev/null; do sleep 60; done; log "finished $l"; done
log "TPUT_ALL_DONE"
