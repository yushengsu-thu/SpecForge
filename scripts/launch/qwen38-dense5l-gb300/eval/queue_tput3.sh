#!/bin/bash
# reordered tail of the throughput queue: 16x1024 5ep on GPU 3 right after the 16x speed gate, no-draft baseline on GPU 1
# after the dense run (the MoE family numbers are wanted first)
E=/scratch/dense5l-eval; log() { echo "[$(date -u +%H:%M:%S)] $*"; }
gpu_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)" -lt 2000 ]; }
launch() { until gpu_free $3; do sleep 20; done; setsid nohup bash $E/tput_one.sh $1 $2 $3 $4 > $E/logs/tput_one_$1.log 2>&1 < /dev/null & log "launched $1 on gpu $3 port $4"; }
( until grep -q SPEED_GATES_DONE $E/logs/speed_gates_fix.log 2>/dev/null; do sleep 30; done; sleep 20; launch moe16x-5ep $E/drafts/moe16x-5ep 3 30504 ) &
( until grep -q -E "TPUT_DONE|SERVER FAILED|NOT HEALTHY" $E/logs/tput_one_dense5l-4ep.log 2>/dev/null; do sleep 60; done; sleep 30; launch none none 1 30503 ) &
wait
for l in dense5l-4ep moe256-4ep official-v1 moe16x-5ep none; do until grep -q -E "TPUT_DONE|SERVER FAILED|NOT HEALTHY" $E/logs/tput_one_$l.log 2>/dev/null; do sleep 60; done; log "finished $l"; done
log "TPUT_ALL_DONE"
