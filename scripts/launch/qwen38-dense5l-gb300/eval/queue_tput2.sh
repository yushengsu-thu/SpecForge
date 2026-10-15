#!/bin/bash
# fifth throughput job: the 16x1024 MoE (5ep) on GPU 1 once the dense 4ep run has released it
E=/scratch/dense5l-eval; log() { echo "[$(date -u +%H:%M:%S)] $*"; }
gpu_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)" -lt 2000 ]; }
until grep -q -E "TPUT_DONE|SERVER FAILED|NOT HEALTHY" $E/logs/tput_one_dense5l-4ep.log 2>/dev/null; do sleep 60; done
sleep 30; until gpu_free 1; do sleep 20; done
setsid nohup bash $E/tput_one.sh moe16x-5ep $E/drafts/moe16x-5ep 1 30504 > $E/logs/tput_one_moe16x-5ep.log 2>&1 < /dev/null & log "launched moe16x-5ep on gpu 1 port 30504"
until grep -q -E "TPUT_DONE|SERVER FAILED|NOT HEALTHY" $E/logs/tput_one_moe16x-5ep.log 2>/dev/null; do sleep 60; done
log "TPUT2_DONE"
