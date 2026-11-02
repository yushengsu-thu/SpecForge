#!/bin/bash
# Groups 4 and 5 on the remaining idle GPUs: dense-g4 GPU 5, moe-g4 GPU 6, dense-g5 GPU 7 now; moe-g5 on GPU 0 once group 1 ends.
cd ${BENCH_DIR:-/scratch/moe8b16/serving-bench}
log(){ echo "[$(date -u +%T)] $*"; }
DENSE=${DENSE:-/scratch/moe8b16/eval/models/jw-dense}; MOE=${MOE:-/scratch/moe8b16/serving-bench/models/moe-run2-ep10}
log "start dense-g4 (GPU 5), moe-g4 (GPU 6), dense-g5 (GPU 7); moe-g5 waits for GPU 0"
GPUS=5 PORT=31544 bash task_sweep.sh dense-g4 $DENSE > logs/run_tasks_dense-g4.log 2>&1 & sleep 40
GPUS=6 PORT=31554 bash task_sweep.sh moe-g4   $MOE   > logs/run_tasks_moe-g4.log   2>&1 & sleep 40
GPUS=7 PORT=31545 bash task_sweep.sh dense-g5 $DENSE > logs/run_tasks_dense-g5.log 2>&1 &
( until grep -q TASKS_ALL_DONE logs/run_tasks_all.log; do sleep 20; done; sleep 15; log "start moe-g5 (GPU 0)"; GPUS=0 PORT=31555 bash task_sweep.sh moe-g5 $MOE > logs/run_tasks_moe-g5.log 2>&1 ) &
wait
for n in dense-g4 moe-g4 dense-g5 moe-g5; do log "$n: $(tail -1 logs/run_tasks_$n.log)"; done
log "GROUPS45_DONE gpu apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
