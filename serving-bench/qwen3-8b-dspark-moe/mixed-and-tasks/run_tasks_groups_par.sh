#!/bin/bash
# Groups 2 and 3 of the per-task experiment on the idle GPUs (user: use GPUs 1-7): four servers at once, each alone on
# its own GPU (dense-g2 GPU 1, moe-g2 GPU 2, dense-g3 GPU 3, moe-g3 GPU 4), staggered starts; group 1 keeps GPU 0.
cd ${BENCH_DIR:-/scratch/moe8b16/serving-bench}
log(){ echo "[$(date -u +%T)] $*"; }
DENSE=${DENSE:-/scratch/moe8b16/eval/models/jw-dense}; MOE=${MOE:-/scratch/moe8b16/serving-bench/models/moe-run2-ep10}
log "start dense-g2 (GPU 1), moe-g2 (GPU 2), dense-g3 (GPU 3), moe-g3 (GPU 4)"
GPUS=1 PORT=31542 bash task_sweep.sh dense-g2 $DENSE > logs/run_tasks_dense-g2.log 2>&1 & sleep 40
GPUS=2 PORT=31552 bash task_sweep.sh moe-g2   $MOE   > logs/run_tasks_moe-g2.log   2>&1 & sleep 40
GPUS=3 PORT=31543 bash task_sweep.sh dense-g3 $DENSE > logs/run_tasks_dense-g3.log 2>&1 & sleep 40
GPUS=4 PORT=31553 bash task_sweep.sh moe-g3   $MOE   > logs/run_tasks_moe-g3.log   2>&1 &
wait
for n in dense-g2 moe-g2 dense-g3 moe-g3; do log "$n: $(tail -1 logs/run_tasks_$n.log)"; done
log "GROUPS_DONE gpu apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
