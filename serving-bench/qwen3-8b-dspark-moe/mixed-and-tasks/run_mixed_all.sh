#!/bin/bash
# Mixed-workload experiment, 3 groups x {dense, MoE} = 6 servers in parallel on 6 EMPTY GPUs, after the per-task groups end.
cd ${BENCH_DIR:-/scratch/moe8b16/serving-bench}
log(){ echo "[$(date -u +%T)] $*"; }
until grep -q GROUPS_DONE logs/run_tasks_groups.log && grep -q GROUPS45_DONE logs/run_tasks_groups45.log; do sleep 30; done; sleep 20
DENSE=${DENSE:-/scratch/moe8b16/eval/models/jw-dense}; MOE=${MOE:-/scratch/moe8b16/serving-bench/models/moe-run2-ep10}
log "start mixed: dense-m1 GPU0, moe-m1 GPU1, dense-m2 GPU2, moe-m2 GPU3, dense-m3 GPU4, moe-m3 GPU5"
GPUS=0 PORT=31560 bash mixed_sweep.sh dense-m1 $DENSE > logs/run_mixed_dense-m1.log 2>&1 & sleep 40
GPUS=1 PORT=31561 bash mixed_sweep.sh moe-m1   $MOE   > logs/run_mixed_moe-m1.log   2>&1 & sleep 40
GPUS=2 PORT=31562 bash mixed_sweep.sh dense-m2 $DENSE > logs/run_mixed_dense-m2.log 2>&1 & sleep 40
GPUS=3 PORT=31563 bash mixed_sweep.sh moe-m2   $MOE   > logs/run_mixed_moe-m2.log   2>&1 & sleep 40
GPUS=4 PORT=31564 bash mixed_sweep.sh dense-m3 $DENSE > logs/run_mixed_dense-m3.log 2>&1 & sleep 40
GPUS=5 PORT=31565 bash mixed_sweep.sh moe-m3   $MOE   > logs/run_mixed_moe-m3.log   2>&1 &
wait
for n in dense-m1 moe-m1 dense-m2 moe-m2 dense-m3 moe-m3; do log "$n: $(tail -1 logs/run_mixed_$n.log)"; done
log "MIXED_ALL_DONE gpu apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
