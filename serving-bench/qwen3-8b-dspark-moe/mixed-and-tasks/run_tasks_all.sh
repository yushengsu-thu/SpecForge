#!/bin/bash
# dense, then MoE (run2 ep10, Triton default fused-MoE config), one server at a time.
cd ${BENCH_DIR:-/scratch/moe8b16/serving-bench}
log(){ echo "[$(date -u +%T)] $*"; }
log "start tasks dense"; GPUS=0 PORT=31530 bash task_sweep.sh dense ${DENSE:-/scratch/moe8b16/eval/models/jw-dense} > logs/run_tasks_dense.log 2>&1; log "end: $(tail -1 logs/run_tasks_dense.log)"; sleep 10
log "start tasks moe";   GPUS=0 PORT=31531 bash task_sweep.sh moe ${MOE:-/scratch/moe8b16/serving-bench/models/moe-run2-ep10} > logs/run_tasks_moe.log 2>&1; log "end: $(tail -1 logs/run_tasks_moe.log)"
log "TASKS_ALL_DONE gpu apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
