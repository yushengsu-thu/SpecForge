#!/bin/bash
# Full V8 protocol, MoE side only, with the TMA-aware tuned config (SGLANG_MOE_CONFIG_DIR=moe-config-tma2), 5 groups split into
# units over 8 GPUs; plus a dense gsm8k cross-check (dense-x1) to measure the box offset vs exp-branch. Every unit = one server +
# one client on one GPU; the sweep scripts refuse a non-empty GPU.
B=/scratch/ckpt-sweep/bench; D=/scratch/ckpt-sweep/models/ours-58000; DENSE=/scratch/moe8b16/eval/models/jw-dense; cd $B; mkdir -p logs
log(){ echo "[$(date -u +%T)] $*"; }
export SGLANG_MOE_CONFIG_DIR=${SGLANG_MOE_CONFIG_DIR:-/scratch/ckpt-sweep/moe-config-tma2}
BIG="gsm8k:1319 math500:500"; SMALL="humaneval:164 mbpp:257 mt-bench:160"; REST="8 16 32 64 128 256"
tasks(){ log "GPU$1 start $2"; BENCH_DIR=$B GPUS=$1 PORT=$((31600+$1)) TASKS="$3" bash task_sweep_sel.sh $2 $4 > logs/run_tasks_$2.log 2>&1; log "GPU$1 done: $(tail -1 logs/run_tasks_$2.log)"; sleep 20; }
mixed(){ log "GPU$1 start $2"; BENCH_DIR=$B GPUS=$1 PORT=$((31640+$1)) CONCS="$3" bash mixed_sweep.sh $2 $4 > logs/run_mixed_$2.log 2>&1; log "GPU$1 done: $(tail -1 logs/run_mixed_$2.log)"; sleep 20; }
P=${PREFIX:-moe-tma2}
( tasks 0 $P-g1-big "$BIG" $D; mixed 0 $P-g1-mrest "$REST" $D; tasks 0 dense-x1 "gsm8k:1319" $DENSE ) & sleep 40
for i in 2 3 4 5; do g=$((i-1)); ( tasks $g $P-g$i-big "$BIG" $D; mixed $g $P-g$i-mrest "$REST" $D ) & sleep 40; done
( mixed 5 $P-g1-m1 "1" $D; mixed 5 $P-g2-m1 "1" $D; tasks 5 $P-g3-small "$SMALL" $D ) & sleep 40
( mixed 6 $P-g3-m1 "1" $D; mixed 6 $P-g4-m1 "1" $D; tasks 6 $P-g4-small "$SMALL" $D ) & sleep 40
( mixed 7 $P-g5-m1 "1" $D; tasks 7 $P-g1-small "$SMALL" $D; tasks 7 $P-g2-small "$SMALL" $D; tasks 7 $P-g5-small "$SMALL" $D ) &
wait; log "TMA2_E2E_DONE"
