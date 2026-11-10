#!/bin/bash
# End-to-end fused-MoE config tuning for a DSpark MoE draft (TMA-aware, up/down separate), then install + verify (+ optional e2e).
#
#   bash tune_moe_draft_e2e.sh                       # tune the default buckets, assemble, verify on one GPU
#   E2E=1 bash tune_moe_draft_e2e.sh                 # ... and run the full 5-group per-task + mixed protocol afterwards
#   BUCKETS="7 56 224 1792" FULL_SPACE=1 bash ...    # custom buckets, full 1920-config search instead of the pruned 648
#
# Inputs (env, with defaults for the Qwen3-8B 2x12288 draft):
#   DRAFT        draft model dir (HF layout, architectures Qwen3MoeDSparkModel)      default /scratch/ckpt-sweep/models/ours-58000
#   TARGET       target model dir                                                   default /scratch/moe2ep/models/Qwen3-8B
#   SHAPE_CFG    stand-in Qwen3MoeForCausalLM config.json dir giving E / topk / moe_intermediate_size / hidden_size / dtype
#   BUCKETS      token counts M to tune = draft tokens per request (7 at block-size 7) x concurrency       default "7 56 112 224 448 896 1792 3584"
#   GPUS         GPUs to use, one tune per GPU at a time                                                   default "0 1 2 3 4 5 6 7"
#   OUT          output config dir -> $OUT/configs/triton_<ver>/E=..json + .._down.json (= SGLANG_MOE_CONFIG_DIR) default /scratch/ckpt-sweep/moe-config-auto
#   BLOCK        --speculative-dspark-block-size                                                            default 7
#   VENV         SGLang venv (c10e0e17 + PR 47181 or newer: TMA fused_moe + _down configs)                 default /scratch/moe8b16/serving-bench/venv
#   BASE         optional earlier config file whose buckets are kept where not re-tuned
#   BENCH_DIR    dir with task_sweep_sel.sh / mixed_sweep.sh / mixed_bench.py for E2E=1                     default /scratch/ckpt-sweep/bench
# Every GPU is checked to be empty (0 compute apps, < 1 GB) before anything is started on it.
set -u
DRAFT=${DRAFT:-/scratch/ckpt-sweep/models/ours-58000}; TARGET=${TARGET:-/scratch/moe2ep/models/Qwen3-8B}
SHAPE_CFG=${SHAPE_CFG:-/scratch/ckpt-sweep/tune/model-cfg}; BUCKETS=${BUCKETS:-"7 56 112 224 448 896 1792 3584"}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}; OUT=${OUT:-/scratch/ckpt-sweep/moe-config-auto}; BLOCK=${BLOCK:-7}
VENV=${VENV:-/scratch/moe8b16/serving-bench/venv}; BENCH_DIR=${BENCH_DIR:-/scratch/ckpt-sweep/bench}; FULL_SPACE=${FULL_SPACE:-0}; E2E=${E2E:-0}
W=${WORK:-/scratch/profile/autotune}; mkdir -p $W/tma-tune $W/logs $W/topk_ids; H=$(dirname "$(readlink -f "$0")")
log(){ echo "[$(date -u +%T)] $*"; }
gpu_clean(){ a=$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | wc -l); m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$a" = 0 ] && [ "$m" -lt 1024 ]; }
TRITON_VER=$($VENV/bin/python -c "import triton; print(triton.__version__.replace('.', '_'))"); CFG=$OUT/configs/triton_$TRITON_VER; mkdir -p $CFG
export PYTHONPATH=${PYTHONPATH:-}:/scratch/ckpt-sweep/tune/pylib   # ray (needed by the tuner module import)

# 1. routing dumps: balanced top-1 over the draft's experts (replace with real dumps if the router is unbalanced)
E=$($VENV/bin/python -c "import json; print(json.load(open('$SHAPE_CFG/config.json'))['num_experts'])")
L=$($VENV/bin/python -c "import json; print(json.load(open('$SHAPE_CFG/config.json')).get('num_hidden_layers', 5))")
MAXM=$(echo $BUCKETS | tr ' ' '\n' | sort -n | tail -1)
$VENV/bin/python - <<PY
import torch; g = torch.Generator().manual_seed(0)
for l in range($L):
    for k in range(20): torch.save(torch.randint(0, $E, (max(4096, $MAXM), 1), generator=g, dtype=torch.int32), "$W/topk_ids/topk_ids_layer%d_idx%d.pt" % (l, k))
PY
log "routing dumps: $L layers x 20 files, E=$E"

# 2. one TMA-aware tune per bucket, one GPU each (waits for a free GPU when buckets > GPUs)
declare -a queue=($BUCKETS); i=0
while [ $i -lt ${#queue[@]} ]; do
  for g in $GPUS; do
    [ $i -ge ${#queue[@]} ] && break
    if [ ! -f $W/gpu$g.busy ] && gpu_clean $g; then
      M=${queue[$i]}; i=$((i+1)); log "GPU $g <- tune M=$M"; echo $M > $W/gpu$g.busy
      ( CUDA_VISIBLE_DEVICES=$g FULL_SPACE=$FULL_SPACE TOPK_DIR=$W/topk_ids SHAPE_CFG=$SHAPE_CFG $VENV/bin/python $H/tune_tma_one.py $M $W/tma-tune/M$M.json > $W/logs/tune_M$M.log 2>&1; rm -f $W/gpu$g.busy ) &
      sleep 5
    fi
  done
  sleep 30
done
wait; log "tuning done: $(ls $W/tma-tune/M*.json | wc -l) buckets"

# 3. assemble up (USE_TMA per bucket) + _down configs
OUTDIR=$CFG TUNE_DIR=$W/tma-tune SHAPE_CFG=$SHAPE_CFG BASE=${BASE:-} $VENV/bin/python $H/assemble_tma2.py || { log "assemble failed"; exit 1; }
ls $CFG

# 4. verify on the first free GPU: server loads both files + TMA, AL at c=1, step time at the tuned concurrencies
for g in $GPUS; do gpu_clean $g && VG=$g && break; done
export SGLANG_MOE_CONFIG_DIR=$OUT
EXTRA_ARGS="" bash $H/profile_segments3.sh $VG moe-autotune $DRAFT $OUT > $W/logs/verify.log 2>&1
grep -E "server up|c=1:|c=32:|c=128:|PROFILE" $W/logs/verify.log
grep -c "MoE TMA is enabled" /scratch/profile/logs/server3_moe-autotune.log

# 5. optional: full protocol (per-task + mixed, 5 groups) with the new config
if [ "$E2E" = 1 ]; then
  log "launching the 5-group protocol with SGLANG_MOE_CONFIG_DIR=$OUT"; SGLANG_MOE_CONFIG_DIR=$OUT PREFIX=moe-autotune bash $H/run_e2e_tma2.sh
fi
log "ALL DONE: configs in $CFG"
