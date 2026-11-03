#!/bin/bash
# Evaluate EVERY uploaded checkpoint of the run (step-1000 ... step-64000, ep6) on jiapingW's five tasks at c=1,
# after training has ended: 16 workers, two per GPU (a c=1 server leaves most of an H200 idle), each pulling the
# next checkpoint from a shared queue, downloading it from HF, serving it, benchmarking, stopping the server and
# deleting the weights. Results: serve/sweep/<ckpt>/<task>_c1.json; summary via summarize_sweep.py.
W=${EVAL_DIR:-/scratch/moe8b16/eval}; RUN=${RUN:-qwen3-8b-moe8x1536-8A1-dspark-run1}; SWEEP=${SWEEP:-sweep}; REPO=yushengsu/qwen3-8b-moe16x768-16A1-dspark
Q=$W/sweep-queue-$SWEEP; mkdir -p $Q $W/serve/$SWEEP $W/models/ckpts
export HF_HOME=/scratch/moe8b16/hf_home HF_HUB_CACHE=/scratch/moe8b16/hf_home/hub HF_TOKEN="${HF_TOKEN:-$(cat /scratch/secrets/hf_token 2>/dev/null)}"
log(){ echo "[$(date -u +%T)] $*"; }
cd $W
# Refuse to start while anything of the training is still on the GPUs.
if pgrep -f 'specforge.cli trai[n]' >/dev/null || [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ]; then
  log "training processes still on the GPUs; not starting"; exit 1
fi
CKPTS=${CKPTS:-"$(for s in $(seq 1000 1000 64000); do echo step-$s; done) ep6"}
for c in $CKPTS; do [ -f $W/serve/$SWEEP/$c/mt-bench_c1.json ] || touch $Q/$c; done   # the queue skips finished ones
log "queued $(ls $Q | wc -l) checkpoints"
worker() {
  local slot=$1 gpu=$((slot % 8)) port=$((31400 + slot))
  while true; do
    local c=$(ls $Q 2>/dev/null | sort -t- -k2 -n | head -1); [ -n "$c" ] || break
    rm -f $Q/$c 2>/dev/null || continue    # whoever removes the file owns the checkpoint
    local dir=$W/models/ckpts/$c
    # One local-dir per checkpoint: 16 concurrent downloads must not share HF's local-dir metadata cache.
    hf download $REPO --include "$RUN/$c/*" --local-dir $W/models/ckpts/dl-$c > $W/serve/$SWEEP/dl-$c.log 2>&1 \
      || { log "slot $slot: download FAILED for $c"; continue; }
    rm -rf $dir; mv $W/models/ckpts/dl-$c/$RUN/$c $dir && rm -rf $W/models/ckpts/dl-$c
    GPU=$gpu PORT=$port MAXRR=8 GRAPH_BS=8 TASKS="gsm8k:1319 math500:500 humaneval:164 mbpp:257 mt-bench:160" CONCS="1" \
      bash $W/serve_eval.sh $SWEEP/$c $dir moe > $W/serve/$SWEEP/$c.log 2>&1
    log "slot $slot gpu $gpu: $c $(grep -oE 'c=1: Output throughput: [0-9.]+ tok/s Average acceptance length: [0-9.]+' $W/serve/$SWEEP/$c.log | sed -E 's/.*length: //' | tr '\n' ' ')"
    rm -rf $dir
  done
  log "slot $slot done"
}
for slot in $(seq 0 15); do worker $slot & sleep 20; done
wait
log "SWEEP_DONE: $(ls -d $W/serve/$SWEEP/*/ | wc -l) checkpoint dirs; GPUs: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
