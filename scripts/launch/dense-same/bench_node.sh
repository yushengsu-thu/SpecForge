#!/bin/bash
# One acceptance-length benchmark mode on the capture node (rank 0), protocol of jiapingW/Qwen3-8b-dspark-SpecForge:
# SGLang DSpark, bf16, TP1, flashinfer target attention, context 16384, max_running_requests 1, CUDA graphs and
# radix cache off, one server per GPU (they share the GPUs with the idle-most-of-the-time capture servers).
# V16 = anchor + 15 draft candidates (--speculative-dspark-block-size 15, i.e. gamma 15 -> verify window 16), V8 =
# anchor + 7 (gamma 7 -> window 8); with the causal draft block the first k candidates are the same as the first k
# of the trained 16-token block.
# usage: bench_node.sh <label> <draft_dir> <V16|V8> <temperature>
#   env: GPUS="0 1 2 3 4 5 6 7"  MAX_PER_DATASET=0 (all)  DATASETS=gsm8k,math500,humaneval,mbpp,mt-bench
set -u
LABEL=${1:?label}; DRAFT=${2:?draft dir}; MODE=${3:?V16|V8}; TEMP=${4:?temperature}
GPUS=${GPUS:-0 1 2 3 4 5 6 7}; MAXN=${MAX_PER_DATASET:-0}; DSETS=${DATASETS:-gsm8k,math500,humaneval,mbpp,mt-bench}
case $MODE in V16) GAMMA=15;; V8) GAMMA=7;; *) echo "mode must be V16 or V8"; exit 2;; esac
BASE=/scratch/dense/bench; OUTD=$BASE/results/$LABEL/${MODE}_t${TEMP}; LOGS=$BASE/logs; mkdir -p $OUTD $LOGS
export HF_HOME=/scratch/hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HF_DATASETS_CACHE=/scratch/hf-cache/datasets FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH=/scratch/dense/SpecForge${PYTHONPATH:+:$PYTHONPATH}
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
PIDS=(); URLS=()
cleanup() { for p in "${PIDS[@]}"; do pkill -TERM -P "$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; done; sleep 5; for p in "${PIDS[@]}"; do pkill -KILL -P "$p" 2>/dev/null; kill -KILL "$p" 2>/dev/null; done; }
trap cleanup EXIT
for g in $GPUS; do
  port=$((31000 + g))
  CUDA_VISIBLE_DEVICES=$g setsid python -m sglang.launch_server --model-path /scratch/models/Qwen3-8B --dtype bfloat16 --tp-size 1 \
    --speculative-algorithm DSPARK --speculative-draft-model-path "$DRAFT" --speculative-dspark-block-size $GAMMA \
    --attention-backend flashinfer --context-length 16384 --max-running-requests 1 --disable-cuda-graph --disable-radix-cache \
    --mem-fraction-static 0.4 --host 127.0.0.1 --port $port --nccl-port $((38000 + g)) > $LOGS/$LABEL-$MODE-t$TEMP-gpu$g.log 2>&1 < /dev/null &
  PIDS+=($!); sleep 15
done
for g in $GPUS; do
  port=$((31000 + g)); ok=0
  for t in $(seq 1 90); do curl -sf -m 3 http://127.0.0.1:$port/health >/dev/null && { ok=1; break; }; grep -qE "Traceback|out of memory" $LOGS/$LABEL-$MODE-t$TEMP-gpu$g.log && break; sleep 10; done
  if [ $ok = 1 ]; then URLS+=("http://127.0.0.1:$port"); else log "server on GPU $g failed"; tail -5 $LOGS/$LABEL-$MODE-t$TEMP-gpu$g.log; fi
done
[ ${#URLS[@]} -gt 0 ] || { log "BENCH_FAILED no healthy server"; exit 1; }
log "$LABEL $MODE t=$TEMP: ${#URLS[@]} servers healthy, draft $DRAFT"
python3 /scratch/dense/SpecForge/scripts/launch/dense-same/bench_accept.py --urls "$(IFS=,; echo "${URLS[*]}")" \
  --tokenizer /scratch/models/Qwen3-8B --temperature "$TEMP" --out-dir "$OUTD" --max-per-dataset "$MAXN" --datasets "$DSETS" \
  --label "$LABEL $MODE t=$TEMP"
rc=$?
grep -hoE "DSpark gamma mismatch[^,]*" $LOGS/$LABEL-$MODE-t$TEMP-gpu*.log | head -1
log "BENCH_DONE $LABEL $MODE t=$TEMP rc=$rc -> $OUTD/summary.json"
exit $rc
