#!/bin/bash
# e2e throughput sweep of one draft with SGLang c10e0e17 + PR #47181 (the serving-bench venv) on DEDICATED GPUs:
# serve, warm up, run bench_accept.py (PR #10) at each client concurrency, stop the server by its process group.
# usage: GPUS=0 PORT=31500 BLOCK=16 TP=1 DP=1 CONCS="1 8 16 32 64 128 256" bash tp_sweep.sh <name> <draft dir|none> [extra server args]
set -u
NAME=$1; DRAFT=$2; shift 2
B=${BENCH_DIR:-/scratch/moe8b16/serving-bench}; OUT=$B/logs; TARGET=${TARGET:-/scratch/moe2ep/models/Qwen3-8B}
GPUS=${GPUS:-0}; PORT=${PORT:-31500}; BLOCK=${BLOCK:-16}; TP=${TP:-1}; DP=${DP:-1}; CONCS=${CONCS:-1 8 16 32 64 128 256}
MAXRR=${MAXRR:-256}; GRAPH_BS=${GRAPH_BS:-256}; MAXTOK=${MAXTOK:-256}; MEMFRAC=${MEMFRAC:-0.8}; REPEAT=${REPEAT:-1}
. $B/venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPUS HF_HOME=${HF_HOME:-/scratch/moe8b16/hf_home} HF_HUB_CACHE=${HF_HUB_CACHE:-/scratch/moe8b16/hf_home/hub}
log(){ echo "[$NAME] $(date -u +%T) $*"; }
# Refuse to start unless every GPU of this run is empty (no compute process, < 1 GB used): the measurement must be clean.
for g in ${GPUS//,/ }; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g | wc -l); used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  if [ "$apps" != 0 ] || [ "$used" -ge 1024 ]; then log "GPU $g NOT clean: $apps compute apps, ${used} MiB used; aborting"; echo "SWEEP_FAILED $NAME (gpu busy)"; exit 1; fi
done
log "GPU $GPUS clean before start: 0 compute apps, ${used} MiB used; other GPU apps on the box: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
SPEC=()
[ "$DRAFT" != none ] && SPEC=(--speculative-algorithm DSPARK --speculative-draft-model-path $DRAFT --speculative-dspark-block-size $BLOCK)
setsid nohup python -m sglang.launch_server --model-path $TARGET --tp-size $TP --dp-size $DP --dtype bfloat16 --attention-backend flashinfer \
  --mem-fraction-static $MEMFRAC --trust-remote-code --host 127.0.0.1 --port $PORT --max-running-requests $MAXRR --cuda-graph-max-bs-decode $GRAPH_BS \
  "${SPEC[@]}" "$@" > $OUT/server_$NAME.log 2>&1 < /dev/null &
PID=$!
healthy(){ curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200; }
for i in $(seq 1 240); do healthy && break; kill -0 $PID 2>/dev/null || break; sleep 5; done
healthy || { log "server FAILED"; grep -iE "error|Traceback" $OUT/server_$NAME.log | tail -5 | cut -c1-240; kill -- -$PID 2>/dev/null; echo "SWEEP_FAILED $NAME"; exit 1; }
log "server healthy; $(grep -oE 'Capture .*CUDA graph end.{0,90}' $OUT/server_$NAME.log | tail -1 | cut -c1-130)"
log "gpu mem: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
R=$OUT/sweep_$NAME.txt; echo "## $NAME draft=$DRAFT block=$BLOCK tp=$TP dp=$DP gpus=$GPUS max_tokens=$MAXTOK $(date -u)" > $R
python $B/bench/bench_accept.py --url http://127.0.0.1:$PORT --concurrency 8 --num-requests 32 --max-tokens $MAXTOK > /dev/null 2>&1   # warm-up
for c in $CONCS; do
  n=$(( c * 4 > 32 ? c * 4 : 32 ))
  for r in $(seq 1 $REPEAT); do echo "== $NAME concurrency=$c requests=$n repeat=$r" | tee -a $R
  timeout 900 python $B/bench/bench_accept.py --url http://127.0.0.1:$PORT --concurrency $c --num-requests $n --max-tokens $MAXTOK 2>&1 | grep -E "concurrency=|spec_accept_length|per_verify_step|Error|error" | tee -a $R; done
done
kill -- -$PID 2>/dev/null; sleep 5; for i in $(seq 1 60); do kill -0 $PID 2>/dev/null || break; sleep 2; done; kill -9 -- -$PID 2>/dev/null; sleep 3
log "server stopped; gpu mem: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
echo "SWEEP_DONE $NAME" | tee -a $R
