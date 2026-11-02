#!/bin/bash
# Mixed-workload e2e throughput of ONE draft (mixed_bench.py: the five tasks shuffled into one stream) at each client
# concurrency, same server settings as task_sweep.sh, on one CLEAN GPU. usage: GPUS=0 PORT=31560 bash mixed_sweep.sh <name> <draft>
set -u
NAME=$1; DRAFT=$2
B=${BENCH_DIR:-/scratch/moe8b16/serving-bench}; OUT=$B/logs/mixed/$NAME; TARGET=${TARGET:-/scratch/moe2ep/models/Qwen3-8B}
GPUS=${GPUS:-0}; PORT=${PORT:-31560}; BLOCK=16; CONCS=${CONCS:-1 8 16 32 64 128 256}
mkdir -p $OUT
log(){ echo "[$NAME] $(date -u +%T) $*"; }
for g in ${GPUS//,/ }; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g | wc -l); used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  if [ "$apps" != 0 ] || [ "$used" -ge 1024 ]; then log "GPU $g NOT clean: $apps compute apps, ${used} MiB used; aborting"; echo "MIXED_FAILED $NAME (gpu busy)"; exit 1; fi
done
log "GPU $GPUS clean before start: 0 compute apps, ${used} MiB used; other GPU apps on the box: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
. $B/venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPUS HF_HOME=${HF_HOME:-/scratch/moe8b16/hf_home} HF_HUB_CACHE=${HF_HUB_CACHE:-/scratch/moe8b16/hf_home/hub} HF_DATASETS_CACHE=/scratch/moe8b16/hf_home/datasets HF_TOKEN="${HF_TOKEN:-$(cat /scratch/secrets/hf_token 2>/dev/null)}"
setsid nohup python -m sglang.launch_server --model-path $TARGET --tp-size 1 --dtype bfloat16 --attention-backend flashinfer \
  --mem-fraction-static 0.8 --trust-remote-code --host 127.0.0.1 --port $PORT --max-running-requests 256 --cuda-graph-max-bs-decode 256 \
  --context-length 16384 --disable-radix-cache --speculative-algorithm DSPARK --speculative-draft-model-path $DRAFT --speculative-dspark-block-size $BLOCK \
  > $OUT/server.log 2>&1 < /dev/null &
PID=$!
healthy(){ curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200; }
for i in $(seq 1 240); do healthy && break; kill -0 $PID 2>/dev/null || break; sleep 5; done
healthy || { log "server FAILED"; grep -iE "error|Traceback" $OUT/server.log | tail -5 | cut -c1-240; kill -- -$PID 2>/dev/null; echo "MIXED_FAILED $NAME"; exit 1; }
log "server healthy; $(grep -a -m1 -oE 'Using (default )?MoE kernel config[^,]*' $OUT/server.log)"
export PYTHONPATH=${SPECFORGE_SRC:-/scratch/moe8b16/eval/SpecForge-serve}
for C in $CONCS; do
  [ -s $OUT/mixed_c$C.json ] && { log "c=$C already done"; continue; }
  env -u CUDA_VISIBLE_DEVICES ${CLIENT_PY:-/scratch/moe2ep/venv/bin/python} $B/mixed_bench.py --model $TARGET --concurrency $C --base-url http://127.0.0.1:$PORT --output-json $OUT/mixed_c$C.json > $OUT/bench_c$C.log 2>&1 \
    && log "$(tail -1 $OUT/bench_c$C.log)" || { log "bench c=$C FAILED"; tail -2 $OUT/bench_c$C.log | cut -c1-200; }
done
kill -- -$PID 2>/dev/null; sleep 5; for i in $(seq 1 60); do kill -0 $PID 2>/dev/null || break; sleep 2; done; kill -9 -- -$PID 2>/dev/null; sleep 3
log "server stopped; gpu mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i ${GPUS%%,*})"
echo "MIXED_DONE $NAME"
