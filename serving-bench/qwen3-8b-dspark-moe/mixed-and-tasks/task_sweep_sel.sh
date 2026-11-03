#!/bin/bash
# Per-task e2e throughput of ONE draft on jiapingW's five tasks (full prompt sets, his generation settings: non-thinking,
# temperature 0, max_new_tokens 2048, block 16, context 16384, radix cache off) at each client concurrency, with the
# SGLang c10e0e17 + PR #47181 venv on one CLEAN H200 (guard below). One server; the client is specforge benchmark from the
# eval checkout. Results: logs/tasks/<name>/<task>_c<C>.json.  usage: GPUS=0 PORT=31530 bash task_sweep.sh <name> <draft dir>
set -u
NAME=$1; DRAFT=$2
B=${BENCH_DIR:-/scratch/moe8b16/serving-bench}; OUT=$B/logs/tasks/$NAME; TARGET=${TARGET:-/scratch/moe2ep/models/Qwen3-8B}
GPUS=${GPUS:-0}; PORT=${PORT:-31530}; BLOCK=7
BIG_CONCS=${BIG_CONCS:-1 8 16 32 64 128 256}; SMALL_CONCS=${SMALL_CONCS:-1 8 16 32 64}
mkdir -p $OUT
log(){ echo "[$NAME] $(date -u +%T) $*"; }
for g in ${GPUS//,/ }; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g | wc -l); used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  if [ "$apps" != 0 ] || [ "$used" -ge 1024 ]; then log "GPU $g NOT clean: $apps compute apps, ${used} MiB used; aborting"; echo "TASKS_FAILED $NAME (gpu busy)"; exit 1; fi
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
healthy || { log "server FAILED"; grep -iE "error|Traceback" $OUT/server.log | tail -5 | cut -c1-240; kill -- -$PID 2>/dev/null; echo "TASKS_FAILED $NAME"; exit 1; }
log "server healthy; $(grep -a -m1 -oE 'Using (default )?MoE kernel config[^,]*' $OUT/server.log); $(grep -a -oE 'Capture draft verify CUDA graph end.{0,60}' $OUT/server.log | tail -1 | cut -c1-90)"
CLI="${CLIENT_PY:-/scratch/moe2ep/venv/bin/python} -m specforge.cli benchmark"
export PYTHONPATH=${SPECFORGE_SRC:-/scratch/moe8b16/eval/SpecForge-serve}
for C in 1 8 16 32 64 128 256; do
  for t in ${TASKS:-gsm8k:1319 math500:500 humaneval:164 mbpp:257 mt-bench:160}; do
    DS=${t%%:*}; N=${t##*:}
    case " $BIG_CONCS " in *" $C "*) ;; *) continue;; esac
    if [ $N -le 300 ]; then case " $SMALL_CONCS " in *" $C "*) ;; *) continue;; esac; fi
    [ -s $OUT/${DS}_c$C.json ] && { log "$DS c=$C: already done"; continue; }
    env -u CUDA_VISIBLE_DEVICES $CLI --model $TARGET --dataset $DS --num-prompts $N --concurrency $C --max-new-tokens 2048 --temperature 0 \
      --timeout-seconds 14400 --base-url http://127.0.0.1:$PORT --output-json $OUT/${DS}_c$C.json > $OUT/bench_${DS}_c$C.log 2>&1 \
      || { log "bench $DS c=$C FAILED"; tail -2 $OUT/bench_${DS}_c$C.log | cut -c1-200; }
    log "$DS c=$C: $(python3 -c "import json; r=json.load(open('$OUT/${DS}_c$C.json')); print('%d reqs, %.0f tok/s, AL %.3f, step %.1f ms' % (r['samples'], r['throughput_tokens_per_second'], r['average_acceptance_length'], 1000*r['latency_seconds']*min($C, r['samples'])/r['spec_verify_count']))" 2>/dev/null)"
  done
done
kill -- -$PID 2>/dev/null; sleep 5; for i in $(seq 1 60); do kill -0 $PID 2>/dev/null || break; sleep 2; done; kill -9 -- -$PID 2>/dev/null; sleep 3
log "server stopped; gpu apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); gpu mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i ${GPUS%%,*})"
echo "TASKS_DONE $NAME"
