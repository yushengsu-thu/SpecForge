#!/bin/bash
# Segment timers only (no torch profiler): serve one draft at block 7 with DSpark's per-step CUDA-event timers, drive it at
# c = 1/8/32/128 with bench_accept prompts, save the records from /server_info.   usage: profile_segments2.sh <gpu> <name> <draft dir> [moe config dir]
set -u
GPU=$1; NAME=$2; DRAFT=$3; CFGDIR=${4:-}; EXTRA_ARGS=${EXTRA_ARGS:-}
P=/scratch/profile; V=/scratch/moe8b16/serving-bench/venv; TARGET=/scratch/moe2ep/models/Qwen3-8B; PORT=$((31720+GPU))
mkdir -p $P/logs $P/results3
log(){ echo "[$(date -u +%T)] [$NAME gpu$GPU] $*"; }
apps=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l); used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$apps" != 0 ] || [ "$used" -ge 1024 ]; then log "GPU $GPU NOT clean ($apps apps, ${used} MiB); abort"; echo "PROFILE_FAILED $NAME"; exit 1; fi
log "GPU $GPU clean: 0 apps, ${used} MiB; other GPU apps on the box: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/scratch/hf_home HF_HUB_OFFLINE=1
export SGLANG_DSPARK_DEBUG_DUMP=core,step_cpu_time,step_gpu_time,draft_gpu_time,target_verify_gpu_time
[ -n "$CFGDIR" ] && export SGLANG_MOE_CONFIG_DIR=$CFGDIR
setsid nohup $V/bin/python -m sglang.launch_server --model-path $TARGET --tp-size 1 --dtype bfloat16 --attention-backend flashinfer \
  --mem-fraction-static 0.8 --trust-remote-code --host 127.0.0.1 --port $PORT --max-running-requests 256 --cuda-graph-max-bs-decode 256 \
  --context-length 16384 --disable-radix-cache --speculative-algorithm DSPARK --speculative-draft-model-path $DRAFT --speculative-dspark-block-size 7 $EXTRA_ARGS \
  > $P/logs/server3_$NAME.log 2>&1 < /dev/null &
PID=$!
for i in $(seq 1 120); do sleep 10; curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; done
curl -sf http://127.0.0.1:$PORT/health > /dev/null || { log "server FAILED"; kill -- -$PID 2>/dev/null; echo "PROFILE_FAILED $NAME"; exit 1; }
log "server up; $(grep -o 'Using [a-zA-Z ]*MoE kernel config[^!]*' $P/logs/server3_$NAME.log | head -1 | cut -c1-100)"
cd /tmp && $V/bin/python $P/bench_accept.py --url http://127.0.0.1:$PORT --concurrency 8 --num-requests 32 --max-tokens 256 > /dev/null 2>&1
for c in 1 8 32 128; do
  n=$(( c*4 > 32 ? c*4 : 32 ))
  curl -sf -X POST http://127.0.0.1:$PORT/set_internal_state -H "Content-Type: application/json" -d '{"server_args": {"dspark_clear_info_records": 1}}' > /dev/null || log "clear failed"
  $V/bin/python $P/bench_accept.py --url http://127.0.0.1:$PORT --concurrency $c --num-requests $n --max-tokens 256 > $P/results3/bench_${NAME}_c$c.txt 2>&1
  curl -s http://127.0.0.1:$PORT/server_info > $P/results3/prof_${NAME}_c$c.json
  log "c=$c: $(grep -o 'output_throughput=[0-9.]* tok/s' $P/results3/bench_${NAME}_c$c.txt) $(grep -o 'spec_accept_length: mean=[0-9.]*' $P/results3/bench_${NAME}_c$c.txt) | server_info bytes $(wc -c < $P/results3/prof_${NAME}_c$c.json) records $(python3 -c "
import json,sys
def f(o):
    if isinstance(o, dict):
        r=o.get('dspark_info_record')
        if isinstance(r, dict) and 'records' in r: return len(r['records'])
        for v in o.values():
            x=f(v)
            if x: return x
    elif isinstance(o, list):
        for v in o:
            x=f(v)
            if x: return x
    return 0
print(f(json.load(open('$P/results3/prof_${NAME}_c$c.json'))))" 2>/dev/null)"
done
kill -- -$PID 2>/dev/null; sleep 5; log "server stopped; gpu mem: $(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader)"
echo "PROFILE_DONE $NAME"
