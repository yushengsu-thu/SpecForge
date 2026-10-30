#!/bin/bash
# Devbox: serve one draft at width 8 with DSpark's per-step GPU timers on, drive it at several concurrencies,
# and save the per-step records (draft_gpu_ms / target_verify_gpu_ms / step_gpu_ms) per concurrency.
# usage: profile_segments.sh <gpu> <draft dir> <port> <tag>
set -u
GPU=$1; DRAFT=$2; PORT=$3; TAG=$4
P=/personal/moe-serving
export PATH=/opt/sglang/bin:$PATH HF_HOME=/scratch/moe-serving/hf HF_HUB_CACHE=/scratch/moe-serving/hf/hub
export SGLANG_DSPARK_DEBUG_DUMP=core,step_cpu_time,step_gpu_time,draft_gpu_time,target_verify_gpu_time
GPU=$GPU DRAFT=$DRAFT PORT=$PORT bash $P/serve.sh --speculative-dspark-block-size 8 --max-running-requests 256 --cuda-graph-max-bs-decode 256 > $P/logs/server_prof_$TAG.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do sleep 10; curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "SERVER FAILED $TAG"; kill $SRV; echo "PROFILE DONE $TAG"; exit 1; }
echo "server up $(date -u +%H:%M:%S)"
cd /tmp && python3 $P/bench_accept.py --url http://127.0.0.1:$PORT --concurrency 8 --num-requests 32 --max-tokens 256 > /dev/null 2>&1
for c in 1 8 32 128; do
  n=$(( c * 4 > 32 ? c * 4 : 32 ))
  curl -sf -X POST http://127.0.0.1:$PORT/set_internal_state -H "Content-Type: application/json" -d '{"server_args": {"dspark_clear_info_records": 1}}' > /dev/null || echo "clear failed"
  echo "== $TAG c=$c n=$n"
  python3 $P/bench_accept.py --url http://127.0.0.1:$PORT --concurrency $c --num-requests $n --max-tokens 256 2>&1 | grep -E "concurrency=|spec_accept|per_verify"
  curl -sf http://127.0.0.1:$PORT/get_internal_state > $P/logs/prof_${TAG}_c$c.json && echo "saved prof_${TAG}_c$c.json ($(wc -c < $P/logs/prof_${TAG}_c$c.json) bytes)"
done
kill $SRV; sleep 3; pkill -9 -f "port $PORT" 2>/dev/null; true
echo "PROFILE DONE $TAG"
