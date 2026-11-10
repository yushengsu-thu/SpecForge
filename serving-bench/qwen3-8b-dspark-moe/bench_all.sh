#!/bin/bash
# Serve one draft at width 8 on one GPU and run specforge benchmark (c=1) over the five datasets.
# usage: bench_all.sh <gpu> <draft dir> <port> <tag>   -> logs/all_g8_<tag>.txt, logs/bench_<tag>_g8_<ds>_c1.json
set -u
GPU=$1; DRAFT=$2; PORT=$3; TAG=$4
P=$(cd "$(dirname "$0")" && pwd); OUT=$P/logs/all_g8_$TAG.txt
export PATH=/opt/sglang/bin:$PATH HF_HOME=/scratch/moe-serving/hf HF_HUB_CACHE=/scratch/moe-serving/hf/hub
GPU=$GPU DRAFT=$DRAFT PORT=$PORT bash $P/serve.sh --speculative-dspark-block-size 8 > $P/logs/server_all_$TAG.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do sleep 10; curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "SERVER FAILED $TAG" | tee -a $OUT; kill $SRV; exit 1; }
: > $OUT
for spec in "gsm8k:100" "math500:100" "humaneval:164" "mbpp:200" "mt-bench:80"; do
  ds=${spec%%:*}; n=${spec#*:}
  echo "== $TAG width=8 $ds num_prompts=$n" | tee -a $OUT
  (cd /tmp && HF_DATASETS_CACHE=/scratch/moe-serving/hfds timeout 3600 specforge benchmark --model Qwen/Qwen3-8B --dataset $ds \
    --num-prompts $n --concurrency 1 --base-url http://127.0.0.1:$PORT --output-json $P/logs/bench_${TAG}_g8_${ds}_c1.json) 2>&1 \
    | grep -E "Dataset:|Output throughput|Average acceptance|verify count|Error|error" | tee -a $OUT
done
kill $SRV; sleep 3; pkill -9 -f "port $PORT" 2>/dev/null; true
echo "DATASETS DONE $TAG" | tee -a $OUT
