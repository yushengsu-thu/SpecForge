#!/usr/bin/env bash
# Host side: serve one draft at width 8 in a container and run specforge benchmark over the datasets.
# usage: bench_datasets.sh <container> <draft dir in container> <port> <tag>
set -u
C=$1; DRAFT=$2; PORT=$3; TAG=$4; W=$HOME/claude-dflash-moe; OUT=$W/logs/datasets_g8_$TAG.txt
docker exec $C bash -c "pkill -f launch_server; sleep 4; pgrep -f launch_server >/dev/null && pkill -9 -f launch_server; true"
docker exec -d $C bash -c "cd /sgl-workspace/sglang && DRAFT=$DRAFT PORT=$PORT bash /work/serve.sh --speculative-dspark-block-size 8 > /work/logs/server_datasets_$TAG.log 2>&1"
for i in $(seq 1 40); do sleep 10; docker exec $C curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; done
: > $OUT
for spec in "math500:100" "humaneval:164" "mbpp:200" "mt-bench:80"; do
  ds=${spec%%:*}; n=${spec#*:}
  echo "== $TAG width=8 $ds num_prompts=$n" | tee -a $OUT
  docker exec $C bash -c "cd /tmp && HF_DATASETS_CACHE=/tmp/hfds_$TAG timeout 3600 specforge benchmark --model Qwen/Qwen3-8B --dataset $ds --num-prompts $n --concurrency 1 --base-url http://127.0.0.1:$PORT --output-json /work/logs/bench_${TAG}_g8_${ds}_c1.json" 2>&1 | grep -E "Dataset:|Output throughput|Average acceptance|verify count|Error|error" | tee -a $OUT
done
docker exec $C bash -c "pkill -f launch_server; true"
echo "DATASETS DONE $TAG" | tee -a $OUT
