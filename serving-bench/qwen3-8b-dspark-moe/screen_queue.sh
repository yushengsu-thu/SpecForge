#!/bin/bash
# Devbox: download and screen checkpoints one after another on one GPU. usage: screen_queue.sh <gpu> <port> <step>...
set -u
GPU=$1; PORT=$2; shift 2
P=/personal/moe-serving
bash $P/dl_steps.sh "$@"
for s in "$@"; do
  [ "$(stat -c %s $P/export2/step-$s/model.safetensors 2>/dev/null)" = 2253170056 ] || { echo "SKIP step-$s (incomplete export)"; continue; }
  bash $P/bench_all.sh $GPU $P/export2/step-$s $PORT moe$s
done
echo "QUEUE DONE gpu$GPU"
