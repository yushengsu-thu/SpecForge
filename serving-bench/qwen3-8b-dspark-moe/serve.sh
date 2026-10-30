#!/bin/bash
# Serve Qwen3-8B with one DSpark draft export on one GPU. env: GPU, DRAFT, PORT; extra args pass through.
set -u
P=$(cd "$(dirname "$0")" && pwd)
export PATH=/opt/sglang/bin:$PATH HF_HOME=/scratch/moe-serving/hf HF_HUB_CACHE=/scratch/moe-serving/hf/hub CUDA_VISIBLE_DEVICES=${GPU:-0}
exec python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B --tp-size 1 --dtype bfloat16 \
  --attention-backend flashinfer --mem-fraction-static 0.8 --trust-remote-code \
  --host 127.0.0.1 --port "${PORT:-30000}" --speculative-algorithm DSPARK \
  --speculative-draft-model-path "${DRAFT:?}" "$@"
