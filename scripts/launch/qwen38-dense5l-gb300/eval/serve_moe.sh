#!/bin/bash
# usage: serve_moe.sh <draft dir> <gpu> <port> [running=16] [mamba=32] [context=40960]
# MoE DSpark drafter (Qwen3MoeDSparkModel) through the PR #941 serving package; otherwise the same flags as serve_dense.sh
D=${1:?draft}; G=${2:?gpu}; P=${3:?port}; RUN=${4:-16}; MAMBA=${5:-32}; CTX=${6:-40960}
export SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang_models PYTHONPATH=/scratch/dense5l-eval/SpecForge:${PYTHONPATH:-}
CUDA_VISIBLE_DEVICES=$G exec python3 -m sglang.launch_server --model-path /scratch/dense5l-eval/models/Qwen3.8-27B --dtype bfloat16 \
  --trust-remote-code --tp-size 1 --attention-backend triton --mem-fraction-static 0.85 --max-running-requests "$RUN" \
  --max-mamba-cache-size "$MAMBA" --context-length "$CTX" --disable-radix-cache --moe-runner-backend triton \
  --speculative-algorithm DSPARK --speculative-draft-model-path "$D" --speculative-draft-model-quantization unquant \
  --speculative-draft-attention-backend triton --speculative-dspark-block-size 7 --speculative-num-steps 1 --speculative-eagle-topk 1 \
  --host 127.0.0.1 --port "$P"
