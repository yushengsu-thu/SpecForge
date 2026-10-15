#!/bin/bash
# usage: serve_e2e.sh <label> <draft dir | none> <gpu> <port>   (same target/flags as the accept-length eval, 32 running for c=32)
L=${1:?label}; D=${2:?draft or none}; G=${3:?gpu}; P=${4:?port}
SPEC=""; [ "$D" != none ] && SPEC="--speculative-algorithm DSPARK --speculative-draft-model-path $D --speculative-draft-model-quantization unquant --speculative-draft-attention-backend triton --speculative-dspark-block-size 7 --speculative-num-steps 1 --speculative-eagle-topk 1"
CUDA_VISIBLE_DEVICES=$G exec python3 -m sglang.launch_server --model-path /scratch/dense5l-eval/models/Qwen3.8-27B --dtype bfloat16 \
  --trust-remote-code --tp-size 1 --attention-backend triton --mem-fraction-static 0.85 --max-running-requests 32 \
  --max-mamba-cache-size 48 --context-length 8192 --disable-radix-cache $SPEC --host 127.0.0.1 --port $P
