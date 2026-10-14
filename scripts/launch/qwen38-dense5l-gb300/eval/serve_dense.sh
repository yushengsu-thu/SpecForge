#!/bin/bash
# usage: serve_dense.sh <draft dir> <gpu> <port>   (bf16 target, DSPARK block 7, triton, mamba cache 160, 32 running)
D=${1:?draft}; G=${2:?gpu}; P=${3:?port}
CUDA_VISIBLE_DEVICES=$G exec python3 -m sglang.launch_server --model-path /scratch/dense5l-eval/models/Qwen3.8-27B --dtype bfloat16 \
  --trust-remote-code --tp-size 1 --attention-backend triton --mem-fraction-static 0.85 --max-running-requests 16 \
  --max-mamba-cache-size 32 --context-length 40960 --disable-radix-cache \
  --speculative-algorithm DSPARK --speculative-draft-model-path "$D" --speculative-draft-model-quantization unquant \
  --speculative-draft-attention-backend triton --speculative-dspark-block-size 7 --speculative-num-steps 1 --speculative-eagle-topk 1 \
  --host 127.0.0.1 --port "$P"
