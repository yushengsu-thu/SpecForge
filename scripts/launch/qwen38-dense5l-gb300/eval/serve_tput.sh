#!/bin/bash
# usage: serve_tput.sh <draft dir | none> <gpu> <port>   (throughput runs: 32 running requests, mamba cache 32 (one slot per request; 48 left the 256-expert server only 23k KV tokens), 40k context,
# otherwise the accept-length server flags; MoE drafters come through the PR #941 package, dense through SGLang's class)
D=${1:?draft or none}; G=${2:?gpu}; P=${3:?port}; RUN=${RUN:-32}; MAMBA=${MAMBA:-32}; CTX=${CTX:-40960}
export SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang_models PYTHONPATH=/scratch/dense5l-eval/SpecForge:${PYTHONPATH:-}
SPEC=""; [ "$D" != none ] && SPEC="--speculative-algorithm DSPARK --speculative-draft-model-path $D --speculative-draft-model-quantization unquant --speculative-draft-attention-backend triton --speculative-dspark-block-size 7 --speculative-num-steps 1 --speculative-eagle-topk 1"
CUDA_VISIBLE_DEVICES=$G exec python3 -m sglang.launch_server --model-path /scratch/dense5l-eval/models/Qwen3.8-27B --dtype bfloat16 \
  --trust-remote-code --tp-size 1 --attention-backend triton --mem-fraction-static 0.85 --max-running-requests "$RUN" \
  --max-mamba-cache-size "$MAMBA" --context-length "$CTX" --disable-radix-cache --moe-runner-backend triton $SPEC --host 127.0.0.1 --port "$P"
