#!/usr/bin/env bash
# One SGLang server for the benchmarks:  serve.sh <draft dir | none> <gpu> <port> <verify width 16|8> <accept|e2e> [mem fraction]
#  accept: jiapingW's acceptance protocol - bf16, TP1, flashinfer target attention, context 16384, one running request,
#          CUDA graphs and radix cache disabled.
#  e2e:    serving throughput - same model flags, CUDA graphs on, 32 running requests, radix cache disabled.
# Use HTTP ports 23xxx (NCCL = port + 2000): ports in the kernel's ephemeral range (32768-60999) collide with the
# thousands of TIME_WAIT sockets the trainer's Mooncake / producer traffic leaves behind (torch's TCPStore bind fails).
# A draft whose config.json lists Qwen3MoeDSparkModel is served through specforge.serving.sglang_models (PYTHONPATH).
set -u
D=${1:?draft dir or none}; G=${2:?gpu}; P=${3:?port}; W=${4:?verify width}; MODE=${5:?accept|e2e}; MEM=${6:-0.80}
REPO=${REPO:-/scratch/SpecForge-spark8b}; TARGET=${TARGET:-/scratch/models/Qwen3-8B}
export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$REPO HF_HOME=/scratch/hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HUGGINGFACE_HUB_CACHE=/scratch/hf-cache/hub TRANSFORMERS_CACHE=/scratch/hf-cache/hub HF_DATASETS_CACHE=/scratch/hf-cache/datasets   # /etc/environment points the HF caches at read-only /cluster-storage
ARGS=(--model-path "$TARGET" --dtype bfloat16 --tp-size 1 --host 127.0.0.1 --port "$P" --nccl-port $((P + 2000))
      --attention-backend flashinfer --context-length 16384 --mem-fraction-static "$MEM" --disable-radix-cache)
if [ "$MODE" = accept ]; then ARGS+=(--max-running-requests 1 --disable-cuda-graph); else ARGS+=(--max-running-requests 32 --cuda-graph-max-bs 32); fi
if [ "$D" != none ]; then
  ARGS+=(--speculative-algorithm DSPARK --speculative-draft-model-path "$D" --speculative-num-steps 1
         --speculative-eagle-topk 1 --speculative-dspark-block-size $((W - 1)))
  if grep -q "Qwen3MoeDSparkModel" "$D/config.json"; then
    export SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang_models; ARGS+=(--moe-runner-backend triton)
  fi
fi
echo "[serve] gpu=$G port=$P draft=$D width=$W mode=$MODE external=${SGLANG_EXTERNAL_MODEL_PACKAGE:-none}"
exec python -m sglang.launch_server "${ARGS[@]}"
