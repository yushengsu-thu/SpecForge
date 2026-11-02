#!/bin/bash
# One memory-bounded SGLang server for Qwen3-8B (BF16) + a DSpark draft on GPU 0 of the TRAINING box (next to the
# capture server, which keeps its own preallocated pool), then gsm8k with jiapingW's generation settings
# (non-thinking, temperature 0, max_new_tokens 2048, block 16, context 16384, radix cache off) at c=1/16/64/128/256.
# The server is killed by its own process group at the end; nothing else on the GPU is touched.
# usage: serve_eval.sh <name> <draft dir|none> [moe]
NAME=$1; DRAFT=$2; MOE=$3
# GPU / PORT / MAXRR (max running requests) / GRAPH_BS come from the caller for the checkpoint sweep.
W=${EVAL_DIR:-/scratch/moe8b16/eval}; OUT=$W/serve/$NAME; TARGET=${TARGET:-/scratch/moe2ep/models/Qwen3-8B}; GPU=${GPU:-0}; PORT=${PORT:-31301}
CODE=$W/SpecForge-serve
mkdir -p $OUT; cd $CODE
. ${VENV:-/scratch/moe2ep/venv}/bin/activate
export PYTHONPATH=$CODE CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/scratch/moe8b16/hf_home HF_HUB_CACHE=/scratch/moe8b16/hf_home/hub HF_DATASETS_CACHE=/scratch/moe8b16/hf_home/datasets
export HF_TOKEN="${HF_TOKEN:-$(cat /scratch/secrets/hf_token 2>/dev/null)}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CUDA-graph capture of the 17-token verify needs ~820 MB of FlashInfer workspace at batch 256, and SGLang pins the
# workspace to 512 MB for Qwen3 (flashinfer_backend.py ignores SGLANG_FLASHINFER_WORKSPACE_SIZE), so graphs are
# captured up to batch 128 only; batches above that run without graphs, the same for both drafts.
GRAPH_BS=${GRAPH_BS:-128}; MAXRR=${MAXRR:-256}
[ "$MOE" = moe ] && export SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang_models
log(){ echo "[$NAME] $(date -u +%T) $*"; }
SPEC=()
if [ "$DRAFT" != none ]; then
  SPEC=(--speculative-algorithm DSPARK --speculative-draft-model-path $DRAFT --speculative-draft-model-quantization unquant
        --speculative-draft-attention-backend flashinfer --speculative-dspark-block-size 16 --speculative-num-steps 1 --speculative-eagle-topk 1)
fi
# 0.35 x 144 GB = 50 GB budget (16 GB weights + 2 GB draft + KV), KV pool capped at 180k tokens: leaves the capture
# server's 34 GB and ~60 GB free on the GPU.
setsid nohup python -m sglang.launch_server --trust-remote-code --model-path $TARGET --mem-fraction-static 0.35 --max-total-tokens 180000 \
  --attention-backend flashinfer --chunked-prefill-size 32768 --max-prefill-tokens 32768 --max-running-requests $MAXRR --cuda-graph-max-bs $GRAPH_BS --context-length 16384 \
  --disable-radix-cache --host 127.0.0.1 --port $PORT "${SPEC[@]}" > $OUT/server.log 2>&1 < /dev/null &
PID=$!
healthy(){ curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200; }
for i in $(seq 1 360); do healthy && break; kill -0 $PID 2>/dev/null || break; sleep 5; done
healthy || { log "server FAILED"; grep -iE "error|Traceback" $OUT/server.log | tail -6 | cut -c1-240; kill -- -$PID 2>/dev/null; echo "[$NAME] DONE"; exit 1; }
log "server healthy: $(grep -m1 -oE 'MoE draft \(Qwen3MoeDSparkModel\).{0,80}|type=Qwen3DSparkModel.{0,40}|type=Qwen3MoeDSparkModel.{0,40}' $OUT/server.log | head -1)"
log "gpu0 memory with server: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0)"
# TASKS="<dataset>:<num prompts> ..." (jiapingW's five: gsm8k:1319 math500:500 humaneval:164 mbpp:257 mt-bench:160),
# CONCS="<concurrency> ...".
# Concurrency sweep stops at 64: above that the numbers are noisy on a shared GPU (and batches above 128 have no
# CUDA graph here).
TASKS=${TASKS:-gsm8k:1319}; CONCS=${CONCS:-1 8 16 32 64}
for t in $TASKS; do DS=${t%%:*}; N=${t##*:}
  for C in $CONCS; do
    python -m specforge.cli benchmark --model $TARGET --dataset $DS --num-prompts $N --concurrency $C \
      --max-new-tokens 2048 --temperature 0 --base-url http://127.0.0.1:$PORT --output-json $OUT/${DS}_c$C.json > $OUT/bench_${DS}_c$C.log 2>&1 \
      || { log "bench $DS c=$C FAILED"; tail -2 $OUT/bench_${DS}_c$C.log | cut -c1-200; }
    log "$DS c=$C: $(grep -iE 'acceptance length|throughput|latency' $OUT/bench_${DS}_c$C.log | tr -s ' ' | tr '\n' ' ' | cut -c1-200)"
  done
done
kill -- -$PID 2>/dev/null; sleep 5; for i in $(seq 1 60); do kill -0 $PID 2>/dev/null || break; sleep 2; done
kill -9 -- -$PID 2>/dev/null; sleep 3
log "server stopped: port listeners=$( (ss -ltn 2>/dev/null) | grep -cE ":$PORT ") gpu0 memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0)"
log "DONE"
