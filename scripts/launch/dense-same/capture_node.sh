#!/bin/bash
# dense-same capture node: 8 spec-capture SGLang servers serving Qwen/Qwen3-8B (bf16, TP1, one per GPU),
# each registering a 100 GiB Mooncake RDMA segment with the master on the trainer node.
# CUDA graphs are off: on 2026-10-21 a capture server died in the decode CUDA-graph runner ("runtime hidden-state mode
# exceeds the fixed CUDA graph capture mode"), and its Mooncake segment took the in-flight features with it.
# usage: [NCCL_BASE=37000] bash capture_node.sh <CAPTURE_IP> <TRAINER_IP> [N=8]      (model: hf download Qwen/Qwen3-8B --local-dir /scratch/models/Qwen3-8B)
set -u
CAP=${1:?capture ip}; TRN=${2:?trainer ip}; N=${3:-8}; NCCL_BASE=${NCCL_BASE:-37000}
MODEL=/scratch/models/Qwen3-8B; LOGS=/scratch/dense/logs
export HF_HOME=/scratch/hf-cache FLASHINFER_DISABLE_VERSION_CHECK=1
export MOONCAKE_GLOBAL_SEGMENT_SIZE=107374182400 MOONCAKE_LOCAL_BUFFER_SIZE=1073741824
export MOONCAKE_LOCAL_HOSTNAME=$CAP MOONCAKE_MASTER_SERVER_ADDR=$TRN:35551 MOONCAKE_METADATA_SERVER=http://$TRN:35880/metadata MOONCAKE_PROTOCOL=rdma
export DISAGG_SERVER_URLS=$(for i in $(seq 0 $((N-1))); do printf "http://%s:%d," "$CAP" $((30000+i)); done | sed 's/,$//')
mkdir -p $LOGS
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
until [ -f $MODEL/config.json ] && ! pgrep -f "hf download Qwen/Qwen3-8B" >/dev/null; do log "waiting for $MODEL download"; sleep 15; done
log "model ready: $(du -sh $MODEL | cut -f1)"
until curl -s -m 3 "http://$TRN:35880/metadata?key=x" >/dev/null 2>&1; do log "waiting for mooncake master on $TRN"; sleep 10; done
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i setsid nohup python -m sglang.launch_server --model-path $MODEL --dtype bfloat16 \
    --skip-tokenizer-init --tp-size 1 --chunked-prefill-size -1 --enable-spec-capture --spec-capture-method dspark --spec-capture-aux-layer-ids 1 9 17 25 33 \
    --host 0.0.0.0 --port $((30000+i)) --nccl-port $((NCCL_BASE+i)) --attention-backend triton --mem-fraction-static 0.85 --disable-radix-cache --disable-cuda-graph \
    --context-length 8208 --max-running-requests 4 --max-total-tokens 65536 > $LOGS/capture-server-$i.log 2>&1 < /dev/null &
  log "capture-server-$i pid $! GPU $i port $((30000+i))"; sleep 20
done
for i in $(seq 0 $((N-1))); do
  for t in $(seq 1 90); do curl -sf -m 3 "http://127.0.0.1:$((30000+i))/health" >/dev/null && break; grep -q -E "Traceback|CUDA out of memory|scheduler died" $LOGS/capture-server-$i.log && { log "capture-server-$i FAILED"; break; }; sleep 10; done
  curl -sf -m 3 "http://127.0.0.1:$((30000+i))/health" >/dev/null && log "capture-server-$i healthy" || log "capture-server-$i NOT healthy"
done
log "CAPTURE_NODE_DONE"
