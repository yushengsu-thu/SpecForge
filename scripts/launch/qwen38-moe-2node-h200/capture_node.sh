#!/bin/bash
# Capture node of the two-node run: N spec-capture SGLang servers (BF16 target), one per GPU, each registering a
# 100 GiB Mooncake segment with the master on the trainer node.
#   usage: bash capture_node.sh <CAPTURE_IP> <TRAINER_IP> [N=8] [PROTOCOL=tcp|rdma]     (MOONCAKE_RDMA_DEVICES optional)
set -u
CAP=${1:?capture ip}; TRN=${2:?trainer ip}; N=${3:-8}; PROTO=${4:-tcp}
TARGET=${TARGET:-/scratch/models/Qwen3.8-27B}
D=/scratch/moe16x1088; mkdir -p $D/logs
VENV=${VENV:-/scratch/moe2ep/venv}; [ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"   # the boxes' sglang 0.5.18 + spec-capture + mooncake env
export PYTHONPATH=/scratch/SpecForge
export HF_HOME=/scratch/hf-cache FLASHINFER_DISABLE_VERSION_CHECK=1
export MOONCAKE_GLOBAL_SEGMENT_SIZE=107374182400 MOONCAKE_LOCAL_BUFFER_SIZE=1073741824
export MOONCAKE_LOCAL_HOSTNAME=$CAP MOONCAKE_MASTER_SERVER_ADDR=$TRN:35551 MOONCAKE_METADATA_SERVER=http://$TRN:35880/metadata MOONCAKE_PROTOCOL=$PROTO
# rdma needs the HCAs listed explicitly (the capture sink refuses an empty MOONCAKE_RDMA_DEVICES): default to every ACTIVE port.
if [ "$PROTO" = rdma ] && [ -z "${MOONCAKE_RDMA_DEVICES:-}" ]; then
  MOONCAKE_RDMA_DEVICES=$(for d in /sys/class/infiniband/*; do grep -q ACTIVE "$d/ports/1/state" 2>/dev/null && basename "$d"; done | paste -sd, -)
fi
export MOONCAKE_RDMA_DEVICES=${MOONCAKE_RDMA_DEVICES:-}
export DISAGG_SERVER_URLS=$(for i in $(seq 0 $((N-1))); do printf "http://%s:%d," "$CAP" $((30000+i)); done | sed 's/,$//')
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
until curl -s -m 3 "http://$TRN:35880/metadata?key=specforge-health-check" 2>/dev/null | grep -q -i -E "not found|value"; do log "waiting for the mooncake master on $TRN"; sleep 10; done
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i setsid nohup python -m sglang.launch_server --model-path "$TARGET" --dtype bfloat16 --trust-remote-code \
    --skip-tokenizer-init --tp-size 1 --chunked-prefill-size -1 --enable-spec-capture --spec-capture-method dspark --spec-capture-aux-layer-ids 5 19 33 47 61 \
    --host 0.0.0.0 --port $((30000+i)) --nccl-port $((37000+i)) --attention-backend triton --mem-fraction-static 0.85 --disable-radix-cache \
    --context-length 8199 --max-running-requests 4 --max-total-tokens 32768 > $D/logs/capture-server-$i.log 2>&1 < /dev/null &
  log "capture-server-$i pid $! GPU $i port $((30000+i)) protocol $PROTO${MOONCAKE_RDMA_DEVICES:+ hcas $MOONCAKE_RDMA_DEVICES}"; sleep 30   # stagger: simultaneous launches race for ports
done
for i in $(seq 0 $((N-1))); do
  for t in $(seq 1 90); do curl -sf -m 3 "http://127.0.0.1:$((30000+i))/health" >/dev/null && break; grep -q -E "Traceback|CUDA out of memory|scheduler died" $D/logs/capture-server-$i.log && { log "capture-server-$i FAILED"; break; }; sleep 10; done
  curl -sf -m 3 "http://127.0.0.1:$((30000+i))/health" >/dev/null && log "capture-server-$i healthy" || log "capture-server-$i NOT healthy"
done
log "CAPTURE_NODE_DONE"
