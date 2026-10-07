#!/usr/bin/env bash
# Wait for the 1-epoch 2-layer run to finish, then launch the 2-epoch continuation reusing its live servers.
set -u
RUN1=qwen3.8-27b-dspark-moe-2layer-regen-mixture-v1-1ep-h200; OUT1=/scratch/outputs/$RUN1; CKPT=$OUT1/$RUN1-step4958
RUN2=qwen3.8-27b-dspark-moe-2layer-regen-mixture-v1-cont2ep-from-1ep-h200; OUT2=/scratch/outputs/$RUN2
RECIPE=/scratch/qwen38-moe-2layer-cont2ep-h200.yaml; LOG=/scratch/logs/chain_cont2ep.log
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
[ "${ALLOW_TRAIN:-1}" = 1 ] || { log "training disabled by user (ALLOW_TRAIN=0)"; exit 0; }
log "chain started; waiting for $RUN1 final checkpoint and trainer exit"
n=0
while true; do
  ckpt_ok=0; if [ -f "$CKPT/training_state.pt" ] && ! ls "$CKPT" 2>/dev/null | grep -q "\.tmp$"; then ckpt_ok=1; fi
  consumers=$(pgrep -fc "qwen38-moe-2layer-1ep-h200.yaml --role consumer" || true)
  if [ "$ckpt_ok" = 1 ] && [ "${consumers:-0}" = 0 ]; then log "run1 done: final checkpoint complete and trainer ranks exited"; break; fi
  if [ "${consumers:-0}" = 0 ] && [ "$ckpt_ok" = 0 ]; then n=$((n+1)); [ $n -ge 10 ] && { log "ABORT: trainer ranks exited but no complete step4958 checkpoint"; exit 1; }; fi
  [ -f /scratch/chain_cont2ep.stop ] && { log "stop file seen, exiting without launching"; exit 0; }
  sleep 120
done
busy=1
for i in $(seq 1 15); do busy=0; for g in 4 5 6 7; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); [ "$u" -lt 2048 ] || busy=1; done; [ $busy = 0 ] && break; log "trainer GPUs still busy ($i/15), waiting"; sleep 120; done
[ $busy = 1 ] && { log "ABORT: GPUs 4-7 not freed after 30 min"; exit 1; }
for p in 30000 30001 30002 30003; do curl -sf --max-time 5 http://127.0.0.1:$p/health > /dev/null || { log "ABORT: capture server :$p unhealthy"; exit 1; }; done
(echo > /dev/tcp/127.0.0.1/35551) 2>/dev/null || { log "ABORT: mooncake master :35551 not listening"; exit 1; }
[ -e "$OUT2/control" ] && { log "ABORT: $OUT2/control exists"; exit 1; }
[ -f /scratch/chain_cont2ep.stop ] && { log "stop file seen, exiting without launching"; exit 0; }
mkdir -p "$OUT2"; cd /scratch/SpecForge-qwen38
export HF_HOME=/scratch/hf-cache MC_TRANSFER_TIMEOUT=300 OMP_NUM_THREADS=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4,5,6,7
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551 MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 MOONCAKE_PROTOCOL=tcp
setsid nohup numactl --preferred=1 python -u -m specforge.cli train -c "$RECIPE" > "$OUT2/launch.log" 2>&1 < /dev/null &
log "launched $RUN2, supervisor pid $!, log $OUT2/launch.log"
