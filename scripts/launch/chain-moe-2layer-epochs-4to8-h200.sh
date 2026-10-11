#!/usr/bin/env bash
# Epochs 4..8 of the 2-layer Qwen3.8-27B DSpark MoE drafter: wait for phase B (epochs 2-3, final step 9916) to finish,
# then run ONE epoch per run, each a weights-only warm start from the previous run's final checkpoint (step 4958 /
# 9916), reusing the live capture servers (:30000-30003) and Mooncake master. Stop file: /scratch/chain_epochs.stop.
set -u
PFX=qwen3.8-27b-dspark-moe-2layer-regen-mixture-v1
PREV_RUN=$PFX-cont2ep-from-1ep-h200; PREV_STEP=9916; PREV_PAT="qwen38-moe-2layer-cont2ep-h200.yaml --role consumer"
TEMPLATE=/scratch/qwen38-moe-2layer-epN-template.yaml  # repo copy: examples/configs/online/disaggregated/external/qwen3.8-27b-dspark-moe-2layer-regen-mixture-v1-epN-from-epN-1-h200.template.yaml; LOG=/scratch/logs/chain_epochs.log; STOP=/scratch/chain_epochs.stop
FIRST_EP=${FIRST_EP:-4}; LAST_EP=${LAST_EP:-8}
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
[ "${ALLOW_TRAIN:-1}" = 1 ] || { log "training disabled (ALLOW_TRAIN=0)"; exit 0; }

wait_run_done(){  # $1 run  $2 final step  $3 consumer pgrep pattern
  local run=$1 step=$2 pat=$3 ckpt=/scratch/outputs/$1/$1-step$2 n=0 ckpt_ok consumers
  log "waiting for $run final checkpoint (step $step) and trainer exit"
  while true; do
    ckpt_ok=0; if [ -f "$ckpt/training_state.pt" ] && ! ls "$ckpt" 2>/dev/null | grep -q "\.tmp$"; then ckpt_ok=1; fi
    consumers=$(pgrep -fc "$pat" || true)
    if [ "$ckpt_ok" = 1 ] && [ "${consumers:-0}" = 0 ]; then log "$run done: final checkpoint complete and trainer ranks exited"; return 0; fi
    if [ "${consumers:-0}" = 0 ] && [ "$ckpt_ok" = 0 ]; then n=$((n+1)); [ $n -ge 10 ] && { log "ABORT: $run trainer ranks exited but no complete step$step checkpoint"; return 1; }; else n=0; fi
    [ -f "$STOP" ] && { log "stop file seen, exiting"; exit 0; }
    sleep 120
  done
}

launch_epoch(){  # $1 epoch number  $2 previous checkpoint dir
  local ep=$1 prev_ckpt=$2 run=$PFX-ep$1-from-ep$(( $1 - 1 ))-h200 out recipe busy i g u p
  out=/scratch/outputs/$run; recipe=/scratch/qwen38-moe-2layer-ep$ep-h200.yaml
  [ -f "$prev_ckpt/training_state.pt" ] || { log "ABORT: warm-start checkpoint missing: $prev_ckpt"; return 1; }
  sed -e "s|__RUN__|$run|g" -e "s|__PREV_CKPT__|$prev_ckpt|g" -e "s|__PROMPT_SEED__|$(( 42 + ep - 1 ))|g" -e "s|__EP__|$ep|g" "$TEMPLATE" > "$recipe"
  grep -q "__" "$recipe" && { log "ABORT: unreplaced placeholder in $recipe"; return 1; }
  busy=1
  for i in $(seq 1 15); do busy=0; for g in 4 5 6 7; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); [ "$u" -lt 2048 ] || busy=1; done; [ $busy = 0 ] && break; log "trainer GPUs still busy ($i/15), waiting"; sleep 120; done
  [ $busy = 1 ] && { log "ABORT: GPUs 4-7 not freed after 30 min"; return 1; }
  for p in 30000 30001 30002 30003; do curl -sf --max-time 5 http://127.0.0.1:$p/health > /dev/null || { log "ABORT: capture server :$p unhealthy"; return 1; }; done
  (echo > /dev/tcp/127.0.0.1/35551) 2>/dev/null || { log "ABORT: mooncake master :35551 not listening"; return 1; }
  [ -e "$out/control" ] && { log "ABORT: $out/control exists"; return 1; }
  [ -f "$STOP" ] && { log "stop file seen, exiting without launching"; exit 0; }
  mkdir -p "$out"; cd /scratch/SpecForge-qwen38 || return 1
  export WANDB_API_KEY=$(cat /root/.wandb_api_key) WANDB_ENTITY=miles_training WANDB_SILENT=true HF_HOME=/scratch/hf-cache MC_TRANSFER_TIMEOUT=300 OMP_NUM_THREADS=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4,5,6,7
  export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551 MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 MOONCAKE_PROTOCOL=tcp
  setsid nohup numactl --preferred=1 python -u -m specforge.cli train -c "$recipe" > "$out/launch.log" 2>&1 < /dev/null &
  log "launched epoch $ep: $run (warm start $prev_ckpt, prompt_seed $(( 42 + ep - 1 ))), supervisor pid $!, log $out/launch.log"
  sleep 600   # let the supervisor spawn the trainer ranks before we start polling for their exit
  return 0
}

log "chain started (epochs $FIRST_EP..$LAST_EP), previous run $PREV_RUN step $PREV_STEP"
wait_run_done "$PREV_RUN" "$PREV_STEP" "$PREV_PAT" || exit 1
for ep in $(seq "$FIRST_EP" "$LAST_EP"); do
  launch_epoch "$ep" "/scratch/outputs/$PREV_RUN/$PREV_RUN-step$PREV_STEP" || exit 1
  PREV_RUN=$PFX-ep$ep-from-ep$(( ep - 1 ))-h200; PREV_STEP=4958; PREV_PAT="qwen38-moe-2layer-ep$ep-h200.yaml --role consumer"
  wait_run_done "$PREV_RUN" "$PREV_STEP" "$PREV_PAT" || exit 1
done
log "ALL DONE: epochs $FIRST_EP..$LAST_EP finished; last checkpoint /scratch/outputs/$PREV_RUN/$PREV_RUN-step$PREV_STEP"
