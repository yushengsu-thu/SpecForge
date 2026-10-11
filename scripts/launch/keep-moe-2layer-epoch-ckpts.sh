#!/usr/bin/env bash
# Hardlink the FINAL (step 4958) checkpoint of every one-epoch continuation run (…-epN-from-epM-h200) into
# /scratch/ckpt-keep/<run>/ so max_checkpoints rotation can never remove it. Intermediate checkpoints are not kept
# (111 GB each). Stop file: /scratch/keep_ckpts_ep.stop.
set -u
KEEP=/scratch/ckpt-keep; LOG=/scratch/logs/keep_ckpts.log; mkdir -p $KEEP
while true; do
  [ -f /scratch/keep_ckpts_ep.stop ] && exit 0
  for d in /scratch/outputs/qwen3.8-27b-dspark-moe-2layer-regen-mixture-v1-ep*-from-ep*-h200/*-step4958/; do
    [ -d "$d" ] || continue; b=$(basename "$d"); run=$(basename "$(dirname "$d")")
    [ -e "$KEEP/$run/$b" ] && continue
    [ -f "$d/training_state.pt" ] || continue
    ls "$d" | grep -q "\.tmp$" && continue
    mkdir -p "$KEEP/$run" && cp -al "$d" "$KEEP/$run/$b" && echo "[$(date -u +%FT%TZ)] kept $run/$b ($(du -sh "$d" | cut -f1))" >> $LOG
  done
  sleep 300
done
