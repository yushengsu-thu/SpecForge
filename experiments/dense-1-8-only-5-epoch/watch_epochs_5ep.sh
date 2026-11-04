#!/bin/bash
# Devbox-only: every 5 min, for each finished epoch checkpoint (step k*10764, k=1..EPOCHS): export to the HF layout with the
# target embedding, normalize the config for SGLang, upload to yushengsu/Qwen3-8B-2layer-Dense-v2/<k>ep (renamed 2026-11-04 from dense-1-8-only-5-epoch), refresh the model
# card. Idempotent (.exported / .uploaded markers). Exits after the last epoch is uploaded, or if the trainer dies.
# Usage: RUN=<run_id> DRAFT=<draft config json> [EPOCHS=5] bash watch_epochs_5ep.sh
W=/scratch/pr1003; RUN=${RUN:?}; DRAFT=${DRAFT:?}; EPOCHS=${EPOCHS:-5}; OUT=$W/outputs/$RUN; EX=$W/exports/$RUN
TARGET=/scratch/models/Qwen3-8B
cd $W/SpecForge && . $W/venv/bin/activate
export HF_HOME=/scratch/hf-cache PYTHONUNBUFFERED=1
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
complete(){ [ -f "$1/training_state.pt" ] && ! ls "$1" 2>/dev/null | grep -q "\.tmp$"; }
log "watcher started: run $RUN, $EPOCHS epochs, draft $DRAFT"
while true; do
  done_n=0
  for k in $(seq 1 $EPOCHS); do
    ck=$OUT/$RUN-step$((k * 10764)); ex=$EX/${k}ep
    if [ -f $ex/.uploaded ]; then done_n=$((done_n + 1)); continue; fi
    complete $ck || continue
    if [ ! -f $ex/.exported ]; then
      rm -rf $ex; mkdir -p $EX; log "epoch $k: export $ck"
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m specforge.cli export --to hf --checkpoint $ck --draft-config $DRAFT \
        --output-dir $ex --embedding-source $TARGET > $EX/export_${k}ep.log 2>&1 || { log "epoch $k: EXPORT FAILED"; tail -3 $EX/export_${k}ep.log; continue; }
      HF_HUB_OFFLINE=1 python scripts/gates/normalize_dflash_export.py --config $ex/config.json --block-size 16 > $EX/normalize_${k}ep.log 2>&1 \
        || { log "epoch $k: NORMALIZE FAILED"; tail -3 $EX/normalize_${k}ep.log; continue; }
      touch $ex/.exported; log "epoch $k: export ready ($(du -sh $ex | cut -f1))"
    fi
    log "epoch $k: uploading"
    if RUN=$RUN DRAFT=$DRAFT EPOCHS=$EPOCHS python $W/upload_epoch_5ep.py $k $ex >> $EX/upload_${k}ep.log 2>&1; then touch $ex/.uploaded; log "epoch $k: UPLOADED"; tail -2 $EX/upload_${k}ep.log
    else log "epoch $k: UPLOAD FAILED (retry next loop)"; tail -3 $EX/upload_${k}ep.log; fi
  done
  [ $done_n -ge $EPOCHS ] && { log "all $EPOCHS epochs uploaded; exiting"; exit 0; }
  if ! pgrep -f "specforge.cli train --config $W/train_$RUN.yaml" > /dev/null && [ $done_n -lt $EPOCHS ]; then
    complete $OUT/$RUN-step$((EPOCHS * 10764)) || { log "trainer gone before the epoch $EPOCHS checkpoint; exiting"; exit 1; }
  fi
  sleep 300
done
