#!/usr/bin/env bash
# Devbox-only: every 5 min, for each finished checkpoint (optimizer step N): export it to the servable HF layout with the
# target embedding, normalize the config for SGLang (block size 16), upload it to
# yushengsu/new-repo-qwen3-8b-moe16x768-16A1-dspark/<run>/step-N (and <run>/epK when N is an epoch end), verify size and
# sha256, then delete the export and every older raw checkpoint (the newest raw checkpoint stays on disk). Idempotent via
# marker files under /scratch/pr1031/uploaded/<run>. Exits after the final step (64,584) is uploaded, or if the trainer dies.
set -u
W=/scratch/pr1031; RUN=pr1031-qwen3-8b-moe8x1536-8A1-dspark-run1; E=$W/SpecForge/experiments/$RUN
OUT=$W/outputs/$RUN; EX=$W/exports/$RUN; DONE=$W/uploaded/$RUN; mkdir -p $EX $DONE
DRAFT=$W/SpecForge/configs/qwen3-8b-moe8x1536-8A1-dspark.json; TARGET=/scratch/models/Qwen3-8B
STEPS_PER_EPOCH=10764; LAST=64584
cd $W/SpecForge && . $W/venv/bin/activate
export HF_HOME=/scratch/hf-cache PYTHONUNBUFFERED=1
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
complete(){ [ -f "$1/training_state.pt" ] && ! ls "$1" 2>/dev/null | grep -q "\.tmp$"; }
steps(){ ls -d $OUT/$RUN-step[0-9]* 2>/dev/null | sed "s/.*-step//" | sort -n; }
trainer_alive(){ pgrep -f "specforge.cli train --config $E/train.yaml" > /dev/null; }
log "watcher started"
while true; do
  for n in $(steps); do
    ck=$OUT/$RUN-step$n; ex=$EX/step-$n
    [ -f $DONE/step-$n ] && continue
    complete $ck || continue
    if [ ! -f $ex/.exported ]; then
      rm -rf $ex; mkdir -p $ex; log "step $n: export $ck"
      CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m specforge.cli export --to hf --checkpoint $ck \
        --draft-config $DRAFT --output-dir $ex --embedding-source $TARGET > $EX/export_$n.log 2>&1 \
        || { log "step $n: EXPORT FAILED"; tail -3 $EX/export_$n.log; continue; }
      CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python scripts/gates/normalize_dflash_export.py --config $ex/config.json --block-size 16 \
        > $EX/normalize_$n.log 2>&1 || { log "step $n: NORMALIZE FAILED"; tail -3 $EX/normalize_$n.log; continue; }
      touch $ex/.exported; log "step $n: export ready ($(du -sh $ex | cut -f1))"
    fi
    ep=""; [ $((n % STEPS_PER_EPOCH)) -eq 0 ] && ep=$((n / STEPS_PER_EPOCH))
    log "step $n: uploading${ep:+ (epoch $ep)}"
    if python $E/upload_ckpt.py $n $ex "$ep" >> $EX/upload_$n.log 2>&1; then
      touch $DONE/step-$n; log "step $n: UPLOADED"; tail -2 $EX/upload_$n.log
      rm -rf $ex $EX/export_$n.log $EX/normalize_$n.log
      for m in $(steps); do
        [ $m -lt $n ] && [ -f $DONE/step-$m ] && { rm -rf $OUT/$RUN-step$m; log "step $m: local checkpoint removed"; }
      done
    else
      log "step $n: UPLOAD FAILED (retry next loop)"; tail -3 $EX/upload_$n.log
    fi
  done
  [ -f $DONE/step-$LAST ] && { log "final step uploaded; exiting"; exit 0; }
  if ! trainer_alive; then
    complete $OUT/$RUN-step$LAST || { log "trainer gone before the final checkpoint; exiting"; exit 1; }
  fi
  sleep 300
done
