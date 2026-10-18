#!/usr/bin/env bash
# Per-epoch export / upload / report loop for the 12-epoch run. Run on the training box:
#   nohup experiments/qwen3-8b-dspark-moe-8expert-1activate/epoch_watcher.sh >/dev/null 2>&1 &
# Every 5 minutes, for each epoch k whose checkpoint (step k*EPOCH_STEPS) is complete:
#   1. copy it to $KEEP (out of the trainer's max_checkpoints rotation),
#   2. export it to the HF layout (+ target tokenizer) under $EXPORTS/<k>ep,
#   3. upload that folder to $HF_REPO/<k>ep,
#   4. regenerate exp_read.me (epoch_report.py), commit it on this branch and push the same
#      text as the HF repo README.
# Stop file: $STOP. Log: $LOG. Everything is idempotent (.exported / .uploaded markers).
set -u
REPO=${REPO:-/scratch/SpecForge-spark8b}
RUN=${RUN:-qwen3-8b-dspark-moe-8expert-1activate-dense-same}
OUT=${OUT:-/scratch/outputs/$RUN}
KEEP=${KEEP:-/scratch/ckpt-keep/$RUN}
EXPORTS=${EXPORTS:-/scratch/exports/$RUN}
HF_REPO=${HF_REPO:-yushengsu/qwen3-8b-dspark-moe-8expert-1activate}
EPOCH_STEPS=${EPOCH_STEPS:-10764}   # checkpoint step of epoch 1; epoch k <-> step k*EPOCH_STEPS
NUM_EPOCHS=${NUM_EPOCHS:-12}       # epochs in THIS run
EPOCH_OFFSET=${EPOCH_OFFSET:-0}    # a continuation run labels its epoch k as k+EPOCH_OFFSET (13ep, ...)
REPORT_ARGS=${REPORT_ARGS:-}       # extra epoch_report.py arguments (continuation segment)
REPORT_RUN=${REPORT_RUN:-}; REPORT_OUT=${REPORT_OUT:-}  # first run of the experiment (the report is always built from it)
TARGET=${TARGET:-/scratch/models/Qwen3-8B}
DRAFT_CONFIG=configs/qwen3-8b-dspark-moe-8x1536-top1-2layer-causal-noconf-b16.json
STOP=${STOP:-$OUT/watcher.stop}
LOG=${LOG:-/scratch/logs/$RUN.watcher.log}
GIT_NAME=${GIT_NAME:-Yusheng Su}; GIT_EMAIL=${GIT_EMAIL:-yushengsu.thu@gmail.com}
REPORT=experiments/qwen3-8b-dspark-moe-8expert-1activate/epoch_report.py
# The box's /etc/environment points HF_HOME at the read-only /cluster-storage; force the writable cache.
export HF_HOME=/scratch/hf-cache HF_DATASETS_CACHE=/scratch/hf-cache/datasets PYTHONUNBUFFERED=1 PYTHONPATH=$REPO
if [ -f /root/.secrets/wandb.env ]; then set -a; . /root/.secrets/wandb.env; set +a; fi
mkdir -p "$(dirname "$LOG")" "$KEEP" "$EXPORTS"
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
complete(){ [ -f "$1/training_state.pt" ] && ! ls "$1" 2>/dev/null | grep -q "\.tmp$"; }

process_epoch(){  # $1 = epoch number within this run; returns 1 when not ready yet or a step failed
  local k=$(( $1 + EPOCH_OFFSET )) step=$(( $1 * EPOCH_STEPS )) ck keep ex
  ck=$OUT/$RUN-step$step; keep=$KEEP/$RUN-step$step; ex=$EXPORTS/${k}ep
  if ! complete "$keep"; then
    complete "$ck" || return 1
    rm -rf "$keep.tmp"
    if cp -a "$ck" "$keep.tmp" && mv "$keep.tmp" "$keep"; then log "epoch $k: checkpoint step $step kept at $keep"
    else log "epoch $k: COPY FAILED"; rm -rf "$keep.tmp"; return 1; fi
  fi
  if [ ! -f "$ex/.exported" ]; then
    rm -rf "$ex"; mkdir -p "$ex"; log "epoch $k: exporting step $step -> $ex"
    (cd "$REPO" && CUDA_VISIBLE_DEVICES= python -m specforge.cli export --to hf --checkpoint "$keep" \
       --draft-config "$DRAFT_CONFIG" --output-dir "$ex" --embedding-source "$TARGET" \
       --embedding-key model.embed_tokens.weight) >> "$LOG" 2>&1 || { log "epoch $k: EXPORT FAILED"; return 1; }
    python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$TARGET').save_pretrained('$ex')" >> "$LOG" 2>&1 || log "epoch $k: WARN tokenizer copy failed"
    touch "$ex/.exported"; log "epoch $k: export ready ($(du -sh "$ex" | cut -f1))"
  fi
  if [ ! -f "$ex/.uploaded" ]; then
    log "epoch $k: uploading -> $HF_REPO/${k}ep"
    if hf upload "$HF_REPO" "$ex" "${k}ep" --exclude ".exported" --exclude ".uploaded" \
         --commit-message "epoch $k export ($RUN step $step)" >> "$LOG" 2>&1; then
      touch "$ex/.uploaded"; log "epoch $k: UPLOADED https://huggingface.co/$HF_REPO/tree/main/${k}ep"
    else log "epoch $k: UPLOAD FAILED (retry next loop)"; return 1; fi
  fi
  return 0
}

report(){  # regenerate exp_read.me, commit when it changed, mirror it as the HF README
  (cd "$REPO" && python3 "$REPORT" --run "${REPORT_RUN:-$RUN}" --out "${REPORT_OUT:-$OUT}" --exports "$EXPORTS" --epoch-steps "$EPOCH_STEPS" \
     --num-epochs "$(( NUM_EPOCHS + EPOCH_OFFSET ))" --hf-repo "$HF_REPO" $REPORT_ARGS --write exp_read.me) >> "$LOG" 2>&1 || { log "REPORT FAILED"; return 1; }
  (cd "$REPO" && if ! git diff --quiet -- exp_read.me; then
     git add exp_read.me && git -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL" commit -q -F - <<MSG && echo committed
exp_read.me: results update ($1)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
   fi) >> "$LOG" 2>&1
  hf upload "$HF_REPO" "$REPO/exp_read.me" README.md --commit-message "README: $1" >> "$LOG" 2>&1 || log "WARN README upload failed"
}

log "watcher started: run=$RUN epoch_steps=$EPOCH_STEPS epochs=$NUM_EPOCHS offset=$EPOCH_OFFSET hf=$HF_REPO"
first=1
while true; do
  [ -f "$STOP" ] && { log "stop file seen, exiting"; exit 0; }
  changed=0; done_epochs=0
  for k in $(seq 1 "$NUM_EPOCHS"); do
    if [ -f "$EXPORTS/$(( k + EPOCH_OFFSET ))ep/.uploaded" ]; then done_epochs=$((done_epochs+1)); continue; fi
    if process_epoch "$k"; then changed=1; done_epochs=$((done_epochs+1)); else break; fi
  done
  if [ "$changed" = 1 ]; then report "epoch $(( done_epochs + EPOCH_OFFSET ))"; elif [ "$first" = 1 ]; then report "launch"; fi
  first=0
  if [ "$done_epochs" -ge "$NUM_EPOCHS" ]; then log "all $NUM_EPOCHS epochs exported and uploaded; exiting"; exit 0; fi
  sleep 300
done
