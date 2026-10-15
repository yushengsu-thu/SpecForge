#!/usr/bin/env bash
# dense-same: export every epoch-end checkpoint to HF format, upload it to yushengsu/dense-same/<ep>ep/, record
# the epoch's metrics in /scratch/dense/epoch_results.json and refresh the HF README. Epoch k ends at optimizer
# step k * STEPS_PER_EPOCH (= training.save_interval); epoch 12 takes the final checkpoint once the trainer exits.
# usage: watch_exports.sh [FIRST_EPOCH=1] [LAST_EPOCH=12]
set -u
FIRST=${1:-1}; LAST=${2:-12}; TOTAL_EPOCHS=12; STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-11100}
RUN_ID=dense-same; BASE=/scratch/dense; OUT=$BASE/outputs/$RUN_ID; EXPORTS=$BASE/exports; RESULTS=$BASE/epoch_results.json
REPO=yushengsu/dense-same; DRAFT_CONFIG=configs/qwen3-8b-dspark-noconf-b16.json; TARGET=/scratch/models/Qwen3-8B
TOOLS=scripts/launch/dense-same/epoch_results.py
export HF_HOME=/scratch/hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HF_TOKEN="$(cat $BASE/hf_token)" CUDA_VISIBLE_DEVICES=
cd $BASE/SpecForge || exit 1
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
trainer_alive() { pgrep -f "specforge.cli trai[n]" >/dev/null; }
latest_target() { local p=$OUT/$RUN_ID-latest; if [ -L "$p" ]; then readlink -f "$p"; elif [ -f "$p" ]; then readlink -f "$(cat "$p")"; fi; }
latest_step() { local t; t=$(latest_target); echo "${t##*-step}"; }
complete() { [ -d "$1" ] && [ -f "$1/training_state.pt" ] || return 1; [ "$(latest_target)" = "$(readlink -f "$1")" ] && return 0; [ "$(latest_step)" -gt "$2" ] 2>/dev/null && return 0; ! trainer_alive && return 0; return 1; }
for ep in $(seq "$FIRST" "$LAST"); do
  step=$((ep * STEPS_PER_EPOCH)); ck=$OUT/$RUN_ID-step$step; dst=$EXPORTS/$RUN_ID-ep$ep
  if [ "$ep" -eq "$TOTAL_EPOCHS" ]; then
    log "epoch $ep: waiting for the trainer to finish (final checkpoint)"
    while trainer_alive; do sleep 120; done; sleep 60
    ck=$(latest_target); step=$(latest_step); log "epoch $ep: final checkpoint $ck"
  else
    log "epoch $ep (step $step): waiting for a complete $ck"
    until complete "$ck" "$step"; do sleep 120; done; sleep 60
  fi
  log "epoch $ep: exporting -> $dst"; rm -rf "$dst"; mkdir -p "$EXPORTS"
  specforge export --to hf --checkpoint "$ck" --draft-config "$DRAFT_CONFIG" --output-dir "$dst" --embedding-source "$TARGET" > "$EXPORTS/export-ep$ep.log" 2>&1 || { log "epoch $ep: EXPORT_FAILED"; tail -5 "$EXPORTS/export-ep$ep.log"; continue; }
  python3 scripts/gates/normalize_dflash_export.py --config "$dst/config.json" --block-size 16 2>&1 | tail -1
  python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$TARGET').save_pretrained('$dst')" 2>&1 | tail -1
  du -sh "$dst" | tail -1
  for try in 1 2 3 4 5 6; do
    if hf upload "$REPO" "$dst" "${ep}ep" --private --repo-type model --commit-message "epoch $ep export (step $step)" > "$EXPORTS/upload-ep$ep.log" 2>&1; then
      url="https://huggingface.co/$REPO/tree/main/${ep}ep"; log "epoch $ep: EPOCH_EXPORT_DONE $url"
      python3 $TOOLS record --results "$RESULTS" --launch-log "$OUT/launch.log" --epoch "$ep" --step "$step" --steps-per-epoch "$STEPS_PER_EPOCH" --hf-url "$url" 2>&1 | tail -1
      python3 $TOOLS hf-readme --results "$RESULTS" --out "$EXPORTS/README.md" && hf upload "$REPO" "$EXPORTS/README.md" README.md --repo-type model --commit-message "README: epoch $ep" > /dev/null 2>&1 && log "epoch $ep: HF README updated"
      break
    fi
    log "epoch $ep: upload attempt $try failed"; sleep 600
  done
done
log "ALL_EPOCHS_EXPORTED ($RUN_ID)"
