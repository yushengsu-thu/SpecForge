#!/usr/bin/env bash
# Export each epoch-end checkpoint (steps 2479 / 4958 / 7437) of the dense 5-layer run: keep a copy of the raw
# checkpoint under /scratch/dense5l/epoch_ckpts/ (the trainer only keeps the last 5), convert it with
# `specforge export`, normalise the config for SGLang (block 7), add the target tokenizer and upload it to folder
# <k>ep/ of the PRIVATE repo RadixArk/qwen38-dspark-dense-5layer together with a refreshed README.
# A checkpoint counts as complete once the trainer has repointed <run_id>-latest at it (CheckpointManager.save does
# that after every rank wrote its state).
#   setsid nohup bash watch_epoch_exports.sh > /scratch/dense5l/logs/watch_epoch_exports.log 2>&1 < /dev/null &
set -u
RUN_ID=${RUN_ID:-qwen3.8-27b-dspark-dense-5layer-3ep-gb300}   # two-node run: RUN_ID=qwen3.8-27b-dspark-dense-5layer-3ep-2node-gb300
D=/scratch/dense5l; RUN_DIR=$D/outputs/$RUN_ID
DRAFT_CONFIG=configs/qwen3.8-27b-dspark.json
TARGET=/scratch/models/Qwen3.8-27B-NVFP4-BF16-LMHead
STEPS_PER_EPOCH=2479; EPOCHS=3
HF_REPO=RadixArk/qwen38-dspark-dense-5layer
export HF_HOME=/scratch/hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HF_DATASETS_CACHE=/scratch/hf-cache/datasets
export HF_TOKEN="$(cat /scratch/secrets/hf_token)"
export CUDA_VISIBLE_DEVICES=
cd /scratch/SpecForge || exit 1
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
latest_target() {
  local p=$RUN_DIR/$RUN_ID-latest
  if [ -L "$p" ]; then readlink -f "$p"; elif [ -f "$p" ]; then readlink -f "$(cat "$p")"; fi
}
complete() {
  [ -d "$1" ] && [ -f "$1/training_state.pt" ] || return 1
  [ "$(latest_target)" = "$(readlink -f "$1")" ] && return 0
  # final step: the trainer exits right after saving; accept the dir once no trainer is left
  [ "$2" = final ] && ! pgrep -f "specforge.cli trai[n]" >/dev/null && return 0
  return 1
}
for k in $(seq 1 "$EPOCHS"); do
  step=$((k * STEPS_PER_EPOCH)); ck="$RUN_DIR/$RUN_ID-step$step"; keep="$D/epoch_ckpts/$RUN_ID-step$step"; dst="$D/exports/$RUN_ID-${k}ep"
  kind=""; [ "$k" -eq "$EPOCHS" ] && kind=final
  log "epoch $k: waiting for a complete $ck"
  until complete "$ck" "$kind"; do sleep 120; done
  sleep 60
  [ -d "$keep" ] || { log "epoch $k: keeping a copy at $keep"; rm -rf "$keep.tmp"; cp -a "$ck" "$keep.tmp" && mv "$keep.tmp" "$keep"; }
  log "epoch $k: exporting $keep -> $dst"
  rm -rf "$dst"; mkdir -p "$D/exports"
  specforge export --to hf --checkpoint "$keep" --draft-config "$DRAFT_CONFIG" --output-dir "$dst" > "$D/exports/export-${k}ep.log" 2>&1 || { log "epoch $k: EXPORT_FAILED (see $D/exports/export-${k}ep.log)"; tail -5 "$D/exports/export-${k}ep.log"; continue; }
  python3 scripts/gates/normalize_dflash_export.py --config "$dst/config.json" --block-size 7 2>&1 | tail -1
  python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$TARGET', trust_remote_code=True).save_pretrained('$dst')" 2>&1 | tail -1
  python3 -c "import json; c=json.load(open('$dst/config.json')); print('config:', c['architectures'], 'layers', c['num_hidden_layers'], 'ffn', c['intermediate_size'], 'block', c.get('block_size'))"
  du -sh "$dst" | tail -1
  python3 scripts/launch/qwen38-dense5l-gb300/make_readme.py --run-dir "$RUN_DIR" --steps-per-epoch "$STEPS_PER_EPOCH" --epochs-done "$k" --out "$D/exports/README.md"
  log "epoch $k: uploading to $HF_REPO/${k}ep (private)"
  for try in 1 2 3 4 5; do
    if hf upload "$HF_REPO" "$dst" "${k}ep" --repo-type model --commit-message "epoch $k export (step $step)" > "$D/exports/upload-${k}ep.log" 2>&1 \
       && hf upload "$HF_REPO" "$D/exports/README.md" README.md --repo-type model --commit-message "README after epoch $k" >> "$D/exports/upload-${k}ep.log" 2>&1; then
      log "epoch $k: EPOCH_EXPORT_DONE https://huggingface.co/$HF_REPO/tree/main/${k}ep"; break
    fi
    log "epoch $k: upload attempt $try failed: $(grep -v -i token "$D/exports/upload-${k}ep.log" | tail -1)"; sleep 600
  done
done
log ALL_EPOCHS_EXPORTED
