#!/usr/bin/env bash
# Export the checkpoints of a Qwen3-8B MoE drafter run as they appear: by default the epoch-end ones (steps 10764 * k,
# uploaded to folder ep<k>/), or with SAVE_EVERY=<n> every interval checkpoint (steps n, 2n, ..., uploaded to
# step-<N>/) plus the recipe's SAVE_STEPS and the trainer's last step, which are epoch ends and go to ep<k>/ too.
# Each export keeps a copy of the raw checkpoint under /scratch/moe8b16/epoch_ckpts/ (all of them in epoch
# mode, the last three in interval mode; the trainer rotates its own after max_checkpoints), converts it with
# `specforge export`, normalises the config for SGLang (block 16 as jiapingW serves it, architectures
# Qwen3MoeDSparkModel so SGLang picks the MoE draft class), adds the target tokenizer and uploads it under $HF_PREFIX
# of the public repo $HF_REPO together with a refreshed README.
# A checkpoint counts as complete once the trainer has repointed <run_id>-latest at it (CheckpointManager.save does
# that after every rank wrote its state).
#   setsid nohup bash watch_epoch_exports.sh > /scratch/moe8b16/logs/watch_epoch_exports.log 2>&1 < /dev/null &
#   RUN_ID=<run> DRAFT_CONFIG=configs/qwen3-8b-moe8x1536-8A1-dspark.json SAVE_EVERY=1000 HF_PREFIX=<run>/ bash ...
set -u
RUN_ID=${RUN_ID:-qwen3-8b-moe16x768-16A1-dspark-2server-dp6-h200}
D=/scratch/moe8b16; RUN_DIR=$D/outputs/$RUN_ID
VENV=${VENV:-/scratch/moe2ep/venv}; [ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"   # the boxes' sglang 0.5.18 + spec-capture + mooncake env
export PYTHONPATH=/scratch/SpecForge
DRAFT_CONFIG=${DRAFT_CONFIG:-configs/qwen3-8b-moe16x768-16A1-dspark.json}
TARGET=${TARGET:-/scratch/moe2ep/models/Qwen3-8B}
STEPS_PER_EPOCH=10764; EPOCHS=${EPOCHS:-6}; EPOCH_OFFSET=${EPOCH_OFFSET:-0}   # EPOCH_OFFSET: epochs finished by earlier phases (warm-start continuation)
SAVE_EVERY=${SAVE_EVERY:-$STEPS_PER_EPOCH}                                    # the recipe's save_interval when it is not an epoch
LAST_STEP=$(( EPOCHS * STEPS_PER_EPOCH / SAVE_EVERY * SAVE_EVERY ))            # the last interval hit
# The trainer also saves its last step (trainer.py), grid or not. The online schedule's horizon is
# floor(num_prompts * epochs / global batch) = 64,586 for 1,420,903 prompts x 6 / 132, not EPOCHS * STEPS_PER_EPOCH
# (64,584): pass FINAL_STEP explicitly.
FINAL_STEP=${FINAL_STEP:-$(( EPOCHS * STEPS_PER_EPOCH ))}
SAVE_STEPS=${SAVE_STEPS:-}                                                     # the recipe's training.save_steps (epoch ends), space-separated
KEEP_RAW=${KEEP_RAW:-1}                                                        # raw copies of interval checkpoints kept (epoch-end copies are kept for good)
HF_REPO=${HF_REPO:-yushengsu/qwen3-8b-moe16x768-16A1-dspark}; HF_PREFIX=${HF_PREFIX:-}   # HF_PREFIX=<folder>/ nests everything, README included
export HF_HOME=/scratch/moe2ep/hf-home HF_HUB_CACHE=/scratch/moe2ep/hf-home/hub HF_DATASETS_CACHE=/scratch/moe2ep/hf-home/datasets
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
epoch_mode=0; [ "$SAVE_EVERY" -eq "$STEPS_PER_EPOCH" ] && epoch_mode=1
# Every step the trainer checkpoints: the interval grid, the recipe's save_steps and its last step, in order.
steps=$(printf '%s\n' $(seq "$SAVE_EVERY" "$SAVE_EVERY" "$LAST_STEP") $SAVE_STEPS "$FINAL_STEP" | sort -n | uniq)
for step in $steps; do
  if [ "$step" -eq "$FINAL_STEP" ]; then label="ep$((EPOCHS + EPOCH_OFFSET))"                        # the last step = the last epoch's end
  elif [ $((step % STEPS_PER_EPOCH)) -eq 0 ]; then label="ep$((step / STEPS_PER_EPOCH + EPOCH_OFFSET))"   # an epoch end
  else label="step-$step"; fi
  # A restarted watcher skips what it already uploaded (the trainer rotates old checkpoints away).
  done_marker="$D/exports/done-$RUN_ID-$label"
  [ -f "$done_marker" ] && { log "$label: already uploaded, skipping"; continue; }
  ck="$RUN_DIR/$RUN_ID-step$step"; keep="$D/epoch_ckpts/$RUN_ID-step$step"; dst="$D/exports/$RUN_ID-$label"
  kind=""; [ "$step" -eq "$FINAL_STEP" ] && kind=final
  log "$label: waiting for a complete $ck"
  until complete "$ck" "$kind"; do sleep 120; done
  sleep 60
  mkdir -p "$D/epoch_ckpts" "$D/exports"
  [ -d "$keep" ] || { log "$label: keeping a copy at $keep"; rm -rf "$keep.tmp"; cp -a "$ck" "$keep.tmp" && mv "$keep.tmp" "$keep"; }
  [ -f "$keep/training_state.pt" ] || { log "$label: COPY_FAILED ($keep has no training_state.pt)"; continue; }
  log "$label: exporting $keep -> $dst"
  exported=0
  for etry in 1 2 3; do
    rm -rf "$dst"
    specforge export --to hf --checkpoint "$keep" --draft-config "$DRAFT_CONFIG" --output-dir "$dst" \
      --embedding-source "$TARGET" --embedding-key model.embed_tokens.weight > "$D/exports/export-$RUN_ID-$label.log" 2>&1 && { exported=1; break; }
    log "$label: export attempt $etry failed: $(tail -1 "$D/exports/export-$RUN_ID-$label.log" | cut -c1-200)"; sleep 300
  done
  [ "$exported" = 1 ] || { log "$label: EXPORT_FAILED (see $D/exports/export-$RUN_ID-$label.log)"; continue; }
  python3 scripts/gates/normalize_dflash_export.py --config "$dst/config.json" --block-size 16 2>&1 | tail -1
  # SGLang picks the draft class by this name; the stock DSparkDraftModel name would load the dense class and drop the experts.
  python3 -c "import json; p='$dst/config.json'; c=json.load(open(p)); c['architectures']=['Qwen3MoeDSparkModel']; json.dump(c, open(p,'w'), indent=2)"
  python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$TARGET', trust_remote_code=True).save_pretrained('$dst')" 2>&1 | tail -1
  python3 -c "import json; c=json.load(open('$dst/config.json')); print('config:', c['architectures'], 'layers', c['num_hidden_layers'], 'experts', c['num_experts'], 'x', c['moe_intermediate_size'], 'top', c['num_experts_per_tok'], 'block', c.get('block_size'))"
  du -sh "$dst" | tail -1
  readme="$D/exports/README-$RUN_ID.md"
  python3 scripts/launch/qwen3-8b-moe16-1node-h200/make_readme.py --run-dir "$RUN_DIR" --draft-config "$DRAFT_CONFIG" \
    --steps-per-epoch "$STEPS_PER_EPOCH" --save-every "$SAVE_EVERY" --save-steps "$SAVE_STEPS" --steps-done "$step" --epochs "$((EPOCHS + EPOCH_OFFSET))" --final-step "$FINAL_STEP" \
    --hf-repo "$HF_REPO" --folder-prefix "$HF_PREFIX" --out "$readme"
  log "$label: uploading to $HF_REPO/$HF_PREFIX$label"
  for try in 1 2 3 4 5; do
    if hf upload "$HF_REPO" "$dst" "$HF_PREFIX$label" --repo-type model --commit-message "$RUN_ID $label export" > "$D/exports/upload-$RUN_ID-$label.log" 2>&1 \
       && hf upload "$HF_REPO" "$readme" "${HF_PREFIX}README.md" --repo-type model --commit-message "$RUN_ID README after $label" >> "$D/exports/upload-$RUN_ID-$label.log" 2>&1; then
      log "$label: EPOCH_EXPORT_DONE https://huggingface.co/$HF_REPO/tree/main/$HF_PREFIX$label"
      touch "$done_marker"
      if [ "$epoch_mode" = 0 ]; then   # interval mode: the upload is the archive; keep the epoch-end raw copies, the last $KEEP_RAW interval ones, no export dirs
        rm -rf "$dst"
        for old in $(ls -d "$D/epoch_ckpts/$RUN_ID-step"* 2>/dev/null | sed "s|.*-step||" | awk -v e="$STEPS_PER_EPOCH" '$1 % e != 0' | sort -n | head -n -"$KEEP_RAW"); do rm -rf "$D/epoch_ckpts/$RUN_ID-step$old"; done
      fi
      break
    fi
    log "$label: upload attempt $try failed: $(grep -v -i token "$D/exports/upload-$RUN_ID-$label.log" | tail -1)"; sleep 600
  done
done
log ALL_EPOCHS_EXPORTED
