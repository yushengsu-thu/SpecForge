#!/bin/bash
# The run's last optimizer step was 64586 (the online schedule's 6-epoch horizon, floor(1,420,903 x 6 / 132)), not the
# 6 x 10,764 = 64,584 the export watcher waited for. Export that final checkpoint exactly as the watcher would and
# upload it as ep6/.
set -u
RUN_ID=qwen3-8b-moe8x1536-8A1-dspark-run1; D=/scratch/moe8b16; RUN_DIR=$D/outputs/$RUN_ID; STEP=64586; label=ep6
HF_REPO=yushengsu/qwen3-8b-moe16x768-16A1-dspark; HF_PREFIX=$RUN_ID/
DRAFT_CONFIG=configs/qwen3-8b-moe8x1536-8A1-dspark.json; TARGET=${TARGET:-/scratch/moe2ep/models/Qwen3-8B}
. ${VENV:-/scratch/moe2ep/venv}/bin/activate
export PYTHONPATH=/scratch/SpecForge HF_HOME=/scratch/moe2ep/hf-home HF_HUB_CACHE=/scratch/moe2ep/hf-home/hub HF_DATASETS_CACHE=/scratch/moe2ep/hf-home/datasets
export HF_TOKEN="${HF_TOKEN:-$(cat /scratch/secrets/hf_token 2>/dev/null)}" CUDA_VISIBLE_DEVICES=
cd /scratch/SpecForge || exit 1
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
ck=$RUN_DIR/$RUN_ID-step$STEP; keep=$D/epoch_ckpts/$RUN_ID-step$STEP; dst=$D/exports/$RUN_ID-$label
[ -f "$ck/training_state.pt" ] || { log "no complete checkpoint at $ck"; exit 1; }
mkdir -p "$D/epoch_ckpts" "$D/exports"
[ -d "$keep" ] || { log "$label: keeping a copy at $keep"; rm -rf "$keep.tmp"; cp -a "$ck" "$keep.tmp" && mv "$keep.tmp" "$keep"; }
log "$label: exporting $keep -> $dst"
rm -rf "$dst"
specforge export --to hf --checkpoint "$keep" --draft-config "$DRAFT_CONFIG" --output-dir "$dst" \
  --embedding-source "$TARGET" --embedding-key model.embed_tokens.weight > "$D/exports/export-$RUN_ID-$label.log" 2>&1 \
  || { log "$label: EXPORT_FAILED"; tail -5 "$D/exports/export-$RUN_ID-$label.log"; exit 1; }
python3 scripts/gates/normalize_dflash_export.py --config "$dst/config.json" --block-size 16 2>&1 | tail -1
python3 -c "import json; p='$dst/config.json'; c=json.load(open(p)); c['architectures']=['Qwen3MoeDSparkModel']; json.dump(c, open(p,'w'), indent=2)"
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$TARGET', trust_remote_code=True).save_pretrained('$dst')" 2>&1 | tail -1
python3 -c "import json; c=json.load(open('$dst/config.json')); print('config:', c['architectures'], 'layers', c['num_hidden_layers'], 'experts', c['num_experts'], 'x', c['moe_intermediate_size'], 'top', c['num_experts_per_tok'], 'block', c.get('block_size'))"
du -sh "$dst" | tail -1
readme="$D/exports/README-$RUN_ID.md"
python3 scripts/launch/qwen3-8b-moe16-1node-h200/make_readme.py --run-dir "$RUN_DIR" --draft-config "$DRAFT_CONFIG" \
  --steps-per-epoch 10764 --save-every 1000 --steps-done "$STEP" --epochs 6 --hf-repo "$HF_REPO" --folder-prefix "$HF_PREFIX" --out "$readme"
log "$label: uploading to $HF_REPO/$HF_PREFIX$label"
hf upload "$HF_REPO" "$dst" "$HF_PREFIX$label" --repo-type model --commit-message "$RUN_ID $label export (step $STEP, final)" > "$D/exports/upload-$RUN_ID-$label.log" 2>&1 \
  && hf upload "$HF_REPO" "$readme" "${HF_PREFIX}README.md" --repo-type model --commit-message "$RUN_ID README after $label" >> "$D/exports/upload-$RUN_ID-$label.log" 2>&1 \
  && { touch "$D/exports/done-$RUN_ID-$label"; log "$label: EPOCH_EXPORT_DONE https://huggingface.co/$HF_REPO/tree/main/$HF_PREFIX$label"; rm -rf "$dst"; } \
  || { log "$label: upload FAILED: $(grep -v -i token "$D/exports/upload-$RUN_ID-$label.log" | tail -1)"; exit 1; }
