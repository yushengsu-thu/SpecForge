#!/usr/bin/env bash
# Epochs 3..8 (B step 9916 = epoch 3, one-epoch runs step 4958 = epochs 4-8): when the final checkpoint is kept, export it to the servable HF layout,
# normalize, add tokenizer + README, upload to RadixArk/qwen38-dspark-moe-2layer/<ep>ep (private repo).
set -u
export HF_HOME=/scratch/hf-cache PYTHONUNBUFFERED=1
PFX=qwen3.8-27b-dspark-moe-2layer-regen-mixture-v1
JOBS=("$PFX-cont2ep-from-1ep-h200 9916 3"); for ep in 4 5 6 7 8; do JOBS+=("$PFX-ep$ep-from-ep$((ep-1))-h200 4958 $ep"); done
LOG=/scratch/logs/auto_export_upload.log
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
export_one(){
  local run=$1 step=$2 ep=$3
  local CK=/scratch/ckpt-keep/$run/$run-step$step
  local E=/scratch/exports/qwen38-dspark-moe-2layer-${ep}ep-step$step
  local REPO=RadixArk/qwen38-dspark-moe-2layer
  if [ ! -f "$E/.normalized" ]; then
    rm -rf "$E"; mkdir -p "$E"
    log "exporting $run step $step -> $E"
    (cd /scratch/SpecForge-qwen38 && CUDA_VISIBLE_DEVICES= python -m specforge.cli export --to hf --checkpoint "$CK" \
       --draft-config configs/qwen3.8-27b-dspark-moe-v2-2layer.json --output-dir "$E") >> "$LOG" 2>&1 || { log "EXPORT FAILED $run $step"; return 1; }
    (cd /scratch/SpecForge-qwen38 && CUDA_VISIBLE_DEVICES= python scripts/gates/normalize_dflash_export.py --config "$E/config.json" --block-size 7) >> "$LOG" 2>&1 || { log "NORMALIZE FAILED $run $step"; return 1; }
    python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('/scratch/models/Qwen3.8-27B', trust_remote_code=True).save_pretrained('$E')" >> "$LOG" 2>&1 || log "WARN tokenizer copy failed for $E"
    touch "$E/.normalized"
    log "export ready: $(du -sh "$E" | cut -f1), arch=$(python3 -c "import json;print(json.load(open('$E/config.json'))['architectures'])")"
  fi
  python3 /scratch/make_readme.py "$run" "$step" "$ep" "$E" >> "$LOG" 2>&1 || log "WARN README generation failed for $E"
  log "uploading $E -> $REPO/${ep}ep"
  if hf upload "$REPO" "$E" "${ep}ep" --exclude ".normalized" --exclude ".uploaded" --commit-message "epoch $ep export: $run step $step" >> "$LOG" 2>&1; then
    touch "$E/.uploaded"; log "UPLOADED https://huggingface.co/$REPO/tree/main/${ep}ep"; python3 /scratch/make_top_readme.py > /scratch/hf-upload/README-top.md && hf upload "$REPO" /scratch/hf-upload/README-top.md README.md --commit-message "update epoch table" >> "$LOG" 2>&1
  else
    log "UPLOAD FAILED $REPO ${ep}ep (will retry next loop)"
  fi
}
log "auto export/upload v4 (epochs 4-8) started"
while true; do
  [ -f /scratch/auto_export_upload_v4.stop ] && { log "v4 stop file seen"; exit 0; }
  alldone=1
  for j in "${JOBS[@]}"; do
    set -- $j; run=$1; step=$2; ep=$3
    E=/scratch/exports/qwen38-dspark-moe-2layer-${ep}ep-step$step
    [ -f "$E/.uploaded" ] && continue
    alldone=0
    CK=/scratch/ckpt-keep/$run/$run-step$step
    if [ -f "$CK/training_state.pt" ] && ! ls "$CK" | grep -q "\.tmp$"; then export_one "$run" "$step" "$ep"; fi
  done
  [ "$alldone" = 1 ] && { log "v4: all epoch 4-8 exports uploaded; exiting"; exit 0; }
  sleep 300
done
