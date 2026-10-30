#!/bin/bash
# Devbox setup after a reprovision: PR #47181 from the patch in /personal, SpecForge's benchmark CLI installed with
# --no-deps (its pins would replace the image's torch/transformers/flashinfer), then the models. Marker: logs/SETUP2_DONE.
set -u
export PATH=/opt/sglang/bin:/usr/local/cuda/bin:$PATH
P=/personal/moe-serving; S=/scratch/moe-serving; mkdir -p $P/logs $S/hf
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
rm -f $P/logs/SETUP2_DONE
vers() { pip list 2>/dev/null | grep -iE "^(torch |transformers |flashinfer-python |sglang |sglang-kernel )" | tr -s " " | tr "\n" ";"; echo; }
log "image versions: $(vers)"
git config --global --add safe.directory '*'
cd /sgl-workspace/sglang || { log "FAIL: no sglang checkout"; exit 1; }
if git log --oneline -3 | grep -q "Serve MoE FFN drafts"; then log "PR already applied"; else
  git -c user.name=devbox -c user.email=devbox@local am -3 $P/pr47181.patch >/dev/null 2>&1 && log "PR applied: $(git log --oneline -1)" || { git am --abort 2>/dev/null; log "FAIL: git am"; }
fi
pip install -q --no-deps -e $P/SpecForge > $P/logs/pip_specforge2.log 2>&1 || log "FAIL: pip --no-deps"
for m in click datasets tqdm pydantic yaml psutil requests safetensors huggingface_hub accelerate typing_extensions; do
  python3 -c "import $m" 2>/dev/null || { pkg=$m; [ $m = yaml ] && pkg=pyyaml; [ $m = huggingface_hub ] && pkg=huggingface-hub; [ $m = typing_extensions ] && pkg=typing-extensions; pip install -q "$pkg" >> $P/logs/pip_specforge2.log 2>&1 && log "installed $pkg"; }
done
if specforge benchmark --help > /dev/null 2> $P/logs/specforge_help.err; then log "specforge CLI ok"; else log "FAIL: specforge CLI"; tail -3 $P/logs/specforge_help.err; fi
log "versions after: $(vers)"
python3 -c "from sglang.srt.models.dflash import DFlashMoE, DFlashMoEDraftModel; print('DFlashMoE import ok')" 2>&1 | tail -1
export HF_HOME=$S/hf
python3 - <<PY 2>&1 | tail -2
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
snapshot_download("jiapingW/Qwen3-8b-dspark-SpecForge", local_dir="$P/dense", allow_patterns=["config.json", "model.safetensors", "README.md"])
print("downloads ok")
PY
bash $P/dl_steps.sh 52000
du -sh $S/hf $P/dense $P/export2/step-52000 2>&1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo
touch $P/logs/SETUP2_DONE; log "SETUP2 DONE"
