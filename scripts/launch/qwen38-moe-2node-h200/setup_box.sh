#!/bin/bash
# One-time setup of an 8x H200 devbox (image lmsysorg/sglang:v0.5.18-cu130) for the 16 x 1088 MoE drafter run.
# Idempotent. The trainer node expects /scratch/secrets/hf_token and /scratch/secrets/wandb_key (mode 600) to exist.
#   setsid nohup bash setup_box.sh > /scratch/moe16x1088/logs/setup_box.log 2>&1 < /dev/null &
#   SKIP_DATA=1 on the capture node (it never reads the corpus).
set -u
BR=${BR:-dspark-v2-moe}
REPO=${REPO:-https://github.com/yushengsu-thu/SpecForge.git}
TARGET=${TARGET:-Qwen/Qwen3.8-27B}
D=/scratch/moe16x1088
VENV=${VENV:-/scratch/moe2ep/venv}; [ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"   # the boxes' sglang 0.5.18 + spec-capture + mooncake env
export PYTHONPATH=/scratch/SpecForge
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
export HF_HOME=/scratch/hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HF_DATASETS_CACHE=/scratch/hf-cache/datasets
[ -f /scratch/secrets/hf_token ] && export HF_TOKEN="$(cat /scratch/secrets/hf_token)"
mkdir -p /scratch/models /scratch/data /scratch/cache $D/outputs $D/logs $D/exports $D/epoch_ckpts /scratch/hf-cache
log "arch $(uname -m); $(python3 --version); sglang $(python3 -c 'import sglang; print(sglang.__version__)') torch $(python3 -c 'import torch; print(torch.__version__)') transformers $(python3 -c 'import transformers; print(transformers.__version__)')"
# 1. code
[ -d /scratch/SpecForge/.git ] || { log "clone $BR"; git clone -q -b "$BR" "$REPO" /scratch/SpecForge || { log SETUP_FAILED_CLONE; exit 1; }; }
cd /scratch/SpecForge && git fetch -q origin "$BR" && git checkout -q "$BR" && git reset -q --hard "origin/$BR" && log "code $(git log -1 --format='%h %s') (python: $(command -v python3))"
# 2. python env: specforge comes from PYTHONPATH (no pip install, so the shared venv stays as it is), plus the
#    few packages the env may lack
for p in zstandard yaml accelerate yunchang tensorboard wandb datasets; do
  python3 -c "import $p" 2>/dev/null || { pkg=$p; [ "$p" = yaml ] && pkg=pyyaml; log "pip install $pkg"; pip install -q "$pkg" 2>&1 | grep -v -i warning | tail -1; }
done
python3 -c "import mooncake" 2>/dev/null || { log "pip install mooncake"; pip install -q mooncake-transfer-engine 2>&1 | tail -1; }
command -v mooncake_master >/dev/null || log "WARNING: mooncake_master not on PATH"
# 3. spec-capture patch on the installed sglang (editable /sgl-workspace checkout on this image)
SG=$(python3 -c "import sglang,os; print(os.path.dirname(sglang.__file__))")
python3 -c "import sglang.srt.spec_capture_sink" 2>/dev/null || { log "apply spec-capture patch to $SG"; rm -f "$SG/.spec_capture_patch.applied"; bash /scratch/SpecForge/scripts/apply_sglang_spec_capture_patch.sh --target v0.5.18 2>&1 | tail -1; }
python3 -c "import specforge, accelerate, yunchang, zstandard, datasets, transformers, wandb, mooncake, sglang.srt.spec_capture_sink; print('env ok')" || { log SETUP_FAILED_ENV; exit 1; }
command -v specforge >/dev/null || log "specforge CLI missing (python -m specforge.cli works)"
# 4. teacher: the BF16 target (Hopper cannot run the NVFP4 checkpoint v2 trained against)
M=/scratch/models/$(basename "$TARGET")
[ -f "$M/config.json" ] || { log "download $TARGET"; hf download "$TARGET" --local-dir "$M" 2>&1 | grep -v -i token | tail -1; }
log "target: $(du -sh "$M" | cut -f1) at $M"
# 5. corpus -> slim {id, conversations} jsonl (status == success rows only); SKIP_DATA=1 on a capture-only node
# An earlier run left the prepared JSONL on shared storage; copying it beats re-downloading the 12.5 GB of shards.
PREPARED=${PREPARED:-/personal/specforge-qwen38-backup/data/qwen38_regen_mixture_v1_train.jsonl}
[ -n "${SKIP_DATA:-}" ] || [ -f /scratch/data/qwen38_regen_mixture_v1_train.jsonl ] || [ ! -f "$PREPARED" ] || { log "copy prepared corpus from $PREPARED"; cp "$PREPARED" /scratch/data/qwen38_regen_mixture_v1_train.jsonl.tmp && mv /scratch/data/qwen38_regen_mixture_v1_train.jsonl.tmp /scratch/data/qwen38_regen_mixture_v1_train.jsonl; }
[ -n "${SKIP_DATA:-}" ] || [ -f /scratch/data/qwen38_regen_mixture_v1_train.jsonl ] || {
  log "download Regen-Mixture-v1"; hf download RadixArk/Qwen3.8-27B-Regen-Mixture-v1 --repo-type dataset --local-dir /scratch/data/regen-mixture-v1 2>&1 | grep -v -i token | tail -1
  log "prepare jsonl ($(ls /scratch/data/regen-mixture-v1/data | wc -l) shards)"
  python3 /scratch/SpecForge/scripts/launch/qwen38-moe-2node-h200/prep_regen_mixture.py --src /scratch/data/regen-mixture-v1/data --out /scratch/data/qwen38_regen_mixture_v1_train.jsonl 2>&1 | tail -6
}
[ -n "${SKIP_DATA:-}" ] || log "jsonl rows: $(wc -l < /scratch/data/qwen38_regen_mixture_v1_train.jsonl) ($(du -sh /scratch/data/qwen38_regen_mixture_v1_train.jsonl | cut -f1))"
log SETUP_DONE
