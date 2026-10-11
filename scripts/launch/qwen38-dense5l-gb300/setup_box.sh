#!/bin/bash
# One-time setup of a 4x GB300 devbox (image lmsysorg/sglang:v0.5.18-cu130) for the dense 5-layer drafter run.
# Idempotent. Expects /scratch/secrets/hf_token and /scratch/secrets/wandb_key (mode 600) to exist already.
#   setsid nohup bash setup_box.sh > /scratch/dense5l/logs/setup_box.log 2>&1 < /dev/null &
set -u
BR=${BR:-yusheng/dense-5layer-3ep-gb300}
REPO=${REPO:-https://github.com/yushengsu-thu/SpecForge.git}
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
export HF_HOME=/scratch/hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HF_DATASETS_CACHE=/scratch/hf-cache/datasets
export HF_TOKEN="$(cat /scratch/secrets/hf_token)"
mkdir -p /scratch/models /scratch/data /scratch/cache /scratch/dense5l/outputs /scratch/dense5l/logs /scratch/dense5l/exports /scratch/dense5l/epoch_ckpts /scratch/hf-cache
log "arch $(uname -m); $(python3 --version); sglang $(python3 -c 'import sglang; print(sglang.__version__)') torch $(python3 -c 'import torch; print(torch.__version__)') transformers $(python3 -c 'import transformers; print(transformers.__version__)')"
# 1. code
[ -d /scratch/SpecForge/.git ] || { log "clone $BR"; git clone -q -b "$BR" "$REPO" /scratch/SpecForge || { log SETUP_FAILED_CLONE; exit 1; }; }
cd /scratch/SpecForge && git fetch -q origin "$BR" && git checkout -q "$BR" && git reset -q --hard "origin/$BR" && log "code $(git log -1 --format='%h %s')"
# 2. python env: specforge without deps (torch / sglang / transformers are the image's pinned versions), plus the
#    few packages the image lacks
pip install -q -e /scratch/SpecForge --no-deps 2>&1 | grep -v -i warning | tail -1
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
# 4. teacher: NVFP4 target with the bf16 lm_head
[ -f /scratch/models/Qwen3.8-27B-NVFP4-BF16-LMHead/config.json ] || { log "download NVFP4 target"; hf download RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead --local-dir /scratch/models/Qwen3.8-27B-NVFP4-BF16-LMHead 2>&1 | grep -v -i token | tail -1; }
log "target: $(du -sh /scratch/models/Qwen3.8-27B-NVFP4-BF16-LMHead | cut -f1)"
# 5. corpus -> slim {id, conversations} jsonl (status == success rows only); SKIP_DATA=1 on a capture-only node
[ -n "${SKIP_DATA:-}" ] || [ -f /scratch/data/qwen38_regen_mixture_v1_train.jsonl ] || {
  log "download Regen-Mixture-v1"; hf download RadixArk/Qwen3.8-27B-Regen-Mixture-v1 --repo-type dataset --local-dir /scratch/data/regen-mixture-v1 2>&1 | grep -v -i token | tail -1
  log "prepare jsonl ($(ls /scratch/data/regen-mixture-v1/data | wc -l) shards)"
  python3 /scratch/SpecForge/scripts/launch/qwen38-dense5l-gb300/prep_regen_mixture.py --src /scratch/data/regen-mixture-v1/data --out /scratch/data/qwen38_regen_mixture_v1_train.jsonl 2>&1 | tail -6
}
[ -n "${SKIP_DATA:-}" ] || log "jsonl rows: $(wc -l < /scratch/data/qwen38_regen_mixture_v1_train.jsonl) ($(du -sh /scratch/data/qwen38_regen_mixture_v1_train.jsonl | cut -f1))"
log SETUP_DONE
