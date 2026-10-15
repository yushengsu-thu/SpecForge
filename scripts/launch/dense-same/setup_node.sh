#!/bin/bash
# dense-same node setup (idempotent). usage: bash setup_node.sh <0|1>   (0 = capture node, 1 = trainer node)
# Puts everything under /scratch/dense (node-local, wiped on release): the branch checkout, Qwen/Qwen3-8B, the
# corpus (trainer only), secrets copied from /personal/dense-secrets (hf_token, wandb_key; cluster storage that
# survives devbox releases), and the SGLang spec-capture patch of this branch.
set -u
RANK=${1:?rank 0 or 1}; BRANCH=${BRANCH:-dense-same}; BASE=/scratch/dense; SF=$BASE/SpecForge
export HF_HOME=/scratch/hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HF_DATASETS_CACHE=/scratch/hf-cache/datasets
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
mkdir -p $BASE/logs $BASE/data $BASE/outputs $BASE/exports /scratch/models /scratch/hf-cache
[ -d $SF/.git ] || { log "clone $BRANCH"; git clone -q -b $BRANCH https://github.com/yushengsu-thu/SpecForge.git $SF; }
cd $SF && git fetch -q origin $BRANCH && git reset -q --hard origin/$BRANCH && log "code $(git log -1 --format=%h)"
bash scripts/apply_sglang_spec_capture_patch.sh 2>&1 | tail -1
python -c "import sglang.srt.spec_capture_sink" && log "spec-capture sink ok"
[ -f /scratch/models/Qwen3-8B/config.json ] || { log "download Qwen/Qwen3-8B"; hf download Qwen/Qwen3-8B --local-dir /scratch/models/Qwen3-8B > $BASE/logs/dl-qwen3-8b.log 2>&1; }
log "model: $(du -sh /scratch/models/Qwen3-8B | cut -f1)"
if [ "$RANK" = 1 ]; then
  python -c "import specforge" 2>/dev/null && [ "$(python -c "import specforge,os;print(os.path.dirname(os.path.dirname(specforge.__file__)))")" = "$SF" ] || { log "pip install -e"; pip install -q -e $SF --no-deps 2>&1 | grep -v -i warning | tail -1; }
  # The lmsysorg/sglang:v0.5.18-cu130 image ships torch, sglang and the CUDA-13 build of Mooncake
  # (mooncake-transfer-engine-cuda13 0.3.12.post1). Never pip install plain mooncake-transfer-engine: the PyPI
  # wheel is a CUDA-12 build and a different version from the capture node's, and feature reads then fail with
  # "get_into failed (status -900)". The trainer additionally needs these pure-Python packages.
  python -c "import mooncake" 2>/dev/null || { log "pip install mooncake-transfer-engine-cuda13==0.3.12.post1"; pip install -q --no-deps "mooncake-transfer-engine-cuda13==0.3.12.post1" 2>&1 | tail -1; }
  python -c "import wandb, accelerate, zstandard, yaml, liger_kernel" 2>/dev/null || { log "pip install trainer deps"; pip install -q wandb accelerate zstandard pyyaml liger-kernel 2>&1 | grep -v -i warning | tail -1; }
  python -c "import specforge, mooncake, liger_kernel, wandb, accelerate, zstandard; print('trainer env ok')" || { log "ENV_FAILED"; exit 1; }
  [ -f $BASE/data/perfectblend_train_regen_temperature0_no_think.jsonl ] || { log "download corpus"; hf download yaro1214/open_data perfectblend_train_regen_temperature0_no_think.jsonl --repo-type dataset --local-dir $BASE/data > $BASE/logs/dl-dataset.log 2>&1; }
  log "corpus: $(du -sh $BASE/data/perfectblend_train_regen_temperature0_no_think.jsonl | cut -f1)"
  for f in hf_token wandb_key; do [ -f $BASE/$f ] || { [ -f /personal/dense-secrets/$f ] && { cp /personal/dense-secrets/$f $BASE/$f; chmod 600 $BASE/$f; log "secret $f restored from /personal"; } || log "MISSING secret $BASE/$f (write it, mode 600)"; }; done
fi
log "SETUP_DONE rank $RANK $(hostname) $(hostname -I | cut -d' ' -f1)"
