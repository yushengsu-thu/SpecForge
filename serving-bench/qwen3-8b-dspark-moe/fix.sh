#!/bin/bash
# Devbox: use a node-local HF cache (the box's default is the read-only cluster cache), fetch Qwen3-8B and the dense
# draft, make serve.sh/bench_all.sh use that cache, and clear stale benchmark outputs. Marker: logs/FIX_DONE.
set -u
export PATH=/opt/sglang/bin:$PATH
P=/personal/moe-serving; S=/scratch/moe-serving
export HF_HOME=$S/hf HF_HUB_CACHE=$S/hf/hub; unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
mkdir -p $HF_HUB_CACHE $P/logs; rm -f $P/logs/FIX_DONE
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
python3 - <<PY 2>&1 | tail -3
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
print("target at", p)
snapshot_download("jiapingW/Qwen3-8b-dspark-SpecForge", local_dir="$P/dense", allow_patterns=["config.json", "model.safetensors", "README.md"])
print("downloads ok")
PY
sed -i 's#HF_HOME=${HF_HOME:-/scratch/moe-serving/hf}#HF_HOME=/scratch/moe-serving/hf HF_HUB_CACHE=/scratch/moe-serving/hf/hub#' $P/serve.sh $P/bench_all.sh
grep -c HF_HUB_CACHE $P/serve.sh $P/bench_all.sh
rm -f $P/logs/all_g8_dense.txt $P/logs/all_g8_moe52000.txt
du -sh $S/hf $P/dense 2>&1
touch $P/logs/FIX_DONE; log "FIX DONE"
