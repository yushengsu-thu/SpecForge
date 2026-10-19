#!/usr/bin/env bash
# End-to-end serving throughput against one running server:  e2e_one.sh <label> <port> <outdir>
# `specforge benchmark` on gsm8k and mt-bench (first turn), concurrency 1 / 8 / 32 with 100 / 200 / 200 prompts,
# 512 new tokens, greedy, thinking off. Writes <outdir>/e2e_<label>_<dataset>_c<c>.json (output tok/s, accept length).
set -u
L=${1:?label}; P=${2:?port}; O=${3:?outdir}; REPO=${REPO:-/scratch/SpecForge-spark8b}; TARGET=${TARGET:-/scratch/models/Qwen3-8B}
export PYTHONPATH=$REPO HF_HOME=/scratch/hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HUGGINGFACE_HUB_CACHE=/scratch/hf-cache/hub TRANSFORMERS_CACHE=/scratch/hf-cache/hub HF_DATASETS_CACHE=/scratch/hf-cache/datasets   # /etc/environment points the HF caches at read-only /cluster-storage
cd "$REPO" || exit 1
for ds in gsm8k mt-bench; do
  for c in 1 8 32; do
    n=100; [ "$c" -gt 1 ] && n=200
    python -m specforge.cli benchmark --model "$TARGET" --dataset "$ds" --num-prompts "$n" --concurrency "$c" \
      --max-new-tokens 512 --base-url "http://127.0.0.1:$P" --output-json "$O/e2e_${L}_${ds}_c${c}.json" 2>&1 \
      | grep -E "throughput|acceptance|Error|Traceback" | sed "s/^/[$L $ds c=$c] /"
  done
done
echo "[$L] E2E_DONE"
