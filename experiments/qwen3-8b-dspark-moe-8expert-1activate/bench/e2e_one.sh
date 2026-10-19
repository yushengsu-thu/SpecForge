#!/usr/bin/env bash
# End-to-end serving throughput against one running server:  e2e_one.sh <label> <port> <outdir>
# `specforge benchmark` on the five acceptance-length tasks (GSM8K, MATH500, HumanEval, MBPP, MT-Bench first turn;
# E2E_DATASETS overrides the list), concurrency 1 / 8 / 32 with 100 / 200 / 200 prompts (a smaller dataset wraps
# around), 512 new tokens, greedy, thinking off. Writes <outdir>/e2e_<label>_<dataset>_c<c>.json (output tok/s,
# accept length). REPO / TARGET / HFC (writable HF cache root) default to the specforge-qwen-h200 layout.
set -u
L=${1:?label}; P=${2:?port}; O=${3:?outdir}; REPO=${REPO:-/scratch/SpecForge-spark8b}; TARGET=${TARGET:-/scratch/models/Qwen3-8B}
HFC=${HFC:-/scratch/hf-cache}; DSS=${E2E_DATASETS:-gsm8k math500 humaneval mbpp mt-bench}
export PYTHONPATH=$REPO HF_HOME=$HFC HF_HUB_CACHE=$HFC/hub HUGGINGFACE_HUB_CACHE=$HFC/hub TRANSFORMERS_CACHE=$HFC/hub HF_DATASETS_CACHE=$HFC/datasets   # /etc/environment points the HF caches at read-only /cluster-storage
cd "$REPO" || exit 1
for ds in $DSS; do
  for c in 1 8 32; do
    n=100; [ "$c" -gt 1 ] && n=200
    python -m specforge.cli benchmark --model "$TARGET" --dataset "$ds" --num-prompts "$n" --concurrency "$c" \
      --max-new-tokens 512 --base-url "http://127.0.0.1:$P" --output-json "$O/e2e_${L}_${ds}_c${c}.json" 2>&1 \
      | grep -E "throughput|acceptance|Error|Traceback" | sed "s/^/[$L $ds c=$c] /"
  done
done
echo "[$L] E2E_DONE"
