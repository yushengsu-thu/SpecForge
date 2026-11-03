#!/bin/bash
# Tune the Triton fused-MoE config for E=2, N=12288 (bf16, top-1) on all 8 H200s. Refuses to start unless every GPU is empty.
V=/scratch/moe8b16/serving-bench/venv/bin; T=/scratch/moe8b16/serving-bench/sglang/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py
cd /scratch/ckpt-sweep/tune || exit 1
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
echo "[$(date -u +%H:%M:%S) UTC] GPU check: compute apps=$apps, max mem used=${mem} MiB"
if [ "$apps" != "0" ] || [ "$mem" -gt 1000 ]; then echo "GPUs NOT clean, abort"; exit 2; fi
export PYTHONPATH=/scratch/ckpt-sweep/tune/pylib
setsid nohup $V/python $T --model /scratch/ckpt-sweep/tune/model-cfg --tp-size 1 --dtype auto --tune \
  --batch-sizes 1 4 7 14 28 56 112 224 448 896 1792 3584 > tune.log 2>&1 < /dev/null &
echo "launched pid $! at $(date -u +%H:%M:%S) UTC"
