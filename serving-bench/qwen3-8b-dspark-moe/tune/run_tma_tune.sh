#!/bin/bash
# One TMA-aware tune per GPU for the draft token counts M = 7 x c (c = 1, 8, 16, 32, 64, 128, 256, 512). Each GPU must be empty.
P=/scratch/profile; V=/scratch/moe8b16/serving-bench/venv; cd $P; mkdir -p tma-tune logs
export PYTHONPATH=/scratch/ckpt-sweep/tune/pylib
MS=(7 56 112 224 448 896 1792 3584)
for g in 0 1 2 3 4 5 6 7; do
  M=${MS[$g]}
  apps=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l); used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$apps" != 0 ] || [ "$used" -ge 1024 ]; then echo "[$(date -u +%T)] GPU $g NOT clean ($apps apps, ${used} MiB) -> skip M=$M"; continue; fi
  echo "[$(date -u +%T)] GPU $g clean -> tune M=$M"
  CUDA_VISIBLE_DEVICES=$g setsid nohup $V/bin/python $P/tune_tma_one.py $M $P/tma-tune/M$M.json > $P/logs/tmatune_M$M.log 2>&1 < /dev/null &
  sleep 5
done
echo "[$(date -u +%T)] all launched"
