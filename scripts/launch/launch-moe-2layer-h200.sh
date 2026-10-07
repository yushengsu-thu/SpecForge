#!/usr/bin/env bash
# Launch the 2-layer Qwen3.8 MoE DSpark run on this node. Refuses to start if any required GPU
# is in use, a specforge train process exists, or the run already has a control dir.
set -u
RECIPE=${1:-/scratch/qwen38-moe-2layer-1ep-h200.yaml}
RUN=$(grep -E "^run_id:" "$RECIPE" | awk "{print \$2}"); OUT=/scratch/outputs/$RUN
NEED=${GPUS:-0 1 2 3 4 5 6 7}
for g in $NEED; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); [ "$used" -lt 1024 ] || { echo "GPU $g busy ($used MiB), abort"; exit 1; }; done
pgrep -f "specforge.cli tr[a]in" > /dev/null && { echo "a specforge train process is already running, abort"; exit 1; }
[ -e "$OUT/control" ] && { echo "control dir exists: $OUT/control, abort"; exit 1; }
mkdir -p "$OUT" /scratch/logs; cd /scratch/SpecForge-qwen38
export HF_HOME=/scratch/hf-cache MC_TRANSFER_TIMEOUT=300 SGLANG_SPEC_CAPTURE_SINK_CLIENTS=${SINK_CLIENTS:-4} OMP_NUM_THREADS=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SGLANG_STOP_FILE=/scratch/logs/stop_capture_$RUN
rm -f "$SGLANG_STOP_FILE"
setsid nohup numactl --preferred=1 python -u -m specforge.cli train -c "$RECIPE" > "$OUT/launch.log" 2>&1 < /dev/null &
echo "launched $RUN, supervisor pid $!, $(date -u +%FT%TZ), log $OUT/launch.log"
