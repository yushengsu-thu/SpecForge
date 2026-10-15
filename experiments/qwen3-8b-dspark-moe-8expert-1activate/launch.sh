#!/usr/bin/env bash
# Launch (or dry-run) the 12-epoch Qwen3-8B DSpark MoE 8-expert/top-1 run on one 8xH200 box
# (managed_local). Box paths are passed as config overrides; extra key=value arguments are
# appended (e.g. training.max_steps=1 tracking.report_to=none for a schedule/cache dry run).
#   RUN=<run id> OUT=<output dir> experiments/qwen3-8b-dspark-moe-8expert-1activate/launch.sh [overrides...]
# Stop: kill -TERM "$(cat $OUT/launcher.pid)" (workers checkpoint and drain on SIGTERM).
set -euo pipefail
REPO=${REPO:-/scratch/SpecForge-spark8b}
RUN=${RUN:-qwen3-8b-dspark-moe-8expert-1activate-dense-same}
RECIPE=examples/configs/online/disaggregated/managed-local/qwen3-8b-dspark-moe-8expert-1activate-dense-same-12ep.yaml
DATA=${DATA:-/scratch/data/perfectblend/perfectblend_train_regen_temperature0_no_think.jsonl}
TARGET=${TARGET:-/scratch/models/Qwen3-8B}
CACHE=${CACHE:-/scratch/cache/qwen3-8b-dspark-perfectblend-8192}
OUT=${OUT:-/scratch/outputs/$RUN}
EXTRA=("$@")

# The box's /etc/environment points HF_HOME at the read-only /cluster-storage; force the writable cache.
export HF_HOME=/scratch/hf-cache HF_DATASETS_CACHE=/scratch/hf-cache/datasets MC_TRANSFER_TIMEOUT=300 SGLANG_SPEC_CAPTURE_SINK_CLIENTS=4 \
       OMP_NUM_THREADS=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$REPO PYTHONUNBUFFERED=1
# W&B credentials live outside the repository: /root/.secrets/wandb.env (WANDB_API_KEY=...)
if [ -f /root/.secrets/wandb.env ]; then set -a; . /root/.secrets/wandb.env; set +a; fi
export SGLANG_STOP_FILE=$OUT/stop_capture
mkdir -p "$OUT"; rm -f "$SGLANG_STOP_FILE"
if [ -f "$OUT/launcher.pid" ] && kill -0 "$(cat "$OUT/launcher.pid")" 2>/dev/null; then
  echo "already running: pid $(cat "$OUT/launcher.pid")"; exit 1
fi
cd "$REPO"
echo "=== $RUN launch $(date -u +%FT%TZ) commit $(git rev-parse --short HEAD) overrides: ${EXTRA[*]:-none}" >> "$OUT/launch.log"
nohup numactl --preferred=1 python -u -m specforge.cli train -c "$RECIPE" \
  model.target_model_path="$TARGET" data.train_data_path="$DATA" data.cache_dir="$CACHE" \
  run_id="$RUN" output_dir="$OUT" deployment.disaggregated.control_dir="$OUT/control" \
  ${EXTRA[@]+"${EXTRA[@]}"} >> "$OUT/launch.log" 2>&1 &
echo $! > "$OUT/launcher.pid"
echo "launched $RUN pid $! -> $OUT/launch.log"
