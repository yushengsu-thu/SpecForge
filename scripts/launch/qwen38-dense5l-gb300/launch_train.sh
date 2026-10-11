#!/bin/bash
# Launch the dense 5-layer DSpark drafter training on this 4x GB300 box.   usage: launch_train.sh smoke|full
# smoke: SMOKE_STEPS (default 3) optimizer steps, no checkpoints, no W&B, separate output dir (same tokenizer cache).
MODE=${1:?smoke|full}
BR=${BR:-yusheng/dense-5layer-3ep-gb300}
D=/scratch/dense5l
cd /scratch/SpecForge || exit 1
git fetch -q origin "$BR" && git checkout -q "$BR" 2>/dev/null; git reset -q --hard "origin/$BR" && echo "code: $(git log -1 --format=%h) $BR"
python3 -c "import specforge, mooncake, sglang.srt.spec_capture_sink" || { echo "env not ready (run setup_box.sh first)"; exit 1; }
export HF_HOME=/scratch/hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HF_DATASETS_CACHE=/scratch/hf-cache/datasets
export HF_TOKEN="$(cat /scratch/secrets/hf_token)"
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK
R=examples/configs/online/disaggregated/managed-local/qwen3.8-27b-dspark-dense-5layer-3ep-gb300.yaml
RUN=qwen3.8-27b-dspark-dense-5layer-3ep-gb300
if [ "$MODE" = smoke ]; then
  RUN=$RUN-smoke; OUT=$D/outputs/$RUN; rm -rf "$OUT"; mkdir -p "$OUT"
  OV="training.max_steps=${SMOKE_STEPS:-3} training.save_interval=1000000 tracking.report_to=none run_id=$RUN output_dir=$OUT deployment.disaggregated.control_dir=$OUT/control deployment.disaggregated.consumer_state_dir=$OUT/consumer-state"
else
  OUT=$D/outputs/$RUN; mkdir -p "$OUT"
  export WANDB_API_KEY="$(cat /scratch/secrets/wandb_key)" WANDB_ENTITY=miles_training WANDB_RUN_ID=qwen38-dspark-dense-5layer-3ep-gb300-v2 WANDB_RESUME=allow
  OV=""
fi
echo "[$(date -u +%H:%M:%S)] launching $MODE run $RUN (overrides: $OV)"
setsid nohup python -m specforge.cli train -c "$R" $OV > "$OUT/launch.log" 2>&1 < /dev/null &
echo "launcher pid $! log $OUT/launch.log"
