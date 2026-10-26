#!/bin/bash
# One-node variant: launch the 16 x 1088 MoE drafter training on one 8x H200 box (4 capture servers + DP4).
#   usage: launch_train.sh smoke|full
# smoke: SMOKE_STEPS (default 3) optimizer steps, no checkpoints, no W&B, separate output dir (same tokenizer cache).
MODE=${1:?smoke|full}
# The body runs from main() so that the git reset below cannot change the file bash is still reading (bash reads
# scripts incrementally; the first launch after a push otherwise runs with the previous version's settings).
main() {
  BR=${BR:-dspark-v2-moe}
  D=/scratch/moe16x1088; mkdir -p $D/outputs
  VENV=${VENV:-/scratch/moe2ep/venv}; [ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"   # the boxes' sglang 0.5.18 + spec-capture + mooncake env
  export PYTHONPATH=/scratch/SpecForge
  cd /scratch/SpecForge || exit 1
  git fetch -q origin "$BR" && git checkout -q "$BR" 2>/dev/null; git reset -q --hard "origin/$BR" && echo "code: $(git log -1 --format=%h) $BR"
  python3 -c "import specforge, mooncake, sglang.srt.spec_capture_sink" || { echo "env not ready (run setup_box.sh first)"; exit 1; }
  export HF_HOME=/scratch/hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HF_DATASETS_CACHE=/scratch/hf-cache/datasets
  export HF_TOKEN="$(cat /scratch/secrets/hf_token)"
  unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK
  R=examples/configs/online/disaggregated/managed-local/qwen3.8-27b-moe-dspark-4server-dp4-h200.yaml
  RUN=qwen3.8-27b-moe-dspark-4server-dp4-h200
  PATHS="model.target_model_path=/scratch/models/Qwen3.8-27B data.train_data_path=/scratch/data/qwen38_regen_mixture_v1_train.jsonl data.cache_dir=/scratch/cache/qwen3.8-27b-regen-mixture-v1"
  if [ "$MODE" = smoke ]; then
    RUN=$RUN-smoke; OUT=$D/outputs/$RUN; rm -rf "$OUT"; mkdir -p "$OUT"
    OV="$PATHS training.max_steps=${SMOKE_STEPS:-3} training.save_interval=1000000 tracking.report_to=none run_id=$RUN output_dir=$OUT deployment.disaggregated.control_dir=$OUT/control deployment.disaggregated.consumer_state_dir=$OUT/consumer-state"
  else
    OUT=$D/outputs/$RUN; mkdir -p "$OUT"
    export WANDB_API_KEY="$(cat /scratch/secrets/wandb_key)" WANDB_ENTITY=${WANDB_ENTITY:-eigent_radixark_training} WANDB_RUN_ID=${WANDB_RUN_ID:-qwen38-moe-16x1088-4ep-1node-h200} WANDB_RESUME=allow
    OV="$PATHS tracking.report_to=wandb tracking.wandb_project=${WANDB_PROJECT:-specforge-qwen38-moe-16x1088} tracking.wandb_name=$RUN tracking.wandb_dir=$OUT/wandb output_dir=$OUT deployment.disaggregated.control_dir=$OUT/control deployment.disaggregated.consumer_state_dir=$OUT/consumer-state"
  fi
  echo "[$(date -u +%H:%M:%S)] launching $MODE run $RUN (overrides: $OV)"
  setsid nohup python -m specforge.cli train -c "$R" $OV > "$OUT/launch.log" 2>&1 < /dev/null &
  echo "launcher pid $! log $OUT/launch.log"
}
main "$@"
