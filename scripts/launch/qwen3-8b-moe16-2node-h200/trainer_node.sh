#!/bin/bash
# Trainer node of the two-node run: render the recipe with the real IPs, start the mooncake master, wait for the
# capture servers, launch the trainer (smoke: SMOKE_STEPS steps, no W&B / checkpoints).
#   usage: bash trainer_node.sh <CAPTURE_IP> <TRAINER_IP> [smoke|full] [PROTOCOL=tcp|rdma]     (N=8 capture servers)
#   continuation after a crash (weights-only warm start, fresh AdamW / warm-up / shuffle, as in Kan's phases):
#     WARMSTART=/scratch/moe16x1088/epoch_ckpts/<run>-step<N> NUM_EPOCHS=<epochs left> PROMPT_SEED=43 WANDB_STEP_OFFSET=<N> \
#     RUN_SUFFIX=-cont-from-<k>ep bash trainer_node.sh ... full rdma      (then watch_epoch_exports.sh with EPOCH_OFFSET=<k>)
set -u
CAP=${1:?capture ip}; TRN=${2:?trainer ip}; MODE=${3:-full}; PROTO=${4:-tcp}; N=${N:-8}
RUN=qwen3-8b-moe16x768-dspark-8server-dp8-2node-h200${RUN_SUFFIX:-}
WANDB_PROJECT=${WANDB_PROJECT:-specforge-qwen3-8b-moe16x768}
D=/scratch/moe8b16; mkdir -p $D/logs $D/outputs
VENV=${VENV:-/scratch/moe2ep/venv}; [ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"   # the boxes' sglang 0.5.18 + spec-capture + mooncake env
export PYTHONPATH=/scratch/SpecForge
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
export HF_HOME=/scratch/hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HF_DATASETS_CACHE=/scratch/hf-cache/datasets
cd /scratch/SpecForge || exit 1
python3 -c "import specforge, mooncake" || { log "env not ready (run setup_box.sh first)"; exit 1; }
[ -f /scratch/moe2ep/data/perfectblend_train_regen_temperature0_no_think.jsonl ] || { log "MISSING /scratch/moe2ep/data/perfectblend_train_regen_temperature0_no_think.jsonl"; exit 1; }
R=$D/recipe-$RUN.yaml
sed -e "s/CAPTURE_IP/$CAP/g" -e "s/TRAINER_IP/$TRN/g" -e "s/mooncake_protocol: rdma/mooncake_protocol: $PROTO/" \
    -e "s#cache/dataset/perfectblend_train_regen_temperature0_no_think.jsonl#/scratch/moe2ep/data/perfectblend_train_regen_temperature0_no_think.jsonl#" \
    -e "s#cache_dir: cache/qwen3-8b-dspark-perfectblend-8192#cache_dir: /scratch/moe2ep/cache/qwen3-8b-dspark-perfectblend-8192#" \
    -e "s#target_model_path: Qwen/Qwen3-8B#target_model_path: /scratch/moe2ep/models/Qwen3-8B#" \
    examples/configs/online/disaggregated/external/qwen3-8b-moe16x768-dspark-8server-dp8-2node-h200.yaml > "$R"
# rdma needs the HCAs listed explicitly (the capture sink refuses an empty MOONCAKE_RDMA_DEVICES): default to every ACTIVE port.
if [ "$PROTO" = rdma ] && [ -z "${MOONCAKE_RDMA_DEVICES:-}" ]; then
  MOONCAKE_RDMA_DEVICES=$(for d in /sys/class/infiniband/*; do grep -q ACTIVE "$d/ports/1/state" 2>/dev/null && basename "$d"; done | paste -sd, -)
fi
export MOONCAKE_RDMA_DEVICES=${MOONCAKE_RDMA_DEVICES:-}
[ -n "${MOONCAKE_RDMA_DEVICES:-}" ] && echo "    mooncake_rdma_devices: $MOONCAKE_RDMA_DEVICES" >> "$R"
log "recipe rendered: $R (capture $CAP, trainer $TRN, protocol $PROTO)"
pgrep -f "mooncake_master --enable_http_metadata_serve[r]" >/dev/null || { CUDA_VISIBLE_DEVICES= setsid nohup mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=0.0.0.0 --rpc_port=35551 --http_metadata_server_port=35880 --metrics_port=35903 --default_kv_lease_ttl=300000 > $D/logs/mooncake_master.log 2>&1 < /dev/null & log "mooncake_master pid $!"; sleep 3; }
for i in $(seq 0 $((N-1))); do until curl -sf -m 3 "http://$CAP:$((30000+i))/health" >/dev/null; do log "waiting for capture server $i on $CAP"; sleep 20; done; done; log "all $N capture servers healthy"
export HF_TOKEN="$(cat /scratch/secrets/hf_token)"
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK
if [ "$MODE" = smoke ]; then
  OUT=$D/outputs/$RUN-smoke; rm -rf "$OUT"; mkdir -p "$OUT"
  OV="training.max_steps=${SMOKE_STEPS:-3} training.save_interval=1000000 tracking.report_to=none run_id=$RUN-smoke output_dir=$OUT deployment.disaggregated.control_dir=$OUT/control deployment.disaggregated.consumer_state_dir=$OUT/consumer-state"
else
  OUT=$D/outputs/$RUN; mkdir -p "$OUT"
  export WANDB_API_KEY="$(cat /scratch/secrets/wandb_key)" WANDB_ENTITY=${WANDB_ENTITY:-eigent_radixark_training} WANDB_RUN_ID=${WANDB_RUN_ID:-qwen3-8b-moe16x768-6ep-2node-h200} WANDB_RESUME=allow
  OV="tracking.report_to=wandb tracking.wandb_project=$WANDB_PROJECT tracking.wandb_name=$RUN tracking.wandb_dir=$OUT/wandb output_dir=$OUT deployment.disaggregated.control_dir=$OUT/control deployment.disaggregated.consumer_state_dir=$OUT/consumer-state"
fi
if [ -n "${WARMSTART:-}" ]; then
  [ -f "$WARMSTART/training_state.pt" ] || { log "MISSING $WARMSTART/training_state.pt"; exit 1; }
  OV="$OV model.draft_checkpoint_path=$WARMSTART training.num_epochs=${NUM_EPOCHS:?} training.prompt_seed=${PROMPT_SEED:?} run_id=$RUN"
  export SPECFORGE_WANDB_STEP_OFFSET=${WANDB_STEP_OFFSET:-0}
fi
log "launching $MODE: $RUN ($OV)"
setsid nohup python -m specforge.cli train -c "$R" $OV > "$OUT/launch.log" 2>&1 < /dev/null &
echo "launcher pid $! log $OUT/launch.log"
