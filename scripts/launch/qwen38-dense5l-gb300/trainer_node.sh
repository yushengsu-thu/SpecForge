#!/bin/bash
# Trainer node of the two-node run: render the recipe with the real IPs, start the mooncake master, wait for the
# capture servers, launch the trainer (smoke: SMOKE_STEPS steps, no W&B / checkpoints).
#   usage: bash trainer_node.sh <CAPTURE_IP> <TRAINER_IP> [smoke|full] [PROTOCOL=tcp|rdma]
set -u
CAP=${1:?capture ip}; TRN=${2:?trainer ip}; MODE=${3:-full}; PROTO=${4:-tcp}
RUN=qwen3.8-27b-dspark-dense-5layer-3ep-2node-gb300
D=/scratch/dense5l; mkdir -p $D/logs $D/outputs
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
export HF_HOME=/scratch/hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HF_DATASETS_CACHE=/scratch/hf-cache/datasets
cd /scratch/SpecForge || exit 1
python3 -c "import specforge, mooncake" || { log "env not ready (run setup_box.sh first)"; exit 1; }
[ -f /scratch/data/qwen38_regen_mixture_v1_train.jsonl ] || { log "MISSING /scratch/data/qwen38_regen_mixture_v1_train.jsonl (run setup_box.sh)"; exit 1; }
R=$D/recipe-$RUN.yaml
sed -e "s/CAPTURE_IP/$CAP/g" -e "s/TRAINER_IP/$TRN/g" -e "s/MOONCAKE_PROTOCOL/$PROTO/g" examples/configs/online/disaggregated/external/qwen3.8-27b-dspark-dense-5layer-3ep-2node-gb300.yaml > "$R"
[ -n "${MOONCAKE_RDMA_DEVICES:-}" ] && echo "    mooncake_rdma_devices: $MOONCAKE_RDMA_DEVICES" >> "$R"
log "recipe rendered: $R (capture $CAP, trainer $TRN, protocol $PROTO)"
pgrep -f "mooncake_master --enable_http_metadata_serve[r]" >/dev/null || { CUDA_VISIBLE_DEVICES= setsid nohup mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=0.0.0.0 --rpc_port=35551 --http_metadata_server_port=35880 --metrics_port=35903 --default_kv_lease_ttl=300000 > $D/logs/mooncake_master.log 2>&1 < /dev/null & log "mooncake_master pid $!"; sleep 3; }
for i in 0 1 2 3; do until curl -sf -m 3 "http://$CAP:$((30000+i))/health" >/dev/null; do log "waiting for capture server $i on $CAP"; sleep 20; done; done; log "all 4 capture servers healthy"
export HF_TOKEN="$(cat /scratch/secrets/hf_token)"
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK
if [ "$MODE" = smoke ]; then
  OUT=$D/outputs/$RUN-smoke; rm -rf "$OUT"; mkdir -p "$OUT"
  OV="training.max_steps=${SMOKE_STEPS:-3} training.save_interval=1000000 tracking.report_to=none run_id=$RUN-smoke output_dir=$OUT deployment.disaggregated.control_dir=$OUT/control deployment.disaggregated.consumer_state_dir=$OUT/consumer-state"
else
  OUT=$D/outputs/$RUN; mkdir -p "$OUT"
  # same W&B run as the one-node launch (resumed), same HF repo (watch_epoch_exports.sh with RUN_ID=$RUN)
  export WANDB_API_KEY="$(cat /scratch/secrets/wandb_key)" WANDB_ENTITY=miles_training WANDB_RUN_ID=qwen38-dspark-dense-5layer-3ep-gb300-v2 WANDB_RESUME=allow
  OV=""
fi
log "launching $MODE: $RUN ($OV)"
setsid nohup python -m specforge.cli train -c "$R" $OV > "$OUT/launch.log" 2>&1 < /dev/null &
echo "launcher pid $! log $OUT/launch.log"
