#!/bin/bash
# dense-same epochs 13-18 (same scheme as the dense-1/8 run): wait until the 12-epoch run has finished cleanly,
# then launch a weights-only warm start from its final (epoch-12) checkpoint: AdamW state reset, 1% warm-up to
# 2e-4 (1/3 of the original peak), cosine to 0 over 6 epochs, training.prompt_seed 54 (epochs 1-12 used 42).
# A new online producer needs a fresh run id / channel / store, so the segment runs as dense-same-ep13-18; W&B
# resumes the same run (WANDB_RUN_ID dense-same-r6) with SPECFORGE_TRACKING_STEP_OFFSET=133200 (= 12 x 11,100),
# so the curve continues instead of restarting at step 0. Starts the epoch 13-18 export watcher afterwards.
# usage: bash continue_node.sh <CAPTURE_IP> <TRAINER_IP>      (run on the trainer node, rank 1)
set -u
CAP=${1:?capture ip}; TRN=${2:?trainer ip}
BASE=/scratch/dense; SF=$BASE/SpecForge
PREV=dense-same; PREV_OUT=$BASE/outputs/$PREV; RUN=dense-same-ep13-18; OUT=$BASE/outputs/$RUN
STEPS_PER_EPOCH=11100; OFFSET=$((12 * STEPS_PER_EPOCH)); W_RUN_ID=${WANDB_RUN_ID:-dense-same-r6}
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
prev_alive() { pgrep -f "specforge.cli trai[n].*recipe-${PREV}[.]yaml" >/dev/null; }
prev_ckpt() { local p=$PREV_OUT/$PREV-latest; if [ -L "$p" ]; then readlink -f "$p"; elif [ -f "$p" ]; then readlink -f "$(cat "$p")"; fi; }
log "waiting for $PREV to finish (final checkpoint step >= $OFFSET, trainer exited)"
while :; do
  ck=$(prev_ckpt); s=${ck##*-step}
  if ! prev_alive && [ -n "$ck" ] && [ -f "$ck/training_state.pt" ] && [ "${s:-0}" -ge "$OFFSET" ] 2>/dev/null; then break; fi
  sleep 300
done
log "$PREV finished; warm start from $ck"; sleep 120
for i in $(seq 1 30); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && break; log "GPUs still busy, waiting"; sleep 20; done
cd $SF || exit 1
R=$BASE/recipe-$RUN.yaml
sed -e "s/CAPTURE_IP/$CAP/g" -e "s/TRAINER_IP/$TRN/g" examples/configs/online/disaggregated/external/qwen3-8b-dspark-dense-same-2node-h200.yaml > "$R"
log "recipe rendered: $R ($(git log -1 --format=%h))"
pgrep -f "mooncake_master --enable_http_metadata_server" >/dev/null || { CUDA_VISIBLE_DEVICES= setsid nohup mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=0.0.0.0 --rpc_port=35551 --http_metadata_server_port=35880 --metrics_port=35903 --default_kv_lease_ttl=300000 > $BASE/logs/mooncake_master.log 2>&1 < /dev/null & sleep 3; log "mooncake_master restarted"; }
for i in $(seq 0 7); do until curl -sf -m 3 "http://$CAP:$((30000+i))/health" >/dev/null; do log "waiting for capture server $i on $CAP"; sleep 20; done; done; log "all 8 capture servers healthy"
export HF_HOME=/scratch/hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HF_DATASETS_CACHE=/scratch/hf-cache/datasets HF_TOKEN="$(cat $BASE/hf_token)"
export WANDB_API_KEY="$(cat $BASE/wandb_key)" WANDB_ENTITY=eigent_radixark_training WANDB_RUN_ID=$W_RUN_ID WANDB_RESUME=allow
export SPECFORGE_TRACKING_STEP_OFFSET=$OFFSET
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK
OV="model.draft_checkpoint_path=$ck training.num_epochs=6 training.learning_rate=0.0002 training.warmup_ratio=0.01 training.prompt_seed=54"
OV="$OV run_id=$RUN output_dir=$OUT deployment.disaggregated.control_dir=$OUT/control deployment.disaggregated.consumer_state_dir=$OUT/consumer-state tracking.wandb_dir=$OUT/wandb"
launch() {
  mkdir -p $OUT; log "launching $RUN: $OV (W&B $W_RUN_ID, step offset $OFFSET)"
  setsid nohup python -m specforge.cli train -c "$R" $OV >> $OUT/launch.log 2>&1 < /dev/null &
  log "trainer launcher pid $!"
}
launch
# one automatic retry if the launch itself fails before the first optimizer step (fresh output dir each time)
for i in $(seq 1 60); do sleep 30; grep -qE "^step [0-9]+:" $OUT/launch.log 2>/dev/null && break; pgrep -f "specforge.cli trai[n].*recipe-${RUN}[.]yaml" >/dev/null || break; done
if ! grep -qE "^step [0-9]+:" $OUT/launch.log 2>/dev/null; then
  log "LAUNCH_FAILED (no optimizer step); log kept as $OUT.failed1; retrying once"; tail -20 $OUT/launch.log
  pkill -f "specforge.cli trai[n].*recipe-${RUN}[.]yaml"; sleep 30; mv $OUT $OUT.failed1; launch
fi
cd $SF && RUN_ID=$RUN EPOCH_OFFSET=12 TOTAL_EPOCHS=18 GLOBAL_STEP_OFFSET=$OFFSET setsid nohup bash scripts/launch/dense-same/watch_exports.sh 13 18 >> $BASE/logs/watch_exports_ep13-18.log 2>&1 < /dev/null &
log "export watcher for epochs 13-18 pid $!"
log "CONTINUATION_LAUNCHED"
