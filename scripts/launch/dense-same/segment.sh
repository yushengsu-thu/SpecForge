#!/bin/bash
# dense-same: launch one weights-only continuation segment (same scheme as the epoch 13-18 continuation and the
# dense-1/8 run: AdamW reset, short warm-up, cosine to 0), optionally after a previous segment has finished.
# A new online producer needs a fresh run id / channel / store, so every segment has its own RUN id; W&B resumes the
# same run (WANDB_RUN_ID, default dense-same-r6) with SPECFORGE_TRACKING_STEP_OFFSET=STEP_OFFSET so the curve continues.
# Starts the segment's export watcher (epochs EPOCH_OFFSET+1 .. TOTAL_EPOCHS) and its producer reaper.
# usage (trainer node, rank 1):  RUN=... NUM_EPOCHS=... LR=... SEED=... STEP_OFFSET=... EPOCH_OFFSET=... TOTAL_EPOCHS=... \
#          { CKPT=<checkpoint dir> | PREV_RUN=<run id> PREV_MIN_STEP=<final step> } [WARMUP=0.01] bash segment.sh <CAP_IP> <TRN_IP>
set -u
CAP=${1:?capture ip}; TRN=${2:?trainer ip}
RUN=${RUN:?}; NUM_EPOCHS=${NUM_EPOCHS:?}; LR=${LR:?}; SEED=${SEED:?}; STEP_OFFSET=${STEP_OFFSET:?}
EPOCH_OFFSET=${EPOCH_OFFSET:?}; TOTAL_EPOCHS=${TOTAL_EPOCHS:?}; WARMUP=${WARMUP:-0.01}; W_RUN_ID=${WANDB_RUN_ID:-dense-same-r6}
BASE=/scratch/dense; SF=$BASE/SpecForge; OUT=$BASE/outputs/$RUN
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
if [ -z "${CKPT:-}" ]; then
  PREV_RUN=${PREV_RUN:?CKPT or PREV_RUN}; PREV_MIN_STEP=${PREV_MIN_STEP:?}; PREV_OUT=$BASE/outputs/$PREV_RUN
  prev_alive() { pgrep -f "specforge.cli trai[n].*recipe-${PREV_RUN}[.]yaml" >/dev/null; }
  prev_ckpt() { local p=$PREV_OUT/$PREV_RUN-latest; if [ -L "$p" ]; then readlink -f "$p"; elif [ -f "$p" ]; then readlink -f "$(cat "$p")"; fi; }
  log "waiting for $PREV_RUN to finish (final checkpoint step >= $PREV_MIN_STEP, trainer exited)"
  while :; do
    ck=$(prev_ckpt); s=${ck##*-step}
    if ! prev_alive && [ -n "$ck" ] && [ -f "$ck/training_state.pt" ] && [ "${s:-0}" -ge "$PREV_MIN_STEP" ] 2>/dev/null; then break; fi
    if ! prev_alive && [ "${s:-0}" -lt "$PREV_MIN_STEP" ] 2>/dev/null && [ -f "$PREV_OUT/launch.log" ] && grep -q "ChildFailedError" "$PREV_OUT/launch.log"; then
      log "PREV_FAILED: $PREV_RUN exited at checkpoint step ${s:-none} < $PREV_MIN_STEP; not launching $RUN"; exit 1; fi
    sleep 300
  done
  CKPT=$ck; log "$PREV_RUN finished; warm start from $CKPT"; sleep 120
fi
[ -f "$CKPT/training_state.pt" ] || { log "LAUNCH_FAILED: no checkpoint at $CKPT"; exit 1; }
for i in $(seq 1 30); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && break; log "GPUs still busy, waiting"; sleep 20; done
cd $SF || exit 1
R=$BASE/recipe-$RUN.yaml
sed -e "s/CAPTURE_IP/$CAP/g" -e "s/TRAINER_IP/$TRN/g" examples/configs/online/disaggregated/external/qwen3-8b-dspark-dense-same-2node-h200.yaml > "$R"
log "recipe rendered: $R ($(git log -1 --format=%h))"
pgrep -f "mooncake_master --enable_http_metadata_server" >/dev/null || { CUDA_VISIBLE_DEVICES= setsid nohup mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=0.0.0.0 --rpc_port=35551 --http_metadata_server_port=35880 --metrics_port=35903 --default_kv_lease_ttl=300000 > $BASE/logs/mooncake_master.log 2>&1 < /dev/null & sleep 3; log "mooncake_master restarted"; }
for i in $(seq 0 7); do until curl -sf -m 3 "http://$CAP:$((30000+i))/health" >/dev/null; do log "waiting for capture server $i on $CAP"; sleep 20; done; done; log "all 8 capture servers healthy"
export HF_HOME=/scratch/hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HF_DATASETS_CACHE=/scratch/hf-cache/datasets HF_TOKEN="$(cat $BASE/hf_token)"
export WANDB_API_KEY="$(cat $BASE/wandb_key)" WANDB_ENTITY=eigent_radixark_training WANDB_RUN_ID=$W_RUN_ID WANDB_RESUME=allow
export SPECFORGE_TRACKING_STEP_OFFSET=$STEP_OFFSET
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK
OV="model.draft_checkpoint_path=$CKPT training.num_epochs=$NUM_EPOCHS training.learning_rate=$LR training.warmup_ratio=$WARMUP training.prompt_seed=$SEED"
OV="$OV run_id=$RUN output_dir=$OUT deployment.disaggregated.control_dir=$OUT/control deployment.disaggregated.consumer_state_dir=$OUT/consumer-state tracking.wandb_dir=$OUT/wandb"
launch() {
  mkdir -p $OUT; log "launching $RUN: $OV (W&B $W_RUN_ID, step offset $STEP_OFFSET)"
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
cd $SF && RUN_ID=$RUN EPOCH_OFFSET=$EPOCH_OFFSET TOTAL_EPOCHS=$TOTAL_EPOCHS GLOBAL_STEP_OFFSET=$STEP_OFFSET setsid nohup bash scripts/launch/dense-same/watch_exports.sh $((EPOCH_OFFSET + 1)) $TOTAL_EPOCHS >> $BASE/logs/watch_exports_$RUN.log 2>&1 < /dev/null &
log "export watcher for epochs $((EPOCH_OFFSET + 1))-$TOTAL_EPOCHS pid $! (log $BASE/logs/watch_exports_$RUN.log)"
setsid nohup bash scripts/launch/dense-same/reap_producer.sh $RUN >> $BASE/logs/reap_$RUN.log 2>&1 < /dev/null &
log "producer reaper pid $!"
log "CONTINUATION_LAUNCHED $RUN"
