#!/bin/bash
# dense-same trainer node: render the recipe with the real IPs, make sure mooncake_master runs, wait for the 8
# capture servers, launch the 12-epoch trainer (one W&B run: fixed WANDB_RUN_ID dense-same-r5, display name
# dense-same, + WANDB_RESUME=allow, so a
# restart continues the same curve) and the per-epoch export/upload watcher.
# usage: bash trainer_node.sh <CAPTURE_IP> <TRAINER_IP> [section.field=value ...]
#        RESUME_FROM=<checkpoint dir> bash trainer_node.sh ...   -> full resume (optimizer + step) after a crash
# secrets: /scratch/dense/hf_token and /scratch/dense/wandb_key (mode 600), never passed on the command line.
set -u
CAP=${1:?capture ip}; TRN=${2:?trainer ip}; shift 2
RUN_ID=dense-same; BASE=/scratch/dense; SF=$BASE/SpecForge; OUT=$BASE/outputs/$RUN_ID
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
mkdir -p $BASE/logs $OUT; cd $SF || exit 1
R=$BASE/recipe-$RUN_ID.yaml
sed -e "s/CAPTURE_IP/$CAP/g" -e "s/TRAINER_IP/$TRN/g" examples/configs/online/disaggregated/external/qwen3-8b-dspark-dense-same-2node-h200.yaml > "$R"
log "recipe rendered: $R ($(git log -1 --format=%h))"
pgrep -f "mooncake_master --enable_http_metadata_server" >/dev/null || { CUDA_VISIBLE_DEVICES= setsid nohup mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=0.0.0.0 --rpc_port=35551 --http_metadata_server_port=35880 --metrics_port=35903 --default_kv_lease_ttl=300000 > $BASE/logs/mooncake_master.log 2>&1 < /dev/null & sleep 3; log "mooncake_master started"; }
for i in $(seq 0 7); do until curl -sf -m 3 "http://$CAP:$((30000+i))/health" >/dev/null; do log "waiting for capture server $i on $CAP"; sleep 20; done; done; log "all 8 capture servers healthy"
export HF_HOME=$BASE/../hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HF_DATASETS_CACHE=/scratch/hf-cache/datasets HF_TOKEN="$(cat $BASE/hf_token)"
export WANDB_API_KEY="$(cat $BASE/wandb_key)" WANDB_ENTITY=eigent_radixark_training WANDB_RUN_ID=${WANDB_RUN_ID:-dense-same-r5} WANDB_RESUME=allow
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK
OV="$*"; [ -n "${RESUME_FROM:-}" ] && OV="$OV training.resume_from=$RESUME_FROM"
log "launching $RUN_ID ${OV:+overrides: $OV}"
setsid nohup python -m specforge.cli train -c "$R" $OV >> $OUT/launch.log 2>&1 < /dev/null &
log "trainer launcher pid $! log $OUT/launch.log"
pgrep -f "dense-same/watch_exports.s[h]" >/dev/null || { setsid nohup bash scripts/launch/dense-same/watch_exports.sh >> $BASE/logs/watch_exports.log 2>&1 < /dev/null & log "export watcher pid $!"; }
