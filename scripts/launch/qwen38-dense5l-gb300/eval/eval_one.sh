#!/bin/bash
# usage: eval_one.sh <k>   -> serves drafts/dense5l-<k>ep on GPU k-1, port 3020k, runs all datasets at c=8, stops the server
K=${1:?k}; E=/scratch/dense5l-eval; G=$((K-1)); P=$((30200+K)); N=dense5l-${K}ep
export HF_HOME=$E/hf-cache HF_HUB_CACHE=$E/hf-cache/hub HUGGINGFACE_HUB_CACHE=$E/hf-cache/hub HF_DATASETS_CACHE=$E/hf-cache/datasets HF_TOKEN="$(cat /personal/specforge-qwen38-backup/hf_token)"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
setsid nohup bash $E/serve_dense.sh $E/drafts/$N $G $P > $E/logs/serve_$N.log 2>&1 < /dev/null &
SP=$!; log "server $N pid $SP gpu $G port $P"
for t in $(seq 1 120); do curl -sf -m 3 http://127.0.0.1:$P/health >/dev/null && break; grep -q -E "Traceback|CUDA out of memory|scheduler died" $E/logs/serve_$N.log && { log "SERVER FAILED"; tail -5 $E/logs/serve_$N.log; exit 1; }; sleep 10; done
curl -sf -m 3 http://127.0.0.1:$P/health >/dev/null || { log "SERVER NOT HEALTHY"; exit 1; }
log "server healthy; eval start"
python3 $E/eval_drafters_lite.py --name $N --base-url http://127.0.0.1:$P --model $E/models/Qwen3.8-27B \
  --datasets gsm8k,mt_bench,math500,traces,aime26 --concurrency 8 --timeout 7200 --out $E/logs/eval_$N.json 2>&1 | grep -v -iE "warning|hf_[a-z]"
log "eval finished; stopping server"; pkill -TERM -P $SP 2>/dev/null; kill -TERM $SP 2>/dev/null; sleep 10; pkill -9 -f "launch_server.*--port $P\$" 2>/dev/null; log "EVAL_ONE_DONE $N"
