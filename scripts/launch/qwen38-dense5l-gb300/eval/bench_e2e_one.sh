#!/bin/bash
# usage: bench_e2e_one.sh <label> <draft dir | none> <gpu> <port>
# PR 943 protocol: `specforge benchmark` gsm8k + mt-bench, c = 1 / 8 / 32 (100 prompts at c=1, 200 otherwise), 512 new tokens,
# greedy; reports output tok/s and (with a drafter) the acceptance length. Results: logs/e2e_<label>_<ds>_c<c>.json
L=${1:?label}; D=${2:?draft}; G=${3:?gpu}; P=${4:?port}; E=/scratch/dense5l-eval; REPO=$E/SpecForge
export HF_HOME=$E/hf-cache HF_HUB_CACHE=$E/hf-cache/hub HUGGINGFACE_HUB_CACHE=$E/hf-cache/hub HF_DATASETS_CACHE=$E/hf-cache/datasets HF_TOKEN="$(cat /personal/specforge-qwen38-backup/hf_token)" PYTHONPATH=$REPO
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
setsid nohup bash $E/serve_e2e.sh $L $D $G $P > $E/logs/serve_e2e_$L.log 2>&1 < /dev/null &
SP=$!; log "server $L pid $SP gpu $G port $P draft=$D"
for t in $(seq 1 120); do curl -sf -m 3 http://127.0.0.1:$P/health >/dev/null && break; grep -q -E "Traceback|CUDA out of memory|scheduler died" $E/logs/serve_e2e_$L.log && { log "SERVER FAILED"; tail -5 $E/logs/serve_e2e_$L.log; exit 1; }; sleep 10; done
curl -sf -m 3 http://127.0.0.1:$P/health >/dev/null || { log "SERVER NOT HEALTHY"; exit 1; }
log "server healthy; bench start"
cd $REPO
for ds in gsm8k mt-bench; do for c in 1 8 32; do n=100; [ "$c" -gt 1 ] && n=200; out=$E/logs/e2e_${L}_${ds}_c$c.json
  log "--- $L $ds c=$c ($n prompts) ---"
  python3 -m specforge.cli benchmark --model $E/models/Qwen3.8-27B --dataset $ds --num-prompts $n --concurrency $c --max-new-tokens 512 \
    --base-url http://127.0.0.1:$P --trust-remote-code --output-json $out 2>&1 | grep -E "throughput|accept|completed|Error|Traceback" | grep -v -iE "hf_[a-z]" | tail -4
done; done
log "bench finished; stopping server"; pkill -TERM -P $SP 2>/dev/null; kill -TERM $SP 2>/dev/null; sleep 10; pkill -9 -f "launch_server.*--port $P\$" 2>/dev/null; log "E2E_DONE $L"
