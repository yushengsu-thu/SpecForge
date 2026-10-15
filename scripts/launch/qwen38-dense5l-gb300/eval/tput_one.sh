#!/bin/bash
# usage: tput_one.sh <label> <draft dir | none> <gpu> <port>
# Output tokens/s at concurrency 1 / 8 / 32 on the five accept-length datasets (same prompts, chat template, thinking
# flags and max_new_tokens as the accept-length eval; greedy). Prompt counts: c=1 -> gsm8k 64, math500 32, mt_bench 32x2,
# aime26 8, traces 32; c=8 and c=32 -> gsm8k 256, math500 128, mt_bench 80x2, aime26 30 (all), traces 128.
L=${1:?label}; D=${2:?draft}; G=${3:?gpu}; P=${4:?port}; E=/scratch/dense5l-eval
export HF_HOME=$E/hf-cache HF_HUB_CACHE=$E/hf-cache/hub HUGGINGFACE_HUB_CACHE=$E/hf-cache/hub HF_DATASETS_CACHE=$E/hf-cache/datasets HF_TOKEN="$(cat /personal/specforge-qwen38-backup/hf_token)"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
setsid nohup bash $E/serve_tput.sh $D $G $P > $E/logs/serve_tput_$L.log 2>&1 < /dev/null &
SP=$!; log "server $L pid $SP gpu $G port $P draft=$D"
for t in $(seq 1 120); do curl -sf -m 3 http://127.0.0.1:$P/health >/dev/null && break; grep -q -E "Traceback|CUDA out of memory|scheduler died" $E/logs/serve_tput_$L.log && { log "SERVER FAILED"; tail -5 $E/logs/serve_tput_$L.log; exit 1; }; sleep 10; done
curl -sf -m 3 http://127.0.0.1:$P/health >/dev/null || { log "SERVER NOT HEALTHY"; exit 1; }
grep -a -o -E "max_running_requests is capped to [0-9]+[^)]*\)" $E/logs/serve_tput_$L.log | head -1
log "server healthy; throughput runs start"
for c in 1 8 32; do
  if [ "$c" -eq 1 ]; then N="--gsm8k-n 64 --math500-n 32 --mt-bench-n 32 --aime26-n 8 --traces-n 32"; else N="--gsm8k-n 256 --math500-n 128 --mt-bench-n 80 --aime26-n 30 --traces-n 128"; fi
  log "--- $L c=$c ---"
  python3 $E/eval_drafters_lite2.py --name ${L}_c$c --base-url http://127.0.0.1:$P --model $E/models/Qwen3.8-27B \
    --datasets gsm8k,math500,mt_bench,aime26,traces --concurrency $c $N --timeout 7200 --out $E/logs/tput_${L}_c$c.json 2>&1 | grep -v -iE "warning|hf_[a-z]"
done
log "runs finished; stopping server"; pkill -TERM -P $SP 2>/dev/null; kill -TERM $SP 2>/dev/null; sleep 10; pkill -9 -f "launch_server.*--port $P\$" 2>/dev/null; log "TPUT_DONE $L"
