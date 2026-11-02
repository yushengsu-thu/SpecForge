#!/usr/bin/env bash
# Accept length + same-GPU e2e of the dense-1-8-only-5-epoch exports (1ep..5ep) on specforge-qwen-h200.
# Accept length: jiapingW protocol (5 tasks, 2400 requests, temperature 0, 2048 tokens, one request at a time, CUDA graphs
# off), widths 8 and 16, four servers per model (eight for the last one). E2E: the best epoch (pooled request-average tau at
# width 16) vs the 2-epoch run's 2ep export, served one after the other on the same GPU (three GPUs per width, one with the
# order reversed), baseline without draft on GPUs 0-1; concurrency 1/16/64/128, 512 new tokens, CUDA graphs on, 128 running
# requests. Reuses /scratch/conc-sweep-epoch12/{serve.sh,sweep2_one.sh,stop_servers.py,stop_port.py} and the exp branch's
# bench_accept.py, exactly as /scratch/dense-eval (2026-10-27/28) and /scratch/ep6-serving (2026-10-31) did.
set -u
W=/scratch/pr1003/eval5ep; DR=$W/drafts; R=$W/results; LG=$W/logs; RE=$W/results_e2e; LE=$W/logs_e2e; mkdir -p $DR $R $LG $RE $LE
C=/scratch/conc-sweep-epoch12; EXP=/scratch/SpecForge-spark8b
BS=$EXP/experiments/qwen3-8b-dspark-moe-8expert-1activate/bench
EXPORTS=/scratch/pr1003/exports/dense-1-8-only-5-epoch; REF2EP=/scratch/pr1003/exports/dense-1-8-only-2-epoch/2ep
MARK=/scratch/IN_USE_dense-1-8-only-5-epoch.txt
export HF_HOME=/scratch/hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HUGGINGFACE_HUB_CACHE=/scratch/hf-cache/hub TRANSFORMERS_CACHE=/scratch/hf-cache/hub HF_DATASETS_CACHE=/scratch/hf-cache/datasets
LOG=$W/run.log; log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOG; }
stop_servers(){ python3 $C/stop_servers.py >> $LOG 2>&1; }
abort(){ log "ABORT: $*"; stop_servers; rm -f $MARK; exit 1; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)" -lt 1000 ]; }
all_free(){ for g in 0 1 2 3 4 5 6 7; do gpu_free $g || return 1; done; }
wait_healthy(){
  for p in "$@"; do
    for i in $(seq 1 120); do curl -sf -m 3 http://127.0.0.1:$p/health > /dev/null && break; sleep 10; done
    curl -sf -m 3 http://127.0.0.1:$p/health > /dev/null || { log "port $p NOT healthy"; return 1; }
  done
}
serve(){  # <draft dir> <gpu> <width> <label>
  REPO=$EXP setsid nohup bash $C/serve.sh "$1" "$2" $((23100 + $2)) "$3" accept 0.80 > $LG/serve_$4_g$2.log 2>&1 < /dev/null &
}
run_round(){  # <name> <model:width:gpu,gpu,...>...
  local name=$1; shift
  local ports=() c M WD GS g
  for c in "$@"; do IFS=: read -r M WD GS <<< "$c"; for g in ${GS//,/ }; do serve $DR/$M $g $WD $M-v$WD; ports+=($((23100 + g))); done; done
  wait_healthy "${ports[@]}" || abort "$name servers"
  log "$name: ${#ports[@]} servers up, benchmarking"
  local pids=()
  for c in "$@"; do IFS=: read -r M WD GS <<< "$c"
    local urls=""; for g in ${GS//,/ }; do urls+="${urls:+,}http://127.0.0.1:$((23100 + g))"; done
    (cd $EXP && PYTHONPATH=$EXP python $BS/bench_accept.py --servers "$urls" --label "accept-$M-v$WD" \
      --meta "{\"model\": \"$M\", \"width\": $WD, \"draft_dir\": \"$DR/$M\"}" --out $R/accept_${M}_v$WD.json > $LG/accept_${M}_v$WD.out 2>&1) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  stop_servers; sleep 5
  log "$name done: $(ls $R/accept_*.json 2>/dev/null | wc -l) result files"
}
one(){  # <gpu> <label> <draft dir|none> <width> "<datasets>": serve (e2e mode), sweep, stop, wait for the GPU to be free again
  local G=$1 L=$2 D=$3 WD=$4 DS=$5 p=$((23100 + $1))
  MAXRUN=128 REPO=$EXP setsid nohup bash $C/serve.sh "$D" $G $p $WD e2e 0.80 > $LE/serve_${L}_g$G.log 2>&1 < /dev/null &
  wait_healthy $p || { log "gpu $G: $L server NOT healthy"; python3 $C/stop_port.py $p >> $LOG 2>&1; return 1; }
  log "gpu $G: $L up"
  bash $C/sweep2_one.sh "$L" $p $RE "$DS" > $LE/sweep_${L}_g$G.out 2>&1
  log "gpu $G: $L done"
  python3 $C/stop_port.py $p >> $LOG 2>&1
  for i in $(seq 1 60); do gpu_free $G && return 0; sleep 5; done
  log "gpu $G: still busy after the stop"; return 1
}
pair(){  # <gpu> <first label> <first draft> <second label> <second draft> <width>
  one $1 "$2-g$1" "$3" $6 "$ALL" && one $1 "$4-g$1" "$5" $6 "$ALL"
}
ALL="gsm8k math500 humaneval mbpp mt-bench"

log "start pid $$"
all_free || abort "GPUs not free: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
printf 'dense-1-8-only-5-epoch accept-length + e2e eval (dense-1/8 session), started %s, uses all 8 GPUs for ~2.5 h, work dir %s, log %s\n' "$(date -u +%FT%TZ)" $W $LOG > $MARK
for k in 1 2 3 4 5; do [ -f $EXPORTS/${k}ep/model.safetensors ] || abort "missing export ${k}ep"; ln -sfn $EXPORTS/${k}ep $DR/dense5ep-ep$k; done
[ -f $REF2EP/model.safetensors ] || abort "missing reference export $REF2EP"
ln -sfn $REF2EP $DR/dense2ep-ep2

log "smoke test: ep5, width 16, GPU 0, 2 conversations each of gsm8k and mt-bench"
serve $DR/dense5ep-ep5 0 16 smoke
wait_healthy 23100 || abort "smoke server (see $LG/serve_smoke_g0.log)"
(cd $EXP && PYTHONPATH=$EXP python $BS/bench_accept.py --servers http://127.0.0.1:23100 --label smoke --limit 2 --datasets gsm8k,mt-bench \
  --out $W/smoke.json > $LG/smoke.out 2>&1) || abort "smoke bench (see $LG/smoke.out)"
stop_servers; sleep 5
python3 - >> $LOG 2>&1 <<'PY' || abort "smoke result (see $LOG)"
import json
d = json.load(open("/scratch/pr1003/eval5ep/smoke.json"))
tau = d["pooled"]["request_macro_tau"]
print("smoke: requests", d["pooled"]["requests"], "pooled request-average tau", tau, "errors", len(d.get("errors", [])))
assert not d.get("errors") and tau and 1.5 < tau < 16.5
PY
log "smoke OK"

log "accept length"
run_round "round 1 (ep1/ep2, width 8)" "dense5ep-ep1:8:0,1,2,3" "dense5ep-ep2:8:4,5,6,7"
run_round "round 2 (ep3/ep4, width 8)" "dense5ep-ep3:8:0,1,2,3" "dense5ep-ep4:8:4,5,6,7"
run_round "round 3 (ep5, width 8)" "dense5ep-ep5:8:0,1,2,3,4,5,6,7"
python3 $W/summarize_accept.py > $W/summary.md 2>> $LOG && log "WIDTH8_DONE: summary written to $W/summary.md"
run_round "round 4 (ep1/ep2, width 16)" "dense5ep-ep1:16:0,1,2,3" "dense5ep-ep2:16:4,5,6,7"
run_round "round 5 (ep3/ep4, width 16)" "dense5ep-ep3:16:0,1,2,3" "dense5ep-ep4:16:4,5,6,7"
run_round "round 6 (ep5, width 16)" "dense5ep-ep5:16:0,1,2,3,4,5,6,7"
python3 $W/summarize_accept.py > $W/summary.md 2>> $LOG && log "ACCEPT_DONE: summary written to $W/summary.md"

BEST=$(python3 - <<'PY'
import json
R = "/scratch/pr1003/eval5ep/results"
score = {}
for k in range(1, 6):
    v16 = json.load(open(f"{R}/accept_dense5ep-ep{k}_v16.json")); v8 = json.load(open(f"{R}/accept_dense5ep-ep{k}_v8.json"))
    score[k] = (v16["pooled"]["request_macro_tau"], v8["pooled"]["request_macro_tau"])
print(max(score, key=score.get))
PY
)
[ -n "$BEST" ] || abort "could not pick the best epoch"
log "best epoch by pooled request-average tau at width 16: ep$BEST"
echo $BEST > $W/best_epoch.txt
python3 $W/make_summarize_e2e.py $BEST >> $LOG 2>&1 || abort "summarize_e2e.py generation"

log "e2e throughput: dense5ep-ep$BEST vs dense2ep-ep2 on the same GPUs, baseline (no draft) on GPUs 0-1"
for i in $(seq 1 60); do all_free && break; sleep 5; done
all_free || abort "GPUs not free before e2e"
NEW=$DR/dense5ep-ep$BEST; OLD=$DR/dense2ep-ep2
one 0 baseline none 16 "gsm8k math500" &
one 1 baseline none 16 "humaneval mbpp mt-bench" &
pair 2 dense2ep-v8 $OLD dense5ep-v8 $NEW 8 &
pair 3 dense5ep-v8 $NEW dense2ep-v8 $OLD 8 &
pair 6 dense2ep-v8 $OLD dense5ep-v8 $NEW 8 &
pair 4 dense2ep-v16 $OLD dense5ep-v16 $NEW 16 &
pair 5 dense5ep-v16 $NEW dense2ep-v16 $OLD 16 &
pair 7 dense2ep-v16 $OLD dense5ep-v16 $NEW 16 &
wait
log "e2e done: $(ls $RE/e2e_*.json 2>/dev/null | wc -l) result files"
stop_servers
python3 $W/summarize_e2e.py $RE > $W/summary_e2e.md 2>> $LOG && log "E2E_DONE: summary written to $W/summary_e2e.md"
rm -f $MARK
log "ALL_DONE"
