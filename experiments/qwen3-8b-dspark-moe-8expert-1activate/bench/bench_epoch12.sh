#!/usr/bin/env bash
# Epoch-12 benchmark of the MoE drafter against the dense-1/8 drafter on this 8x H200 box, run by continue_to_18.sh
# after the 12-epoch training run has finished and before epochs 13-18 start.
#  1. drafts: MoE = this run's 12ep HF export; dense-1/8 = yushengsu/dense-1-8 epoch-12 (waits up to DENSE_WAIT_S for it);
#     both normalized for SGLang (scripts/gates/normalize_dflash_export.py --block-size 16).
#  2. acceptance length, jiapingW's protocol (bench_accept.py): 8 servers = {MoE, dense} x {V16, V8} x 2 replicas.
#  3. end-to-end throughput (e2e_one.sh): baseline (no draft), MoE V16, dense V16, MoE V8, dense V8 on GPUs 0-4.
#  4. tables (summarize.py) -> /scratch/bench12/epoch12/summary.md + summary.json; copied into the repo
#     (experiments/.../bench/results_epoch12/) and committed; uploaded to the MoE HF repo under bench/epoch12/.
# Log: /scratch/bench12/epoch12/bench.log. Exit 0 = all phases ran (individual errors are inside the results).
set -u
REPO=${REPO:-/scratch/SpecForge-spark8b}; BREPO=$REPO   # serving classes + bench scripts are on the branch
RUN=${RUN:-qwen3-8b-dspark-moe-8expert-1activate-dense-same}
MOE_SRC=${MOE_SRC:-/scratch/exports/$RUN/12ep}
DENSE_REPO=${DENSE_REPO:-yushengsu/dense-1-8}; DENSE_DIR=${DENSE_DIR:-epoch-12}; DENSE_WAIT_S=${DENSE_WAIT_S:-7200}
HF_MOE=${HF_MOE:-yushengsu/qwen3-8b-dspark-moe-8expert-1activate}
W=/scratch/bench12/epoch12; DR=$W/drafts; R=$W/results; LG=$W/logs
BS=experiments/qwen3-8b-dspark-moe-8expert-1activate/bench
mkdir -p "$DR" "$R" "$LG"
LOG=$W/bench.log
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
export HF_HOME=/scratch/hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HUGGINGFACE_HUB_CACHE=/scratch/hf-cache/hub TRANSFORMERS_CACHE=/scratch/hf-cache/hub HF_DATASETS_CACHE=/scratch/hf-cache/datasets PYTHONPATH=$BREPO   # /etc/environment points the HF caches at read-only /cluster-storage
stop_servers(){
  python3 - <<'PY'
import os, signal, time
pids = []
for pid in os.listdir("/proc"):
    if not pid.isdigit(): continue
    try: cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode()
    except Exception: continue
    if "sglang.launch_server" in cmd and " --port 231" in cmd: pids.append(int(pid))
for p in pids:
    try: os.kill(p, signal.SIGTERM)
    except Exception: pass
time.sleep(15)
for p in pids:
    try: os.kill(p, signal.SIGKILL)
    except Exception: pass
print("stopped", len(pids), "server processes")
PY
}
wait_healthy(){  # ports...
  for p in "$@"; do
    for i in $(seq 1 90); do curl -sf -m 3 "http://127.0.0.1:$p/health" >/dev/null && break; sleep 10; done
    curl -sf -m 3 "http://127.0.0.1:$p/health" >/dev/null || { log "server on port $p NOT healthy"; return 1; }
  done
}
# ---- 1. drafts ---------------------------------------------------------------------------------------------
cd "$BREPO" || exit 1
log "bench start; MoE export $MOE_SRC"
for i in $(seq 1 60); do [ -f "$MOE_SRC/.exported" ] && break; sleep 30; done
[ -f "$MOE_SRC/.exported" ] || { log "ABORT: MoE 12ep export missing"; exit 1; }
rm -rf "$DR/moe-12ep" && cp -r "$MOE_SRC" "$DR/moe-12ep" && rm -f "$DR/moe-12ep/.exported" "$DR/moe-12ep/.uploaded"
python scripts/gates/normalize_dflash_export.py --config "$DR/moe-12ep/config.json" --block-size 16 >> "$LOG" 2>&1 || { log "ABORT: normalize MoE failed"; exit 1; }
t0=$(date +%s); DENSE_OK=0
while [ $(( $(date +%s) - t0 )) -lt "$DENSE_WAIT_S" ]; do
  if python3 -c "
from huggingface_hub import HfApi; import sys
fs = HfApi().list_repo_files('$DENSE_REPO'); sys.exit(0 if '$DENSE_DIR/model.safetensors' in fs and '$DENSE_DIR/config.json' in fs else 1)" 2>/dev/null; then DENSE_OK=1; break; fi
  log "waiting for $DENSE_REPO/$DENSE_DIR ..."; sleep 300
done
if [ "$DENSE_OK" = 1 ]; then
  python3 -c "
from huggingface_hub import snapshot_download; snapshot_download('$DENSE_REPO', allow_patterns=['$DENSE_DIR/*'], local_dir='$W/dl')" >> "$LOG" 2>&1
  rm -rf "$DR/dense18-12ep" && cp -r "$W/dl/$DENSE_DIR" "$DR/dense18-12ep"
  python scripts/gates/normalize_dflash_export.py --config "$DR/dense18-12ep/config.json" --block-size 16 >> "$LOG" 2>&1 || DENSE_OK=0
fi
log "drafts ready: MoE yes, dense-1/8 $([ $DENSE_OK = 1 ] && echo yes || echo NO)"
# ---- 2. acceptance length ----------------------------------------------------------------------------------
stop_servers >> "$LOG" 2>&1
declare -a CFG=()      # model:draftdir:width:gpuA:gpuB
CFG+=("MoE:$DR/moe-12ep:16:0:1" "MoE:$DR/moe-12ep:8:2:3")
[ "$DENSE_OK" = 1 ] && CFG+=("dense-1/8:$DR/dense18-12ep:16:4:5" "dense-1/8:$DR/dense18-12ep:8:6:7")
PORTS=()
for c in "${CFG[@]}"; do IFS=: read -r M D WD GA GB <<< "$c"
  for g in $GA $GB; do p=$((23100 + g)); PORTS+=($p)
    setsid nohup bash $BS/serve.sh "$D" "$g" "$p" "$WD" accept 0.80 > "$LG/serve_accept_g$g.log" 2>&1 < /dev/null &
  done
done
wait_healthy "${PORTS[@]}" || { log "accept servers failed"; }
log "accept phase: ${#CFG[@]} configs on ${#PORTS[@]} servers"
APIDS=()
for c in "${CFG[@]}"; do IFS=: read -r M D WD GA GB <<< "$c"; tag=$(echo "$M" | tr -d '/-' | tr 'A-Z' 'a-z')
  python $BS/bench_accept.py --servers "http://127.0.0.1:$((23100 + GA)),http://127.0.0.1:$((23100 + GB))" \
    --label "accept-$tag-v$WD" --meta "{\"model\": \"$M\", \"width\": $WD, \"draft_dir\": \"$D\"}" \
    --out "$R/accept_${tag}_v$WD.json" > "$LG/accept_${tag}_v$WD.out" 2>&1 &
  APIDS+=($!)
done
for p in "${APIDS[@]}"; do wait "$p"; done
log "accept phase done"
stop_servers >> "$LOG" 2>&1
# ---- 3. end-to-end throughput ------------------------------------------------------------------------------
declare -a E2E=("baseline:none:16:0" "moe-v16:$DR/moe-12ep:16:1" "moe-v8:$DR/moe-12ep:8:3")
[ "$DENSE_OK" = 1 ] && E2E+=("dense18-v16:$DR/dense18-12ep:16:2" "dense18-v8:$DR/dense18-12ep:8:4")
PORTS=()
for c in "${E2E[@]}"; do IFS=: read -r L D WD G <<< "$c"; p=$((23100 + G)); PORTS+=($p)
  setsid nohup bash $BS/serve.sh "$D" "$G" "$p" "$WD" e2e 0.80 > "$LG/serve_e2e_$L.log" 2>&1 < /dev/null &
done
wait_healthy "${PORTS[@]}" || { log "e2e servers failed"; }
log "e2e phase: ${#E2E[@]} servers"
EPIDS=()
for c in "${E2E[@]}"; do IFS=: read -r L D WD G <<< "$c"
  bash $BS/e2e_one.sh "$L" $((23100 + G)) "$R" > "$LG/e2e_$L.out" 2>&1 &
  EPIDS+=($!)
done
for p in "${EPIDS[@]}"; do wait "$p"; done
log "e2e phase done"
stop_servers >> "$LOG" 2>&1
# ---- 4. tables, repo, HF -----------------------------------------------------------------------------------
python $BS/summarize.py "$R" --md "$W/summary.md" --json "$W/summary.json" > /dev/null 2>> "$LOG"
cd "$REPO" && mkdir -p "$BS/results_epoch12" && cp "$W/summary.md" "$W/summary.json" "$R"/*.json "$BS/results_epoch12/" 2>/dev/null
git add "$BS/results_epoch12" && git -c user.name="Yusheng Su" -c user.email=yushengsu.thu@gmail.com commit -q -m "bench: epoch-12 acceptance length and e2e throughput, MoE vs dense-1/8

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >> "$LOG" 2>&1 && log "results committed $(git rev-parse --short HEAD)"
hf upload "$HF_MOE" "$W/summary.md" bench/epoch12/summary.md --commit-message "epoch-12 benchmark summary" >> "$LOG" 2>&1
hf upload "$HF_MOE" "$W/summary.json" bench/epoch12/results.json --commit-message "epoch-12 benchmark results" >> "$LOG" 2>&1
log "BENCH_DONE"
