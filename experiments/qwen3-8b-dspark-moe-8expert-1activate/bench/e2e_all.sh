#!/usr/bin/env bash
# Standalone end-to-end throughput run for the epoch-12 drafts on all five acceptance-length tasks, for a box other than
# the training box (whose GPUs are busy with epochs 13-18):   W=<work dir> bash e2e_all.sh
#   W holds drafts/{moe-12ep,dense18-12ep} (the normalized drafts of bench_epoch12.sh), models/Qwen3-8B (revision
#   b968826d, the target used everywhere), hf-cache/ (writable HF cache with the five benchmark datasets), moe-cfg/
#   and SpecForge/ (a `git archive` of this branch; REPO overrides it).
# Same servers and flags as bench_epoch12.sh's e2e phase: baseline (no draft) on GPU 0, MoE V16 GPU 1, dense-1/8 V16
# GPU 2, MoE V8 GPU 3, dense-1/8 V8 GPU 4; serve.sh e2e mode at mem fraction 0.80; e2e_one.sh against each server.
# The borrowed E=8,N=1536 H200 fused-MoE tile config (Mixtral's E=8,N=1792 file, as on the training box) sits in
# $W/moe-cfg/configs/triton_<ver>/ and is selected with SGLANG_MOE_CONFIG_DIR, so the box's SGLang install is untouched.
# Results: $W/results/e2e_*.json, tables: $W/summary_e2e.md / .json, log: $W/logs/e2e_all.log.
set -u
W=${W:?work dir}
export REPO=${REPO:-$W/SpecForge} TARGET=${TARGET:-$W/models/Qwen3-8B} HFC=${HFC:-$W/hf-cache}
export SGLANG_MOE_CONFIG_DIR=${SGLANG_MOE_CONFIG_DIR:-$W/moe-cfg}
BS=$REPO/experiments/qwen3-8b-dspark-moe-8expert-1activate/bench; R=$W/results; LG=$W/logs; mkdir -p "$R" "$LG"
LOG=$LG/e2e_all.log; log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
stop_servers(){   # this work dir's SGLang servers only (their --model-path is under $W)
  W="$W" python3 - <<'PY'
import os, signal, time
w = os.environ["W"]; pids = []
for pid in os.listdir("/proc"):
    if not pid.isdigit(): continue
    try: cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode()
    except Exception: continue
    if "sglang.launch_server" in cmd and f"--model-path {w}/" in cmd: pids.append(int(pid))
for sig in (signal.SIGTERM, signal.SIGKILL):
    for p in pids:
        try: os.killpg(os.getpgid(p), sig)
        except Exception:
            try: os.kill(p, sig)
            except Exception: pass
    time.sleep(15 if sig == signal.SIGTERM else 2)
print("stopped", len(pids), "server processes")
PY
}
wait_healthy(){   # all ports answer /health within 20 min
  local t0=$(date +%s) p
  for p in "$@"; do
    until curl -sf -m 5 "http://127.0.0.1:$p/health" > /dev/null 2>&1; do
      [ $(( $(date +%s) - t0 )) -gt 1200 ] && { log "port $p not healthy after 20 min"; return 1; }
      sleep 10
    done
  done
}
declare -a E2E=("baseline:none:16:0" "moe-v16:$W/drafts/moe-12ep:16:1" "dense18-v16:$W/drafts/dense18-12ep:16:2"
                "moe-v8:$W/drafts/moe-12ep:8:3" "dense18-v8:$W/drafts/dense18-12ep:8:4")
log "start: REPO=$REPO TARGET=$TARGET MOE_CFG=$SGLANG_MOE_CONFIG_DIR datasets=${E2E_DATASETS:-gsm8k math500 humaneval mbpp mt-bench}"
PORTS=()
for c in "${E2E[@]}"; do IFS=: read -r L D WD G <<< "$c"; p=$((23100 + G)); PORTS+=($p)
  setsid nohup bash $BS/serve.sh "$D" "$G" "$p" "$WD" e2e 0.80 > "$LG/serve_e2e_$L.log" 2>&1 < /dev/null &
done
wait_healthy "${PORTS[@]}" || { log "e2e servers failed"; stop_servers >> "$LOG" 2>&1; exit 1; }
log "e2e: ${#E2E[@]} servers up"
EPIDS=()
for c in "${E2E[@]}"; do IFS=: read -r L D WD G <<< "$c"
  bash $BS/e2e_one.sh "$L" $((23100 + G)) "$R" > "$LG/e2e_$L.out" 2>&1 &
  EPIDS+=($!)
done
for p in "${EPIDS[@]}"; do wait "$p"; done
log "e2e done: $(ls "$R"/e2e_*.json 2>/dev/null | wc -l) result files"
stop_servers >> "$LOG" 2>&1
python3 $BS/summarize.py "$R" --md "$W/summary_e2e.md" --json "$W/summary_e2e.json" > /dev/null 2>> "$LOG"
log "E2E_ALL_DONE"
