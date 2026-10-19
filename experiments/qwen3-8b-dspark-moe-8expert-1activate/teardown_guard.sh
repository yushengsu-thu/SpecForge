#!/usr/bin/env bash
# teardown_guard.sh <output dir> [grace seconds]
# After a managed_local run's consumer has finished (control/refs.jsonl.consumer_done), the producer's post-run cleanup
# re-reads the whole refs.jsonl on every 1024-ref poll (quadratic in the ref count: 33 GB / 17M refs after 12 epochs),
# so the launcher, the producer, the capture servers and Mooncake never exit. Everything that matters (final
# checkpoint, W&B summary) is written before that point. This guard waits for consumer_done, gives the run
# <grace> seconds (default 900) to exit by itself, then SIGTERMs the launcher and SIGKILLs its process tree.
set -u
OUT=${1:?output dir}; GRACE=${2:-900}; LOG=/scratch/logs/$(basename "$OUT").teardown_guard.log
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
log "guard started for $OUT (grace ${GRACE}s)"
until [ -f "$OUT/control/refs.jsonl.consumer_done" ]; do sleep 300; done
log "consumer_done seen"; sleep "$GRACE"
PID=$(cat "$OUT/launcher.pid" 2>/dev/null)
if [ -z "$PID" ] || [ ! -d "/proc/$PID" ]; then log "launcher already exited"; exit 0; fi
TREE=$(python3 - "$PID" <<'PY'
import os, sys
root = int(sys.argv[1]); kids = {}
for p in os.listdir("/proc"):
    if p.isdigit():
        try: kids.setdefault(int(open(f"/proc/{p}/stat").read().split()[3]), []).append(int(p))
        except Exception: pass
out, stack = [], [root]
while stack:
    p = stack.pop(); out.append(p); stack += kids.get(p, [])
print(" ".join(map(str, out)))
PY
)
log "launcher $PID still alive ${GRACE}s after consumer_done; stopping tree: $TREE"
kill -TERM "$PID" 2>/dev/null; sleep 60
for p in $TREE; do [ -d "/proc/$p" ] && kill -KILL "$p" 2>/dev/null; done
log "done; GPUs: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
