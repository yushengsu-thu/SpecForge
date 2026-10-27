#!/bin/bash
# One box: the 3-step smoke, then the full six-epoch run and the per-epoch export watcher once the smoke exited cleanly.
D=/scratch/moe8b16; S=/scratch/SpecForge/scripts/launch/qwen3-8b-moe16-1node-h200; RUN=qwen3-8b-moe16x768-16A1-dspark-2server-dp6-h200
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p $D/logs; cd /scratch/SpecForge && bash $S/launch_1node.sh smoke
sleep 30; while pgrep -f "specforge.cli trai[n]" >/dev/null; do sleep 20; done
OUT=$D/outputs/$RUN-smoke
if ! ls -d "$OUT"/*-step3 >/dev/null 2>&1; then log "SMOKE_FAILED (no step3 checkpoint)"; grep -iE 'Traceback|Error' $OUT/launch.log | grep -v -iE 'warning' | tail -3 | cut -c1-200; exit 1; fi
log "smoke ok (step3 checkpoint present)"
# The smoke's capture servers release their ports a few seconds after the trainer exits; the managed-local launcher refuses a busy port.
for i in $(seq 1 30); do (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -qE ":3000[01] " || break; sleep 5; done
bash $S/launch_1node.sh full
sleep 60; setsid nohup bash $S/watch_epoch_exports.sh > $D/logs/watch_epoch_exports.log 2>&1 < /dev/null &
log "FULL_LAUNCHED"
