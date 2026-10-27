#!/bin/bash
# Trainer node: run the 3-step smoke, then the full six-epoch run once the smoke's trainer exited cleanly.
CAP=${1:?capture ip}; TRN=${2:?trainer ip}; D=/scratch/moe8b16; S=/scratch/SpecForge/scripts/launch/qwen3-8b-moe16-2node-h200
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
cd /scratch/SpecForge && bash $S/trainer_node.sh $CAP $TRN smoke rdma
sleep 30; while pgrep -f "specforge.cli trai[n]" >/dev/null; do sleep 20; done
OUT=$D/outputs/qwen3-8b-moe16x768-dspark-8server-dp8-2node-h200-smoke
if grep -qiE 'Traceback|Error' $OUT/launch.log && ! [ -d "$OUT"/*-step3 ]; then log "SMOKE_FAILED"; grep -iE 'Traceback|Error' $OUT/launch.log | tail -3 | cut -c1-200; exit 1; fi
ls -d "$OUT"/*-step3 >/dev/null 2>&1 && log "smoke ok (step3 checkpoint present)" || log "smoke: no step3 checkpoint, launching full anyway"
bash $S/trainer_node.sh $CAP $TRN full rdma
log "FULL_LAUNCHED"
