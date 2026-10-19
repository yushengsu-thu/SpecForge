#!/bin/bash
# Full acceptance-length suite for one draft checkpoint on the capture node (rank 0): temperature 0 and 1, verify
# width 16 and 8, 2400 requests each (jiapingW/Qwen3-8b-dspark-SpecForge protocol), then the comparison table.
# usage: bench_all.sh <label> <draft_dir>        results: /scratch/dense/bench/results/<label>/<V16|V8>_t<0|1>/
set -u
LABEL=${1:?label}; DRAFT=${2:?draft dir}; SF=/scratch/dense/SpecForge/scripts/launch/dense-same
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
for T in 0 1; do for MODE in V16 V8; do
  if [ -f /scratch/dense/bench/results/$LABEL/${MODE}_t$T/summary.json ]; then log "$LABEL $MODE t=$T already done"; continue; fi
  bash $SF/bench_node.sh "$LABEL" "$DRAFT" "$MODE" "$T" || log "$LABEL $MODE t=$T FAILED"
done; done
python3 $SF/bench_table.py --results /scratch/dense/bench/results > /scratch/dense/bench/results/bench_table.md 2>&1
log "ALL_BENCH_DONE $LABEL"
