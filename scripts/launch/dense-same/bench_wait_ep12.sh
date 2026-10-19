#!/bin/bash
# On the capture node (rank 0): wait until no other benchmark suite is running and the epoch-12 export has been
# uploaded to yushengsu/dense-same/12ep, download it, and run the full acceptance-length suite on it.
# usage: bench_wait_ep12.sh            log: /scratch/dense/bench/logs/bench_wait_ep12.out
set -u
export HF_HOME=/scratch/hf-cache HF_HUB_CACHE=/scratch/hf-cache/hub HF_TOKEN="$(cat /personal/dense-secrets/hf_token)"
M=/scratch/dense/bench/models; SF=/scratch/dense/SpecForge/scripts/launch/dense-same
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
log "waiting for yushengsu/dense-same/12ep/model.safetensors"
until python3 -c "
from huggingface_hub import HfApi
import sys; sys.exit(0 if '12ep/model.safetensors' in HfApi().list_repo_files('yushengsu/dense-same') else 1)" 2>/dev/null; do sleep 600; done
log "12ep is on HF"
while pgrep -f "bench_all[.]sh" >/dev/null; do log "another suite is running, waiting"; sleep 300; done
rm -rf $M/dl12 && python3 -m huggingface_hub.cli.hf download yushengsu/dense-same --include "12ep/*" --local-dir $M/dl12 > /dev/null 2>&1 \
  && rm -rf $M/dense-same-12ep && mv $M/dl12/12ep $M/dense-same-12ep && rm -rf $M/dl12 || { log "DOWNLOAD_FAILED"; exit 1; }
log "downloaded $(du -sh $M/dense-same-12ep | cut -f1); starting suite"
bash $SF/bench_all.sh dense-same-12ep $M/dense-same-12ep
log "EP12_BENCH_DONE"
