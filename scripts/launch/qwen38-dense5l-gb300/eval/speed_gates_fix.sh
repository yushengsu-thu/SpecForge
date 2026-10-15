#!/bin/bash
# Re-run the MoE speed gates with the HF cache env exported (sglang.bench_serving's random dataset pulls ShareGPT via
# huggingface_hub; without HF_HOME it writes to the read-only default cache and the bench dies before the server is used).
E=/scratch/dense5l-eval
export HF_HOME=$E/hf-cache HF_HUB_CACHE=$E/hf-cache/hub HUGGINGFACE_HUB_CACHE=$E/hf-cache/hub HF_DATASETS_CACHE=$E/hf-cache/datasets HF_TOKEN="$(cat /personal/specforge-qwen38-backup/hf_token)"
export SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang_models PYTHONPATH=$E/SpecForge
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
gpu_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)" -lt 2000 ]; }
run_gate() { # <gpu> <port> <results dir> <label> <draft dir> <log>
  cd $E/SpecForge && GPU=$1 PORT=$2 MAMBA_CACHE=48 MAX_RUNNING=32 PY=python3 RESULTS=$3 ACC_LEN=5.0 CONCS="1 8 32" \
    bash scripts/moe_speed/speed_test_moe_draft.sh $E/models/Qwen3.8-27B $4=$5 > $6 2>&1; cat $3/summary.tsv; }
until grep -q POOL_DONE $E/logs/pool_16x_b.log 2>/dev/null; do sleep 60; done
until gpu_free 0; do sleep 20; done
log "256exp speed gate on GPU 0"; run_gate 0 30311 $E/moe_speed_256 moe256-4ep $E/drafts/moe256-4ep $E/logs/speed_gate_256b.log; log "SPEED_256_DONE"
until grep -q FAMILIES_PIPELINE_DONE $E/logs/queue_families.log 2>/dev/null; do sleep 60; done
until gpu_free 3; do sleep 20; done
log "16x speed gate on GPU 3"; run_gate 3 30312 $E/moe_speed_16x moe16x-5ep $E/drafts/moe16x-5ep $E/logs/speed_gate_16xb.log; log "SPEED_16X_DONE"
log "SPEED_GATES_DONE"
