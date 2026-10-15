#!/bin/bash
# 256-expert first (accept-length 1ep..4ep on 4 GPUs, then its e2e + speed gate on GPUs 0/1), 16x1024 next
# (accept-length 1ep..5ep split over the GPUs as they free up, then its e2e + speed gate)
E=/scratch/dense5l-eval; log() { echo "[$(date -u +%H:%M:%S)] $*"; }
until grep -q "DOWNLOAD_DONE moe256" $E/logs/download_families.log 2>/dev/null; do sleep 60; done
printf '%s\n' "moe256-4ep $E/drafts/moe256-4ep" "moe256-3ep $E/drafts/moe256-3ep" "moe256-2ep $E/drafts/moe256-2ep" "moe256-1ep $E/drafts/moe256-1ep" > $E/jobs_256.txt
log "256exp accept-length evals start"; PORT_BASE=30400 bash $E/run_pool.sh $E/jobs_256.txt; log "256exp accept-length evals done"
until grep -q "DOWNLOAD_DONE moe16x" $E/logs/download_families.log 2>/dev/null; do sleep 60; done
# 16x1024: three jobs on GPUs 2/3 right away; two more on GPUs 0/1 after the 256exp bench
printf '%s\n' "moe16x-5ep $E/drafts/moe16x-5ep" "moe16x-4ep $E/drafts/moe16x-4ep" "moe16x-3ep $E/drafts/moe16x-3ep" > $E/jobs_16x_a.txt
printf '%s\n' "moe16x-2ep $E/drafts/moe16x-2ep" "moe16x-1ep $E/drafts/moe16x-1ep" > $E/jobs_16x_b.txt
setsid nohup env GPUS="2 3" PORT_BASE=30420 bash $E/run_pool.sh $E/jobs_16x_a.txt > $E/logs/pool_16x_a.log 2>&1 < /dev/null &
export SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang_models PYTHONPATH=$E/SpecForge
setsid nohup bash $E/bench_e2e_one.sh moe256-4ep $E/drafts/moe256-4ep 0 30221 > $E/logs/e2e_moe256-4ep.log 2>&1 < /dev/null &
cd $E/SpecForge && GPU=1 PORT=30311 MAMBA_CACHE=48 MAX_RUNNING=32 PY=python3 RESULTS=$E/moe_speed_256 ACC_LEN=5.0 CONCS="1 8 32" bash scripts/moe_speed/speed_test_moe_draft.sh $E/models/Qwen3.8-27B moe256-4ep=$E/drafts/moe256-4ep > $E/logs/speed_gate_256.log 2>&1
until grep -q -E "E2E_DONE|SERVER FAILED|NOT HEALTHY" $E/logs/e2e_moe256-4ep.log 2>/dev/null; do sleep 30; done
log "256EXP_DONE (accept-length + e2e + speed gate)"
sleep 20; env GPUS="0 1" PORT_BASE=30440 bash $E/run_pool.sh $E/jobs_16x_b.txt > $E/logs/pool_16x_b.log 2>&1
until grep -q POOL_DONE $E/logs/pool_16x_a.log 2>/dev/null; do sleep 60; done
log "16x accept-length evals done"
setsid nohup bash $E/bench_e2e_one.sh moe16x-5ep $E/drafts/moe16x-5ep 2 30222 > $E/logs/e2e_moe16x-5ep.log 2>&1 < /dev/null &
GPU=3 PORT=30312 MAMBA_CACHE=48 MAX_RUNNING=32 PY=python3 RESULTS=$E/moe_speed_16x ACC_LEN=5.0 CONCS="1 8 32" bash scripts/moe_speed/speed_test_moe_draft.sh $E/models/Qwen3.8-27B moe16x-5ep=$E/drafts/moe16x-5ep > $E/logs/speed_gate_16x.log 2>&1
until grep -q -E "E2E_DONE|SERVER FAILED|NOT HEALTHY" $E/logs/e2e_moe16x-5ep.log 2>/dev/null; do sleep 30; done
log "16X_DONE"; log "FAMILIES_PIPELINE_DONE"
