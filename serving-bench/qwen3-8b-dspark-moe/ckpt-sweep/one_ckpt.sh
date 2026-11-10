#!/bin/bash
# Evaluate ONE checkpoint of a run on jiapingW's five tasks at c=1 (same body as a sweep_all.sh worker),
# for re-running a checkpoint whose download or server failed. Usage: one_ckpt.sh <ckpt> <gpu> <port>
W=${EVAL_DIR:-/scratch/moe8b16/eval}; RUN=${RUN:-qwen3-8b-moe8x1536-8A1-dspark-run2}; SWEEP=${SWEEP:-sweep-run2}; REPO=yushengsu/qwen3-8b-moe16x768-16A1-dspark
c=$1; gpu=$2; port=$3; LOG=$W/serve/sweep_all-$SWEEP.log
export HF_HOME=/scratch/moe8b16/hf_home HF_HUB_CACHE=/scratch/moe8b16/hf_home/hub HF_TOKEN="${HF_TOKEN:-$(cat /scratch/secrets/hf_token 2>/dev/null)}"
cd $W
dir=$W/models/ckpts/$c
ok=0
for i in 1 2 3; do
  hf download $REPO --include "$RUN/$c/*" --local-dir $W/models/ckpts/dl-$c > $W/serve/$SWEEP/dl-$c.log 2>&1 && { ok=1; break; }
  sleep 60
done
[ $ok = 1 ] || { echo "[$(date -u +%T)] extra: download FAILED for $c (3 tries)" >> $LOG; exit 1; }
rm -rf $dir; mv $W/models/ckpts/dl-$c/$RUN/$c $dir && rm -rf $W/models/ckpts/dl-$c
GPU=$gpu PORT=$port MAXRR=8 GRAPH_BS=8 TASKS="gsm8k:1319 math500:500 humaneval:164 mbpp:257 mt-bench:160" CONCS="1" \
  bash $W/serve_eval.sh $SWEEP/$c $dir moe > $W/serve/$SWEEP/$c.log 2>&1
echo "[$(date -u +%T)] extra gpu $gpu: $c $(grep -oE 'c=1: Output throughput: [0-9.]+ tok/s Average acceptance length: [0-9.]+' $W/serve/$SWEEP/$c.log | sed -E 's/.*length: //' | tr '\n' ' ')" >> $LOG
rm -rf $dir
echo "[$(date -u +%T)] EXTRA_DONE $c; GPU $gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $gpu)" >> $LOG
