#!/bin/bash
# Devbox-only: start the 5-epoch dense-1/8 run (PR #1003 venv, same recipe as dense-1-8-only-2-epoch with num_epochs 5)
# and the per-epoch export/upload watcher (uploads to yushengsu/dense-1-8-only-5-epoch), both detached.
W=/scratch/pr1003; RUN=dense-1-8-only-5-epoch
DRAFT=$W/configs/qwen3-8b-dspark-dense-1-8-2layer-causal-noconf-b16.json
cd $W/SpecForge && . $W/venv/bin/activate
export HF_HOME=/scratch/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
set -a; . /root/.secrets/wandb.env; set +a   # WANDB_API_KEY only; never echoed
mkdir -p $W/logs $W/outputs/$RUN && rm -rf $W/outputs/$RUN/control
setsid nohup python -m specforge.cli train --config $W/train_$RUN.yaml > $W/logs/train_$RUN.log 2>&1 < /dev/null &
echo "train pid $! ($RUN) $(date -u +%FT%TZ)"
RUN=$RUN DRAFT=$DRAFT EPOCHS=5 setsid nohup bash $W/watch_epochs_5ep.sh > $W/logs/watch_$RUN.log 2>&1 < /dev/null &
echo "watcher pid $!"
