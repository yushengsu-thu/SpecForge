#!/usr/bin/env bash
# Devbox-only (specforge-qwen-h200): start the 6-epoch MoE DSpark run (2 capture servers on GPU 0-1, 6 DDP trainers on
# GPU 2-7) and the checkpoint watcher that exports and uploads every checkpoint to the Hub. Secrets are read from
# /scratch/secrets (W&B key, HF token) and never written anywhere else.
set -u
W=/scratch/pr1031; RUN=pr1031-qwen3-8b-moe8x1536-8A1-dspark-run1; E=$W/SpecForge/experiments/$RUN
mkdir -p $W/logs $W/outputs/$RUN
cd $W/SpecForge && . $W/venv/bin/activate
export HF_HOME=/scratch/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export WANDB_API_KEY="$(tr -d '\n' < /scratch/secrets/wandb_key)" WANDB_ENTITY=eigent_radixark_training
setsid nohup python -m specforge.cli train --config $E/train.yaml > $W/logs/train_$RUN.log 2>&1 < /dev/null &
echo "train pid $!"
# The watcher talks to the Hub, so it must not inherit the trainer's offline flags; it does not need the W&B key.
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE -u WANDB_API_KEY setsid nohup bash $E/watch_ckpts.sh > $W/logs/watch_$RUN.log 2>&1 < /dev/null &
echo "watcher pid $!"
