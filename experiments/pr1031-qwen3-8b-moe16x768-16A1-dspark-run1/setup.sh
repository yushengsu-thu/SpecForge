#!/bin/bash
# Devbox-only (specforge-exp-branch, lmsysorg/sglang:v0.5.18-cu130 image): this branch's code in its own venv, with a
# venv-local SGLang 0.5.18 carrying the spec-capture patch, plus the training-only packages the image lacks (wandb,
# accelerate, Liger). The target model and the data are reused from /scratch/moe2ep on that box. Idempotent.
set -uo pipefail
W=/scratch/pr1031; BR=exp/pr1031-qwen3-8b-moe16x768-16A1-dspark-6ep; mkdir -p $W/logs $W/outputs $W/cache && cd $W
echo "== code $(date -u +%T)"
[ -d SpecForge ] || git clone -q --branch $BR https://github.com/yushengsu-thu/SpecForge.git SpecForge
cd SpecForge && git fetch -q origin && git checkout -q $BR && git merge -q --ff-only origin/$BR
echo "head: $(git log -1 --format='%h %s')"; cd $W
echo "== venv $(date -u +%T)"
[ -d venv ] || python3 -m venv --system-site-packages venv
. venv/bin/activate
pip install -q --disable-pip-version-check --ignore-installed --no-deps sglang==0.5.18 2>&1 | tail -1
pip install -q --disable-pip-version-check --no-deps -e SpecForge 2>&1 | tail -1
pip install -q --disable-pip-version-check accelerate 'liger-kernel>=0.8.3,<0.9' wandb 2>&1 | grep -v "dependency resolver\|requires\|^\s*$" | tail -2
(cd SpecForge && bash scripts/apply_sglang_spec_capture_patch.sh 2>&1 | tail -2)
python -c "import specforge, sglang, torch, mooncake.store, liger_kernel, wandb, yaml, accelerate; print('specforge', specforge.__file__, '| sglang', sglang.__file__, sglang.__version__, '| liger', liger_kernel.__file__, '| gpus', torch.cuda.device_count())"
echo "mooncake_master: $(command -v mooncake_master || echo MISSING)"
for p in /scratch/moe2ep/models/Qwen3-8B/config.json /scratch/moe2ep/data/perfectblend_train_regen_temperature0_no_think.jsonl /scratch/secrets/wandb_key /scratch/secrets/hf_token; do
  [ -f $p ] && echo "ok: $p" || echo "MISSING: $p"
done
echo "SETUP DONE $(date -u +%T)"
