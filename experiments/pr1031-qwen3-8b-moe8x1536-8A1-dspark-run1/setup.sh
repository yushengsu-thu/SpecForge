#!/bin/bash
# Devbox-only (specforge-qwen-h200, lmsysorg/sglang:v0.5.18-cu130 image): this branch's code in its own venv, with a
# venv-local SGLang 0.5.18 carrying the spec-capture patch. Idempotent.
set -uo pipefail
W=/scratch/pr1031; BR=exp/pr1031-qwen3-8b-moe8x1536-8A1-dspark-6ep; mkdir -p $W/logs $W/outputs $W/cache && cd $W
echo "== code $(date -u +%T)"
[ -d SpecForge ] || git clone -q --branch dev-moe-ffn https://github.com/yushengsu-thu/SpecForge.git SpecForge
cd SpecForge && git fetch -q origin
if git rev-parse -q --verify origin/$BR > /dev/null; then
  git checkout -q $BR 2>/dev/null || git checkout -q -b $BR origin/$BR
  git merge -q --ff-only origin/$BR
else
  echo "branch $BR not on origin yet; staying on $(git branch --show-current)"
fi
echo "head: $(git log -1 --format='%h %s')"; cd $W
echo "== venv $(date -u +%T)"
[ -d venv ] || python3 -m venv --system-site-packages venv
. venv/bin/activate
pip install -q --disable-pip-version-check --ignore-installed --no-deps sglang==0.5.18 2>&1 | tail -1
pip install -q --disable-pip-version-check --no-deps -e SpecForge 2>&1 | tail -1
(cd SpecForge && bash scripts/apply_sglang_spec_capture_patch.sh 2>&1 | tail -2)
python -c "import specforge, sglang, torch, mooncake.store, liger_kernel, wandb, yaml, accelerate; print('specforge', specforge.__file__, '| sglang', sglang.__file__, sglang.__version__, '| liger', liger_kernel.__file__, '| gpus', torch.cuda.device_count())"
echo "mooncake_master: $(command -v mooncake_master || echo MISSING)"
echo "SETUP DONE $(date -u +%T)"
