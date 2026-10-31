#!/bin/bash
# Devbox-only (specforge-qwen-h200, SGLang 0.5.18 image): PR #1003 code (dev-moe-draft @ cde2f059) in its own venv.
set -uo pipefail
W=/scratch/pr1003; mkdir -p $W/logs $W/outputs $W/cache && cd $W
echo "== code $(date -u +%T)"
[ -d SpecForge ] || git clone -q --branch dev-moe-draft https://github.com/yushengsu-thu/SpecForge.git SpecForge
cd SpecForge && git checkout -q cde2f059 && git checkout -q -- . && echo "head: $(git log -1 --format='%h %s')"; cd $W
echo "== venv $(date -u +%T)"
[ -d venv ] || python3 -m venv --system-site-packages venv
. venv/bin/activate
pip install -q --disable-pip-version-check --ignore-installed --no-deps sglang==0.5.18 2>&1 | tail -1
pip install -q --disable-pip-version-check --no-deps -e SpecForge 2>&1 | tail -1
(cd SpecForge && bash scripts/apply_sglang_spec_capture_patch.sh 2>&1 | tail -2)
python -c "import specforge, sglang, torch, mooncake.store, liger_kernel, wandb, yaml, accelerate; print('specforge', specforge.__file__, '| sglang', sglang.__file__, sglang.__version__, '| gpus', torch.cuda.device_count())"
echo "mooncake_master: $(command -v mooncake_master || echo MISSING)"
echo "SETUP DONE $(date -u +%T)"
