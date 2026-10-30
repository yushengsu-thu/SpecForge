#!/bin/bash
# Fetch MoE-run exports into export2/step-N (skips complete ones). usage: dl_steps.sh <step>...
set -u
P=$(cd "$(dirname "$0")" && pwd)
R=yushengsu/qwen3-8b-moe16x768-16A1-dspark; D=qwen3-8b-moe8x1536-8A1-dspark-run1
dl() {
  s=$1; d=$P/export2/step-$s; mkdir -p $d
  if [ "$(stat -c %s $d/model.safetensors 2>/dev/null)" = 2253170056 ]; then echo "step-$s: already complete"; return; fi
  for f in config.json tokenizer_config.json chat_template.jinja tokenizer.json model.safetensors; do
    curl -fsSL --retry 5 --retry-delay 10 -C - -o $d/$f "https://huggingface.co/$R/resolve/main/$D/step-$s/$f" || echo "FAILED step-$s $f"
  done
  echo "step-$s: $(stat -c %s $d/model.safetensors) bytes"
}
for s in "$@"; do dl $s & done; wait
