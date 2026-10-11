#!/usr/bin/env python3
"""Top-level README for yushengsu/qwen38-dspark-moe-2layer: one folder per epoch (1-8), status from upload markers."""
import glob, os, re
descs = {1: "phase A (from scratch), step 4958 = end of epoch 1",
         2: "phase B (weights-only warm start from the epoch-1 checkpoint), run-local step 5000 = end of epoch 2",
         3: "phase B, run-local step 9916 = end of epoch 3"}
for ep in range(4, 9):
    descs[ep] = (f"one-epoch run `ep{ep}-from-ep{ep-1}` (weights-only warm start from the epoch-{ep-1} checkpoint, "
                 f"prompt_seed {42 + ep - 1}), step 4958 = end of epoch {ep}")
rows = []
for ep in range(1, 9):
    done = glob.glob(f"/scratch/exports/qwen38-dspark-moe-2layer-{ep}ep-step*/.uploaded")
    status = "pending (training in progress)"
    if done:
        d = os.path.dirname(done[0]); acc = "n/a"
        try:
            m = re.search(r"train/acc ([0-9.]+)", open(os.path.join(d, "README.md")).read()); acc = m.group(1) if m else acc
        except Exception:
            pass
        status = f"uploaded (train/acc {acc})"
    rows.append(f"| [`{ep}ep/`](./{ep}ep) | {descs[ep]} | {status} |")
print(f"""---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags: [specforge, dspark, speculative-decoding, moe, draft-model, qwen3.8]
---
# qwen38-dspark-moe-2layer

2-layer MoE DSpark draft models for **Qwen3.8-27B**: an ablation of Kan Wu's 5-layer Qwen3.8-27B DSpark MoE drafter
(SpecForge branch `kan/moe-3-qwen38`) with the draft reduced from 5 to 2 decoder layers and nothing else changed
(512 routed experts x 512 wide, top-10 softmax routing, sigmoid-gated shared expert of width 2048, router centering;
8.48B total / about 1.08B active parameters). Planned training length: 8 epochs (phase A = epoch 1, phase B = epochs
2-3, then one run per epoch for epochs 4-8, each warm-started from the previous epoch's final checkpoint).

One servable export per training epoch, each in its own folder (config.json, model.safetensors, tokenizer and a README
with the training metrics of that checkpoint):

| folder | phase / checkpoint | status |
|---|---|---|
{chr(10).join(rows)}

- Training recipes and code: https://github.com/yushengsu-thu/SpecForge/tree/exp/qwen3.8-27b-dspark-moe-2layer
- Live training metrics: https://wandb.ai/miles_training/specforge-2%20layer%20moe
- Data: `RadixArk/Qwen3.8-27B-Regen-Mixture-v1`; target served in bf16 (`Qwen/Qwen3.8-27B`) on 8x H200.

Serving needs an SGLang build with the `Qwen3MoeDSparkModel` draft class (upstream SGLang 0.5.18 only ships the dense
`Qwen3DSparkModel`). Download one folder, e.g. `hf download yushengsu/qwen38-dspark-moe-2layer --include "1ep/*"`, and pass
it as `--speculative-draft-model-path` together with `--speculative-algorithm DSPARK --speculative-dspark-block-size 7
--speculative-draft-model-quantization unquant --attention-backend triton --speculative-draft-attention-backend triton`.

Accept-length evaluations (GSM8K / MATH500 / MT-Bench, accepted drafts per verify step) will be added as they are measured.
""")
