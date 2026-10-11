#!/usr/bin/env python3
"""Write a model-card README.md for an exported 2-layer Qwen3.8-27B DSpark MoE drafter checkpoint."""
import json, re, sys

run, step, ep, export_dir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
log_path = f"/scratch/outputs/{run}/launch.log"
SAFE = {"__builtins__": {}, "nan": float("nan"), "inf": float("inf"), "True": True, "False": False, "None": None}

best = None
for line in open(log_path, errors="replace"):
    m = re.match(r"step (\d+): (\{.*\})\s*$", line)
    if not m:
        continue
    s = int(m.group(1))
    if s <= step and (best is None or s > best[0]):
        try:
            d = eval(m.group(2), SAFE)
            d = {(k[6:] if k.startswith('train/') else k): v for k, v in d.items()}
            best = (s, d)
        except Exception:
            pass
metrics = best[1] if best else {}
mstep = best[0] if best else None
cfg = json.load(open(f"{export_dir}/config.json"))
pos = [metrics.get(f"accuracy_position_{i}") for i in range(7)]
pos_txt = " / ".join(f"{p:.3f}" if isinstance(p, float) else "n/a" for p in pos)
def g(k, fmt="{:.3f}"):
    v = metrics.get(k)
    return fmt.format(v) if isinstance(v, (int, float)) else "n/a"
if ep == 1:
    phase = "A (epoch 1, from scratch)"
elif ep in (2, 3):
    phase = f"B (2-epoch run, weights-only warm start from the epoch-1 checkpoint; run-local step {step} = end of epoch {ep})"
else:
    phase = (f"C (one-epoch run `ep{ep}-from-ep{ep-1}`, weights-only warm start from the epoch-{ep-1} checkpoint, "
             f"prompt_seed {42 + ep - 1}; run-local step {step} = end of epoch {ep})")
readme = f"""---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags: [specforge, dspark, speculative-decoding, moe, draft-model, qwen3.8]
---
# qwen38-dspark-moe-2layer-{ep}ep

DSpark draft model for **Qwen3.8-27B** with a **2-layer MoE** backbone: an ablation of Kan Wu's 5-layer
Qwen3.8-27B DSpark MoE drafter (SpecForge branch `kan/moe-3-qwen38`) with the draft reduced from 5 to 2 decoder
layers and nothing else changed. Export of training phase {phase}, SpecForge checkpoint `{run}-step{step}`.

## Architecture

- Backbone: 2 full-attention layers, hidden 5120, 32 query / 8 KV heads (head dim 128), block size 7,
  Markov head rank 256, confidence head, mask token 248070, target capture layers 5/19/33/47/61 (concatenated and
  projected once by `fc`, shared by both layers).
- FFN per layer: {cfg.get('n_routed_experts', cfg.get('num_experts'))} routed experts of width {cfg.get('moe_intermediate_size')},
  top-{cfg.get('num_experts_per_tok')} softmax routing with renormalisation, one sigmoid-gated shared expert of width
  {cfg.get('shared_expert_intermediate_size')}. Router input centering is folded into `gate.bias` at export.
- 8.48B total parameters (experts 8.12B), about 1.08B active per token.
- `config.json` `architectures`: `{cfg.get('architectures')}`.

## Training

- Code: SpecForge `kan/moe-3-qwen38` @ `abf003e9` plus the config/recipes on
  https://github.com/yushengsu-thu/SpecForge/tree/exp/qwen3.8-27b-dspark-moe-2layer (no source changes).
- Data: `RadixArk/Qwen3.8-27B-Regen-Mixture-v1` (1,269,290 conversations, chat template qwen3.5, max_length 8192);
  one epoch = 4,958 optimizer steps at global batch 256 (batch 4 x accumulation 16 x 4 ranks).
- Epoch plan (8 epochs total): phase A = epoch 1 from scratch; phase B = epochs 2-3 in one run (warm start from A);
  epochs 4-8 = one run per epoch, each a weights-only warm start from the previous epoch's final checkpoint with
  prompt_seed 42 + (epoch - 1), i.e. the prompt order a single 8-epoch run would use.
- Objective: DSpark (CE 0.1 + L1 0.9 + confidence 1.0, 512 anchors, loss decay gamma 4), lr 5e-4 constant after 4%
  warm-up (re-warmed in every run), AdamW CPU-offloaded, FSDP FULL_SHARD, bf16, seed 42.
- Hardware: one 8x H200 node; target served in bf16 (`Qwen/Qwen3.8-27B`) by 4 TP1 SGLang 0.5.18 capture servers
  (Kan's run used the NVFP4 build on B200).
- Training metrics at logged step {mstep}: train/acc {g('acc')}, loss {g('loss', '{:.2f}')}, ce {g('ce_loss', '{:.2f}')},
  l1 {g('l1_loss', '{:.2f}')}, per-position accuracy {pos_txt}, experts unused {g('moe/experts_unused_frac')},
  load_max_ratio {g('moe/load_max_ratio', '{:.1f}')}.
- Live curves: https://wandb.ai/miles_training/specforge-2%20layer%20moe

## Serving

This drafter needs an SGLang build with the `Qwen3MoeDSparkModel` draft class (Qwen3-style attention + Qwen3.5-MoE
FFN); upstream SGLang 0.5.18 only ships the dense `Qwen3DSparkModel`. Launch like the dense Qwen3.8 drafter:
`--speculative-algorithm DSPARK --speculative-dspark-block-size 7 --speculative-draft-model-path <this repo>
--speculative-draft-model-quantization unquant --attention-backend triton --speculative-draft-attention-backend triton`.

Accept-length evaluation (GSM8K / MATH500 / MT-Bench, accepted drafts per verify step) will be added here once measured.

## Reference points

Kan's 5-layer MoE drafter (same recipe, NVFP4 target, B200): train/acc 0.594 after 1 epoch, 0.617 after 3 epochs;
accepted drafts/step on GSM8K 4.18 (1 ep) and 4.56 (3 ep) versus 4.33 for the official dense `RadixArk/Qwen3.8-27B-DSpark`.
"""
open(f"{export_dir}/README.md", "w").write(readme)
print(f"README written for {run} step {step} (metrics from logged step {mstep})")
