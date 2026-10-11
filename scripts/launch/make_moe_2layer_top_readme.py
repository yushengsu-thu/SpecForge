#!/usr/bin/env python3
"""Model card of RadixArk/qwen38-dspark-moe-2layer, in the same format as RadixArk/qwen38-dspark-dense-5layer
(one table: folder / cumulative optimizer step / train/acc over the last 100 steps / train/ce_loss / status).
Metrics come from each run's launch.log; status from the .uploaded marker of the export directory."""
import ast, os, re
from pathlib import Path

PFX = "qwen3.8-27b-dspark-moe-2layer-regen-mixture-v1"
# epoch -> (run id, run-local checkpoint step, cumulative optimizer step over the chained runs)
EPOCHS = {1: (f"{PFX}-1ep-h200", 4958, 4958),
          2: (f"{PFX}-cont2ep-from-1ep-h200", 5000, 4958 + 5000),
          3: (f"{PFX}-cont2ep-from-1ep-h200", 9916, 4958 + 9916)}
for k in range(4, 9):
    EPOCHS[k] = (f"{PFX}-ep{k}-from-ep{k-1}-h200", 4958, 4958 * k)
STEP_RE = re.compile(r"^step (\d+): (\{.*\})\s*$")
SAFE = {"__builtins__": {}, "nan": float("nan"), "inf": float("inf"), "True": True, "False": False, "None": None}

def metrics(log_path):
    out = {}
    p = Path(log_path)
    if not p.exists():
        return out
    for line in p.read_text(errors="replace").splitlines():
        m = STEP_RE.match(line)
        if not m:
            continue
        try:
            d = eval(m.group(2), SAFE)
        except Exception:
            continue
        out[int(m.group(1))] = {k.split("/")[-1]: v for k, v in d.items() if isinstance(v, (int, float))}
    return out

def at_epoch_end(ms, end, window=100):
    steps = sorted(s for s in ms if end - window < s <= end)
    if not steps:
        return None
    acc = [ms[s]["acc"] for s in steps if ms[s].get("acc") is not None]
    ce = [ms[s]["ce_loss"] for s in steps if ms[s].get("ce_loss") is not None]
    return {"acc": sum(acc) / len(acc) if acc else None, "ce": sum(ce) / len(ce) if ce else None}

rows, ckpts = [], []
cache = {}
for k in range(1, 9):
    run, step, cum = EPOCHS[k]
    export_dir = f"/scratch/exports/qwen38-dspark-moe-2layer-{k}ep-step{step}"
    if os.path.exists(f"{export_dir}/.uploaded"):
        if run not in cache:
            cache[run] = metrics(f"/scratch/outputs/{run}/launch.log")
        m = at_epoch_end(cache[run], step)
        acc = f"{m['acc']:.3f}" if m and m["acc"] is not None else "n/a"
        ce = f"{m['ce']:.2f}" if m and m["ce"] is not None else "n/a"
        rows.append(f"| `{k}ep/` | {cum} | {acc} | {ce} | uploaded |")
    else:
        rows.append(f"| `{k}ep/` | {cum} | | | pending |")
    ckpts.append(f"`{k}ep/` = `{run}-step{step}`")

body = f"""---
license: other
base_model: Qwen/Qwen3.8-27B
tags: [speculative-decoding, dspark, draft-model, sglang]
---

# Qwen3.8-27B DSpark 2-layer MoE drafter (8 epochs on Regen-Mixture-v1, one export per epoch)

Private. A MoE DSpark draft model for `Qwen/Qwen3.8-27B`: Kan Wu's 5-layer MoE drafter (SpecForge branch
`kan/moe-3-qwen38`) with the draft reduced from 5 to 2 decoder layers and nothing else changed (2 full-attention layers,
hidden 5120, 32 / 8 heads of dim 128, per layer 512 routed experts of width 512 with top-10 softmax routing plus one
sigmoid-gated shared expert of width 2048, router centering folded into `gate.bias` at export, target layers
5/19/33/47/61, Markov rank 256, confidence head, 8.48B total / about 1.08B active parameters), trained from scratch with
Kan's MoE recipe (`qwen3.8-27b-dspark-moe-regen-mixture-v1-1ep-v3.yaml`) for eight epochs as chained runs (epoch 1,
epochs 2-3, then one run per epoch, each a weights-only warm start from the previous epoch's final checkpoint with
prompt_seed 42 + epoch): global batch 256, lr 5e-4 constant after 4% warm-up (re-warmed in every run), DSpark CE 0.1 /
L1 0.9 / confidence 1.0, training block 7, bf16 teacher `Qwen/Qwen3.8-27B`, corpus `RadixArk/Qwen3.8-27B-Regen-Mixture-v1`
(1,269,290 conversations, max_length 8192, qwen3.5 chat template), one 8x H200 node (4 bf16 capture servers + 4 FSDP
ranks, Mooncake TCP). It is the 2-layer ablation of Kan's 5-layer MoE drafter (train/acc 0.594 after 1 epoch, 0.617
after 3) on the same corpus and recipe; the same-corpus dense reference is `RadixArk/qwen38-dspark-dense-5layer`.

| folder | optimizer step | train/acc (last 100 steps) | train/ce_loss | status |
|---|---|---|---|---|
{chr(10).join(rows)}

Optimizer step is cumulative over the chained runs; the exported SpecForge checkpoints are {", ".join(ckpts)}.

Each folder is a complete SGLang-servable export (`architectures: Qwen3MoeDSparkModel`, block_size 7, tokenizer
included). Serving needs the `Qwen3MoeDSparkModel` draft class (upstream SGLang 0.5.18 only ships the dense
`Qwen3DSparkModel`; the patch is `patches/sglang/v0.5.18/dspark-moe-draft.patch` on the branch below):

```
python -m sglang.launch_server --model-path Qwen/Qwen3.8-27B --speculative-algorithm DSPARK \\
  --speculative-draft-model-path <local copy of 8ep/> --speculative-dspark-block-size 7 \\
  --speculative-draft-model-quantization unquant --attention-backend triton --speculative-draft-attention-backend triton
```

Code and recipe: https://github.com/yushengsu-thu/SpecForge/tree/exp/qwen3.8-27b-dspark-moe-2layer
(`configs/qwen3.8-27b-dspark-moe-v2-2layer.json`, `examples/configs/online/disaggregated/*/qwen3.8-27b-dspark-moe-2layer-*.yaml`,
`scripts/launch/`). Training curves (one W&B run per chained run):
https://wandb.ai/miles_training/specforge-2%20layer%20moe
"""
print(body, end="")
