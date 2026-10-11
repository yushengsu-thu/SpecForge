#!/usr/bin/env python3
"""Write the model card of RadixArk/qwen38-dspark-dense-5layer from the run's launch.log (train/acc at each epoch end)."""
import argparse, ast, re
from pathlib import Path

STEP_RE = re.compile(r"^step (\d+): (\{.*\})\s*$")

def metrics(log_path):
    out = {}
    for line in Path(log_path).read_text(errors="replace").splitlines():
        m = STEP_RE.match(line)
        if not m:
            continue
        try:
            d = ast.literal_eval(m.group(2))
        except Exception:
            continue
        out[int(m.group(1))] = {k.split("/")[-1]: v for k, v in d.items() if isinstance(v, (int, float))}
    return out

def at_epoch_end(ms, end, window=100):
    steps = sorted(s for s in ms if end - window < s <= end)
    if not steps:
        return None
    acc = [ms[s].get("acc") for s in steps if ms[s].get("acc") is not None]
    ce = [ms[s].get("ce_loss") for s in steps if ms[s].get("ce_loss") is not None]
    return {"step": steps[-1], "acc": sum(acc) / len(acc) if acc else None, "ce": sum(ce) / len(ce) if ce else None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps-per-epoch", type=int, default=2479)
    ap.add_argument("--epochs-done", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ms = metrics(Path(a.run_dir) / "launch.log")
    rows = []
    for k in range(1, a.epochs + 1):
        end = k * a.steps_per_epoch
        if k <= a.epochs_done:
            m = at_epoch_end(ms, end)
            acc = f"{m['acc']:.3f}" if m and m["acc"] is not None else "n/a"
            ce = f"{m['ce']:.2f}" if m and m["ce"] is not None else "n/a"
            rows.append(f"| `{k}ep/` | {end} | {acc} | {ce} | uploaded |")
        else:
            rows.append(f"| `{k}ep/` | {end} | | | pending |")
    body = f"""---
license: other
base_model: Qwen/Qwen3.8-27B
tags: [speculative-decoding, dspark, draft-model, sglang]
---

# Qwen3.8-27B DSpark dense 5-layer drafter (8 epochs on Regen-Mixture-v1, one export per epoch)

Private. A dense DSpark draft model for `Qwen/Qwen3.8-27B` with the architecture of the official
`RadixArk/Qwen3.8-27B-DSpark` drafter (5 full-attention layers, hidden 5120, FFN 17408, 32 / 8 heads of dim 128,
target layers 5/19/33/47/61, Markov rank 256, confidence head, 1.86B parameters), trained from scratch with Kan's
dense recipe (`kan/qwen38-dspark-ablation`, `qwen3.8-27b-dspark-regen-mixture-v1-3ep.yaml`) for eight epochs in one
continuous run (a folder per epoch end): global batch 512, lr 5e-4 constant after ~300 warm-up steps, DSpark CE 0.1 / L1 0.9 / confidence 1.0, training block 7,
NVFP4 teacher `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`, corpus `RadixArk/Qwen3.8-27B-Regen-Mixture-v1`
(1,269,290 conversations, max_length 8192, qwen3.5 chat template), two 4x GB300 nodes (4 NVFP4 capture servers on one, 4 FSDP ranks on the other, Mooncake RDMA).
It is the same-corpus, same-recipe dense reference for the MoE drafters trained on this data (Kan's 512 x 512 top-10,
the 16 x 1024 top-2 run of SpecForge PR #943, the 256-expert and 2-layer ablations).

| folder | optimizer step | train/acc (last 100 steps) | train/ce_loss | status |
|---|---|---|---|---|
{chr(10).join(rows)}

Each folder is a complete SGLang-servable export (`architectures: Qwen3DSparkModel`, block_size 7, tokenizer included):

```
python -m sglang.launch_server --model-path Qwen/Qwen3.8-27B --speculative-algorithm DSPARK \\
  --speculative-draft-model-path <local copy of 8ep/> --speculative-dspark-block-size 7 \\
  --speculative-draft-model-quantization unquant --attention-backend triton --speculative-draft-attention-backend triton
```

Code and recipe: https://github.com/yushengsu-thu/SpecForge/tree/yusheng/dense-5layer-3ep-gb300
(`configs/qwen3.8-27b-dspark.json`, `examples/configs/online/disaggregated/managed-local/qwen3.8-27b-dspark-dense-5layer-3ep-gb300.yaml`,
`scripts/launch/qwen38-dense5l-gb300/`). Training curves:
https://wandb.ai/miles_training/specforge-qwen38-dspark-dense-5layer/runs/qwen38-dspark-dense-5layer-3ep-gb300-v2
"""
    Path(a.out).write_text(body)
    print(f"wrote {a.out} ({a.epochs_done} epoch(s) done)")

if __name__ == "__main__":
    main()
