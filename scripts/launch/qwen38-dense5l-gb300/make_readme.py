#!/usr/bin/env python3
"""Write the model card of RadixArk/qwen38-dspark-dense-5layer: training setting + train/acc per epoch (from launch.log)."""
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
    acc = [ms[s]["acc"] for s in steps if ms[s].get("acc") is not None]
    ce = [ms[s]["ce_loss"] for s in steps if ms[s].get("ce_loss") is not None]
    return {"acc": sum(acc) / len(acc) if acc else None, "ce": sum(ce) / len(ce) if ce else None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps-per-epoch", type=int, default=2479)
    ap.add_argument("--epochs-done", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ms = metrics(Path(a.run_dir) / "launch.log")
    results = {}
    for k in range(1, a.epochs_done + 1):
        m = at_epoch_end(ms, k * a.steps_per_epoch)
        if m and m["acc"] is not None:
            results[k] = m
    best = max(results, key=lambda k: results[k]["acc"]) if results else None
    rows = []
    for k in range(1, a.epochs + 1):
        if k in results:
            star = " **(best)**" if k == best else ""
            rows.append(f"| {k} | `{k}ep/` | {k * a.steps_per_epoch} | {results[k]['acc']:.3f}{star} | {results[k]['ce']:.2f} |")
        elif k <= a.epochs_done:
            rows.append(f"| {k} | `{k}ep/` | {k * a.steps_per_epoch} | n/a | n/a |")
        else:
            rows.append(f"| {k} | `{k}ep/` | {k * a.steps_per_epoch} | (training) | |")
    best_line = f"Highest train/acc so far: **{results[best]['acc']:.3f}** at epoch {best} (`{best}ep/`)." if best else "No epoch finished yet."
    body = f"""---
license: other
base_model: Qwen/Qwen3.8-27B
tags: [speculative-decoding, dspark, draft-model, sglang]
---

# Qwen3.8-27B DSpark dense 5-layer drafter

Private. One folder per finished epoch (`1ep/` … `8ep/`), each a complete SGLang-servable draft model
(`Qwen3DSparkModel`, block_size 7, tokenizer included).

## Training setting

| | |
|---|---|
| Target model | `Qwen/Qwen3.8-27B` (capture / teacher: `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`, NVFP4 body + bf16 lm_head) |
| Draft model | **dense** DSpark drafter (no MoE): 5 full-attention layers, hidden 5120, FFN 17408 (SwiGLU), 32 query / 8 KV heads of dim 128, target layers 5/19/33/47/61, Markov rank 256, confidence head, YaRN 32 / 8192; 1.86B params. Same architecture as the official `RadixArk/Qwen3.8-27B-DSpark`. |
| Data | `RadixArk/Qwen3.8-27B-Regen-Mixture-v1` (1,269,290 conversations, max_length 8192, qwen3.5 chat template) |
| Recipe | Kan's dense recipe: global batch 512 (2,479 steps / epoch), AdamW lr 5e-4 constant after ~300 warm-up steps, grad clip 1.0, DSpark CE 0.1 / L1 0.9 / confidence 1.0, 512 anchors, decay gamma 4, training block 7, seed 42 |
| Schedule | 8 epochs in one continuous run (19,832 steps), from scratch |
| Hardware | 2 x 4 GB300 (one node: 4 NVFP4 capture servers; other node: 4 FSDP ranks; features over Mooncake RDMA), ~28 s / step |

## train/acc per epoch

train/acc = mean draft-token accuracy over the 7 block positions, averaged over the last 100 steps of the epoch.

| epoch | folder | step | train/acc | ce loss |
|---|---|---|---|---|
{chr(10).join(rows)}

{best_line}

## Links

- Code / recipe: https://github.com/yushengsu-thu/SpecForge/tree/yusheng/dense-5layer-3ep-gb300
- Training curves: https://wandb.ai/miles_training/specforge-qwen38-dspark-dense-5layer/runs/qwen38-dspark-dense-5layer-3ep-gb300-v2
- Serve: `python -m sglang.launch_server --model-path Qwen/Qwen3.8-27B --speculative-algorithm DSPARK --speculative-draft-model-path <local copy of Nep/> --speculative-dspark-block-size 7 --speculative-draft-model-quantization unquant --attention-backend triton --speculative-draft-attention-backend triton`
"""
    Path(a.out).write_text(body)
    print(f"wrote {a.out} ({a.epochs_done} epoch(s) done; best epoch {best})")

if __name__ == "__main__":
    main()
