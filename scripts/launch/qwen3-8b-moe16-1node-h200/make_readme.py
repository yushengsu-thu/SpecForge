#!/usr/bin/env python3
"""Write the model card of a Qwen3-8B MoE drafter run: training setting + train/acc per uploaded checkpoint (from launch.log).

Epoch mode (--save-every == --steps-per-epoch): one row per epoch, folders <k>ep/.  Interval mode: one row per
interval checkpoint, folders step-<N>/.  --folder-prefix is the sub-folder of the repo the run is uploaded to.
"""
import argparse, ast, json, re
from pathlib import Path

STEP_RE = re.compile(r"^step (\d+): (\{.*\})\s*$")
# Parameter counts of the 5-layer draft: everything but the experts and the router (embeddings, attention,
# projector, Markov head, norms) measured on the 16 x 768 config, then experts and router from the geometry.
NON_EXPERT_PARAMS = 371_442_944
LAYERS, HIDDEN, FFN = 5, 4096, 12288

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

def at_step(ms, end, window=100):
    steps = sorted(s for s in ms if end - window < s <= end)
    if not steps:
        return None
    acc = [ms[s]["acc"] for s in steps if ms[s].get("acc") is not None]
    ce = [ms[s]["ce_loss"] for s in steps if ms[s].get("ce_loss") is not None]
    return {"acc": sum(acc) / len(acc) if acc else None, "ce": sum(ce) / len(ce) if ce else None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--draft-config", default="configs/qwen3-8b-moe16x768-16A1-dspark.json")
    ap.add_argument("--steps-per-epoch", type=int, default=10764)
    ap.add_argument("--save-every", type=int, default=0, help="checkpoint interval; default = one epoch")
    ap.add_argument("--steps-done", type=int, required=True, help="step of the checkpoint just uploaded")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--hf-repo", default="yushengsu/qwen3-8b-moe16x768-16A1-dspark")
    ap.add_argument("--folder-prefix", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.draft_config).read_text())
    E, d = cfg["num_experts"], cfg["moe_intermediate_size"]
    aux = cfg["dflash_config"]["moe_router_aux_loss_coef"]
    bias_rate = cfg["dflash_config"]["moe_router_bias_update_rate"]
    total = NON_EXPERT_PARAMS + LAYERS * (E * 3 * HIDDEN * d + HIDDEN * E)
    active = NON_EXPERT_PARAMS + LAYERS * (3 * HIDDEN * d + HIDDEN * E)
    save_every = a.save_every or a.steps_per_epoch
    epoch_mode = save_every == a.steps_per_epoch
    last_step = a.epochs * a.steps_per_epoch // save_every * save_every
    final_step = a.epochs * a.steps_per_epoch
    steps = list(range(save_every, last_step + 1, save_every))
    if final_step not in steps:
        steps.append(final_step)  # the trainer saves its last step too, uploaded as final/
    ms = metrics(Path(a.run_dir) / "launch.log")
    results = {}
    for step in steps:
        if step > a.steps_done:
            break
        m = at_step(ms, step)
        if m and m["acc"] is not None:
            results[step] = m
    best = max(results, key=lambda s: results[s]["acc"]) if results else None
    rows = []
    for step in steps:
        folder = f"{step // a.steps_per_epoch}ep/" if epoch_mode else ("final/" if step == final_step else f"step-{step}/")
        epoch = f"{step / a.steps_per_epoch:.2f}"
        if step in results:
            star = " **(best)**" if step == best else ""
            rows.append(f"| {epoch} | `{folder}` | {step} | {results[step]['acc']:.3f}{star} | {results[step]['ce']:.2f} |")
        elif step <= a.steps_done:
            rows.append(f"| {epoch} | `{folder}` | {step} | n/a | n/a |")
        else:
            rows.append(f"| {epoch} | `{folder}` | {step} | (training) | |")
    if epoch_mode:
        folders = f"One folder per finished epoch (`1ep/` … `{a.epochs}ep/`)"
    else:
        folders = f"One folder per checkpoint, every {save_every} steps (`step-{save_every}/` … `step-{last_step}/`, {a.steps_per_epoch} steps per epoch) plus the last step as `final/` (step {final_step})"
    where = f" under `{a.folder_prefix}`" if a.folder_prefix else ""
    def folder_of(step):
        return "%dep" % (step // a.steps_per_epoch) if epoch_mode else ("final" if step == final_step else "step-%d" % step)
    best_line = (f"Highest train/acc so far: **{results[best]['acc']:.3f}** at step {best} (`{a.folder_prefix}{folder_of(best)}/`)."
                 if best else "No checkpoint uploaded yet.")
    body = f"""---
license: other
base_model: Qwen/Qwen3-8B
tags: [speculative-decoding, dspark, draft-model, moe, sglang]
---

# Qwen3-8B DSpark MoE drafter ({E} experts x {d}, top-1)

{folders}{where}, each a complete SGLang-servable draft model (`Qwen3MoeDSparkModel`, block_size 16, tokenizer included).

## Training setting

| | |
|---|---|
| Target model | `Qwen/Qwen3-8B` (BF16 capture / teacher, as in jiapingW/Qwen3-8b-dspark-SpecForge) |
| Draft model | jiapingW/Qwen3-8b-dspark-SpecForge's dense draft architecture with the FFN replaced by a Qwen3.5-MoE FFN: 5 full-attention layers, hidden 4096, **{E} routed experts x {d} (= {FFN} / {E}), one per token, no shared expert**, 32 query / 8 KV heads of dim 128, target layers 1/9/17/25/33, Markov rank 256, no confidence head, causal, block 16; {total / 1e9:.2f}B params, {active / 1e9:.2f}B active per token. Router balanced with a selection bias (step {bias_rate} once per optimizer step) plus a {aux} batch-wise aux loss. |
| Data | PerfectBlend regenerated by Qwen3-8B at temperature 0, non-thinking (`yaro1214/open_data`, 1,420,903 conversations, max_length 8192, qwen3-instruct template, no system prompt) |
| Recipe | jiapingW's protocol: global batch 132 (6 ranks x 2 x 11; {a.steps_per_epoch} steps / epoch), AdamW lr 6e-4 cosine with a 4% warm-up, grad clip 1.0, DSpark CE 0.1 / L1 0.9, no confidence head, 512 anchors, decay gamma 7, NO_SHARD DDP, training block 16, seed 42 |
| Schedule | {a.epochs} epochs in one continuous run ({a.epochs * a.steps_per_epoch} steps), from scratch; a checkpoint every {save_every} steps |
| Hardware | one 8x H200 node: 2 BF16 capture servers + 6 DDP ranks, features over Mooncake |

## train/acc per checkpoint

train/acc = mean draft-token accuracy over the block positions, averaged over the 100 steps before the checkpoint.

| epoch | folder | step | train/acc | ce loss |
|---|---|---|---|---|
{chr(10).join(rows)}

{best_line}

## Links

- Code / recipe: https://github.com/sgl-project/SpecForge/pull/1022 (config; recipe and scripts on branch `dspark-v2-moe-launch`); dense reference: https://huggingface.co/jiapingW/Qwen3-8b-dspark-SpecForge
- Serve (the MoE draft class comes from SpecForge's serving package until SGLang ships it):
  `SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang_models python -m sglang.launch_server --model-path Qwen/Qwen3-8B --speculative-algorithm DSPARK --speculative-draft-model-path <local copy of a checkpoint folder> --speculative-dspark-block-size 16 --speculative-draft-model-quantization unquant --attention-backend triton --speculative-draft-attention-backend triton`
"""
    Path(a.out).write_text(body)
    print(f"wrote {a.out} (step {a.steps_done} done; best step {best})")

if __name__ == "__main__":
    main()
