#!/usr/bin/env python3
"""Upload one epoch export to yushengsu/dense-1-8-only-5-epoch/<k>ep, verify sizes and sha256, refresh the model card.
Usage: RUN=<run_id> DRAFT=<draft config json> [EPOCHS=5] python upload_epoch_5ep.py <k> <export folder>"""
import ast, hashlib, json, os, re, sys
for _v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):  # inherited from the trainer launch; the upload needs the network
    os.environ.pop(_v, None)
from huggingface_hub import HfApi

k, folder = int(sys.argv[1]), sys.argv[2]
REPO = "yushengsu/dense-1-8-only-5-epoch"
RUN = os.environ["RUN"]
EPOCHS = int(os.environ.get("EPOCHS", "5"))
LOG = f"/scratch/pr1003/logs/train_{RUN}.log"
CFG = json.load(open(os.environ["DRAFT"]))
IS_MOE = "moe_preset" in CFG
api = HfApi(token=open("/personal/specforge-qwen38-backup/hf_token").read().strip())
api.upload_folder(repo_id=REPO, folder_path=folder, path_in_repo=f"{k}ep", ignore_patterns=[".exported", ".uploaded"],
                  commit_message=f"epoch {k} export ({RUN} step {k * 10764})")
for name in ("config.json", "model.safetensors"):
    local = os.path.join(folder, name)
    info = api.get_paths_info(REPO, [f"{k}ep/{name}"])[0]
    size_ok = info.size == os.path.getsize(local)
    sha_ok = True
    if getattr(info, "lfs", None):
        sha_ok = info.lfs.sha256 == hashlib.sha256(open(local, "rb").read()).hexdigest()
    print(f"{k}ep/{name}: size match {size_ok}, sha256 match {sha_ok}")
    if not (size_ok and sha_ok):
        sys.exit(1)


def epoch_metrics(step):
    best = None
    for line in open(LOG, errors="replace"):
        for piece in re.split(r"(?=step \d+: \{)", line):  # some log lines hold two entries back to back
            m = re.match(r"step (\d+): (\{.*\})", piece)
            if m and int(m.group(1)) <= step:
                best = (int(m.group(1)), m.group(2))
    if not best:
        return "n/a"
    try:
        d = ast.literal_eval(best[1])
    except (ValueError, SyntaxError):  # a garbled log line must not block the upload
        return f"step {best[0]}: n/a"
    acc = d.get("train/acc", d.get("acc")); ce = d.get("train/ce_loss", d.get("ce_loss")); loss = d.get("train/loss", d.get("loss"))
    return f"step {best[0]}: acc {acc:.4f}, ce_loss {ce:.4f}, loss {loss:.4f}"


uploaded = sorted(int(p.split("ep/")[0]) for p in api.list_repo_files(REPO) if re.match(r"^\dep/model\.safetensors$", p))
rows = "\n".join(f"| `{e}ep/` | {e * 10764:,} | {epoch_metrics(e * 10764)} |" for e in uploaded)
ffn = ("a mixture-of-experts FFN per layer: 8 experts of width 1536, top-1 routing (one expert per token, the same active FFN "
       "width as dense-1/8), DeepSeek-style selection bias (step 0.001) plus a batch-wise load-balance loss (0.005), no shared expert"
       if IS_MOE else "one SwiGLU MLP of width 1536 per layer")
kind = "2-layer MoE (8 x 1536, top-1)" if IS_MOE else "dense-1/8"
card = f"""---
library_name: sglang
tags:
- speculative-decoding
- dspark
- qwen3
---

# Qwen3-8B DSpark {kind} draft, {EPOCHS}-epoch schedule

A DSpark draft model for [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), trained with the code of SpecForge
PR [#1003](https://github.com/sgl-project/SpecForge/pull/1003) (commit cde2f059) with the hyperparameters of
[yushengsu/dense-1-8](https://huggingface.co/yushengsu/dense-1-8) on a {EPOCHS}-epoch cosine schedule (run `{RUN}`).
One folder per epoch, each directly servable by SGLang 0.5.18
(`--speculative-algorithm DSPARK --speculative-draft-model-path <folder>`).

| Folder | Optimizer step | Last logged training metrics |
|---|---|---|
{rows}

## Architecture

2 decoder layers (all full attention, causal inside the draft block), residual hidden 4096, GQA 8 heads / 2 KV heads
(head_dim 128), {ffn}, DSpark projector over target layers 1/9/17/25/33, vanilla Markov head (rank 256), block size 16,
no confidence head, target embedding and lm_head.

## Training

- Data: Open Perfect Blend prompts with Qwen3-8B responses (temperature 0, thinking off), 1,420,909 rows, `qwen3-instruct`
  template without a system prompt, max length 8192.
- Online DSpark training (2 SGLang capture servers + 6 DDP trainer ranks on one 8x H200 node), global batch 132
  (2 x 11 x 6), AdamW, lr 6e-4, cosine over {EPOCHS} epochs ({EPOCHS * 10764:,} steps) with 4% warmup, max_grad_norm 1.0,
  loss 0.1 CE + 0.9 L1, 512 anchors, loss_decay_gamma 7.0, bf16 without Liger kernels, seed / prompt_seed 42.
- Exported with `specforge export --to hf` (frozen Qwen3-8B embedding) and normalized for SGLang
  (`scripts/gates/normalize_dflash_export.py --block-size 16`).
"""
api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=REPO, commit_message=f"README: epoch {k}")
print("README updated; epochs on the Hub:", uploaded)
