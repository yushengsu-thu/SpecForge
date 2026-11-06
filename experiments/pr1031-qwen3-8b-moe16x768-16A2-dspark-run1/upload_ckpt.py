#!/usr/bin/env python3
"""Upload one exported checkpoint to the public Hub repo yushengsu/Qwen3-8B-5layer-MoE-16E2A-v1 as step-N/ (and epK/ when step N ends
prompt epoch K, per the producer's schedule record), verify it, refresh the repo README.

usage: upload_ckpt.py N FOLDER        upload the export in FOLDER as step N
       upload_ckpt.py --epoch-of N    print K when step N ends prompt epoch K, else nothing
       upload_ckpt.py --last-step     print the run's final optimizer step
"""
import ast
import hashlib
import json
import os
import re
import sys

for _v in (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
):  # the launcher's offline flags must not reach the uploader
    os.environ.pop(_v, None)

W = "/scratch/pr1031"
RUN = "pr1031-qwen3-8b-moe16x768-16A2-dspark-run1"
REPO = "yushengsu/Qwen3-8B-5layer-MoE-16E2A-v1"
LOG = f"{W}/logs/train_{RUN}.log"
SCHEDULE = f"{W}/outputs/{RUN}/control/refs.jsonl.schedule.json"


def schedule():
    """The producer's online schedule record plus the epoch-end steps the consumer derives from it."""
    from specforge.training.schedule import resolve_online_epoch_end_steps

    with open(SCHEDULE) as stream:
        s = json.load(stream)
    ends = resolve_online_epoch_end_steps(
        num_prompts=s["num_prompts"],
        prompt_epochs=s["prompt_epochs"],
        dp_size=s["dp_size"],
        batch_size=s["batch_size"],
        accumulation_steps=s["accumulation_steps"],
    )
    s["epoch_of"] = {step: epoch for epoch, step in enumerate(ends, start=1)}
    s["steps_per_epoch"] = s["num_prompts"] / (
        s["dp_size"] * s["batch_size"] * s["accumulation_steps"]
    )
    return s


if sys.argv[1] == "--epoch-of":
    print(schedule()["epoch_of"].get(int(sys.argv[2]), ""))
    sys.exit(0)
if sys.argv[1] == "--last-step":
    print(schedule()["total_steps"])
    sys.exit(0)

from huggingface_hub import HfApi

n, folder = int(sys.argv[1]), sys.argv[2]
sched = schedule()
ep = sched["epoch_of"].get(n)
api = HfApi(token=open("/scratch/secrets/hf_token").read().strip())
api.create_repo(REPO, private=False, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


targets = [f"step-{n}"] + ([f"ep{ep}"] if ep else [])
for t in targets:
    api.upload_folder(
        repo_id=REPO,
        folder_path=folder,
        path_in_repo=t,
        ignore_patterns=[".exported"],
        commit_message=f"{RUN}: step {n}" + (f" = end of epoch {ep}" if ep else ""),
    )
    for name in ("config.json", "model.safetensors"):
        local = os.path.join(folder, name)
        info = api.get_paths_info(REPO, [f"{t}/{name}"])[0]
        ok = info.size == os.path.getsize(local)
        if getattr(info, "lfs", None):
            ok = ok and info.lfs.sha256 == sha256(local)
        print(f"{t}/{name}: verified {ok}")
        if not ok:
            sys.exit(1)


def metrics_at(step):
    """Last logged training metrics at or before `step` (some log lines hold two entries back to back)."""
    best = None
    for line in open(LOG, errors="replace"):
        for piece in re.split(r"(?=step \d+: \{)", line):
            m = re.match(r"step (\d+): (\{.*\})", piece)
            if m and int(m.group(1)) <= step:
                best = (int(m.group(1)), m.group(2))
    if not best:
        return "n/a"
    try:
        d = ast.literal_eval(best[1])
    except (ValueError, SyntaxError):
        return f"step {best[0]}: n/a"
    g = lambda k: d.get(f"train/{k}", d.get(k))
    acc, ce, walk, bal = (
        g("acc"),
        g("ce_loss"),
        g("dspark/hard_label/walk_accepted_length"),
        g("moe/step_balance"),
    )
    f = lambda v, p=4: "n/a" if v is None else f"{v:.{p}f}"
    return f"step {best[0]}: acc {f(acc)}, ce_loss {f(ce)}, walk accept length {f(walk, 3)}, step_balance {f(bal, 3)}"


files = api.list_repo_files(REPO)
steps = sorted(
    {
        int(m.group(1))
        for p in files
        if (m := re.match(r"^step-(\d+)/model\.safetensors$", p))
    }
)
spe, last = sched["steps_per_epoch"], sched["total_steps"]
rows = "\n".join(
    f"| `step-{s}/` | {s:,} | {s / spe:.2f} | "
    + (f"`ep{sched['epoch_of'][s]}/`" if s in sched["epoch_of"] else "")
    + f" | {metrics_at(s)} |"
    for s in steps
)
card = f"""# {REPO.split('/')[1]}: Qwen3-8B DSpark draft, MoE 16 experts x 768, top-2 (run `{RUN}`)

Qwen3-8B DSpark draft with the MoE FFN of SpecForge PR [#1031](https://github.com/sgl-project/SpecForge/pull/1031)
(branch `exp/pr1031-qwen3-8b-moe16x768-16A2-dspark-6ep` of yushengsu-thu/SpecForge, on top of dev-moe-ffn 3543fa2a):
5 decoder layers, GQA 32 heads / 8 KV heads, 16 experts x 768 per layer (the dense FFN's 12288 split 16 ways), top-2,
batch-wise load-balance loss 0.005, expert bias 0.001 per step, DSpark projector over target layers 1/9/17/25/33, block
size 16. Recipe as jiapingW/Qwen3-8b-dspark-SpecForge: PerfectBlend regen (temperature 0, thinking off, {sched['num_prompts']:,}
prompts), `qwen3-instruct`, max length 8192, global batch 132 (2 x 11 x 6 ranks), lr 6e-4 cosine over 6 epochs ({last:,}
steps) with 4% warmup, 0.1 CE + 0.9 L1, 512 anchors, loss_decay_gamma 7.0, Liger RMSNorm on, seed 42. One 8x H200 node
(specforge-exp-branch): 2 SGLang capture servers + 6 DDP trainers. W&B: eigent_radixark_training/Qwen3-8B-5layer-MoE-16E2A-v1.

Each folder is an export at one optimizer step (every 1,000 steps plus every epoch end, {spe:,.2f} steps per epoch),
normalized for SGLang 0.5.18. Epoch ends are duplicated as `ep1/` .. `ep6/`. Serving needs the MoE serving package of PR #1016.

| Folder | Optimizer step | Epoch | Also as | Last logged training metrics |
|---|---|---|---|---|
{rows}
"""
api.upload_file(
    path_or_fileobj=card.encode(),
    path_in_repo="README.md",
    repo_id=REPO,
    commit_message=f"{RUN}: README after step {n}",
)
print(f"README updated; steps on the Hub: {steps}")
