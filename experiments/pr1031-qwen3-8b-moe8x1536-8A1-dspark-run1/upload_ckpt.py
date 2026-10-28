#!/usr/bin/env python3
"""Upload one exported checkpoint to the public Hub repo under <run>/step-N (and <run>/epK), verify it, refresh the run README."""
import ast, hashlib, os, re, sys

for _v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):  # the launcher's offline flags must not reach the uploader
    os.environ.pop(_v, None)
from huggingface_hub import HfApi

n, folder, ep = int(sys.argv[1]), sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
REPO = "yushengsu/new-repo-qwen3-8b-moe16x768-16A1-dspark"
RUN = "pr1031-qwen3-8b-moe8x1536-8A1-dspark-run1"
LOG = f"/scratch/pr1031/logs/train_{RUN}.log"
STEPS_PER_EPOCH, LAST = 10764, 64584
api = HfApi(token=open("/scratch/secrets/hf_token").read().strip())
api.create_repo(REPO, private=False, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


targets = [f"{RUN}/step-{n}"] + ([f"{RUN}/ep{ep}"] if ep else [])
for t in targets:
    api.upload_folder(repo_id=REPO, folder_path=folder, path_in_repo=t, ignore_patterns=[".exported"],
                      commit_message=f"{RUN}: step {n}" + (f" = end of epoch {ep}" if ep else ""))
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
    acc, ce, walk, bal = g("acc"), g("ce_loss"), g("dspark/hard_label/walk_accepted_length"), g("moe/step_balance")
    f = lambda v, p=4: "n/a" if v is None else f"{v:.{p}f}"
    return f"step {best[0]}: acc {f(acc)}, ce_loss {f(ce)}, walk accept length {f(walk, 3)}, step_balance {f(bal, 3)}"


files = api.list_repo_files(REPO)
steps = sorted({int(m.group(1)) for p in files if (m := re.match(rf"^{re.escape(RUN)}/step-(\d+)/model\.safetensors$", p))})
rows = "\n".join(f"| `{RUN}/step-{s}/` | {s:,} | {s / STEPS_PER_EPOCH:.2f} | " + (f"`ep{s // STEPS_PER_EPOCH}/`" if s % STEPS_PER_EPOCH == 0 else "") +
                 f" | {metrics_at(s)} |" for s in steps)
card = f"""# {RUN}

Qwen3-8B DSpark draft with the MoE FFN of SpecForge PR [#1031](https://github.com/sgl-project/SpecForge/pull/1031)
(branch `exp/pr1031-qwen3-8b-moe8x1536-8A1-dspark-6ep` of yushengsu-thu/SpecForge, on top of dev-moe-ffn 3543fa2a):
5 decoder layers, GQA 32 heads / 8 KV heads, 8 experts x 1536 per layer, top-1, batch-wise load-balance loss 0.005, expert
bias 0.001 per step, DSpark projector over target layers 1/9/17/25/33, block size 16. Recipe as jiapingW/Qwen3-8b-dspark-SpecForge:
PerfectBlend regen (temperature 0, thinking off, 1,420,909 rows), `qwen3-instruct`, max length 8192, global batch 132
(2 x 11 x 6 ranks), lr 6e-4 cosine over 6 epochs ({LAST:,} steps) with 4% warmup, 0.1 CE + 0.9 L1, 512 anchors,
loss_decay_gamma 7.0, Liger RMSNorm on, seed 42. One 8x H200 node: 2 SGLang capture servers + 6 DDP trainers.

Each folder is an export at one optimizer step (every 1,196 steps = 1/9 epoch, plus the last step), normalized for
SGLang 0.5.18. Epoch ends are duplicated as `ep1/` .. `ep6/`. Serving needs the MoE serving package of PR #1016.

| Folder | Optimizer step | Epoch | Also as | Last logged training metrics |
|---|---|---|---|---|
{rows}
"""
api.upload_file(path_or_fileobj=card.encode(), path_in_repo=f"{RUN}/README.md", repo_id=REPO, commit_message=f"{RUN}: README after step {n}")
print(f"README updated; steps on the Hub: {steps}")
