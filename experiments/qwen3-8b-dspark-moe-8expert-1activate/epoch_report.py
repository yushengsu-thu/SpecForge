#!/usr/bin/env python3
"""Regenerate exp_read.me for the Qwen3-8B DSpark MoE 8-expert/top-1 (dense-same) run.

Sources, all optional (without them the file reports "not launched"):
  <out>/launch.log                console metrics ``step N: {...}`` (one line per rank, deduped)
  <out>/<run>-step<k*E>/          epoch checkpoints (mtime -> wall-clock per epoch)
  <out>/control/**/*.json         the online schedule payload (num_prompts, total_steps)
  <exports>/<k>ep/.uploaded       which epochs are on the HF repo
"""
import argparse
import ast
import datetime as dt
import glob
import json
import os
import re
import statistics
import time

# several ranks may print on one physical line; the metric dict is flat, so stop at the first "}"
STEP_RE = re.compile(r"step (\d+): (\{[^{}]*\})")
LAUNCH_RE = re.compile(r"^=== (\S+) launch (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)")
BRANCH = "exp/qwen3-8b-dspark-moe-8expert-1activate-dense-same"
BASE = "spark-Qwen3-8B @ 7cd9696d, includes the NO_SHARD MoE aux-loss fix c55cdd10"
RECIPE = ("examples/configs/online/disaggregated/managed-local/"
          "qwen3-8b-dspark-moe-8expert-1activate-dense-same-12ep.yaml")
WANDB = ("specforge-qwen3-8b-dspark", "qwen3-8b-dspark-moe-8expert-1activate-dense-same")
DENSE_REF = "https://huggingface.co/jiapingW/Qwen3-8b-dspark-SpecForge"


def utc(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ")


def parse_log(path):
    """Return ({step: {metric: value}}, [launch timestamps]) from the console log."""
    steps, launches = {}, []
    if not os.path.exists(path):
        return steps, launches
    with open(path, errors="replace") as fh:
        for line in fh:
            m = LAUNCH_RE.match(line)
            if m:
                launches.append(dt.datetime.strptime(m.group(2), "%Y-%m-%dT%H:%M:%SZ")
                                .replace(tzinfo=dt.timezone.utc).timestamp())
                continue
            for step_text, dict_text in STEP_RE.findall(line):
                step = int(step_text)
                if step in steps:
                    continue
                try:
                    payload = ast.literal_eval(re.sub(r"\b(nan|inf)\b", "None", dict_text))
                except (ValueError, SyntaxError):
                    continue
                steps[step] = {k: float(v) for k, v in payload.items()
                               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return steps, launches


def schedule_payload(out):
    for path in glob.glob(os.path.join(out, "control", "**", "*.json"), recursive=True):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "total_steps" in data:
            return data
    return {}


def ckpt_time(out, run, step):
    state = os.path.join(out, f"{run}-step{step}", "training_state.pt")
    if os.path.exists(state) and not glob.glob(os.path.join(out, f"{run}-step{step}", "*.tmp")):
        return os.path.getmtime(state)
    return None


def alive(out):
    try:
        with open(os.path.join(out, "launcher.pid")) as fh:
            os.kill(int(fh.read().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def fmt(value, digits=4):
    return "-" if value is None else f"{value:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exports", default=None)
    ap.add_argument("--keep", default=None)
    ap.add_argument("--epoch-steps", type=int, default=10683)
    ap.add_argument("--num-epochs", type=int, default=12)
    ap.add_argument("--hf-repo", default="yushengsu/qwen3-8b-dspark-moe-8expert-1activate")
    ap.add_argument("--write", default=None, help="file to write (default: stdout)")
    args = ap.parse_args()

    out, run, E, N = args.out, args.run, args.epoch_steps, args.num_epochs
    exports = args.exports or f"/scratch/exports/{run}"
    keep = args.keep or f"/scratch/ckpt-keep/{run}"
    steps, launches = parse_log(os.path.join(out, "launch.log"))
    sched = schedule_payload(out)
    total = int(sched.get("total_steps") or N * E)
    total_src = "schedule payload" if sched.get("total_steps") else "estimate (num_epochs x epoch_steps)"
    num_prompts = sched.get("num_prompts")
    ordered = sorted(steps)
    last = ordered[-1] if ordered else 0
    now = time.time()

    # ---- status -------------------------------------------------------------
    if not launches and not ordered:
        status = "Not launched yet."
    else:
        first_launch, last_launch = launches[0], launches[-1]
        elapsed_h = (now - first_launch) / 3600
        rate = last / elapsed_h if elapsed_h > 0 and last else None
        eta = utc(now + (total - last) / rate * 3600) if rate and last < total else "-"
        cur_epoch = min(N, last // E + 1)
        state = ("finished" if last >= total else
                 ("running" if alive(out) else "NOT RUNNING (launcher gone)"))
        status = (f"**{state}** - optimizer step {last:,} / {total:,} "
                  f"({100 * last / total:.1f}%), epoch {cur_epoch}/{N}; launched {utc(first_launch)}"
                  + (f" (relaunched {utc(last_launch)}, {len(launches)} launches)" if len(launches) > 1 else "")
                  + f"; {elapsed_h:.1f} h elapsed"
                  + (f", {rate:,.0f} steps/h, ETA {eta}" if rate else "")
                  + f". total_steps from {total_src}"
                  + (f", num_prompts {int(num_prompts):,}" if num_prompts else "") + ".")

    # ---- per-epoch rows -------------------------------------------------------
    loss_keys = sorted({k for s in ordered[-200:] for k in steps[s]
                        if "loss" in k and "position" not in k})
    rows, prev_t = [], (launches[0] if launches else None)
    for k in range(1, N + 1):
        lo, hi = (k - 1) * E, (k * E if k < N else max(total, k * E))
        window = [steps[s] for s in ordered if lo < s <= hi]
        if not window:
            if last < lo:
                break
            rows.append((k, hi, 0, None, None, {}, None, None, None, None))
            continue
        accs = [m["acc"] for m in window if "acc" in m]
        tail = accs[-20:]
        losses = {key: (statistics.fmean([m[key] for m in window if key in m]) if any(key in m for m in window) else None,
                        statistics.fmean([m[key] for m in window[-20:] if key in m]) if any(key in m for m in window[-20:]) else None)
                  for key in loss_keys}
        lr_end = next((m["lr"] for m in reversed(window) if "lr" in m), None)
        t = ckpt_time(out, run, k * E) or ckpt_time(keep, run, k * E)
        hours = (t - prev_t) / 3600 if (t and prev_t) else None
        if t:
            prev_t = t
        complete = last >= hi
        uploaded = os.path.exists(os.path.join(exports, f"{k}ep", ".uploaded"))
        rows.append((k, hi, len(window), statistics.fmean(accs) if accs else None,
                     statistics.fmean(tail) if tail else None, losses, lr_end, hours, complete, uploaded))

    # ---- document -------------------------------------------------------------
    doc = []
    w = doc.append
    w("# Qwen3-8B DSpark MoE drafter: 8 experts, top-1, dense-same hyperparameters")
    w("")
    w(f"Branch `{BRANCH}` (base `{BASE}`). Recipe `{RECIPE}`.  ")
    w(f"Checkpoints: https://huggingface.co/{args.hf_repo} (private; one folder `<k>ep/` per epoch, HF layout + tokenizer).  ")
    w(f"W&B: project `{WANDB[0]}`, run `{WANDB[1]}` (one run for all {N} epochs, so the curves are continuous).  ")
    w(f"Reference dense run: {DENSE_REF} (training/launch.yaml, step 39,000 checkpoint).")
    w("")
    w("## Status")
    w("")
    w(status)
    w("")
    w("## Setup")
    w("")
    w("### Target and data")
    w("")
    w("| item | setting |")
    w("|---|---|")
    w("| target model | Qwen/Qwen3-8B (36 layers), served by SGLang TP1 bf16 capture servers |")
    w("| data | `yaro1214/open_data` / `perfectblend_train_regen_temperature0_no_think.jsonl`, 1,420,909 rows, 6.32 GB: Open Perfect Blend prompts regenerated by Qwen3-8B at temperature 0 with thinking off |")
    w("| chat_template / system prompt | qwen3-instruct / none |")
    w("| max_length | 8192 |")
    w("| build_dataset_num_proc / dataloader_num_workers | 64 / 8 |")
    w("")
    w("### Draft model (`configs/qwen3-8b-dspark-moe-8x1536-top1-noconf-b16.json`)")
    w("")
    w("| item | setting |")
    w("|---|---|")
    w("| architecture | DSparkDraftModel, block_size 16, 5 layers (all full attention), hidden 4096, 32 heads / 8 KV heads (GQA), head_dim 128 |")
    w("| MoE FFN | 8 routed SwiGLU experts of width 1536, top-1 softmax routing without top-k renormalisation (`num_experts_per_tok` 1, `norm_topk_prob` false), no shared expert, router centering `sample` (momentum 0.99), auxiliary balance loss 0.001, `grouped_mm` dispatch, preset `qwen3_5_moe` |")
    w("| comparison arms | full dense (`dense-same`): MLP 12288; dense-1/8 (`dense-1/8`): MLP 1536; this MoE: 8 x 1536 top-1, so active MLP = dense-1/8 and total MLP = full dense; residual hidden, attention and depth identical in all three |")
    w("| target feature layers | 1, 9, 17, 25, 33 |")
    w("| Markov head / confidence head | vanilla, rank 256 / off |")
    w("| projector / mask_token_id | dspark / 151669 |")
    w("| rope_theta / max_position_embeddings | 1e6 / 40960 |")
    w("| vocab | 151936, untied; frozen target embedding (`model.embed_tokens.weight`) |")
    w("| size | see the parameter table below (`experiments/.../epoch_report.py --params`) |")
    w("")
    w("### Training (the dense recipe, applied unchanged unless listed under deviations)")
    w("")
    w("| item | setting |")
    w("|---|---|")
    w("| strategy | dspark, online disaggregated capture (Mooncake TCP) |")
    w(f"| epochs / optimizer steps | {N} in one run / {total:,} ({total_src}) |")
    w("| global batch | 133 = batch_size 1 x accumulation_steps 19 x 7 DDP ranks (same as the dense-1/8 arm) |")
    w("| optimizer | AdamW, lr 6e-4, cosine over the whole run, warmup 4%, FP32 master params |")
    w("| max_grad_norm | 1.0 |")
    w("| num_anchors / loss_decay_gamma / objective_chunk_blocks | 512 / 7.0 / 128 |")
    w("| loss | 0.1 CE + 0.9 L1, confidence loss 0 |")
    w("| sharding / precision / attention | NO_SHARD (DDP) / bf16 + Liger kernels / flex_attention |")
    w(f"| checkpoints | every 1187 steps, newest 12 kept by the trainer; epoch k = step k x {E:,} (copied out of the rotation, exported, uploaded) |")
    w("| seed / prompt_seed | 42 / 42 |")
    w("| hardware | 1 node x 8 H200: one capture server on GPU 0, seven trainer ranks on GPUs 1-7 |")
    w("")
    w("### Deviations from the dense run")
    w("")
    w("- Global batch 133 instead of 128: 128 does not divide over 7 trainer ranks; batch_size 1 x accumulation 19 x 7 = 133 is the closest and is the quantum the dense-1/8 arm uses. The first launch used the branch's 2-server + 6-rank layout (2 x 11 x 6 = 132); it was stopped after 200 steps because the trainer, not capture, was the bottleneck, and relaunched from scratch with one capture server and seven ranks.")
    w(f"- {N} epochs in one run with one cosine schedule (the dense run planned 6 epochs and stopped at step 40,960).")
    w("- Draft attention stays bidirectional inside the block: `is_causal`, `dspark_flatten_projection`, `dspark_cache_projection` and `ddp_bucket_cap_mb` (unmerged PR #942) are not used on this branch.")
    w("- The dense FFN (12288) is replaced by 8 x 1536 top-1 experts (no shared expert); every other draft-config field is the dense skeleton.")
    w("")
    w("## Per-epoch results")
    w("")
    loss_cols = "".join(f" | {key} (mean / end)" for key in loss_keys)
    w(f"| epoch | end step | logged points | train/acc (epoch mean) | train/acc (end of epoch){loss_cols} | lr (end) | wall-clock (h) | checkpoint |")
    w("|---|---|---|---|---|" + "---|" * len(loss_keys) + "---|---|---|")
    if not rows:
        w(f"| - | - | - | - | - |{' - |' * len(loss_keys)} - | - | no optimizer step logged yet |")
    for (k, hi, n, acc_mean, acc_end, losses, lr_end, hours, complete, uploaded) in rows:
        loss_cells = "".join(f" | {fmt(losses.get(key, (None, None))[0])} / {fmt(losses.get(key, (None, None))[1])}" for key in loss_keys)
        if complete:
            ck = (f"[{k}ep](https://huggingface.co/{args.hf_repo}/tree/main/{k}ep)" if uploaded
                  else "complete, export/upload pending")
        else:
            ck = "in progress" if n else "-"
        w(f"| {k} | {hi:,} | {n} | {fmt(acc_mean)} | {fmt(acc_end)}{loss_cells} | "
          f"{fmt(lr_end, 6)} | {fmt(hours, 1)} | {ck} |")
    w("")
    w("## How the numbers are computed")
    w("")
    w(f"- Metrics come from the trainer console log (`step N: {{...}}`, one line every `log_interval` = 10 optimizer steps, rank-deduplicated); the same values are on W&B under `train/*`.")
    w(f"- Epoch k covers optimizer steps ((k-1) x {E:,}, k x {E:,}]; the last epoch runs to total_steps. The producer keeps 1,420,903 of the 1,420,909 rows (6 have fewer than 2 trainable tokens), so one epoch is 1,420,903 / 133 = 10,683.48 optimizer steps and total_steps = 128,201; the k x {E:,} windows end at most 5 steps before the true epoch boundaries.")
    w("- *epoch mean* averages all logged points in the window; *end of epoch* averages the last 20 logged points (200 optimizer steps, about 26k samples).")
    w("- *wall-clock* is the time between consecutive epoch checkpoints (epoch 1: since launch); it includes the dataset cache build for epoch 1.")
    w("- Exports are `specforge export --to hf` of the epoch checkpoint with the frozen Qwen3-8B embedding, plus the Qwen3-8B tokenizer; the raw training checkpoints (optimizer state included) stay on the box under `/scratch/ckpt-keep/`.")
    w("")
    w(f"_Generated {utc(now)} by `experiments/qwen3-8b-dspark-moe-8expert-1activate/epoch_report.py`; do not edit by hand._")
    text = "\n".join(doc) + "\n"
    if args.write:
        with open(args.write, "w") as fh:
            fh.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
