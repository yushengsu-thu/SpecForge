#!/usr/bin/env python3
"""Regenerate exp_read.me for the Qwen3-8B DSpark MoE 8-expert/top-1 (dense-same) run.

Sources, all optional (without them the file reports "not launched"):
  <out>/launch.log                console metrics ``step N: {...}`` (one line per rank, deduped)
  <out>/<run>-step<k*E>/          epoch checkpoints (mtime -> wall-clock per epoch)
  <out>/control/**/*.json         the online schedule payload (num_prompts, total_steps)
  <exports>/<k>ep/.uploaded       which epochs are on the HF repo
  <cont out>/continuation.json    a warm-started continuation's recipe (lr, warmup_ratio, prompt_seed, step_offset,
                                  warm_start); one --cont-out/--cont-run/--cont-epoch-offset triple per continuation
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
BASE = "spark-Qwen3-8B @ 7cd9696d + upstream PR stack #957-#963 (as on dense-1/8) + #966 cost-balanced consumer dispatch"
RECIPE = ("examples/configs/online/disaggregated/managed-local/"
          "qwen3-8b-dspark-moe-8expert-1activate-dense-same-12ep.yaml")
WANDB = ("specforge-qwen3-8b-dspark", "qwen3-8b-dspark-moe-8expert-1activate-dense-same")
DENSE_REF = "https://huggingface.co/jiapingW/Qwen3-8b-dspark-SpecForge"


def utc(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ")


def parse_log(path, plain_step_offset=0):
    """Return ({step: {metric: value}}, [launch timestamps]) from the console log.

    Every logged step is printed twice: the trainer's console dict (`acc`, `lr`, `ce_loss`, ... at the run's own
    step) and the tracker's dict (`train/acc`, ..., `perf/*` at the W&B step, which a warm-started continuation
    offsets by SPECFORGE_LOG_STEP_OFFSET). Both are merged per step with the `train/` prefix stripped (first value
    wins, as several ranks may print the same step); `plain_step_offset` moves a continuation's console dicts onto
    the W&B step axis.
    """
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
                try:
                    payload = ast.literal_eval(re.sub(r"\b(nan|inf)\b", "None", dict_text))
                except (ValueError, SyntaxError):
                    continue
                tracker = any(k.startswith("train/") for k in payload)
                merged = steps.setdefault(int(step_text) + (0 if tracker else plain_step_offset), {})
                for k, v in payload.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        merged.setdefault(k[len("train/"):] if k.startswith("train/") else k, float(v))
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
    ap.add_argument("--epoch-steps", type=int, default=10764)
    ap.add_argument("--num-epochs", type=int, default=12)
    ap.add_argument("--hf-repo", default="yushengsu/qwen3-8b-dspark-moe-8expert-1activate")
    ap.add_argument("--write", default=None, help="file to write (default: stdout)")
    ap.add_argument("--cont-out", action="append", default=[],
                    help="output dir of a warm-started continuation run (repeat once per continuation, in order)")
    ap.add_argument("--cont-run", action="append", default=[])
    ap.add_argument("--cont-epoch-offset", action="append", type=int, default=[],
                    help="epochs completed before that continuation")
    args = ap.parse_args()

    out, run, E, N = args.out, args.run, args.epoch_steps, args.num_epochs
    exports = args.exports or f"/scratch/exports/{run}"
    keep = args.keep or f"/scratch/ckpt-keep/{run}"
    steps, launches = parse_log(os.path.join(out, "launch.log"))
    sched = schedule_payload(out)
    if not (len(args.cont_out) == len(args.cont_run) == len(args.cont_epoch_offset)):
        ap.error("give one --cont-out, --cont-run and --cont-epoch-offset per continuation")
    starts = [0] + list(args.cont_epoch_offset)                 # epochs completed before each segment
    M = starts[1] if len(starts) > 1 else N                    # epochs of the first run
    main_total = int(sched.get("total_steps") or M * E)
    total_src = "schedule payload" if sched.get("total_steps") else "estimate (num_epochs x epoch_steps)"
    # segments: the first run, then each continuation; a continuation logs its tracker dicts at base + its own step
    # (SPECFORGE_LOG_STEP_OFFSET = steps of all earlier segments), and its console dicts are moved there too
    segs = [{"out": out, "run": run, "keep": keep, "first": 0, "n": M, "base": 0, "total": main_total, "meta": {}}]
    for j, (c_out, c_run, c_off) in enumerate(zip(args.cont_out, args.cont_run, args.cont_epoch_offset)):
        n = (starts[j + 2] if j + 2 < len(starts) else N) - c_off
        base = segs[-1]["base"] + segs[-1]["total"]
        c_steps, c_launches = parse_log(os.path.join(c_out, "launch.log"), plain_step_offset=base)
        steps.update({s: v for s, v in c_steps.items() if s not in steps})
        launches += c_launches
        try:
            with open(os.path.join(c_out, "continuation.json")) as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            meta = {}
        segs.append({"out": c_out, "run": c_run, "keep": f"/scratch/ckpt-keep/{c_run}", "first": c_off, "n": n,
                     "base": base, "total": int(schedule_payload(c_out).get("total_steps") or n * E), "meta": meta})
    cont_segs = segs[1:]
    total = sum(s["total"] for s in segs)
    ends = []
    for s in segs:
        ends += [s["base"] + i * E for i in range(1, s["n"])] + [s["base"] + s["total"]]
    ends = ends[:N]

    def seg_of(k):
        return next(s for s in reversed(segs) if k > s["first"])

    def recipe(s):
        m = s["meta"]
        lr, wu, seed = m.get("lr", 6e-4), m.get("warmup_ratio", 0.04), m.get("prompt_seed", 42 + s["first"])
        return f"re-warmup {wu:.0%} to {f'{lr:.0e}'.replace('e-0', 'e-')}, cosine to 0, prompt_seed {seed}"
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
        cur_epoch = min(N, next((i + 1 for i, e in enumerate(ends) if last < e), N))
        state = ("finished" if last >= total else
                 ("running" if alive(next((s["out"] for s in reversed(segs) if os.path.exists(os.path.join(s["out"], "launcher.pid"))), out))
                  else "NOT RUNNING (launcher gone)"))
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
        lo, hi = (ends[k - 2] if k > 1 else 0), ends[k - 1]
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
        s = seg_of(k)
        t = ckpt_time(s["out"], s["run"], (k - s["first"]) * E) or ckpt_time(s["keep"], s["run"], (k - s["first"]) * E)
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
    w("# Qwen3-8B DSpark MoE drafter: 2 layers, 1/4 attention width, 8 experts x 1536 top-1, dense-same hyperparameters")
    w("")
    w(f"Branch `{BRANCH}` (base `{BASE}`). Recipe `{RECIPE}`.  ")
    w(f"Checkpoints: https://huggingface.co/{args.hf_repo} (private; one folder `<k>ep/` per epoch, HF layout + tokenizer).  ")
    w(f"W&B: project `{WANDB[0]}`, run `{WANDB[1]}` (id 1ia117wn; one W&B run for all {N} epochs, so the curves are continuous).  ")
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
    w("| chat_template / system prompt | qwen3-instruct / none: the rows carry no system turn and PR #959 removes the template's implicit `You are a helpful assistant.`, matching the reference run |")
    w("| max_length | 8192 |")
    w("| build_dataset_num_proc / dataloader_num_workers | 64 / 8 |")
    w("")
    w("### Draft model (`configs/qwen3-8b-dspark-moe-8x1536-top1-2layer-causal-noconf-b16.json`)")
    w("")
    w("| item | setting |")
    w("|---|---|")
    w("| architecture | DSparkDraftModel, block_size 16, **2 layers** (all full attention), residual hidden 4096, **GQA 8 heads / 2 KV heads**, head_dim 128 (attention width 1024, 1/4 of the 5-layer skeleton), `is_causal: true` (PR #960); same structure as the dense-1/8 arm |")
    w("| MoE FFN | per layer 8 routed SwiGLU experts of width 1536 (2 layers, 16 experts in total), top-1 softmax routing without top-k renormalisation (`num_experts_per_tok` 1, `norm_topk_prob` false), no shared expert, router centering `sample` (momentum 0.99), auxiliary balance loss 0.001, `grouped_mm` dispatch, preset `qwen3_5_moe` |")
    w("| comparison arm | dense-1/8 (`dense-1/8`, `configs/qwen3-8b-dspark-dense-1-8-2layer-causal-noconf-b16.json`): the same 2-layer, 8/2-head skeleton with one dense MLP of width 1536; this MoE replaces that MLP by 8 x 1536 top-1 experts, so the active MLP width is identical (1536) and the MLP parameters are 8x (12288 = the full dense MLP width) |")
    w("| target feature layers | 1, 9, 17, 25, 33 (kept at 5 although the draft has 2 layers; the dspark projector is 5 x 4096 -> 4096) |")
    w("| Markov head / confidence head | vanilla, rank 256 / off |")
    w("| projector / mask_token_id | dspark / 151669 |")
    w("| rope_theta / max_position_embeddings | 1e6 / 40960 |")
    w("| vocab | 151936, untied; frozen target embedding (`model.embed_tokens.weight`) |")
    w("| size | about 0.49B parameters in total (2 x (attention 10.5M + experts 151M + router) + projector 84M + Markov head and norms 78M), about 0.22B active per token; the dense-1/8 arm is about 0.22B |")
    w("")
    w("### Training (the dense recipe, applied unchanged unless listed under deviations)")
    w("")
    w("| item | setting |")
    w("|---|---|")
    w("| strategy | dspark, online disaggregated capture (Mooncake TCP) |")
    w(f"| epochs / optimizer steps | {N} in one run / {total:,} ({total_src}) |" if not cont_segs else
      f"| epochs / optimizer steps | {N} = {M} in the first run ({main_total:,} steps, one cosine)"
      + "".join(f" + {s['n']} in a warm-started continuation ({s['total']:,} steps, {recipe(s)})" for s in cont_segs)
      + f" = {total:,} |")
    w("| global batch | 132 = batch_size 2 x accumulation_steps 11 x 6 DDP ranks (the dense-1/8 arm uses 133 = 1 x 19 x 7) |")
    w("| optimizer | AdamW, lr 6e-4, cosine over the whole run, warmup 4%, FP32 master params |")
    w("| max_grad_norm | 1.0 |")
    w("| num_anchors / loss_decay_gamma / objective_chunk_blocks | 512 / 7.0 / 128 |")
    w("| loss | 0.1 CE + 0.9 L1, confidence loss 0 |")
    w("| sharding / precision / attention | NO_SHARD (DDP, bucket 128 MB, one gradient sync per accumulation window, PR #958) / bf16 + Liger kernels / flex_attention; flatten_vocab_projection and cache_vocab_projection on (PR #963); consumer_dispatch cost_balanced (PR #966, same window membership, balanced per-rank cost); four optimizer quanta in flight |")
    w(f"| checkpoints | every 1196 steps, newest 12 kept by the trainer; epoch k = step k x {E:,} (copied out of the rotation, exported, uploaded) |")
    w("| seed / prompt_seed | 42 / 42 |")
    w("| hardware | 1 node x 8 H200: capture servers on GPUs 0-1, trainer ranks on GPUs 2-7 |")
    w("")
    w("### Deviations from the dense run")
    w("")
    w("- Global batch 132 instead of 128: 128 does not divide over 6 trainer ranks; 2 x 11 x 6 = 132 is the branch's standard for the 2-server + 6-rank layout (the dense-1/8 arm runs 1 x 19 x 7 = 133). A 1-server + 7-rank layout was profiled and rejected: the single capture server ran stop-and-go against the producer window and did not beat two servers.")
    if not cont_segs:
        w(f"- {N} epochs in one run with one cosine schedule (the dense run planned 6 epochs and stopped at step 40,960).")
    else:
        w(f"- {N} epochs: {M} in one run with one cosine schedule, then {len(cont_segs)} warm-started continuation run(s); "
          "`deployment.managed_local` cannot resume a run, so each continuation starts from the previous run's final "
          "checkpoint (draft weights only; fresh AdamW state) and logs into the same W&B run with its steps offset by "
          "all earlier steps (`SPECFORGE_LOG_STEP_OFFSET`). (The dense run planned 6 epochs and stopped at step 40,960.)")
        for s in cont_segs:
            w(f"  - epochs {s['first'] + 1}-{s['first'] + s['n']} (`{s['run']}`): warm start from the epoch-{s['first']} "
              f"checkpoint, {recipe(s)}, W&B steps offset by {s['base']:,}.")
    w("- The upstream PR stack #957-#963 is merged (same stack as the dense-1/8 branch) plus #966 (cost-balanced consumer dispatch, a scheduling-only change that keeps each optimizer window's sample membership), so `is_causal: true`, `flatten_vocab_projection`, `cache_vocab_projection`, `ddp_bucket_cap_mb: 128`, one DDP gradient sync per window and the system-prompt removal apply exactly as in the reference run.")
    w("- Structure shrunk like the dense-1/8 arm: 2 layers instead of 5 and GQA 8/2 heads instead of 32/8; the dense FFN is replaced by 8 x 1536 top-1 experts (no shared expert); every other draft-config field is the dense skeleton.")
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
    w(f"- Epoch k covers optimizer steps ((k-1) x {E:,}, k x {E:,}]; the last epoch runs to total_steps. The producer keeps 1,420,903 of the 1,420,909 rows (6 have fewer than 2 trainable tokens), so one epoch is 1,420,903 / 132 = 10,764.42 optimizer steps and total_steps = 129,173; the k x {E:,} windows end at most 5 steps before the true epoch boundaries.")
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
