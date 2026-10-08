#!/usr/bin/env python3
"""Convert a SpecForge MoE drafter *serving export* into a *trainable* warm-start checkpoint.

Why this exists
---------------
``specforge export --to hf`` writes the drafter for SGLang. For an MoE drafter whose router
centres its input (``moe_router_center: sample`` in the draft JSON: ``TopKRouter`` keeps an EMA
of the router-input mean ``mu``), the export folds that centering into an fp32
``layers.N.mlp.gate.bias`` (= ``-W @ mu``) and drops the EMA buffers, because the serving
loader has no EMA. The trainer, however, warm-starts ``model.draft_checkpoint_path`` with a
strict key match against its own module: it expects ``layers.N.mlp.gate.input_mean``
(fp32 ``[hidden_size]``) and ``layers.N.mlp.gate.input_mean_steps`` (int64 scalar) and rejects
the bias. ``mu`` (``hidden_size`` values) cannot be recovered from the bias (``n_experts``
values), so this script drops the bias and re-initialises the EMA at zero; with momentum 0.99
the EMA re-converges within the first optimizer steps. Measured on Kan's 3-epoch Qwen3.8-27B
DSpark MoE drafter (``RadixArk/qwen38-dspark-moe-3ep-cont-step9916``): the continued run logged
train/acc 0.628 at step 10 against 0.617-0.638 at the end of his run.

Everything else is copied unchanged (same names, dtypes, shapes); the output is re-sharded
(``--shard-gb``) and gets ``config.json`` plus the tokenizer files. Tensors are streamed shard
by shard, so peak memory is about one shard.

Only the ``qwen3_5_moe`` preset (aux-loss balancing, softmax routing) is handled: there the
export's ``gate.bias`` can only be the folded centering. For ``deepseek_v4`` (``noaux_tc``)
``gate.bias`` is the trainable selection bias and must be renamed, not dropped, so the script
refuses such exports.

Usage
-----
    python scripts/convert_dspark_moe_export_to_trainable.py \
        --src /scratch/drafts/qwen38-moe-3ep --dst /scratch/drafts/qwen38-moe-3ep-trainable
    # then in the recipe:  model.draft_checkpoint_path: /scratch/drafts/qwen38-moe-3ep-trainable
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import re
import shutil
import struct
import sys

COPY_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")
GATE_WEIGHT = re.compile(r"^layers\.(\d+)\.mlp\.gate\.weight$")


def read_headers(src: str) -> dict[str, tuple[str, str, list[int]]]:
    """{tensor name: (file, dtype, shape)} from the safetensors headers, without loading data."""
    out: dict[str, tuple[str, str, list[int]]] = {}
    files = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    if not files:
        sys.exit(f"no *.safetensors under {src}")
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if name in out:
                sys.exit(f"tensor {name} appears in two files: {out[name][0]} and {f}")
            out[name] = (f, meta["dtype"], meta["shape"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="HF export directory (specforge export --to hf)")
    ap.add_argument("--dst", required=True, help="output directory for the trainable checkpoint")
    ap.add_argument("--shard-gb", type=float, default=8.0, help="max shard size in GiB (default 8)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty --dst")
    a = ap.parse_args()

    config = json.load(open(os.path.join(a.src, "config.json")))
    draft_cfg = config.get("dspark_config") or config.get("dflash_config") or {}
    preset = config.get("moe_preset")
    center = draft_cfg.get("moe_router_center", config.get("moe_router_center", "none"))
    hidden = int(config["hidden_size"])
    if preset != "qwen3_5_moe":
        sys.exit(f"refusing: moe_preset={preset!r}; only qwen3_5_moe exports are handled (see docstring)")
    if center != "sample":
        sys.exit(f"refusing: moe_router_center={center!r}; only the 'sample' centering mode is handled")

    headers = read_headers(a.src)
    layers = sorted({int(m.group(1)) for k in headers if (m := GATE_WEIGHT.match(k))})
    bias_keys = sorted(k for k in headers if k.endswith(".mlp.gate.bias"))
    if not layers:
        sys.exit("no layers.N.mlp.gate.weight found: not an MoE drafter export?")
    add = {}
    for n in layers:
        add[f"layers.{n}.mlp.gate.input_mean"] = ("F32", [hidden])
        add[f"layers.{n}.mlp.gate.input_mean_steps"] = ("I64", [])
    already = [k for k in add if k in headers]
    if already:
        sys.exit(f"{a.src} already has EMA buffers ({already[:2]}...): it is a trainable checkpoint, not an export")

    total = sum(
        _nbytes(dtype, shape) for k, (_, dtype, shape) in headers.items() if k not in bias_keys
    ) + sum(_nbytes(d, s) for d, s in add.values())
    print(f"source tensors: {len(headers)} in {len({f for f, _, _ in headers.values()})} file(s); layers: {layers}")
    print(f"drop ({len(bias_keys)}): {bias_keys}")
    print(f"add  ({len(add)}): {sorted(add)}")
    print(f"output: {len(headers) - len(bias_keys) + len(add)} tensors, {total / 1e9:.1f} GB, "
          f"shards of <= {a.shard_gb} GiB -> {a.dst}")
    if a.dry_run:
        return 0

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if os.path.isdir(a.dst) and os.listdir(a.dst) and not a.force:
        sys.exit(f"{a.dst} is not empty; pass --force to overwrite")
    os.makedirs(a.dst, exist_ok=True)
    limit = int(a.shard_gb * 1024**3)
    keep = [k for k in sorted(headers) if k not in bias_keys]
    plan = [(k, "src") for k in keep] + [(k, "add") for k in sorted(add)]
    shards: list[list[str]] = []
    cur: dict[str, "torch.Tensor"] = {}
    size = 0
    weight_map: dict[str, str] = {}
    pending: list[dict] = []

    def flush() -> None:
        nonlocal cur, size
        if cur:
            pending.append(cur)
            cur, size = {}, 0

    with contextlib.ExitStack() as stack:
        handles = {f: stack.enter_context(safe_open(f, framework="pt")) for f in {h[0] for h in headers.values()}}
        for k, kind in plan:
            if kind == "src":
                t = handles[headers[k][0]].get_tensor(k)
            elif k.endswith("input_mean"):
                t = torch.zeros(hidden, dtype=torch.float32)
            else:
                t = torch.zeros((), dtype=torch.long)
            nb = t.numel() * t.element_size()
            if size + nb > limit and cur:
                flush()
            cur[k] = t.contiguous()
            size += nb
            # write finished shards eagerly so memory stays at about one shard
            while pending:
                _write_shard(pending.pop(0), a.dst, shards, weight_map, save_file, placeholder=True)
        flush()
        while pending:
            _write_shard(pending.pop(0), a.dst, shards, weight_map, save_file, placeholder=True)

    # rename placeholder shard files to the final model-XXXXX-of-NNNNN names
    n = len(shards)
    final_map = {}
    for i, tmp in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
        os.replace(os.path.join(a.dst, tmp), os.path.join(a.dst, name))
        for k, v in weight_map.items():
            if v == tmp:
                final_map[k] = name
    index = {"metadata": {"total_size": total}, "weight_map": final_map}
    with open(os.path.join(a.dst, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    for f in COPY_FILES:
        p = os.path.join(a.src, f)
        if os.path.exists(p):
            shutil.copy(p, a.dst)
    print(f"DONE: {len(final_map)} tensors in {n} shard(s) -> {a.dst}")
    return 0


def _nbytes(dtype: str, shape: list[int]) -> int:
    width = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1, "BOOL": 1, "F64": 8}[dtype]
    n = 1
    for s in shape:
        n *= s
    return n * width


def _write_shard(tensors, dst, shards, weight_map, save_file, placeholder=False):
    tmp = f"shard-{len(shards):05d}.safetensors.part"
    save_file(tensors, os.path.join(dst, tmp), metadata={"format": "pt"})
    shards.append(tmp)
    for k in tensors:
        weight_map[k] = tmp
    print(f"wrote {tmp}: {len(tensors)} tensors", flush=True)


if __name__ == "__main__":
    sys.exit(main())
