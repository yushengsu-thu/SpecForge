#!/usr/bin/env python3
"""Prepare RadixArk/Qwen3.8-27B-Regen-Mixture-v1 for SpecForge training.

Decompresses every data/*.jsonl.zst shard and writes one JSONL with the slim rows
Kan's recipe trains on ({"id", "conversations"}), keeping only rows whose
regeneration status is "success" and whose conversations are non-empty.

  python prep_regen_mixture.py --src /scratch/data/regen-mixture-v1/data \
      --out /scratch/data/qwen38_regen_mixture_v1_train.jsonl
"""
import argparse
import glob
import io
import json
import os
import sys

import zstandard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    shards = sorted(glob.glob(os.path.join(args.src, "*.jsonl.zst")))
    kept = dropped = 0
    per_split = {}
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as out:
        for shard in shards:
            split = os.path.basename(shard).split("-")[0]
            n_kept = 0
            with open(shard, "rb") as fh:
                reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
                for line in reader:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    conv = row.get("conversations") or []
                    if row.get("status", "success") != "success" or not conv:
                        dropped += 1
                        continue
                    out.write(json.dumps({"id": row.get("id"), "conversations": conv}, ensure_ascii=False) + "\n")
                    kept += 1
                    n_kept += 1
            per_split[split] = per_split.get(split, 0) + n_kept
            print(f"{os.path.basename(shard)}: kept {n_kept}", flush=True)
    os.replace(tmp, args.out)
    print(json.dumps({"kept": kept, "dropped": dropped, "per_split": per_split}, indent=2))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e9:.1f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
