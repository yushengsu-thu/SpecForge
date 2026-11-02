#!/usr/bin/env python3
"""Write summarize_e2e.py for this comparison from the 2026-10-28 one (same tables, other labels):  make_summarize_e2e.py <best epoch>"""
import sys
best = sys.argv[1]
src = open("/scratch/dense-eval/summarize_e2e.py").read()
for old, new in [
    ('of dense 2ep vs batch-wise MoE 2ep', f'of the 2-epoch run (2ep) vs the 5-epoch run (ep{best})'),
    ('DENSE, MOE = "dense2ep", "bw2ep"', 'DENSE, MOE = "dense2ep", "dense5ep"'),
    ('DENSE_NAME, MOE_NAME = "dense 2ep", "batch-wise MoE 2ep (selection bias)"', f'DENSE_NAME, MOE_NAME = "dense 2ep (2-epoch run)", "dense 5-epoch run ep{best}"'),
    ('"dense first"', '"2ep first"'),
    ('"MoE first"', '"5-epoch first"'),
    ("MoE throughput vs dense", "5-epoch run throughput vs 2ep"),
    ("MoE time per verify step vs dense", "5-epoch run time per verify step vs 2ep"),
    ("MoE accept length vs dense", "5-epoch run accept length vs 2ep"),
    ("MoE cell = MoE (difference vs dense)", "5-epoch cell = 5-epoch run (difference vs 2ep)"),
    ("vs dense)", "vs 2ep)"),
    ("| dense, spread between GPUs", "| 2ep, spread between GPUs"),
]:
    if old not in src:
        raise SystemExit(f"not found: {old}")
    src = src.replace(old, new)
open("/scratch/pr1003/eval5ep/summarize_e2e.py", "w").write(src)
print("summarize_e2e.py written for ep" + best)
