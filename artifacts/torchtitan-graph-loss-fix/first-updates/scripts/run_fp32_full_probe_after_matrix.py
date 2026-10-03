"""Correctness-only follow-up. Launch only after exclusive timing has ended."""
import json
import os
import subprocess
from pathlib import Path

root = Path('/scratch/specforge-torchtitan-20261003')
name = 'first-update-fp32-c2-full'
source = root / 'first-update-source-candidate2'
env = {
    **os.environ, 'CUDA_VISIBLE_DEVICES': '0,1', 'OMP_NUM_THREADS': '1',
    'COLLECT_FLOAT32': '1', 'COLLECT_TINY_HEAD_DIM': '16',
    'COLLECT_DISABLE_GRAPH_PASSES': '0', 'COLLECT_KEEP_FLEX_PASSES': '0',
    'COLLECT_DRIVER': str(root / 'first-update-driver-candidate1/scripts/benchmark_training_backends.py'),
    'COLLECT_DIRECTORY': str(root / name),
    'PROBE_PRESERVE_CASTS': '1', 'PROBE_DIVISION_ROUNDING': '1',
}
command = [
    str(root / 'venv/bin/python'), '-m', 'torch.distributed.run', '--standalone',
    '--nproc-per-node=2', str(root / 'collect_titan_first_updates.py'),
    '--specforge-root', str(source), '--backend', 'torchtitan', '--algorithm', 'dflash2',
    '--tiny', '--seq-length', '128', '--num-anchors', '8', '--objective-chunk-blocks', '2',
    '--warmup-steps', '1', '--steps', '1', '--attention', 'eager',
    '--compile', '--cuda-graphs', '--titan-engine', 'graph', '--graph-inductor', 'full',
    '--output', str(root / f'{name}.json'),
]
(root / f'{name}-command.json').write_text(json.dumps({
    'command': command,
    'environment': {key: env[key] for key in env if key.startswith(('COLLECT_', 'PROBE_')) or key in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS')},
}, indent=2) + '\n')
with (root / f'{name}.log').open('w') as log:
    subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
subprocess.run([
    str(root / 'venv/bin/python'), str(root / 'compare_titan_first_updates.py'),
    str(root / 'first-update-fp32-c2-native'), str(root / name),
    '--output', str(root / 'first-update-fp32-c2-native-vs-full.json'),
], check=True)
