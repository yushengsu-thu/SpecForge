"""Bounded forward-loss preflight for the corrected shared buffer contract."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path('/scratch/specforge-torchtitan-20261003')
source = root / 'extension-bench-source-p5'
driver = root / 'extension-bench-driver-p5'
output = root / 'extension-bench-p5-preflight'
output.mkdir(exist_ok=True)
sys.path.insert(0, str(driver / 'scripts'))
from training_backend_matrix import plan_trials, source_hashes

trials = [trial for trial in plan_trials(
    root / 'venv/bin/python', source, driver, output, 1, 1, 1, '4,5',
    python213=root / 'venv213/bin/python',
) if trial['case'] in ('fsdp214', 'titan-cuda')]
plan = {
    'purpose': 'Correctness preflight only; no steady performance claim from two windows',
    'trials': trials,
    'source_sha256': source_hashes(source),
    'driver_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted((driver / 'scripts').glob('*.py'))},
    'first_forward_tolerances': {'atol': 1e-5, 'rtol': 1e-4},
}
(output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '4,5', 'OMP_NUM_THREADS': '1',
       'PYTHONPATH': str(source), 'SPECFORGE_DFLASH_FUSED_HEAD': '1',
       'SPECFORGE_DFLASH2_FUSED_CONV': '1',
       'SPECFORGE_FLEX_ATTENTION_BACKEND': ''}
launches = []
for trial in trials:
    if (output / f"{trial['name']}.json").exists():
        raise RuntimeError('Preflight output already exists; refusing reuse')
    assert source_hashes(source) == plan['source_sha256']
    assert all(hashlib.sha256((driver / 'scripts' / name).read_bytes()).hexdigest() == digest
               for name, digest in plan['driver_sha256'].items())
    started = time.time()
    with (output / f"{trial['name']}.log").open('w') as stream:
        result = subprocess.run(trial['command'], cwd=source, env=env,
                                stdout=stream, stderr=subprocess.STDOUT, timeout=900)
    launches.append({'name': trial['name'], 'exit_code': result.returncode,
                     'started': started, 'finished': time.time()})
    (output / 'launches.json').write_text(json.dumps(launches, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError(f"Failed {trial['name']}: {result.returncode}")
checks = []
for algorithm in ('dflash', 'dflash2', 'dspark'):
    fsdp, titan = [json.loads((output / f'{algorithm}-{case}-1.json').read_text())
                   for case in ('fsdp214', 'titan-cuda')]
    ref, actual = fsdp['losses_including_warmup'][0], titan['losses_including_warmup'][0]
    error = abs(ref-actual)
    check = {'algorithm': algorithm,
             'comparison_contract_equal': fsdp['comparison_contract'] == titan['comparison_contract'],
             'reference_loss': ref, 'candidate_loss': actual,
             'first_forward_absolute_error': error,
             'first_forward_passed': error <= 1e-5 + 1e-4 * abs(ref)}
    checks.append(check)
report = {'checks': checks, 'passed': all(item['comparison_contract_equal'] and item['first_forward_passed'] for item in checks)}
(output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
if not report['passed']:
    raise SystemExit(1)
