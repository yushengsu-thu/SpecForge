"""Public CLI CUDA graph capture/replay and exact DCP continuation checks."""
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

root = Path('/scratch/specforge-torchtitan-20261003')
source = root / 'graph-loss-source-candidate4'
out = root / 'native-cuda-loss-fix-tp2'
out.mkdir(exist_ok=True)
(out / 'source-manifest.json').write_text(json.dumps({
    'core_commit': '5a35ff35b0ba939242f294907892a2d8955a52b9',
    'source_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted((source / 'specforge').rglob('*.py'))}
}, indent=2) + '\n')
base = yaml.safe_load((root / 'graph-probes/graph-fused-dp1.yaml').read_text())
base['deployment']['trainer']['nproc_per_node'] = 2
base['output_dir'] = str(out)
base['training']['torchtitan']['engine'] = 'trainer'
base['training']['torchtitan']['compile'] = True
base['training']['torchtitan']['disable_cuda_graphs'] = False
base['training']['save_interval'] = 1
base['training']['num_epochs'] = 4
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='4,5', PYTHONPATH=str(source),
                   OMP_NUM_THREADS='1', SPECFORGE_DFLASH_FUSED_HEAD='1',
                   SPECFORGE_DFLASH2_FUSED_CONV='1')
results = {}
for layout, tp, cp in [('tp2', 2, 1)]:
    for phase in ['full', 'cut', 'resume']:
        config = copy.deepcopy(base)
        config['run_id'] = f'{layout}-{phase}'
        config['training']['tp_size'] = tp
        config['training']['torchtitan']['cp_size'] = cp
        config['training']['max_steps'] = 1 if phase == 'cut' else 3
        if phase == 'resume':
            config['training']['resume_from'] = str(out / f'{layout}-cut/checkpoint/step-1')
        filename = out / f'{layout}-{phase}.yaml'
        filename.write_text(yaml.safe_dump(config))
        with (out / f'{layout}-{phase}.log').open('w') as log:
            run = subprocess.run([str(root / 'venv/bin/python'), '-m', 'torch.distributed.run',
                                  '--standalone', '--nproc-per-node=2', '-m', 'specforge.cli',
                                  'train', '--config', str(filename)], cwd=source,
                                 env=environment, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=600)
        print(json.dumps({'layout': layout, 'phase': phase, 'exit_code': run.returncode}), flush=True)
        if run.returncode:
            results[layout] = {'runtime_failed': phase}
            break
    else:
        states = {}
        for phase in ['full', 'resume']:
            filename = out / f'{layout}-{phase}.pt'
            dcp_to_torch_save(out / f'{layout}-{phase}/checkpoint/step-3', filename)
            states[phase] = torch.load(filename, map_location='cpu', weights_only=False)
        differences = []
        counts = {'tensors': 0, 'values': 0}
        def compare(left, right, name=''):
            if isinstance(left, torch.Tensor):
                counts['tensors'] += 1
                if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
                    differences.append(name)
            elif isinstance(left, dict):
                if left.keys() != right.keys(): differences.append(name + '/keys')
                for key in left.keys() & right.keys(): compare(left[key], right[key], name + '/' + str(key))
            elif isinstance(left, (tuple, list)):
                if len(left) != len(right): differences.append(name + '/length')
                for index, (a, b) in enumerate(zip(left, right)): compare(a, b, name + '/' + str(index))
            else:
                counts['values'] += 1
                if left != right: differences.append(name)
        compare(states['full'], states['resume'])
        exports = {}
        for phase in ['full', 'resume']:
            exports[phase] = {}
            for path in (out / f'{layout}-{phase}/draft').glob('*.safetensors'):
                exports[phase].update(load_file(path))
        export_equal = bool(exports['full']) and exports['full'].keys() == exports['resume'].keys()
        export_equal = export_equal and all(torch.equal(value, exports['resume'][key]) for key, value in exports['full'].items())
        results[layout] = {'checkpoint_bitwise_equal': not differences, 'differences': differences,
                           'compared': counts, 'export_keys': len(exports['full']),
                           'resume_export_bitwise_equal': export_equal,
                           'capture_logged': 'Recorded CUDA graph' in (out / f'{layout}-full.log').read_text()}
    (out / 'verification.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results[layout]), flush=True)
assert all(r.get('checkpoint_bitwise_equal') and r.get('resume_export_bitwise_equal') and r.get('capture_logged') for r in results.values())
