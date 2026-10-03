"""Corrected GraphTrainer DCP continuation and evaluation isolation checks."""
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
source = root / 'graph-loss-source-candidate3'
out = root / 'graph-loss-lifecycle'
out.mkdir(exist_ok=True)
base = yaml.safe_load((root / 'graph-probes/graph-fused-dp1.yaml').read_text())
base['deployment']['trainer']['nproc_per_node'] = 2
base['output_dir'] = str(out)
base['training']['torchtitan'].update(engine='graph', graph_inductor='full')
base['training'].update(save_interval=1, num_epochs=4, objective_chunk_blocks=2)
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='4,5', PYTHONPATH=str(source),
                   OMP_NUM_THREADS='1', SPECFORGE_DFLASH_FUSED_HEAD='1',
                   SPECFORGE_DFLASH2_FUSED_CONV='1')
manifest = []
for phase in ['full', 'cut', 'resume', 'eval']:
    config = copy.deepcopy(base)
    config['run_id'] = phase
    config['training']['max_steps'] = 1 if phase == 'cut' else 3
    if phase == 'resume':
        config['training']['resume_from'] = str(out / 'cut/checkpoint/step-1')
    if phase == 'eval':
        config['data']['eval_hidden_states_path'] = config['data']['hidden_states_path']
        config['training']['eval_interval'] = 1
    filename = out / f'{phase}.yaml'
    filename.write_text(yaml.safe_dump(config))
    command = [str(root / 'venv/bin/python'), '-m', 'torch.distributed.run',
               '--standalone', '--nproc-per-node=2', '-m', 'specforge.cli',
               'train', '--config', str(filename)]
    with (out / f'{phase}.log').open('w') as log:
        run = subprocess.run(command, cwd=source, env=environment,
                             stdout=log, stderr=subprocess.STDOUT, timeout=900)
    manifest.append(dict(phase=phase, command=command, exit_code=run.returncode))
    (out / 'launch-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest[-1]), flush=True)
    if run.returncode:
        raise RuntimeError(f'{phase} failed; inspect log')

states, exports = {}, {}
for phase in ['full', 'resume', 'eval']:
    filename = out / f'{phase}.pt'
    dcp_to_torch_save(out / f'{phase}/checkpoint/step-3', filename)
    states[phase] = torch.load(filename, map_location='cpu', weights_only=False)
    exports[phase] = {}
    for path in (out / f'{phase}/draft').glob('*.safetensors'):
        exports[phase].update(load_file(path))

reports = {}
for phase in ['resume', 'eval']:
    differences, counts = [], {'tensors': 0, 'values': 0}
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
    compare(states['full'], states[phase])
    export_equal = bool(exports['full']) and exports['full'].keys() == exports[phase].keys()
    export_equal = export_equal and all(torch.equal(value, exports[phase][key]) for key, value in exports['full'].items())
    reports[phase] = dict(checkpoint_bitwise_equal=not differences, differences=differences,
                          compared=counts, export_keys=len(exports['full']),
                          export_bitwise_equal=export_equal)
reports['source_sha256'] = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((source/'specforge').rglob('*.py'))}
(out / 'verification.json').write_text(json.dumps(reports, indent=2)+'\n')
print(json.dumps({k:v for k,v in reports.items() if k!='source_sha256'}, indent=2))
assert all(reports[p]['checkpoint_bitwise_equal'] and reports[p]['export_bitwise_equal'] for p in ['resume','eval'])
