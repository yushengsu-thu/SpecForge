"""CPU verification of the completed real-TCP consumer and no-op resume."""

import hashlib
import importlib.metadata
import json
import os
import re
import sqlite3
from pathlib import Path

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

root = Path('/scratch/specforge-torchtitan-20261003/mooncake-native-real')
stream = root / 'stream'
output = stream / 'output/native-online'
repo = root.parent / 'SpecForge-titan-extensions'
source = json.loads((root / 'source-hashes.json').read_text())
for name, expected in source['files'].items():
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == expected, name
before = json.loads((root / 'before-noop-resume.json').read_text())
connection = sqlite3.connect(stream / 'consumer.sqlite')
acked = sorted(row[0] for row in connection.execute('select sample_id from acked'))
marker = {key: json.loads(value) for key, value in connection.execute('select k,v from marker')}
connection.close()
assert acked == sorted(f'native-{i}' for i in range(12))
assert marker == before['marker'] == {'global_step': 3, 'optimizer_durable': True}
assert hashlib.sha256(json.dumps(acked).encode()).hexdigest() == before['ack_ids_sha256']
checkpoint_hashes = {
    str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((output / 'checkpoint/step-3').rglob('*')) if path.is_file()
}
assert checkpoint_hashes == before['checkpoint_sha256']
assert hashlib.sha256((output / 'draft/model.safetensors').read_bytes()).hexdigest() == before['export_sha256']
assert Path(str(stream / 'refs.jsonl') + '.consumer_done').stat().st_mtime_ns > before['consumer_done_mtime_ns']
dcp_to_torch_save(output / 'checkpoint/step-3', output / 'verify-state.pt')
state = torch.load(output / 'verify-state.pt', weights_only=False, map_location='cpu')
assert state['train_state']['step'] == 3
contract = state['train_state']['specforge_resume_contract']
assert contract['stream_natural_steps'] == 3 and contract['schedule_total_steps'] == 100
assert state['dataloader'] == {f'dp_rank_{rank}': {'committed_step': 3, 'dp_world_size': 2} for rank in range(2)}
train_log = re.sub(r'\x1b\[[0-9;]*m', '', (root / 'train.log').read_text())
resume_log = re.sub(r'\x1b\[[0-9;]*m', '', (root / 'completed-resume.log').read_text())
losses = {int(step): float(loss) for step, loss in re.findall(r'step:\s+(\d+)\s+loss:\s+([0-9.]+)', train_log)}
assert sorted(losses) == [1, 2, 3]
assert not re.findall(r'step:\s+(\d+)\s+loss:', resume_log)
assert "'dispatched': 0, 'skipped': 12" in resume_log

# Verify physical deletion through a fresh real Mooncake client. Read only
# whitelisted endpoint settings locally; no environment dump is archived.
endpoint_settings = json.loads((root / 'environment.json').read_text())
os.environ.update(endpoint_settings)
from specforge.config import load_config
from specforge.training.disaggregated import _mooncake_store
cfg = load_config(str(root.parent / 'cli-smoke213/train.yaml'))
store = _mooncake_store(cfg, retain_on_release=True)
removed_keys = 0
try:
    for line in (stream / 'refs.jsonl').read_text().splitlines():
        ref = json.loads(line)
        for name in ref['feature_keys']:
            key = store._tkey(ref['sample_id'], ref['metadata']['generation'], name)
            assert not store._store_exists(key), key
            removed_keys += 1
finally:
    store.close()
assert removed_keys == 48
result = {
    'passed': True,
    'core_commit': source['core_commit'],
    'matched_source_files': len(source['files']),
    'torch': torch.__version__,
    'torchtitan': importlib.metadata.version('torchtitan'),
    'mooncake': importlib.metadata.version('mooncake-transfer-engine-cuda13'),
    'algorithm': 'dflash2',
    'parallelism': 'DP2/FSDP2',
    'attention': 'eager',
    'transport': 'real Mooncake TCP, same host, pageable host receive',
    'memcpy_shortcut_disabled_on_consumer': endpoint_settings['MC_STORE_MEMCPY'] == '0',
    'kernel_policy': 'Shipping defaults; neither fused-head nor fused-convolution environment switch overridden',
    'producer_natural_steps': 3,
    'lr_schedule_horizon': 100,
    'optimizer_steps': 3,
    'logged_losses': losses,
    'unique_durable_acks': 12,
    'physically_removed_tensor_objects': removed_keys,
    'checkpoint_steps': [1, 2, 3],
    'hf_export_exists': True,
    'completed_checkpoint_resume': {
        'optimizer_steps_executed': 0,
        'dispatched_samples': 0,
        'skipped_acked_samples': 12,
        'ledger_unchanged': True,
        'checkpoint_bytes_unchanged': True,
        'hf_export_bytes_unchanged': True,
        'consumer_done_republished': True,
    },
    'boundaries': ['Synthetic published features; no SGLang capture server', 'One host TCP; no RDMA or multi-node validation', 'Tiny two-layer fixture; no performance or training-quality claim'],
}
(root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
