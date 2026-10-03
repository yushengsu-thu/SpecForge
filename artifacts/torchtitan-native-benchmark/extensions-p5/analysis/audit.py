"""CPU-only final audit of all immutable p5 records; run after all36 finish."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'measured-driver/scripts'))
from training_backend_matrix import validate_and_summarize

raw = root / 'raw'
plan = json.loads((raw / 'matrix-plan.json').read_text())
launches = json.loads((raw / 'launch-manifest.json').read_text())
assert len(launches) == len(plan['trials']) == 36
assert all(item['exit_code'] == 0 for item in launches)
records = {
    (trial['algorithm'], trial['case'], trial['repeat']):
    json.loads((raw / f"{trial['name']}.json").read_text())
    for trial in plan['trials']
}
summary = validate_and_summarize(records, plan['tolerances'], repeats=3)
assert summary == json.loads((raw / 'comparison.json').read_text())
source = json.loads((root / 'provenance/extension-bench-p5-source-check.json').read_text())
driver = json.loads((root / 'provenance/extension-bench-p5-driver-provenance.json').read_text())
assert plan['source_sha256'] == source['source_sha256']
assert plan['driver_sha256'] == driver['files']
for name, expected in driver['files'].items():
    assert hashlib.sha256((root / 'measured-driver/scripts' / name).read_bytes()).hexdigest() == expected
for record in records.values():
    assert record['source_sha256'] == source['source_sha256']
    assert record['benchmark_sha256'] == driver['files']['benchmark_training_backends.py']
    assert record['recipe_helpers_sha256'] == driver['files']['training_benchmark_recipes.py']
    assert record['runtime']['kernel_environment']['SPECFORGE_FLEX_ATTENTION_BACKEND'] == ''

p4 = Path(sys.argv[1]) if len(sys.argv) > 1 else root.parent / 'extension-bench-p4-evidence/raw'
prior_checks = []
for trial in plan['trials']:
    name = trial['name']
    previous = json.loads((p4 / f'{name}.json').read_text())['comparison_contract']
    current = records[(trial['algorithm'], trial['case'], trial['repeat'])]['comparison_contract']
    assert all(current[key] == value for key, value in previous.items()), name
    prior_checks.append(name)

audit = {
    'core_commit': source['core_commit'],
    'benchmark_code_commit': driver['benchmark_code_commit'],
    'trials_completed': len(launches),
    'all_child_exits_zero': True,
    'observed_orchestrator_exit_code': None,
    'orchestrator_exit_status_note': 'SSH monitor exited255 during relay rollout; original remote process survived. Final child exits and aggregate are observed, while the orchestrator exit status is unavailable.',
    'all_168_core_source_hashes_equal': len(source['source_sha256']) == 168,
    'all_driver_files_exact_committed_bytes': True,
    'all_36_nonpersistent_buffer_and_algorithm_identities_pass': True,
    'all_36_legacy_persistent_state_and_input_contracts_equal_to_p4': prior_checks,
    'aggregate_exactly_reproduced': True,
    'graph_gate_passed': summary['graph_gate_passed'],
    'graph_pairs_failed': sum(not item['passed'] for item in summary['gates']),
    'original_fsdp_backend_with_benchmark_fp32_buffer_normalization': True,
    'limitation': 'Implementation timing and first-window objective checks are not matched-quality, gradient, or convergence evidence.',
}
(root / 'analysis/audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2))
