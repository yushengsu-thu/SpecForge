"""Recheck immutable trial payloads using the hardened identity validator."""
from pathlib import Path
import ast
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from training_backend_matrix import validate_and_summarize

raw = ROOT / 'raw'
plan = json.loads((raw / 'matrix-plan.json').read_text())
records = {
    (trial['algorithm'], trial['case'], trial['repeat']):
    json.loads((raw / f"{trial['name']}.json").read_text())
    for trial in plan['trials']
}
summary = validate_and_summarize(records, plan['tolerances'], repeats=3)
original = json.loads((raw / 'comparison.json').read_text())
assert summary == original, 'Hardened analysis changed the frozen result'
files = {}
for name in ['benchmark_training_backends.py', 'training_benchmark_recipes.py', 'training_backend_matrix.py']:
    measured = (ROOT / 'measured-driver/scripts' / name).read_bytes()
    current = (HERE / name).read_bytes()
    assert hashlib.sha256(measured).hexdigest() == plan['driver_sha256'][name]
    files[name] = {
        'measured_sha256': hashlib.sha256(measured).hexdigest(),
        'post_analysis_sha256': hashlib.sha256(current).hexdigest(),
        'ast_equal': ast.dump(ast.parse(measured)) == ast.dump(ast.parse(current)),
    }
audit = {
    'core_commit': '5a35ff35b0ba939242f294907892a2d8955a52b9',
    'benchmark_code_commit': '56130f347a9d2c0b67a7f9330f3359b22e34fcf5',
    'trials': len(records),
    'identity_validation_passed': True,
    'complete_summary_equal_to_frozen_execution': summary == original,
    'graph_gate_passed': summary['graph_gate_passed'],
    'files': files,
}
(HERE / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
(HERE / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2))
