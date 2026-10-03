"""CPU-only counterfactual: fresh FP32 RoPE vs BF16-rounded-then-FP32 RoPE."""
import hashlib
import importlib.metadata
import inspect
import json
import platform
from pathlib import Path

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

root = Path(__file__).resolve().parent
torch.set_num_threads(1)
config = Qwen3Config.from_dict(json.loads((root / "config.json").read_text()))
fresh = Qwen3RotaryEmbedding(config)
legacy = Qwen3RotaryEmbedding(config).to(dtype=torch.bfloat16).to(dtype=torch.float32)
position_ids = torch.arange(4096).reshape(1, -1)
x = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
fresh_cos, fresh_sin = fresh(x, position_ids)
legacy_cos, legacy_sin = legacy(x, position_ids)
frequency_error = (fresh.inv_freq - legacy.inv_freq).abs()
phase_error = frequency_error.unsqueeze(-1) * position_ids.float()

def buffers(module):
    return {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            "values": value.tolist(),
        }
        for name, value in module.named_buffers()
    }

source = Path(inspect.getfile(Qwen3RotaryEmbedding))
report = {
    "device": "cpu",
    "python": platform.python_version(),
    "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
    "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
    "source": {"file": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
    "config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
    "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "positions": [0, 4095],
    "fresh_state_dict_keys": list(fresh.state_dict()),
    "legacy_state_dict_keys": list(legacy.state_dict()),
    "fresh_nonpersistent_buffer_names": sorted(fresh._non_persistent_buffers_set),
    "fresh_buffers": buffers(fresh),
    "legacy_buffers": buffers(legacy),
    "inv_freq_numel": fresh.inv_freq.numel(),
    "inv_freq_different_elements": int((fresh.inv_freq != legacy.inv_freq).sum()),
    "inv_freq_max_absolute_difference": frequency_error.max().item(),
    "phase_max_absolute_difference_radians": phase_error.max().item(),
    "cos_shape": list(fresh_cos.shape),
    "cos_different_elements": int((fresh_cos != legacy_cos).sum()),
    "cos_max_absolute_difference": (fresh_cos - legacy_cos).abs().max().item(),
    "sin_max_absolute_difference": (fresh_sin - legacy_sin).abs().max().item(),
}
(root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key not in {"fresh_buffers", "legacy_buffers"}}, indent=2))
