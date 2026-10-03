# Isolated BF16 indexed-gradient repeatability

This probe isolates `weight[indices]` backward for a BF16 64-by-4 table. The
objective multiplies selected rows by fixed BF16 gradient seeds and sums in
FP32. It includes no model adapter, parameter cache, checkpoint, evaluation,
optimizer, SimpleFSDP, or collective. Each case runs the same inputs 100 times.

All compilations use `emulate_precision_casts=True` and
`eager_numerics.division_rounding=True`. The full compiled case calls TorchTitan's
`full_inductor_compilation_pass` on a joint `make_fx` forward/backward graph.
The GPU is NVIDIA H200, physical GPU 1; Torch is 2.14.0+cu130.

| Indexed rows | Execution | Deterministic algorithms | Distinct gradients / 100 | Max difference from first gradient |
|---:|---|---|---:|---:|
| 320 | Eager | False | 1 | 0 |
| 320 | Full joint Inductor | False | 16 | 0.00006103515625 |
| 320 | Full joint Inductor | True | 1 | 0 |
| 4096 | Eager | False | 1 | 0 |
| 4096 | Full joint Inductor | False | 100 | 0.000732421875 |
| 4096 | Full joint Inductor | True | 1 | 0 |

Every forward loss is bitwise identical across repeats. At both sizes,
deterministic full Inductor's gradient SHA256 equals the eager gradient SHA256.
The JSON also records maximum absolute error against the sum of BF16 seeds
accumulated in FP64. The deterministic/eager errors are 2.384185791015625e-5
and 5.316734313964844e-5 for 320 and 4096 rows, respectively.

The installed Torch lowering source (`torch/_inductor/lowering.py:4977–4979`)
selects ATen fallback when deterministic algorithms are enabled; its ordinary
path creates a scatter with `atomic_add` for accumulation (`5043–5049`).
This and the isolated repeatability result identify a real compiled indexed
backward nondeterminism mechanism. They do not alone prove that a full-model
checkpoint/evaluation difference is cured; that requires repeating its lifecycle
test with the selected policy.

Files include exact source as text, raw stdout/stderr, result JSON, and a hash
manifest. Runtime/source hashes are also recorded in `../selector-lifecycle-audit-versions.json`.

```bash
CUDA_VISIBLE_DEVICES=1 python selector_index_repeat_probe.py.txt result.json
```
