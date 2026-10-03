# Shared-activation BF16 fanout arithmetic probe

This standalone probe demonstrates that compilation boundaries can change the
BF16 backward addition tree even when forward values match and eager precision
boundaries are preserved. It contains no SpecForge model, optimizer, data,
SimpleFSDP, or distributed collective. It therefore does **not** establish the
cause or acceptability of any full-model loss trajectory difference.

The input is a shared BF16 activation with 64 elements, each equal to 0.5. Two
layers each use that activation in two branches. Their coefficients are
`[[1, BF16(0.004)], [-1, BF16(0.004)]]`; stored BF16(0.004) equals
0.003997802734375. The scalar objective sums the final activation in FP32.
All cases produce exactly the same loss, **0.37890625**.

The independent exact derivative, computed by summing the stored coefficients
in FP64, is **0.00799560546875 per activation element**. This is an exact-arithmetic
reference for the linear operations, not a finite-difference derivative of a
quantized forward map.

| Execution | Explicit per-layer backward merge | Gradient per element | Absolute error from FP64 reference |
|---|---|---:|---:|
| Eager | No | 0.0078125 | 0.00018310546875 |
| `torch.compile` on each layer | No | 0.01171875 | 0.00372314453125 |
| TorchTitan full joint Inductor pass | No | 0.0078125 | 0.00018310546875 |
| Eager | Yes | 0.01171875 | 0.00372314453125 |
| `torch.compile` on each layer | Yes | 0.01171875 | 0.00372314453125 |
| TorchTitan full joint Inductor pass | Yes | 0.01171875 | 0.00372314453125 |

**CPU and NVIDIA H200 results are identical for all six cases.** All compilations
use `emulate_precision_casts=True` and `eager_numerics.division_rounding=True`.
The full joint case uses TorchTitan's `full_inductor_compilation_pass` on a
`make_fx` forward/backward trace; it does not instantiate the full trainer.

The custom autograd function returns two copies of the activation and explicitly
adds their two gradients in BF16 before returning to the shared activation.
It makes the three execution modes agree, but in this cancellation-sensitive
fixture that common answer is about 20.33 times farther from the exact derivative.
Consequently, forcing a gradient merge purely to match another backend's values
would select an addition tree; it would not by itself correct a mathematical bug.

For this particular four-term addition, the usual round-to-nearest BF16 absolute
error bound is `gamma_3 * sum(abs(coefficients))`, where unit roundoff is
`u=2^-8` and `gamma_3=3u/(1-3u)`. It is approximately **0.02381022457**;
both measured errors lie within it. This is only a bound for this small sum,
not a bound for an optimizer update, a neural network, or a 30-step loss trajectory.
The deliberately cancelling coefficients make relative error an unsuitable
sole criterion here.

## Files and reproduction

- `context_fanout_probe.py.txt`: exact source used for both CPU and GPU runs.
- `context_fanout_probe_cpu.json` and `context_fanout_probe_cuda.json`: all six
  cases plus the actual joint FX graphs before Inductor.
- `context_fanout_probe_cpu.log`: raw redirected stdout/stderr from the CPU run.
- `context_fanout_probe_cuda.stdout.log`: six stdout records reconstructed from
  the saved result JSON and checked against the six tool-captured stdout lines;
  this is not a separate rerun or a raw stderr capture.
- `context_fanout_probe_versions.json`: Python, Torch, CUDA runtime, TorchTitan,
  Triton, GPU, and SHA256 hashes of the source and relevant installed compiler files.
- `manifest.json`: hashes of every other file in this directory.

Run in the recorded Torch 2.14 / TorchTitan 0.3 environment:

```bash
CUDA_VISIBLE_DEVICES= python context_fanout_probe.py.txt cpu cpu.json
CUDA_VISIBLE_DEVICES=1 python context_fanout_probe.py.txt cuda cuda.json
```

The GPU run used physical GPU 1 exclusively. No full-model speed, convergence,
quality, or backend-equivalence conclusion is supported by this probe.
