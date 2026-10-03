# Preliminary extension probes and deferred FP8

These results are diagnostic history. They are **not** the final repeated
`p2` performance matrix and must not be pooled with it. The early probes did
not establish exclusive whole-node isolation. All use synthetic cached
features, random initial models and frozen synthetic teacher tables; none
establish real-data convergence, model quality or serving speed.

## One-trial performance probes

Each row is one fresh DP2 process pair on H200, BF16 computation, batch 1 per
rank and accumulation 2. The controlled DFlash2 recipe has 632,360,192 trainable
parameters, 4096 context tokens, 512 anchors and block size 16. Fused head and
convolution environment flags are enabled. CPU token IDs/masks and cached GPU
hidden states are identical within a snapshot, as recorded in the raw
comparison contracts.

The executed convolution still differs by compiler path: compiled native
DFlash2 uses the ATen decomposition selected by
`torch.compiler.is_compiling()`, while GraphTrainer traces the wrapped custom
Triton convolution. Native CUDA capture replays the compiled native path.
The flags alone do not establish identical kernels, and comparing these
configurations does not isolate compilation from convolution implementation.

| Snapshot | Configuration | Warmup / steady windows | Steady ms/window | Allocated / reserved GiB | First window s |
| --- | --- | ---: | ---: | ---: | ---: |
| p0 | Compiled native BF16 | 5 / 10 | 213.92 | 16.235 / 17.783 | 3.45 |
| p0 | Native CUDA graph | 5 / 10 | 206.91 | 6.806 / 17.748 | 2.83 |
| p0 | Experimental rowwise FP8 | 5 / 10 | 223.15 | 14.979 / 16.609 | 34.11 |
| p1 | GraphTrainer regional + capture | 10 / 20 | 295.05 | 7.992 / 15.273 | 41.58 |
| p1 | GraphTrainer full Inductor + capture | 10 / 20 | 212.38 | 7.995 / 18.818 | 48.26 |

The FP8 probe was 4.3% slower than its same-snapshot BF16 baseline. It converted
36 draft linear modules, including the feature projection; frozen teacher/head
weights and auxiliary convolution/selector work were not converted. This one
trial does not support a general conclusion about FP8 performance.

The `p1` rows compare two compilation modes with the same source and inputs.
They do not share a frozen source snapshot or the same number of windows with
`p0`. Startup windows include lazy compile/capture costs and shared disk caches
were not cleared. Allocator peaks reset after warmup: graph pools can retain
reserved memory despite a lower live allocated peak. The two figures should
be read together.

- [p0 raw trials and manifests](p0/raw/) retain original bytes.
- [p1 raw trials and manifests](p1/raw/) retain original bytes.
- Files containing `profile` are instrumented diagnostics. Their timing is
  excluded from the table; per-rank operator summaries are retained only to
  explain the experiment. [Profiler source](p0/profile_extension_benchmark.py.txt)
  is archived as text.
- [p0 provenance](p0/source-provenance.json) and
  [p1 provenance](p1/source-provenance.json) verify all recorded source,
  driver and recipe hashes. The text patches reconstruct all 166 `specforge`
  Python modules from core commit
  [`36eb793`](https://github.com/yushengsu-thu/SpecForge/commit/36eb793a4fba7fabb441821e9cb7b3f8279bca97).
  Both reconstructions were applied in temporary directories and every Python
  module hash was checked. Changed modules and all modules hashed by the
  measured driver also have byte-exact `.py.txt` copies.
- Runtime: PyTorch 2.14.0, CUDA 13.0, TorchTitan 0.3.0, Transformers 5.12.1,
  Triton 3.8.0. The pinned TorchTitan source is
  [`086bf6c`](https://github.com/pytorch/torchtitan/commit/086bf6c166ec85c1298eb5596fa9bf95f6a2d840).

## FP8 numerical gate: failed, implementation deferred

The experimental converter and public options were removed from the production
extension. The source in this directory is archival text, not an installed
feature or instructions to enable FP8 in the current backend.

The [original matrix](fp8-deferred/original-matrix/) tested DP2 and TP2 with
TorchAO 0.18.0 `rowwise` and `rowwise_with_gw_hp`, all three algorithms and
three optimizer steps. All four processes wrote reports and exited nonzero
because the numerical gate failed. The later
[final DCP matrix](fp8-deferred/final-dcp-matrix/) repeated DP2 rowwise and TP2
rowwise-with-high-precision-weight-gradients with the final source, including
the compiled feature projection in both BF16 and FP8. Each final process ran
all three algorithms for three steps and again exited nonzero at the unchanged
gate.

The tiny fixture uses hidden size 256, intermediate size 512, two decoder
layers and vocabulary 512; it converts 15 draft linear modules. Fused head and
convolution flags remain enabled; the compiled native DFlash2 decoder uses the
ATen convolution decomposition in both FP8 and its BF16 reference. The BF16
reference uses the same distributed layout, initial weights, input and compile
coverage. Gradients for every trainable parameter and actual Adam updates are
compared, not only forward loss.

The [final summary](fp8-deferred/summary.json) records:

- 18 FP8 optimizer steps completed. Six first steps failed the gate; later
  steps passing do not make the full experiment pass.
- Maximum loss relative error 0.0954%; gradient relative L2 differences
  1.18–7.96%.
- First Adam update relative L2 differences 28.93–36.09%, exceeding the
  strict 25% threshold; update cosine also had to exceed 0.97.
- Final full-parameter relative L2 error after three steps 0.0843–0.1163%.
  This divides by the entire parameter vector and is not an alternative
  measure of optimizer-update agreement.
- Six native model/optimizer DCP save–zero–load checks were bitwise equal.
  Checkpoint roundtrip success does not establish BF16/FP8 equivalence.

The complete gate also requires loss relative error below 1%, gradient
relative L2 below 10% and gradient cosine above 0.99. It was not loosened after
observing failures. Per-parameter gradients, update/parameter errors and
near-zero gradient sign statistics remain in the raw reports.
`bf16_loss`/`fp8_loss` are rank-zero local scalars; `loss_relative_error` is the
maximum over ranks, so it can be nonzero when those two local scalars match.

[Final source](fp8-deferred/final-source/) hashes exactly match every source
hash embedded in the final DCP reports. The original matrix did not embed
execution-time source hashes; [post-original source](fp8-deferred/post-original-source/)
is explicitly the later preservation snapshot and is not claimed to be an
exact reconstruction of that earlier execution. The final DCP matrix is the
source-pinned basis for the numerical ranges above.

## Archive boundaries

[The manifest](archive-manifest.json) records SHA256 for each archived file.
Raw JSON is copied byte-for-byte, including historical commands and scratch
paths. Logs retain only selected completion/failure lines and their original
full-log hashes; machine names, IPs and work paths are redacted if present.
Model tensors, checkpoints, credentials and full machine-environment dumps are
not included. No code in this directory is imported by the training backend.
