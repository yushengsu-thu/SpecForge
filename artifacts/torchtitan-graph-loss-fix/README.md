# TorchTitan GraphTrainer numerical investigation

This archive separates confirmed adapter defects, numerical differences caused
by compilation, lifecycle reproducibility, and timing experiments. All models,
teacher tables and features used here are synthetic fixtures. No serving or
real-data convergence result is implied.

The final correction is SpecForge core commit
`5a35ff35b0ba939242f294907892a2d8955a52b9` (PR #920), on top of
`c70a5f413220e6ce68247afb528664986502300f`. PyTorch is 2.14.0+cu130,
TorchTitan 0.3.0, Transformers 5.12.1 and Triton 3.8.0 on H200.
The convolution route also passed its regression suite on PyTorch 2.13.0.

## Confirmed corrections

| Problem | Correction | Independent evidence |
| --- | --- | --- |
| DFlash2 disabled its fused convolution only under Dynamo, while joint FX tracing retained it. BF16 intermediates therefore had different rounding. | Keep the supported fused Triton route under both tracers; retain device/dtype/group-size fallback checks. | `convolution/`: compiled kernel presence plus eager/compiled forward and all gradients; both PyTorch versions pass. |
| Full compilation could remove BF16 intermediate rounding, and enabling preservation after `make_fx` was too late to attach barriers. | Scope `emulate_precision_casts` and division rounding around training and evaluation, including first trace and backward; restore prior settings on exit. | `focused-tests/`, `convolution/`: joint-trace barrier regression and actual full-Inductor BF16 convolution forward/backward exactness. |
| SimpleFSDP returned a fresh BF16 parameter cast on each read. Its backward converted each contribution to FP32 before addition, unlike FSDP2's shared unsharded BF16 parameter. | Cache materializations for one joint call, including objective checkpoint recomputation; preserve state keys and clear the cache on all exits. | `first-updates/`: the K/V and selector gradient mismatches disappear. Actual SimpleFSDP tracing, DP2 reduction, replay after updates, DCP and nested scopes are tested. |

Compiled checkpoints now record the precision policy, graph materialization
policy and regional/full compiler mode. Missing or changed identities fail
before loading trainer state. Unchanged pure-eager native contracts remain
compatible.

## What remains different

The corrected tiny BF16 native block-compiled versus full-graph comparison has
34/36 first-step parameter gradients bitwise equal. The remaining differences
are in the shared teacher-feature projector (`fc`, `hidden_norm`). They occur
before clipping and exist with raw joint tracing and eager attention, without
graph optimization passes. They are not explained by dropped gradients or
checkpoint corruption.

The whole-model FP32 native eager/raw-joint control has identical first two
losses, first-step gradient relative L2 error `4.910e-8`, and first Adam update
relative L2 error `3.276e-7`. This supports BF16 rounding amplification, while
not proving real-data convergence. See `first-updates/` for all parameters,
initial-value checks, before-clip gradients and optimizer-state comparisons.

Two isolated experiments establish mechanisms without pretending they are a
complete attribution of a 30-step trajectory:

- `forwardops/`: FP32 mean-reduction rounding alone explains the 60 differing
  BF16 elements in a 20,971,520-element H2560 RMSNorm probe. Eager and compiled
  results have comparable accuracy against FP64; feeding compiled variance
  into eager postprocessing reproduces the compiled output exactly.
- `fanout/` and `forwardops/`: different backward compilation boundaries group
  BF16 shared-activation derivatives differently, even when every forward
  output matches. Forcing one grouping merely to match a tolerance can make
  the derivative less accurate against FP64.

The corrected full-size DFlash2 probe still fails the predeclared
`atol=1e-5, rtol=1e-4` native-versus-Graph loss gate. Maximum absolute difference
is `0.0368411541` over 30 windows. The earlier pair's maximum was `0.51697063`;
both paths' arithmetic was corrected, so this is a descriptive before/after
comparison, not a single-variable estimate. **Do not call this a passed
matched-quality gate.** `full-trajectories/` contains raw losses and the failed
gate, and its concurrently run probe timings are excluded from performance
claims.

## Lifecycle and repeatability

`lifecycle/` runs the public training CLI on a tiny DFlash2 DP2 fixture with
gradient accumulation, objective chunks and a nonzero selector ramp:

| Configuration | 3 steps vs 1 + resume to 3 | 3 steps vs evaluation after every step |
| --- | --- | --- |
| Corrected regional GraphTrainer | 152 DCP tensors + 36 exported weights bitwise equal | Same exact result |
| Corrected default full GraphTrainer | Selector weights/moments differ | Selector weights/moments differ |
| Full GraphTrainer with native deterministic debug mode, diagnostic override | 152 DCP tensors + 36 exported weights bitwise equal | Same exact result |

`lifecycle-audit/` shows that an independent uninterrupted full run already
differs at its first step, before any resume or evaluation. Non-selector state,
RNG and data cursors remain identical. An isolated compiled repeated-index
backward produces different gradients on repeated calls with unchanged inputs;
native deterministic mode gives one repeated gradient, identical to eager.
This localizes the lifecycle variation to indexed-gradient atomics rather than
lost checkpoint state. **Production determinism defaults were not changed.**

## Validation and provenance

- `focused-tests/`: 90 selected tests and 306 subtests pass across the combined
  suite and a separately isolated process-group cache suite. The final follow-up
  resume-mode identity change passes all 11 frontend tests and 10 subtests.
- Candidate1: convolution correction plus process-wide precision probe flags.
- Candidate2: production precision scope and shared parameter cache.
- Candidate3: precision/materialization resume identities; arithmetic unchanged.
- Candidate4 / final core commit: add regional/full resume identity; arithmetic
  unchanged. The final timing matrix checks all 168 Python source hashes against
  this committed source before launch.
- Per-directory manifests pin scripts, reports and source provenance. The root
  manifest covers every payload in this archive except itself. Early full-size
  probes did not retain complete shell launch manifests; their README states
  that limitation. First-update and final matrix runs retain commands.

The separately archived final performance matrix includes the official
PyTorch 2.13 FSDP baseline, a PyTorch 2.14 FSDP control, compiled native TorchTitan
with CUDA graphs, and full GraphTrainer. Failed Graph loss gates keep its
timings in a diagnostic section; they do not suppress unrelated FSDP/native
measurements or turn them into matched-convergence claims.
