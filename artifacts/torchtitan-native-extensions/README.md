# TorchTitan runtime extension validation

This archive records correctness and integration checks for the optional native
TorchTitan backend extensions. The earlier native Trainer / DP / TP / CP / PP
evidence remains in [../torchtitan-native](../torchtitan-native/README.md).
Performance measurements belong to the separate benchmark archive.

The runs used H200 GPUs, Python 3.12.3, Torch 2.14.0+cu130, and TorchTitan 0.3.0
at `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`. The core checkout started at
`36eb793a4fba7fabb441821e9cb7b3f8279bca97` and had uncommitted extensions.
The final core implementation was subsequently committed as
`c70a5f413220e6ce68247afb528664986502300f`; that commit is distinct from the
historical evolving source and the separately frozen performance snapshot.
[source-manifest.json](source-manifest.json) contains **collection-time** source
hashes. These are not a frozen per-run source identity: the checkout evolved
during validation, particularly the frontend online/CLI/error/horizon wiring.
The final graph/runtime/kernel semantics were unchanged after their final
checks. The performance archive has its own frozen-source provenance.
The manifest also hashes the corresponding files from the final core commit;
all 32 selected source files match the collection-time snapshot.

## Passing checks

| Check | Result | Evidence |
| --- | --- | --- |
| Native Trainer CUDA Graph capture, DFlash2 DP2 / TP2 / CP2 | Capture logged in all layouts; three uninterrupted steps equal one step + resume to step three | [native-cuda/verification.json](native-cuda/verification.json) |
| Native CUDA Graph complete-state resume | All 152 DCP tensors and 36 HF export keys bitwise equal in each layout; non-tensor counts are recorded per layout | Same report and each `*-full`, `*-cut`, `*-resume` log/YAML |
| GraphTrainer regional, DFlash v1 / DFlash2 / DSpark | Actual public CLI, DP2, three optimizer steps, accumulation two, native DCP and HF export | [graph/matrix-status.json](graph/matrix-status.json), `graph-final-*` logs/YAML |
| GraphTrainer regional versus unoptimized graph oracle | All exported weights bitwise equal for all three algorithms; logged losses equal | [graph/graph-oracle-verification.json](graph/graph-oracle-verification.json) |
| GraphTrainer DFlash2 complete-state resume | All 152 DCP tensors, 564 non-tensor values, and 36 HF export keys bitwise equal | [graph/graph-verification.json](graph/graph-verification.json) |
| GraphTrainer full Inductor | DFlash2 DP2, three optimizer steps, DCP and HF export completed | [graph/graph-full-inductor.log](graph/graph-full-inductor.log) |
| Graph-safe fused convolution/head kernels plus existing numerical tests | 25 tests and 311 subtests passed | [graph/fused-tests.log](graph/fused-tests.log) |
| Final graph focused tests, including functionalization/DCE and preserved FQN metadata | Six tests passed | [graph/graph-metadata-tests.log](graph/graph-metadata-tests.log) |
| Final targeted CPU regression suite | 132 passed, four skipped, 177 subtests passed | [cpu/native-extensions-cpu-final.log](cpu/native-extensions-cpu-final.log) |
| Legacy Torch 2.13 CPU regression | 24 passed, one skipped, four subtests passed | [cpu/native-extensions-legacy213-cpu.log](cpu/native-extensions-legacy213-cpu.log) |
| Periodic offline evaluation | Actual DFlash2 DP2 / TP2 / CP2 and GraphTrainer DP2, evaluation after each of two training steps | `evaluation/eval-native-*.log` |
| Evaluation + DP2 resume | All 152 DCP tensors, 563 non-tensor values and 36 HF tensors bitwise equal, no exclusions | [evaluation/eval-native-resume-verification.json](evaluation/eval-native-resume-verification.json) |
| Retained online shared-store boundary | DFlash v1 / DFlash2 / DSpark DP2, three optimizer steps, 12 samples acknowledged; DFlash2 cut/resume | `online-shared-store/online-native-*.log` |
| Real Mooncake TCP DP2, final core | DFlash2 three steps, 12 durable ACKs, 48 remote objects reclaimed, native DCP/HF export; completed-checkpoint resume skips all 12 refs without updating weights | [mooncake-real/verification.json](mooncake-real/verification.json), [cleanup](mooncake-real/cleanup.json) |
| Online shared-store DP2 resume | All 152 DCP tensors, 562 non-tensor values and 36 HF tensors bitwise equal; three literal fixture identity paths excluded | [online-shared-store/online-native-resume-verification.json](online-shared-store/online-native-resume-verification.json) |

GraphTrainer CLI fixtures use the default fused head/convolution paths, with
tiny randomly initialized two-layer models, hidden size 32, vocabulary 64,
sequence length 32, and eager attention. DFlash2 exercises a changing selector
loss schedule across capture/replays. The graph oracle retains the same
SimpleFSDP representation and forward/loss/backward tracing but disables graph
optimization passes. It is not an independent legacy-FSDP implementation.

Full Inductor is **not bitwise equivalent** to regional compilation: the
DFlash2 final-weight relative L2 difference is `0.0008983321604318917`, and
maximum absolute difference is `0.00405782088637352`. These measured BF16
differences are recorded, not converted into a quality-equivalence claim.

The separate full-size p2 benchmark shows a larger DFlash2 trajectory mismatch:
full GraphTrainer differs from compiled native Trainer by up to approximately
`0.517` loss over 30 optimizer windows. At the largest difference the losses
are about `4.940` versus `4.423` (11.7%). This is non-negligible; the tiny graph
oracle checks above do not establish equivalence to native FSDP2 or matched
training quality. Full GraphTrainer remains experimental.

The default fused-kernel switches also do not guarantee identical kernel routes:
the native compiled decoder takes the ATen convolution decomposition because
`DFlashGroupedConv._fused_convolution` disables the custom kernel while
`torch.compiler.is_compiling()`. GraphTrainer's `make_fx` tracing retains the
custom Triton convolution through `wrap_triton` and functionalization. Full
Inductor then compiles the joint forward/backward graph more broadly than the
regional mode. These execution differences are confirmed by source inspection;
their contribution to the observed loss drift has not been isolated. No causal
explanation or quality-equivalence claim follows from this audit.

GraphTrainer uses SimpleFSDP and the native default graph selective activation
checkpointing policy. `activation_checkpoint=none` disables module-level
checkpoint wrappers, not the graph memory pass. That pass consumes the resolved
`fsdp_reshard_after_forward` policy. This differs from ordinary Trainer's eager
FSDP2 execution and memory scheduling even when the requested sharding matches.

Online evidence starts at the published `SampleRef` boundary with synthetic
variable-length features. The shared-directory fixture replaces only the
Mooncake store constructor; distribution, retained refs, ledger, ACKs, native
optimizer/checkpointing and cleanup run normally. A later real Mooncake TCP
check is recorded separately under [mooncake-real/](mooncake-real/commands.md).
It used final core `c70a5f413220e6ce68247afb528664986502300f`; all 175 SpecForge
and online-fixture file hashes matched the committed checkout before launch.
DFlash2 completed three native DP2 steps with the producer's natural horizon of
three and an independent LR horizon of 100. All 12 refs were durably acknowledged
and all 48 tensor objects were physically removed. Resuming the already-complete
step-three checkpoint dispatched zero refs, skipped all 12 ACKed refs, republished
consumer completion, and left the ledger, checkpoint bytes and exported weights
unchanged. This used real same-host TCP put/get with the consumer memcpy shortcut
disabled and pageable host receive; it did not run SGLang capture, RDMA or multiple
nodes. Owned producer/master processes were stopped and GPUs 6/7 released.
The checkpoint comparison excludes only documented fixture
identity paths when the uninterrupted and resumed online stores use different
directories. It does not exclude model, optimizer, scheduler, RNG or cursor state.

Evaluation and online unit/lifecycle checks are also retained in
[cpu/native-online-eval-regression-final.log](cpu/native-online-eval-regression-final.log)
(66 passed) and [cpu/online-resource-regression.log](cpu/online-resource-regression.log)
(16 passed, one skipped). These selections overlap other checks; their counts
must not be added to claim a unique total.

## Historical failures and boundaries

The logs under `development-failures/` are **failed pre-fix development runs**,
not passing validation:

- `graph-dp2-wrap.log`: non-finite first-step loss before functionalizing Triton
  mutation nodes. Dead-code elimination could discard their output writes.
  The final transform makes data dependencies explicit; the focused joint
  forward/backward regression and all-three-algorithm oracle checks pass.
- `graph-dp2-functional.log`: startup failed because `BaseValidator.Config`
  does not accept `enable`. The frontend now uses `Validator.Config` for the
  disabled validator; subsequent public-CLI runs pass.

These are selected useful failure records, not a complete history of every
development attempt. Warning messages about unavailable optional FlashAttention
do not mean those tests exercised its backend.

This archive does not establish model convergence, matched-quality training
speed, speculative serving acceptance, SGLang capture correctness, multi-node
operation, graph-plus-online support, or FP8 support. FP8 is deferred. GraphTrainer
is DP-only; ordinary Trainer CUDA Graphs exclude pipeline parallelism. Large-model
timings, compilation/capture costs, and reserved graph-pool memory must be read
from the separate performance evidence rather than these tiny correctness runs.

Checkpoint tensors, model weights, SQLite stores, raw feature payloads, and
environment/credential dumps are deliberately absent. Recipes keep their
original devbox-local fixture paths; reconstruct the fixture data and replace
those paths before reproducing elsewhere. The scripts in `scripts/` retain
their original launch and comparison logic, including original scratch paths.
See [commands.md](commands.md) for recorded commands and reproduction boundaries.
[artifact-manifest.json](artifact-manifest.json) hashes the archived files;
after appending further evidence, refresh it with
`python scripts/refresh_manifest.py` from this directory.
