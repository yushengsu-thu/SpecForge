# Native TorchTitan validation

Core implementation: `36eb793a4fba7fabb441821e9cb7b3f8279bca97`.
TorchTitan v0.3.0: `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`.
This archive contains validation artifacts rather than production source changes.

## Numerical checks

`numerical.json` extracts the result arrays from the corresponding raw logs.
Each test compares losses and reconstructed gradients against an unpartitioned
model. The later FP32/default-kernel tests additionally compare native clipping,
fused AdamW updates, frozen weights and optimizer checkpoint parameter names.

- DP2×TP2 and TP2×CP2 FP32, all three algorithms: maximum gradient absolute
  error `2.98e-8`, maximum AdamW update relative L2 error `2.07e-5`.
- TP2 and TP2×CP2 with full activation checkpointing, BF16 and default fused
  kernels pass. BF16 tiny-fixture AdamW first-step update error is about 5–11%
  relative L2; near-zero gradients and BF16 rounding affect the first update.
  This is not bitwise equivalence across parallel layouts.
- Earlier CP checks cover an empty rank and D-PACE with LK-lambda. Those older
  logs record requested loss flags for DSpark too; DSpark still uses its own
  KL/L1/confidence objective, not D-PACE. PP with LK-lambda remains unsupported.

Reproduction examples from the core checkout, in the documented Titan environment:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m tests.test_training.test_torchtitan_parallel --tp 2 --fp32
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m tests.test_training.test_torchtitan_parallel --tp 2 --cp 2 --fp32
SPECFORGE_DFLASH_FUSED_HEAD=1 SPECFORGE_DFLASH2_FUSED_CONV=1 \
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m tests.test_training.test_torchtitan_parallel --tp 2 --cp 2 --ac
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m tests.test_training.test_torchtitan_parallel --tp 1 --cp 2 --anchors 1
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m tests.test_training.test_torchtitan_parallel --tp 1 --cp 2 \
  --loss-type dpace --lk-loss-type lambda
```

The executable defaults to transparent kernels unless the two environment
variables above explicitly enable fused kernels. Missing optimizer fields in
older CP logs mean that check had not yet been added to the test at that time.

## Public CLI and checkpoints

The tiny CLI fixture uses 2 decoder layers, hidden size 32, vocabulary 64,
sequence length 32, block size 4 and real local teacher embedding/head files.
Its randomly initialized weights/features test integration rather than quality.

- `cli/dp2.json`: native DP2 uninterrupted versus resumed DCP and draft export.
- `cli/tp-default-kernels.json`: native TP2, 3 uninterrupted steps versus
  1 step plus 2 resumed steps; all 152 DCP tensors, 560 non-tensor values and
  36 exported draft weights match bitwise. Three compiled steps also pass;
  one `requires_grad` warmup recompile per rank is recorded.
- `cli/cp-default-kernels.json`: the same complete-state resume comparison for CP2.
- `cli/pp2.json`: native PP2 GPipe and 1F1B run and export. The 1F1B interrupted
  continuation matches all DCP state and exported tensors bitwise. Cross-layout
  BF16 weights differ slightly from non-PP, as the file records.
- `cli/pptp-default-kernels.json`: native PP2×TP2 runs 1F1B and 2 optimizer steps
  for DFlash, DFlash2 and DSpark with default kernels, DCP and HF export.

DP2 and plain PP2 resume checks used transparent tiny-fixture kernels. Default
kernel coverage is provided by TP2, CP2 and PP2×TP2. Compilation may select the
kernel's native eager decomposition. Example PP recipes retain devbox-local
fixture paths; they must be replaced with actual offline feature/model paths.

## CPU regression and boundaries

The archived 23-file CPU run has 248 passed, 17 skipped and 162 subtests. A
separate disjoint Domino shared-base run adds 12 passed, 5 skipped and 4 subtests.
The retained native tests include sequential PP loss/all-parameter gradient
equivalence for all three algorithms, including context gradients to the first
projector. An existing legacy DDP test leaks rendezvous environment variables;
running it last avoids interference with the CLI test. All changed files pass
pre-commit. GitHub CI had no reported checks when the PR was updated.

Performance evidence belongs to the separate benchmark PR. These checks do not
establish convergence, serving acceptance, online queue recovery, multi-node
behavior, or unsupported TorchTitan modes such as GraphTrainer/spmd_types/FP8.
