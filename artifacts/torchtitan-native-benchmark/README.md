# Native TorchTitan training benchmark evidence

The production engine is core commit `36eb793a4fba7fabb441821e9cb7b3f8279bca97`.
This archive is evidence for the separate benchmark PR, not a second backend.

- `h200-dp2.json`: audited aggregate of 18 fresh-process trials.
- `dp2/`: all raw DP2 baseline/native trials, including complete per-step timing,
  loss, precision, memory, input fingerprints and source hashes.
- `exploratory/`: DFlash2 native DP2 compilation and TP2 measurements plus logs.
- `tiny/`: six same-input FSDP/native smoke results across all three algorithms.
- `source-provenance.json`: exact measured DP2 source hashes, changes from the
  final engine commit and a reconstruction patch verified against every hash.
- `measured-sources/`: exact executed driver/helper bytes stored as `.py.txt`.
  The original `benchmark_native_torchtitan.py.txt` is the native Trainer driver;
  `benchmark_training_backends.py.txt` was imported only for deterministic
  recipe/model/feature/statistics functions. Its older backend runner was never
  called. The `published-*` files are the renamed single driver and extracted
  recipe-only helper used for the later compilation/TP experiments.

The native timed path is `run_titan -> TimedTrainer.train_step ->
SpecForgeTitanTrainer.train_step -> torchtitan.trainer.Trainer.train_step`,
inside the real native `Trainer.train` loop. The baseline path uses the existing
SpecForge `TrainerCore` and FSDP1 backend. Recorded `benchmark_sha256` and
`recipe_helpers_sha256` match the corresponding exact text artifacts.

The DP2 snapshot predates only formatting, TP-only collective waits, diagnostic
logger filtering and frontend URI/timeout fixes. Those changes do not alter
steady forward/backward/optimizer computation in the measured DP2 path. The
later compilation and TP runs use the exact final engine commit.

Features and frozen teacher tables are synthetic. Repeated-cache losses and
runtime tests do not establish real-data convergence or serving acceptance.
