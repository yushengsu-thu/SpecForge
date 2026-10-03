# Superseded buffer-mismatched H200 timing matrix (p4)

**Superseded:** a post-run audit found that FSDP draft initialization rounded
nonpersistent RoPE `inv_freq` buffers through BF16, while native Titan retained
FP32 values. The prior persistent-state hashes did not cover these buffers.
The same-model FSDP/native comparison is invalid; retain these raw timings only
as diagnostic history. A new complete matrix must preserve/hash the buffers.
The native/Graph pair used the same buffer values, so this does not remove its
failed Graph loss gate. No raw payload or historical analyzer output is changed.

All 36 fresh DP2 processes completed successfully on the same H200 pair (GPUs
4,5), with no concurrent GPU jobs during the measurement window. The matrix
contains DFlash v1, DFlash2 and DSpark; original FSDP1 on Torch 2.13 and 2.14;
native TorchTitan Trainer with compile/CUDA graphs on 2.14; and full GraphTrainer
on 2.14. Each case has three repetitions, 10 warmup and 20 measured windows.
The aggregate is the median of three **per-trial means**. Historical matrices
used other source snapshots/statistics and must not be pooled with this one.

The matrix process returned exit 1 after all child processes exited 0: all nine
native-versus-Graph loss pairs failed the predeclared strict gate
`abs(error) <= 1e-5 + 1e-4 * abs(native_loss)` over all 30 windows. Graph timings
remain separate diagnostics. FSDP/native precision policies differ and their
loss trajectories are recorded separately, not accepted through a loosened gate.
Model fingerprints cover persistent state_dict entries, not nonpersistent buffers.
Recorded precision differences are not proven to explain all FSDP/native divergence.
These are implementation timings, not matched-quality or convergence evidence.

| Algorithm | FSDP Torch 2.13 ms | FSDP Torch 2.14 ms | Native Titan CUDA ms | Graph full diagnostic ms |
| --- | ---: | ---: | ---: | ---: |
| DFlash v1 | 270.530 | 271.654 | 224.426 | 220.909 |
| DFlash2 | 252.552 | 253.112 | 204.400 | 209.701 |
| DSpark | 458.284 | 459.375 | 412.586 | 296.756 |

## Source and evidence

- Core source: `5a35ff35b0ba939242f294907892a2d8955a52b9`. All 168 Python source
  hashes matched the final committed local source before launching. Frozen
  remote source was `extension-bench-source-p4`; it was not modified mid-run.
- Benchmark code commit: `56130f347a9d2c0b67a7f9330f3359b22e34fcf5`.
  `measured-driver/scripts/` contains the exact three executed files. The final
  driver was Black-formatted afterward (identical Python AST), recipe helper
  bytes are unchanged, and the matrix added architecture/recipe identity checks.
  `provenance/extension-bench-p4-driver-provenance.json` records both sets of hashes.
- `raw/matrix-plan.json` contains all 36 exact commands, frozen source/driver
  hashes and predeclared tolerances. `raw/launch-manifest.json` contains child
  exit codes/timestamps; `raw/orchestrator.log` records orchestration.
- `raw/*.json` and `raw/*.log` retain each unmodified trial payload and log.
  Payloads include every warmup/steady latency/loss, runtime versions/precision,
  persistent-model-state/teacher/feature/anchor fingerprints and complete resolved config.
- `raw/comparison.json` is the frozen executed analyzer's output.
  `post-analysis/reanalyze.py` reruns the final identity-hardened analyzer directly
  over the raw payloads. All identity checks passed and its **entire output is
  exactly equal** to the frozen aggregate (`post-analysis/audit.json`).
- No checkpoint weights, feature payloads, environment dumps or credentials are
  included. No FP8 was used. Synthetic cached features exclude teacher capture,
  feature transport, checkpoint/export I/O, evaluation and serving from timings.

## Reproduction

From the benchmark checkout, with isolated environments and frozen source:

```bash
python scripts/training_backend_matrix.py \
  --specforge-root /path/to/frozen-core \
  --driver-root /path/to/frozen-benchmark \
  --python /path/to/torch214/bin/python \
  --python213 /path/to/torch213/bin/python \
  --gpus 4,5 --output /path/to/new-results --execute
```

The exact original commands are in the plan; do not reuse a populated output
folder. CPU-only archived post-analysis is `python post-analysis/reanalyze.py`.
It checks identities and reproduces the failed Graph gate; it does not turn the
matrix into a correctness pass.
