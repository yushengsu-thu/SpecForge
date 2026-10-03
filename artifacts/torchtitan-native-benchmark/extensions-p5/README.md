# Corrected shared-buffer H200 matrix (p5)

This matrix uses the original FSDP1 training backend on Torch 2.13 and 2.14,
native TorchTitan Trainer with compilation/CUDA graphs on 2.14, and full
GraphTrainer on 2.14. It covers DFlash v1, DFlash2 and DSpark, with three fresh
DP2 processes per case (36 total) on the same H200 pair (GPUs 4,5), serially with
no concurrent test or GPU jobs. Each trial has 10 warmup plus 20 steady windows.
Latency is the median of the three per-trial **means**; memory is the maximum
of the three steady peak allocations.

## Observed result

All **36/36 child processes exited 0**, and all identities passed the final audit.
All nine native/Graph pairs failed the strict trajectory gate. Every recorded
persistent-model/teacher/feature/anchor contract matches p4; the new separate
buffer contract is correct for every rank in every p5 trial. The final frozen
aggregate is reproduced exactly by `analysis/audit.py`.

| Algorithm | FSDP Torch 2.13 ms | FSDP Torch 2.14 ms | Native Titan CUDA ms | Native paired latency reduction vs 2.13 |
| --- | ---: | ---: | ---: | ---: |
| DFlash v1 | 270.66 | 271.52 | 224.94 | 16.78% |
| DFlash2 | 252.24 | 253.07 | 204.63 | 18.86% |
| DSpark | 458.01 | 459.24 | 413.00 | 9.90% |

Graph results are **diagnostic only**, excluded from matched-quality speedup claims:

| Algorithm | Graph full ms | Max absolute loss difference from native | Failed windows |
| --- | ---: | ---: | ---: |
| DFlash v1 | 220.73 | 0.040533 | 78/90 |
| DFlash2 | 209.95 | 0.042302 | 82/90 |
| DSpark | 297.20 | 0.005556 | 81/90 |

All trial means for a given configuration fall within a 0.83% max/min range.
The relay-overlap DFlash2 Graph repetition is within 0.044% of its peers; there
is no material transport-related timing outlier. No trials were excluded or
restarted. `analysis/timing-review.json` records every range. GPU processes
were absent on all eight devices after completion.

## Corrected buffer contract

The superseded p4 comparison hashed persistent state only. FSDP initialization
rounded nonpersistent RoPE frequency buffers through BF16, unlike native Titan.
P5 preserves fresh FP32 `rotary_emb.inv_freq` and `original_inv_freq` values in
both paths, explicitly hashes name/dtype/shape/content, compares both ranks
after backend preparation, and checks them again after training. This is a
benchmark-only normalization of initialization. It exercises the original
FSDP backend with a shared model-state reference; it does **not** represent
untouched production-default initialization. Production source is unchanged.

The six full-geometry preflight launches passed the first-window objective check
at `atol=1e-5, rtol=1e-4`. Absolute FSDP214/native differences were 9.54e-6,
4.29e-5 and 2.77e-5 for DFlash v1, DFlash2 and DSpark. This is the pooled objective
of two accumulation microbatches per rank, not a comparison of every individual
forward loss or of gradients. Preflight timings are not performance results.
Its anchor hashes cover 2 windows; the final matrix covers 30 windows.

## Immutable provenance and replay

- Core: `5a35ff35b0ba939242f294907892a2d8955a52b9`; all 168 Python files matched
  the committed local source before launch. Frozen remote source was
  `extension-bench-source-p5`.
- Driver/helper/matrix: **exact committed bytes** from benchmark code
  `8bf591727ad3b3a2287c777997f8c6da352f28cd`, retained in
  `measured-driver/scripts/`. The three recipe JSON files are retained too.
- `raw/matrix-plan.json` contains all 36 exact commands, source/driver hashes
  and predeclared tolerances.
  `raw/launch-manifest.json` retains every child exit, timestamp and relevant environment override.
- Raw JSON/logs retain all warmup/steady losses and timings, configuration,
  runtime versions/precision, memory and initial-state/input/buffer fingerprints.
  No checkpoint weights, feature payloads, broad environment dumps or credentials
  are included. `SPECFORGE_FLEX_ATTENTION_BACKEND=''` selects the existing default
  and is forced/recorded consistently.
- `preflight/` contains the six original logs/results and their report;
  `provenance/run_buffer_preflight.py` is the exact executed runner.
- `analysis/audit.py` replays the frozen analyzer, verifies all 36 source/driver,
  buffer and algorithm identities, and compares every old persistent-state/input
  contract with p4. Run it as
  `python analysis/audit.py /path/to/extensions-p4/raw`.

The SSH monitoring connection closed during an rx relay rollout after 28 trials.
The original remote orchestrator and children continued without restart.
`provenance/ssh-monitoring-event.json` and the process/start-time snapshot retain
that evidence. The overall orchestrator exit status is unavailable after the
connection loss; child exit codes and the final aggregate are observed directly.

## Numerical interpretation

The native/Graph tolerance remains exactly
`abs(error) <= 1e-5 + 1e-4 * abs(native_loss)` for all 30 windows. Graph timings
that fail this gate remain separate diagnostics; the runner deliberately exits 1
when the gate fails after successful child jobs. No threshold was loosened.

FSDP1 retains BF16 parameter/gradient accumulation/reduction with FP32 masters
and Adam state. Native/Graph use FP32 parameter storage and gradient reduction/
accumulation with BF16 forward computation. First-window agreement does not
establish matching updates or trajectories; the FSDP/native differences are
reported separately without a post-hoc gate. These are implementation timings,
not matched-quality, convergence or serving evidence. No FP8 was used.

Synthetic hidden features are cached on GPU. Windows include CPU anchor
preparation, input/mask transfers, forward/backward and optimizer work, while
construction, teacher capture, transport, checkpoint/export and evaluation are
excluded. Fresh processes share compiler caches; warmup is not guaranteed
cold-cache compilation time.
