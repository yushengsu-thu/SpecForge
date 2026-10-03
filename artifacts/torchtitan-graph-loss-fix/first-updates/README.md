# First-update numerical isolation

These are correctness probes, **not performance measurements**. Optimizer hooks
copy full distributed tensors to CPU before clipping and around Adam; their
elapsed times must not be used in a speed comparison.

All successful probes used two H200 GPUs (IDs 2,3; the later full-Graph FP32
follow-up used IDs 0,1), Torch 2.14.0, TorchTitan 0.3.0,
the real DFlash2 architecture, two layers, hidden size 64/head dimension 16,
128 input positions, eight anchors, objective chunks of two, and two accumulation
microbatches per optimizer window. The head-dimension override is necessary
because compiled FlexAttention rejects the standard tiny recipe's dimension 8.
Initial model/teacher/features/anchor hashes, resolved configuration, exact
package versions, compiler overrides and source hashes are in `runs/*.json`.

`commands.json` reconstructs the executed argument lists and records only the
explicit relevant environment variables. The production Trainer's forward,
backward, clipping and optimizer are delegated unchanged. The scratch collector
adds native optimizer pre/post hooks and an optional wrapper around the native
clipper. Optional raw-graph and FP32 ablations are recorded in each result.

## Results

| Comparison, first optimizer window | Max microbatch loss difference | Full gradient relative L2 error | Main finding |
| --- | ---: | ---: | --- |
| candidate1 native block-compiled vs full Graph | 0 | 0.00200072 | Seven parameter gradients differ, including repeated K/V and selector reads. |
| candidate2 native block-compiled vs full Graph | 0 | 0.00159431 | Cache removes K/V/selector differences; only context projector `fc.weight` and `hidden_norm.weight` remain. |
| candidate2 native eager vs regional Graph | 0 | 0.00159431 | Residual is also present without native block compilation. |
| native eager vs minimal regional Graph | 0 | 0.00159431 | Removing rematerialization, bucketing, cleanup and CUDA capture does not remove the residual. |
| eager attention, native eager vs raw joint Graph, BF16 | 0 | 0.00160068 | Residual exists without FlexAttention compilation or optional graph passes. |
| eager attention, native eager vs raw joint Graph, FP32 | 0 | 4.90995e-8 | Gradient max absolute error is 2.79397e-9; post-Adam parameter max error is 3.72529e-9. |
| eager attention, native eager vs full Inductor Graph, FP32 | 0 | 2.94897e-7 | Gradient max absolute error is 1.11759e-8; post-Adam parameter max error is 4.85219e-7. |

The raw-Graph FP32 pair also has bitwise-equal losses in the second window.
The full-Inductor FP32 follow-up uses the same candidate2 source, inputs and
original native control; second-window loss differs by 2.38419e-7 and gradients
by relative L2 3.15558e-7. Its exact executed command and environment are in
`provenance/first-update-fp32-c2-full-command.json`. Full compiler updates are
close in FP32, but not bitwise identical. These results
support BF16 accumulation amplifying the remaining context-gradient difference;
they do not establish bitwise equivalence in BF16 or validate a 30-window loss
trajectory. Candidate2's longer Graph trajectory remains subject to the separate
strict gate; that gate must not be weakened based on these probes.

`reports/*.json` contains complete per-parameter statistics, before/after Adam
parameter and optimizer-state comparisons, and unclipped gradients where
available. Reports combining Adam-state fields have mixed units (`step`, first
and second moments); use their per-tensor statistics, not the aggregate relative
norm, to assess optimizer-state differences.

The fully raw FlexAttention graph is deliberately retained as a negative
ablation: disabling its regional compilation changes attention execution and
introduces a first-forward difference. It cannot independently attribute a
backward discrepancy to graph optimization. The `minimalregional` probe keeps
only functionalization plus FlexAttention annotation/regional compilation.

## Provenance and boundaries

- Candidate1 is the frozen convolution-fix snapshot; candidate2 additionally has
  scoped compiler numerics and per-call SimpleFSDP parameter caching.
  `provenance/source_hashes.json` contains all 166/168 Python source hashes.
- The archived benchmark driver and recipe helper are the exact frozen files
  used for every probe; their hashes are checked against all thirteen run results.
- `scripts/collect_titan_first_updates.py` is the final reusable collector with
  additive ablation options. Early results predate some collector options and
  retain their original collector hash. Exact per-run copies are also retained
  under `provenance/collectors/` for later probes that saved them.
- Initial dimension-8 compilation failure is retained in the logs; it is not a
  successful numerical test.
- The original full tensor snapshots totaled approximately 83 MiB on the devbox;
  the full-Graph FP32 follow-up adds another snapshot directory. All tensor
  snapshots are omitted here. Reports preserve every parameter's numerical comparison; no checkpoint
  or feature payload is included in this archive.
- Synthetic features and a tiny model test implementation numerics; this is not
  convergence, draft quality, or serving evidence.
