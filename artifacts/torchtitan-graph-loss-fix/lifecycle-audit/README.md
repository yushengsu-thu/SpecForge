# CPU audit of full GraphTrainer lifecycle differences

This read-only audit compares the completed candidate-3 DFlash2 checkpoints
under `/scratch/specforge-torchtitan-20261003/graph-loss-lifecycle`.
It extracts separate CPU copies under `graph-loss-lifecycle-audit` and does not
modify the source checkpoints. All comparisons include model, Adam state,
trainer RNG, data cursor, and remaining serialized values.

The source run uses DP2, batch 2, accumulation 2, objective chunk size 2,
BF16 compute with FP32 parameters/reductions, full GraphTrainer Inductor,
and three optimizer updates. The detailed configuration and launch records
remain with the source lifecycle experiment; this is an audit of that experiment,
not a new run.

| Comparison against uninterrupted full run | First divergent checkpoint | First differing tensors |
|---|---:|---|
| Cut after step 1 | None through step 1 | All 152 tensors bitwise equal |
| Periodic evaluation | Step 2 | Successor codebook and its two Adam moments |
| Resume from step 1 | Step 2 | Successor codebook and its two Adam moments |
| Independent uninterrupted repeat | Step 1 | Successor codebook and its two Adam moments |

The evaluation run is also bitwise equal at step 1. For both evaluation and
resume, step 2 differs at only four successor-codebook elements: maximum weight
difference 3.230525180697441e-6 and relative L2 error 0.0003685871434283668.

The independent repeat already differs at seven successor-codebook elements
after step 1: maximum weight difference 9.011186193674803e-6 and relative L2
error 0.00197734097832207. Thus checkpoint loading or evaluation is not necessary
to trigger the observed divergence.

By step 3, all three comparisons differ in the same three selector parameter
tensors and their six Adam moment tensors. All other 143 tensors and all
non-tensor values remain bitwise/equal, including RNG and data cursor states.
Maximum successor-codebook weight differences are 5.241017788648605e-6 for
resume/evaluation, versus 1.8158694729208946e-5 for the independent repeat.

These controls support intrinsic compiled selector-gradient nondeterminism,
rather than proving a resume/cache/evaluation defect. The isolated
indexed-gradient experiment in `selector-index/` establishes a concrete mechanism.

A subsequent four-phase full-Graph diagnostic enabled native TorchTitan
`DebugConfig.deterministic` under
`/scratch/specforge-torchtitan-20261003/graph-loss-lifecycle-full-deterministic`.
Its retained `deterministic-diagnostic-verification.json` reports bitwise equality
for **all 152 DCP tensors and 36 HF export tensors** in both uninterrupted versus
resumed and uninterrupted versus periodic-evaluation comparisons. This supports
deterministic indexed reduction as the explanation and cure in this tiny
lifecycle fixture. **The production default was not changed**, so these exact
results do not apply to its normal full-Graph configuration. This diagnostic
also does not establish native-versus-Graph loss agreement or model quality.

`comparison.json` contains every differing tensor's shape, changed-element count,
maximum absolute difference, relative L2 error, and same-sign FP32 ULP difference.
Large ULP counts on tiny Adam moments should be read alongside absolute and
relative errors, not as a standalone severity measure. The audit script, raw
log, source metadata hashes, extracted CPU-checkpoint hashes, runtime versions,
and installed lowering-source hash are retained for reproducibility.
