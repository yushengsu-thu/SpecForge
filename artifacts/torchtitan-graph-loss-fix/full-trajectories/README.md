# Full-size correctness probes

These are diagnosis runs, not performance trials. Some ran concurrently on different GPU pairs; timings in the raw JSON must not be pooled with the final serial matrix.

The controlled DFlash2 fixture is DP2, five-layer Qwen3-4B geometry, 632.4M trainable parameters, BF16, 4K context, 512 anchors, batch 1/rank, accumulation 2, 10 warmup +20 steps. `comparison_contract` fingerprints match. `comparison.json` retains the unchanged strict numerical gate and its failures.

- `graph-full-preserve`: historical p2 source with compiler cast preservation enabled globally by `probe_graph_precision.py`.
- `*-fixed-strict`: candidate1 adds the DFlash2 fused-convolution compile-route correction; the wrapper enables both cast preservation and division rounding.
- `native-eager-strict`, `graph-regional-strict`: same candidate1 and flags, different compilation scopes.
- `graph-cache-*`: candidate2 adds the shared SimpleFSDP parameter cache and production precision scope; launched directly through the frozen p2 driver, without precision overrides.

The full Graph/native maximum absolute loss difference is 0.0368411541 over 30 optimizer windows (maximum relative difference 0.03930888). This still fails atol 1e-5 + rtol 1e-4; it is not evidence of matched convergence. The historical pre-fix pair had a maximum 0.51697063 difference, but both runtimes' numerical policy changed, so this before/after comparison is not a single-variable causal experiment.

Source hashes and exact runtime flags are embedded in each raw JSON. Complete original shell launch manifests were not retained for these early probes; do not represent reconstructed commands or CPU-thread settings as original launch evidence. The final p4 performance matrix separately retains complete commands and environment.
