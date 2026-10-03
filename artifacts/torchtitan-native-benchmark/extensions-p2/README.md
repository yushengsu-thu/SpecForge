# Compiled native, CUDA Graph and GraphTrainer: frozen H200 matrix

This matrix measures DFlash, DFlash2 and DSpark using three BF16 native
TorchTitan configurations. All runs use the immutable `extension-bench-source-p2`
and `extension-bench-driver-p2` snapshots, with GPUs 4 and 5 reserved for this
serial matrix. No other task-owned GPU validation ran during it.

The 27 trials cover three algorithms, three configurations and three fresh
processes per combination. Each process runs 10 warmup and 20 steady optimizer
windows. Case order is forward, reversed, then forward. See
[raw/launch-manifest.json](raw/launch-manifest.json) for exact launch arguments,
selected environment variables, timestamps, durations and exit codes.

All **27 trials completed successfully** and passed comparison-contract and
source/runtime-hash checks. [Aggregate JSON](aggregate.json) retains the full
summary; the raw reports retain every observation.

| Algorithm | Configuration | Steady ms/window (range) | Change vs compiled native |
| --- | --- | ---: | ---: |
| dflash | bf16 | 230.62 (230.00–230.69) | +0.00% |
| dflash | cuda | 225.20 (224.59–225.67) | -2.35% |
| dflash | graph-full | 222.66 (222.37–222.83) | -3.45% |
| dflash2 | bf16 | 213.69 (213.68–214.27) | +0.00% |
| dflash2 | cuda | 208.52 (207.87–208.71) | -2.42% |
| dflash2 | graph-full | 212.87 (212.67–213.04) | -0.38% |
| dspark | bf16 | 419.72 (419.17–419.86) | +0.00% |
| dspark | cuda | 413.14 (413.10–414.77) | -1.57% |
| dspark | graph-full | 306.63 (306.06–306.86) | -26.94% |

| Algorithm | Configuration | Allocated GiB | Reserved GiB | Build s | First window s (range) |
| --- | --- | ---: | ---: | ---: | ---: |
| dflash | bf16 | 13.069 | 14.350 | 35.10 | 2.32 (2.28–11.35) |
| dflash | cuda | 6.089 | 14.420 | 35.52 | 2.32 (2.28–2.37) |
| dflash | graph-full | 7.102 | 16.877 | 35.77 | 6.84 (6.74–37.55) |
| dflash2 | bf16 | 16.235 | 17.783 | 36.68 | 2.67 (2.65–2.69) |
| dflash2 | cuda | 6.806 | 17.748 | 36.66 | 2.80 (2.78–2.80) |
| dflash2 | graph-full | 7.996 | 18.457 | 37.18 | 12.77 (12.69–12.83) |
| dspark | bf16 | 19.199 | 20.721 | 36.94 | 2.44 (2.44–2.44) |
| dspark | cuda | 6.673 | 21.096 | 37.04 | 2.63 (2.60–2.71) |
| dspark | graph-full | 7.832 | 24.721 | 40.24 | 36.06 (35.84–44.01) |


## Workload and boundaries

Two H200s, PyTorch 2.14.0+cu130, TorchTitan 0.3.0, Transformers 5.12.1 and
Triton 3.8.0. Controlled five-layer Qwen3-4B geometry, DP2, 4K context,
512 anchors, block size 16, objective chunks of 128, batch 1/rank and
accumulation 2. Both production fused-kernel options are enabled. Teacher tables and
two synthetic hidden-state batches per rank are resident; IDs/masks use pinned
CPU memory. Timed windows include anchor preparation, small input transfers,
forward/backward, clipping and AdamW. They exclude feature capture/transport,
file reads, evaluation, checkpoint/export and initial construction.

All configurations use BF16 compute, FP32 parameters/reductions/Adam moments
and requested `SHARD_GRAD_OP` (`reshard_after_forward=never`). The ordinary
Trainer uses FSDP2 and compiles decoder blocks plus the feature projector.
GraphTrainer uses SimpleFSDP and its default graph selective-activation memory
policy, full Inductor compilation and graph CUDA capture. Its graph passes
honor the reshard policy but can schedule memory/communication differently.
These compare complete configurations rather than isolate a single pass.

Enabled options do not imply identical executed kernels: compiled native
DFlash2 uses the convolution's existing ATen fallback under
`torch.compiler.is_compiling()`, whereas GraphTrainer's make-fx trace retains
the wrapped fused Triton convolution. Full Inductor also compiles the joint
backward graph. These are concrete execution/numerical differences; their
individual contributions to the observed trajectory drift were not isolated.

Steady latency is the median of three process medians; ranges span those
medians. Warmup and build time are reported separately. Fresh processes share
the node's disk compiler caches, so startup is not a cold-cache claim.
Allocated and reserved GPU memory are both reported: graph pools retain
reservations even when fewer tensors are live during replay. Statistics reset
after warmup and exclude non-PyTorch driver/NCCL allocations.

Native CUDA and compiled-native reported loss trajectories are bitwise equal
for all paired trials. Full GraphTrainer is numerically different. Maximum
absolute difference across all 30 optimizer windows and all three repeats:

| Algorithm | Native CUDA max loss difference | Full GraphTrainer max loss difference |
| --- | ---: | ---: |
| dflash | 0.00000000 | 0.00992632 |
| dflash2 | 0.00000000 | 0.51697111 |
| dspark | 0.00000000 | 0.00504351 |

DFlash2 reaches a difference of approximately 0.517 at window 16
(native 4.423 versus graph 4.940, about 11.7%). This is not negligible
trajectory drift. Its cause was not isolated; kernel and whole-graph compiler
precision differences are concrete candidates, not a demonstrated explanation.
GraphTrainer timing is therefore not a matched-quality speed result.

The synthetic repeated features are not a quality/convergence dataset. These
results do not establish acceptance, serving speed, multi-node scaling or
end-to-end online training performance. They must not be pooled with the
earlier FSDP/native or exploratory p0/p1 experiments, which used other snapshots.

## Exact measured-source reconstruction

[source-provenance.json](source-provenance.json) hashes all 638 files in the
measured runtime snapshot, excluding tooling caches. Applying
[measured-runtime.patch](measured-runtime.patch) to
`36eb793a4fba7fabb441821e9cb7b3f8279bca97` reconstructs that snapshot; this was
checked byte-for-byte across all files. Start from a clean checkout of that
commit and run `git apply /path/to/measured-runtime.patch`.

The exact measured [driver](benchmark_training_backends.py.txt) and
[recipe helper](training_benchmark_recipes.py.txt) are archived as text. Copy
them into `scripts/benchmark_training_backends.py` and
`scripts/training_benchmark_recipes.py` in a separate benchmark checkout.
The final benchmark PR adds three output-only metadata fields after this
snapshot; its training/timing operations are unchanged. Use these archived
files when comparing the exact recorded hashes.

For each algorithm and fresh trial, launch:

```bash
CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=1 \
SPECFORGE_DFLASH_FUSED_HEAD=1 SPECFORGE_DFLASH2_FUSED_CONV=1 \
torchrun --standalone --nproc-per-node=2 scripts/benchmark_training_backends.py \
  --specforge-root /path/to/reconstructed-runtime \
  --backend torchtitan --algorithm dflash2 --compile \
  --warmup-steps 10 --steps 20 --output /path/to/trial.json
```

Add `--cuda-graphs` for ordinary native capture. Add
`--titan-engine graph --graph-inductor full --cuda-graphs` for GraphTrainer.
The archived [orchestrator](run_extension_benchmarks.py.txt) retains the exact
local launch logic and hardcoded devbox paths; the matrix invocation used
`--tag p2 --algorithms dflash dflash2 dspark --cases bf16 cuda graph-full
--repeats 3 --gpus 4,5`. Its historical FP8 case definitions are unused here.

[summarize_extension_benchmarks.py.txt](summarize_extension_benchmarks.py.txt)
validates all 27 raw reports, matching source/runtime/input contracts, finite
values, correct sample counts and successful launch exits, then generates the
aggregate. [artifact-manifest.json](artifact-manifest.json) hashes this archive.
No weights, checkpoints, live feature payloads or credentials are included.
