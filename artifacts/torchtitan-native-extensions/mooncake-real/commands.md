# Real Mooncake TCP validation

This final check used core `c70a5f413220e6ce68247afb528664986502300f`.
`source-hashes.json` records 175 SpecForge and fixture files, checked against the
remote checkout immediately before training. `verification.json` records the
asserted results; `verify_result.py` retains the CPU verification logic.

The native environment used Python 3.12, Torch 2.14.0+cu130, TorchTitan 0.3.0,
and Mooncake CUDA 13 build `0.3.13.dev0+g4dbe5a4c`. The already-installed Mooncake
package, packaged shared libraries and metadata were copied from `/opt/sglang`
into the isolated native environment; Torch and other dependencies were not
changed. Only the distributed store API was exercised, not Mooncake's
Torch-version-specific EP/PG integrations.

A test-owned localhost master exposed its RPC and HTTP metadata services on
unused ports. A CPU-only producer used the fixture's `--prepare --transport
mooncake` mode to publish 12 synthetic, variable-length samples and hold a 64 MiB
host segment until the scoped `stop` file appeared. The local endpoint JSON
configured TCP, pageable receive buffers and `MC_STORE_MEMCPY=0` for consumers.
That endpoint file is deliberately not archived; no environment dump, credentials,
feature payloads, ledger database or checkpoint weights are included.

The consumer ran the shipping `_OnlineSourceFactory` and MooncakeFeatureStore,
with no constructor substitution. These commands ran from
`/scratch/specforge-torchtitan-20261003/SpecForge-titan-extensions`:

```bash
CUDA_VISIBLE_DEVICES=6,7 PYTHONPATH=. \
  /scratch/specforge-torchtitan-20261003/venv/bin/torchrun \
  --standalone --nproc_per_node=2 \
  tests/test_training/test_torchtitan_online_parallel.py \
  --root /scratch/specforge-torchtitan-20261003/mooncake-native-real/stream \
  --fixture /scratch/specforge-torchtitan-20261003/cli-smoke213 \
  --transport mooncake \
  --mooncake-env /scratch/specforge-torchtitan-20261003/mooncake-native-real/environment.json
```

The command intentionally omits `--steps`: the producer schedule ends at three
steps while the recipe's LR horizon remains 100. The tiny DFlash2 recipe uses
eager attention and shipping fused-head/convolution defaults; neither fused
kernel environment switch was overridden. Both ranks use native FSDP2 and the
native optimizer, scheduler, checkpoint manager and export path. `train.log`
records three optimizer steps and 12 acknowledged samples.

The same command then ran again with this additional argument:

```bash
--resume /scratch/specforge-torchtitan-20261003/mooncake-native-real/stream/output/native-online/checkpoint/step-3
```

`completed-resume.log` records zero dispatched refs, 12 skipped ACKed refs and
successful completion at step three. The CPU verifier compares the ledger,
checkpoint-file hashes and exported weights against `before-noop-resume.json`,
checks the restored data and model steps, confirms natural/LR horizons of 3/100,
and queries a fresh real Mooncake client to confirm all 48 tensor objects are
absent. It also requires consumer completion to be published again.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
  /scratch/specforge-torchtitan-20261003/venv/bin/python \
  /scratch/specforge-torchtitan-20261003/mooncake-native-real/verify_result.py
```

After verification, the producer stopped through its scoped stop file and the
owned master process group received SIGTERM. `cleanup.json` records that neither
owned service remained running and GPUs 6/7 had no compute processes. Log files
in this directory have ANSI color escape sequences removed.

This proves the synthetic-feature consumer and same-host TCP transport path.
It does not establish SGLang capture correctness, RDMA/multi-node behavior,
production-model convergence, or performance.
