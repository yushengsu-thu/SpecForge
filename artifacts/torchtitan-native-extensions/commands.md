# Commands and reproduction scope

Paths below are the original devbox paths. Checkpoint/model/feature payloads
are not included. Use a fresh output directory and reconstruct the tiny fixture
before rerunning; the frontend intentionally rejects accidental reuse of an
existing checkpoint directory without explicit resume.

## Commands retained verbatim in scripts

The archived scripts preserve the actual child command arrays, environment
overrides, generated recipe settings, and checkpoint comparisons:

- `scripts/verify_native_cuda_extensions.py`: native Trainer CUDA Graph DP2,
  TP2 and CP2 public CLI; complete/full/cut/resume comparison. Its source recipe
  is archived as `graph/graph-fused-dp1.yaml` and every generated YAML is under
  `native-cuda/`.
- `scripts/graph_matrix.py`: GraphTrainer regional public CLI, all three
  algorithms, plus DFlash2 cut/resume. Generated recipes are under `graph/`.
- `scripts/graph_oracle_matrix.py` and `scripts/graph_probe.py`: the oracle
  disables GraphTrainer graph optimization passes while preserving tracing and
  SimpleFSDP. This is a fixture-only override, not a public API or shipping path.
- `scripts/verify_graph_resume.py` and `scripts/verify_graph_oracle.py`:
  checkpoint/export comparisons and recorded compile-mode differences.

For example, the native CUDA script executes this command array for each
generated recipe, from the core checkout, with `CUDA_VISIBLE_DEVICES=4,5`,
`OMP_NUM_THREADS=1`, `SPECFORGE_DFLASH_FUSED_HEAD=1`, and
`SPECFORGE_DFLASH2_FUSED_CONV=1`:

```bash
/scratch/specforge-torchtitan-20261003/venv/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m specforge.cli train --config <generated-recipe.yaml>
```

Graph matrix scripts use GPUs `2,3` and the default enabled fused kernels.
The full-Inductor run used the same public CLI shape and
`graph/graph-full-inductor.yaml`; its original top-level shell invocation was
not retained. It is an equivalent reproduction command, not a captured shell
transcript.

## Evaluation commands reported by the executing agent

Each run used `CUDA_VISIBLE_DEVICES=6,7` and the native environment's Python.
The common command was:

```bash
python -m specforge.cli train \
  -c /scratch/specforge-torchtitan-20261003/cli-smoke213/train.yaml \
  output_dir=/scratch/specforge-torchtitan-20261003/eval-native-dp2 \
  run_id=eval-native-dp2 \
  data.eval_hidden_states_path=/scratch/specforge-torchtitan-20261003/cli-smoke213/features \
  training.eval_interval=1
```

TP2 changes the output/run suffix to `tp2` and adds `training.tp_size=2`.
CP2 changes the suffix to `cp2` and adds `training.torchtitan.cp_size=2`.
GraphTrainer changes the suffix to `graph` and adds
`training.torchtitan.engine=graph training.torchtitan.compile=true
training.torchtitan.disable_cuda_graphs=false`.
The resumed DP2 run uses suffix `resumed` and
`training.resume_from=/scratch/specforge-torchtitan-20261003/eval-native-dp2/eval-native-dp2/checkpoint/step-1`.
These were two-step DFlash2 runs. The CLI resolves its configured two-worker
launch; no hand-written independent trainer loop is involved.

## Shared-directory online fixture commands

The executing agent reported the following sequence. The current retained test
driver has since gained producer-horizon and optional real-Mooncake setup;
source hashes in this archive identify its collection-time version, not the
exact earlier shared-directory execution bytes.

```bash
python tests/test_training/test_torchtitan_online_parallel.py \
  --root /scratch/specforge-torchtitan-20261003/online-native-full \
  --fixture /scratch/specforge-torchtitan-20261003/cli-smoke213 --prepare
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  tests/test_training/test_torchtitan_online_parallel.py \
  --root /scratch/specforge-torchtitan-20261003/online-native-full \
  --fixture /scratch/specforge-torchtitan-20261003/cli-smoke213 --steps 3
```

Cut/resume uses root `online-native-resume`, runs `--prepare`, then `--steps 1`,
then another two-worker launch with `--steps 3` and
`--resume /scratch/specforge-torchtitan-20261003/online-native-resume/output/native-online/checkpoint/step-1`.
DFlash v1 and DSpark use separate roots `online-native-dflash` and
`online-native-dspark`, respectively, and `--algorithm dflash|dspark`.
This exercises actual retained feature refs, inbox distribution, SQLite ledger,
optimizer-boundary ACK, native training and DCP; it replaces the Mooncake store
constructor with the retaining shared-directory store. It does not exercise
SGLang capture or real Mooncake transport.

The retained CPU verifier was rerun when collecting the archive, using the
original completed checkpoint outputs (no new GPU training):

```bash
CUDA_VISIBLE_DEVICES="" python verify_native_resume.py \
  --baseline /scratch/specforge-torchtitan-20261003/eval-native-dp2/eval-native-dp2 \
  --resumed /scratch/specforge-torchtitan-20261003/eval-native-resumed/eval-native-resumed \
  --step 2 --output eval-native-resume-verification.json
CUDA_VISIBLE_DEVICES="" python verify_native_resume.py \
  --baseline /scratch/specforge-torchtitan-20261003/online-native-full/output/native-online \
  --resumed /scratch/specforge-torchtitan-20261003/online-native-resume/output/native-online \
  --step 3 --allow-fixture-path-differences \
  --output online-native-resume-verification.json
```

`scripts/verify_native_resume.py` enumerates the three allowed online literal
path exclusions. All other checkpoint leaves and HF tensors are compared.

## CPU and focused kernel checks

The root agent retained this exact legacy-environment command (relative to the
core checkout, with the path to `venv213` resolved on the devbox):

```bash
CUDA_VISIBLE_DEVICES="" /scratch/specforge-torchtitan-20261003/venv213/bin/python \
  -m pytest tests/test_config/test_server_only_online.py \
  tests/test_runtime/test_disagg_online_dp_protocol.py \
  tests/test_runtime/test_disagg_online_shared_plane.py \
  tests/test_runtime/test_cli_config_build.py -q
```

The exact shell invocation for the 132-pass native CPU summary was not
retained. The following is an equivalent focused reproduction selection;
counts can differ as tests evolve and it is not claimed to be that exact run:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/test_config/test_torchtitan_runtime.py \
  tests/test_config/test_torchtitan_frontend.py \
  tests/test_training/test_torchtitan_runtime.py \
  tests/test_training/test_torchtitan_online.py \
  tests/test_training/test_torchtitan_validation.py \
  tests/test_runtime/test_cli_config_build.py
```

The GPU fused-kernel and graph regression selection can be reproduced with:

```bash
python -m pytest -q tests/test_training/test_torchtitan_graph.py \
  tests/test_modeling/test_dflash2_fused_conv.py \
  tests/test_utils/test_dflash2_fused_head.py
```

The later six-test metadata run selected only
`tests/test_training/test_torchtitan_graph.py`. These last two commands describe
the retained test selections; original shell environment transcripts were not
stored. No secrets or environment dumps are archived.

## Final real Mooncake TCP validation

[mooncake-real/commands.md](mooncake-real/commands.md) records the later actual
DP2 consumer, completed-checkpoint resume, CPU verification and cleanup against
final core `c70a5f413220e6ce68247afb528664986502300f`. It uses the shipping Mooncake
store constructor with no transport substitution. Endpoint environment files,
feature payloads and checkpoint weights are not archived.
