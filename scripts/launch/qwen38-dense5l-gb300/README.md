# Dense 5-layer Qwen3.8-27B DSpark drafter, 3 epochs on one 4x GB300 devbox

Box-side scripts for the run described in
`examples/configs/online/disaggregated/managed-local/qwen3.8-27b-dspark-dense-5layer-3ep-gb300.yaml`
(draft config `configs/qwen3.8-27b-dspark.json` = Kan's dense config = the official v1 architecture, training block 7).

| what | where |
|---|---|
| devbox | `rx devbox acquire --gpu gb300 --count 4 --cluster gb300-coreweave --image lmsysorg/sglang:v0.5.18-cu130 --ttl 48h` |
| secrets | `/scratch/secrets/hf_token`, `/scratch/secrets/wandb_key` (mode 600; never committed) |
| setup (code, deps, spec-capture patch, NVFP4 teacher, corpus jsonl) | `setsid nohup bash setup_box.sh > /scratch/dense5l/logs/setup_box.log 2>&1 < /dev/null &` |
| smoke (3 optimizer steps, no W&B, no checkpoints) | `bash launch_train.sh smoke` |
| full run | `bash launch_train.sh full` (W&B entity `miles_training`, project `specforge-qwen38-dspark-dense-5layer`) |
| per-epoch export + upload | `setsid nohup bash watch_epoch_exports.sh > /scratch/dense5l/logs/watch_epoch_exports.log 2>&1 < /dev/null &` |
| outputs | `/scratch/dense5l/outputs/<run_id>/launch.log` (`step N: {...}` lines every 10 steps), checkpoints `<run_id>-step<N>` |
| HF | https://huggingface.co/RadixArk/qwen38-dspark-dense-5layer (private; folders `1ep/`, `2ep/`, `3ep/`) |

`prep_regen_mixture.py` turns the `RadixArk/Qwen3.8-27B-Regen-Mixture-v1` shards into the slim `{id, conversations}`
JSONL Kan's recipes train on. `make_readme.py` refreshes the HF model card from `launch.log` after each epoch.
Devbox gotchas: the container is OOM-killed at ~919 GB host memory (recipe header has the budget); SGLang 0.5.18's
`get_free_port()` can hand two capture servers the same NCCL port, so `launch_plan.py` pins `--nccl-port`;
the rx ssh relay drops connections after ~60 s, so everything long runs under `setsid nohup`.
