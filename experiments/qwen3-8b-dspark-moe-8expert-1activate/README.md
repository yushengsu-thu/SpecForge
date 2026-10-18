# qwen3-8b-dspark-moe-8expert-1activate (dense-same)

Scripts for the 12-epoch Qwen3-8B DSpark MoE (8 experts, top-1) run; setup and results are in
`exp_read.me` at the repository root.

- `launch.sh`: starts the run on an 8xH200 box (managed_local); extra `key=value` arguments are
  config overrides (`RUN=x OUT=y launch.sh training.max_steps=1 tracking.report_to=none` for a dry run).
- `epoch_watcher.sh`: polls for epoch checkpoints, keeps them out of the rotation, exports and
  uploads them to the HF repo (`<k>ep/`), regenerates and commits `exp_read.me`.
- `epoch_report.py`: builds `exp_read.me` from the console log, checkpoints and schedule payload.
- `continue_to_18.sh`: chained after the 12-epoch run; launches epochs 13-18 warm-started from its final
  checkpoint (4% re-warmup to 6e-4, cosine to 0) into the same W&B run with `SPECFORGE_LOG_STEP_OFFSET`,
  then a watcher for 13ep..18ep whose `exp_read.me` covers all 18 epochs.
