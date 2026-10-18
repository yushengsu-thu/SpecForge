#!/usr/bin/env bash
# Epochs 13-18 of the Qwen3-8B DSpark MoE 8x1536 top-1 run, chained after the 12-epoch run on the same box:
#   nohup experiments/qwen3-8b-dspark-moe-8expert-1activate/continue_to_18.sh >/dev/null 2>&1 &
# 1. waits for the 12-epoch launcher to exit and for its final checkpoint (saved at the last step);
# 2. launches a 6-epoch continuation warm-started from it (draft weights only, fresh AdamW, its own 4% re-warmup to
#    6e-4 and cosine to 0, prompt_seed 54), logging into the SAME W&B run (id 1ia117wn, resume=allow) with every
#    optimizer step shifted by the first run's total steps, so the curves continue without a gap;
# 3. once the first run's watcher has finished epoch 12, starts a watcher for epochs 13-18 (HF folders 13ep..18ep)
#    whose exp_read.me covers all 18 epochs.
# Log: /scratch/logs/<run>.chain18.log. Stop file: /scratch/outputs/<run>/chain18.stop
set -u
REPO=${REPO:-/scratch/SpecForge-spark8b}
RUN=${RUN:-qwen3-8b-dspark-moe-8expert-1activate-dense-same}
MAIN_RUN=$RUN   # prefix assignments below re-bind RUN, so refer to the first run by this name
OUT=/scratch/outputs/$RUN
CONT_RUN=$RUN-cont13to18
CONT_OUT=/scratch/outputs/$CONT_RUN
EXP=experiments/qwen3-8b-dspark-moe-8expert-1activate
LOG=/scratch/logs/$RUN.chain18.log; STOP=$OUT/chain18.stop
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
alive(){ [ -f "$1/launcher.pid" ] && kill -0 "$(cat "$1/launcher.pid")" 2>/dev/null; }
complete(){ [ -f "$1/training_state.pt" ] && ! ls "$1" 2>/dev/null | grep -q "\.tmp$"; }
cd "$REPO" || exit 1
log "chain started; waiting for $RUN to finish"
while alive "$OUT"; do [ -f "$STOP" ] && { log "stop file seen"; exit 0; }; sleep 120; done
MAIN_TOTAL=$(python3 - "$OUT" <<'PY'
import glob, json, sys
for p in glob.glob(sys.argv[1] + "/control/**/*.json", recursive=True):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if isinstance(d, dict) and d.get("total_steps"):
        print(int(d["total_steps"])); break
PY
)
MAIN_TOTAL=${MAIN_TOTAL:-129173}
LAST=$(ls -d "$OUT/$RUN"-step* 2>/dev/null | sed 's/.*-step//' | sort -n | tail -1)
CK=$OUT/$RUN-step$LAST
log "launcher exited; total_steps $MAIN_TOTAL, latest checkpoint step $LAST"
if [ -z "$LAST" ] || [ "$LAST" -lt $(( MAIN_TOTAL - 1196 )) ] || ! complete "$CK"; then
  log "ABORT: the 12-epoch run did not reach its end (latest checkpoint step ${LAST:-none} of $MAIN_TOTAL); not continuing"; exit 1
fi
mkdir -p /scratch/ckpt-keep/$RUN
if [ ! -d "/scratch/ckpt-keep/$RUN/$RUN-step$LAST" ]; then cp -a "$CK" "/scratch/ckpt-keep/$RUN/$RUN-step$LAST.tmp" && mv "/scratch/ckpt-keep/$RUN/$RUN-step$LAST.tmp" "/scratch/ckpt-keep/$RUN/$RUN-step$LAST"; fi
CK=/scratch/ckpt-keep/$RUN/$RUN-step$LAST
log "warm-start checkpoint: $CK"
sleep 60
for i in $(seq 1 30); do ss -ltn 2>/dev/null | grep -qE ":(3000[01]|35551|35880) " || break; sleep 10; done
LEFT=$(pgrep -f "sglang::|[s]glang.launch_server|[m]ooncake_master|[s]pecforge.cli|torch.[d]istributed.run")
[ -n "$LEFT" ] && { log "killing leftovers $LEFT"; kill -KILL $LEFT 2>/dev/null; sleep 10; }
[ -e "$CONT_OUT" ] && mv "$CONT_OUT" "$CONT_OUT.stale.$(date -u +%Y%m%dT%H%M%SZ)"
export SPECFORGE_LOG_STEP_OFFSET=$MAIN_TOTAL WANDB_RUN_ID=1ia117wn WANDB_RESUME=allow
RUN=$CONT_RUN OUT=$CONT_OUT $EXP/launch.sh training.num_epochs=6 training.prompt_seed=54 \
  model.draft_checkpoint_path="$CK" >> "$LOG" 2>&1 || { log "ABORT: launch failed"; exit 1; }
log "continuation launched (step offset $MAIN_TOTAL)"
for i in $(seq 1 60); do
  sleep 60
  s=$(grep -oE "step [0-9]+:" "$CONT_OUT/launch.log" 2>/dev/null | tail -1 | tr -dc 0-9)
  if [ -n "$s" ]; then log "continuation logging at global step $s"; break; fi
  alive "$CONT_OUT" || { log "ABORT: continuation launcher died before its first step"; exit 1; }
done
while pgrep -f "^bash $EXP/epoch_watcher[.]sh$" >/dev/null && [ ! -f "$STOP" ]; do sleep 120; done
RUN=$CONT_RUN OUT=$CONT_OUT KEEP=/scratch/ckpt-keep/$CONT_RUN EXPORTS=/scratch/exports/$MAIN_RUN \
  NUM_EPOCHS=6 EPOCH_OFFSET=12 REPORT_RUN=$MAIN_RUN REPORT_OUT=/scratch/outputs/$MAIN_RUN \
  REPORT_ARGS="--cont-out $CONT_OUT --cont-run $CONT_RUN --cont-epoch-offset 12" \
  LOG=/scratch/logs/$CONT_RUN.watcher.log nohup $EXP/epoch_watcher.sh >/dev/null 2>&1 &
log "epoch 13-18 watcher started (pid $!)"
