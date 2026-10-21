#!/usr/bin/env bash
# Epochs 19-24 of the Qwen3-8B DSpark MoE 8x1536 top-1 run, chained after the epochs 13-18 continuation:
#   setsid nohup experiments/qwen3-8b-dspark-moe-8expert-1activate/continue_to_24.sh >/dev/null 2>&1 < /dev/null &
# 1. waits for the 13-18 launcher to exit (teardown_guard.sh stops it after its consumer is done), keeps its final
#    checkpoint, and waits for the 13-18 watcher to finish epoch 18 (export, HF upload, exp_read.me);
# 2. launches a 6-epoch continuation warm-started from that checkpoint (draft weights only, fresh AdamW, its own
#    warmup and cosine to 0) into the SAME W&B run (id 1ia117wn, resume=allow) with every step shifted by all earlier
#    steps (129,173 + 64,586 = 193,759), so the curves continue without a gap. LR / warmup / prompt_seed are read from
#    $SETTINGS right before the launch (default as in epochs 13-18: 6e-4, 4% warmup; prompt_seed 60);
# 3. starts a teardown guard and the epoch 19-24 watcher (HF folders 19ep..24ep; exp_read.me covers all 24 epochs).
# Log: /scratch/logs/<run>.chain24.log. Stop file: /scratch/outputs/<run>/chain24.stop
set -u
REPO=${REPO:-/scratch/SpecForge-spark8b}
MAIN_RUN=${MAIN_RUN:-qwen3-8b-dspark-moe-8expert-1activate-dense-same}
MAIN_OUT=/scratch/outputs/$MAIN_RUN
PREV_RUN=$MAIN_RUN-cont13to18; PREV_OUT=/scratch/outputs/$PREV_RUN
NEXT_RUN=$MAIN_RUN-cont19to24; NEXT_OUT=/scratch/outputs/$NEXT_RUN
EXP=experiments/qwen3-8b-dspark-moe-8expert-1activate
LOG=/scratch/logs/$MAIN_RUN.chain24.log; STOP=$MAIN_OUT/chain24.stop
SETTINGS=${SETTINGS:-$MAIN_OUT/cont19to24.settings}
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
alive(){ [ -f "$1/launcher.pid" ] && kill -0 "$(cat "$1/launcher.pid")" 2>/dev/null; }
complete(){ [ -f "$1/training_state.pt" ] && ! ls "$1" 2>/dev/null | grep -q "\.tmp$"; }
stopped(){ [ -f "$STOP" ] && { log "stop file seen, exiting"; exit 0; }; }
listening(){ python3 -c "
import sys
ports = {int(p) for p in sys.argv[1:]}
for f in ('/proc/net/tcp', '/proc/net/tcp6'):
    for line in open(f).readlines()[1:]:
        x = line.split()
        if x[3] == '0A' and int(x[1].split(':')[1], 16) in ports: sys.exit(0)
sys.exit(1)" "$@"; }
total_steps(){ python3 - "$1" <<'PY'
import glob, json, sys
for p in glob.glob(sys.argv[1] + "/control/**/*.json", recursive=True):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if isinstance(d, dict) and d.get("total_steps"):
        print(int(d["total_steps"])); break
PY
}
cd "$REPO" || exit 1
log "chain started; waiting for $PREV_RUN to finish"
while alive "$PREV_OUT"; do stopped; sleep 120; done
MAIN_TOTAL=$(total_steps "$MAIN_OUT"); MAIN_TOTAL=${MAIN_TOTAL:-129173}
PREV_TOTAL=$(total_steps "$PREV_OUT"); PREV_TOTAL=${PREV_TOTAL:-64586}
OFFSET=$(( MAIN_TOTAL + PREV_TOTAL ))
LAST=$(ls -d "$PREV_OUT/$PREV_RUN"-step* 2>/dev/null | sed 's/.*-step//' | sort -n | tail -1)
CK=$PREV_OUT/$PREV_RUN-step$LAST
log "launcher exited; $PREV_RUN total_steps $PREV_TOTAL, latest checkpoint step ${LAST:-none}; W&B step offset $OFFSET"
if [ -z "$LAST" ] || [ "$LAST" -lt $(( PREV_TOTAL - 1196 )) ] || ! complete "$CK"; then
  log "ABORT: $PREV_RUN did not reach its end (latest checkpoint step ${LAST:-none} of $PREV_TOTAL); not continuing"; exit 1
fi
KEEP=/scratch/ckpt-keep/$PREV_RUN; mkdir -p "$KEEP"
if [ ! -d "$KEEP/$PREV_RUN-step$LAST" ]; then
  cp -a "$CK" "$KEEP/$PREV_RUN-step$LAST.tmp" && mv "$KEEP/$PREV_RUN-step$LAST.tmp" "$KEEP/$PREV_RUN-step$LAST" \
    || { log "ABORT: could not copy $CK to $KEEP"; exit 1; }
fi
CK=$KEEP/$PREV_RUN-step$LAST
log "warm-start checkpoint: $CK"
# epoch 18 has to be exported and uploaded first: the leftover cleanup below kills anything named specforge.cli
while pgrep -f "^bash $EXP/epoch_watcher[.]sh$" > /dev/null; do stopped; sleep 120; done
log "13-18 watcher done: $(tail -1 /scratch/logs/$PREV_RUN.watcher.log 2>/dev/null | cut -c1-160)"
stopped
LR=0.0006; WARMUP=0.04; PROMPT_SEED=60
if [ -f "$SETTINGS" ]; then . "$SETTINGS"; fi
log "recipe: lr $LR, warmup_ratio $WARMUP, prompt_seed $PROMPT_SEED (settings file: $SETTINGS)"
export SPECFORGE_LOG_STEP_OFFSET=$OFFSET WANDB_RUN_ID=1ia117wn WANDB_RESUME=allow
LAUNCHED=0
for attempt in 1 2 3; do
  for i in $(seq 1 30); do listening 30000 30001 35551 35880 || break; sleep 10; done
  LEFT=$(pgrep -f "sglang::|[s]glang.launch_server|[m]ooncake_master|[s]pecforge.cli|torch.[d]istributed.run")
  [ -n "$LEFT" ] && { log "killing leftovers $LEFT"; kill -KILL $LEFT 2>/dev/null; sleep 10; }
  [ -e "$NEXT_OUT" ] && mv "$NEXT_OUT" "$NEXT_OUT.stale.$(date -u +%Y%m%dT%H%M%SZ)"
  RUN=$NEXT_RUN OUT=$NEXT_OUT $EXP/launch.sh training.num_epochs=6 training.prompt_seed="$PROMPT_SEED" \
    training.learning_rate="$LR" training.warmup_ratio="$WARMUP" model.draft_checkpoint_path="$CK" >> "$LOG" 2>&1 \
    || { log "launch attempt $attempt failed to start"; sleep 120; continue; }
  log "continuation launched, attempt $attempt"
  for i in $(seq 1 60); do
    sleep 60
    s=$(grep -oE "step [0-9]+:" "$NEXT_OUT/launch.log" 2>/dev/null | tail -1 | tr -dc 0-9)
    if [ -n "$s" ]; then log "continuation logging at step $s"; LAUNCHED=1; break; fi
    alive "$NEXT_OUT" || { log "attempt $attempt: launcher died before its first step: $(grep -E "Error|error" "$NEXT_OUT/launch.log" | tail -1 | cut -c1-200)"; break; }
  done
  [ "$LAUNCHED" = 1 ] && break
  sleep 120
done
[ "$LAUNCHED" = 1 ] || { log "ABORT: continuation did not start after 3 attempts"; exit 1; }
python3 - "$NEXT_OUT/continuation.json" "$LR" "$WARMUP" "$PROMPT_SEED" "$OFFSET" "$CK" <<'PY'
import json, sys
p, lr, wu, seed, off, ck = sys.argv[1:]
json.dump({"epochs": "19-24", "lr": float(lr), "warmup_ratio": float(wu), "prompt_seed": int(seed),
           "step_offset": int(off), "warm_start": ck}, open(p, "w"), indent=1)
PY
setsid nohup bash $EXP/teardown_guard.sh "$NEXT_OUT" 900 > /dev/null 2>&1 < /dev/null &
log "teardown guard started (pid $!)"
RUN=$NEXT_RUN OUT=$NEXT_OUT KEEP=/scratch/ckpt-keep/$NEXT_RUN EXPORTS=/scratch/exports/$MAIN_RUN \
  NUM_EPOCHS=6 EPOCH_OFFSET=18 REPORT_RUN=$MAIN_RUN REPORT_OUT=$MAIN_OUT \
  REPORT_ARGS="--cont-out $PREV_OUT --cont-run $PREV_RUN --cont-epoch-offset 12 --cont-out $NEXT_OUT --cont-run $NEXT_RUN --cont-epoch-offset 18" \
  LOG=/scratch/logs/$NEXT_RUN.watcher.log nohup $EXP/epoch_watcher.sh > /dev/null 2>&1 < /dev/null &
log "epoch 19-24 watcher started (pid $!)"
