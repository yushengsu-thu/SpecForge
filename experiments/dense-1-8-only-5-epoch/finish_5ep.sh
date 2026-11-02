#!/bin/bash
# End-of-run cleanup for the dense-1-8-only-5-epoch run on specforge-qwen-h200 (shared node).
# Touches only this run's processes: launcher pid 1166605 and its descendants, plus sglang.launch_server /
# sglang:: / mooncake_master / specforge.cli processes whose environment carries VIRTUAL_ENV=/scratch/pr1003/venv.
#   finish_5ep.sh dry    -> print the state and the pids that would be killed; change nothing
#   finish_5ep.sh        -> require last step >= 53810, 5ep uploaded and a shutdown hang (trainer ranks gone and
#                           10 min without log output, or 30 min without log output); then py-spy dump the producer,
#                           kill the tree, wait for GPU memory to drop, remove the IN_USE marker
#   finish_5ep.sh force  -> same without the hang check
RUN=dense-1-8-only-5-epoch; W=/scratch/pr1003; L=$W/logs/train_$RUN.log; LAUNCHER=1166605; FINAL=53820
PYSPY=$W/logs/${RUN}_end_pyspy.txt; MARK=$W/exports/$RUN/5ep/.uploaded; INUSE=/scratch/IN_USE_$RUN.txt
log() { echo "[$(date -u +%FT%TZ)] $*"; }
tree_pids() {
  local ps_out keep=" $LAUNCHER " changed pid ppid args
  ps_out=$(ps -eo pid=,ppid=,args=)
  while read -r pid ppid args; do
    case "$args" in *sglang.launch_server*|*sglang::*|*mooncake_master*|*specforge.cli*) ;; *) continue ;; esac
    tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -q '^VIRTUAL_ENV=/scratch/pr1003/venv$' && keep="$keep$pid "
  done <<< "$ps_out"
  changed=1
  while [ "$changed" = 1 ]; do
    changed=0
    while read -r pid ppid args; do
      case "$keep" in *" $pid "*) continue ;; esac
      case "$keep" in *" $ppid "*) keep="$keep$pid "; changed=1 ;; esac
    done <<< "$ps_out"
  done
  for pid in $keep; do kill -0 "$pid" 2>/dev/null && echo "$pid"; done | sort -n
}
state() {
  last=$(grep -E "^step [0-9]+:" $L | tail -1 | grep -oE "^step [0-9]+" | awk '{print $2}')
  ranks=$(ps -eo pid,args | grep -cE "python -u -m specforge[.]cli train")
  age=$(( $(date +%s) - $(stat -c %Y $L) ))
  pids=$(tree_pids); nprocs=$(echo "$pids" | grep -c .)
  log "last step ${last:-?}/$FINAL, 5ep uploaded: $([ -f $MARK ] && echo yes || echo no), trainer ranks $ranks, log silent ${age}s, run processes $nprocs"
}
mode=${1:-run}
state
if [ "$mode" = dry ]; then
  for p in $pids; do echo "  $p $(ps -o etime=,args= -p $p | cut -c1-140)"; done
  echo "GPU memory.used (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
  exit 0
fi
[ "${last:-0}" -ge $((FINAL-10)) ] || { log "REFUSING: last step ${last:-?} < $((FINAL-10))"; exit 1; }
[ -f $MARK ] || { log "REFUSING: 5ep not uploaded yet"; exit 1; }
if [ "$mode" != force ]; then
  { [ "$ranks" = 0 ] && [ $age -ge 600 ]; } || [ $age -ge 1800 ] || { log "REFUSING: not a shutdown hang yet (ranks $ranks, silent ${age}s)"; exit 1; }
fi
if [ "$nprocs" = 0 ]; then
  log "no run processes left: clean exit"
else
  [ -f $L.final ] || cp $L $L.final
  : > $PYSPY
  PYSPY_BIN=$(command -v py-spy || ls $W/venv/bin/py-spy 2>/dev/null)
  for p in $(ps -eo pid=,args= | grep -E "python -m specforge[.]cli train --config" | grep -v torch.distributed | awk '{print $1}'); do
    echo "$pids" | grep -qx "$p" || continue
    echo "=== pid $p: $(ps -o args= -p $p | cut -c1-120)" >> $PYSPY
    if [ -n "$PYSPY_BIN" ]; then timeout 60 $PYSPY_BIN dump --pid $p >> $PYSPY 2>&1; else echo "(py-spy unavailable)" >> $PYSPY; fi
  done
  log "py-spy dump -> $PYSPY ($(wc -l < $PYSPY) lines)"
  log "killing $nprocs pids: $(echo $pids | tr '\n' ' ')"
  kill -TERM $pids 2>/dev/null; sleep 30
  pids=$(tree_pids)
  [ -n "$pids" ] && { log "SIGKILL $(echo $pids | tr '\n' ' ')"; kill -KILL $pids 2>/dev/null; sleep 10; }
  log "remaining run processes: $(tree_pids | grep -c .)"
fi
for i in $(seq 1 30); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{busy=1} END{exit busy}' && break; sleep 5; done
log "GPU memory.used (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
[ -f $INUSE ] && rm -f $INUSE && log "removed $INUSE"
log "done"
