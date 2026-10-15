#!/bin/bash
# usage: run_pool.sh <jobs file>   (lines: <name> <draft dir>); runs eval_any.sh jobs on GPUs 0-3, one per GPU, until all done
J=${1:?jobs}; E=/scratch/dense5l-eval; declare -A busy; port=${PORT_BASE:-30400}; GPUS=${GPUS:-"0 1 2 3"}
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
while read -r name draft; do
  [ -z "$name" ] && continue
  while :; do
    for g in $GPUS; do
      pid=${busy[$g]:-}; if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        port=$((port+1)); setsid nohup bash $E/eval_any.sh $name $draft $g $port > $E/logs/eval_one_$name.log 2>&1 < /dev/null &
        busy[$g]=$!; log "launched $name on gpu $g port $port (pid ${busy[$g]})"; sleep 30; continue 3
      fi
    done
    sleep 60
  done
done < "$J"
for g in $GPUS; do pid=${busy[$g]:-}; [ -n "$pid" ] && wait "$pid" 2>/dev/null; done
# wait for any still running (setsid children are not our children; poll the logs)
while read -r name draft; do [ -z "$name" ] && continue; until grep -q -E "EVAL_ONE_DONE|SERVER FAILED|NOT HEALTHY" $E/logs/eval_one_$name.log 2>/dev/null; do sleep 60; done; done < "$J"
log "POOL_DONE"
