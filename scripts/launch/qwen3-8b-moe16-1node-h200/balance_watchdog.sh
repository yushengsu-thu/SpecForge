#!/bin/bash
# Router-collapse watchdog: every minute read the latest logged step of the run and alert when moe/step_balance
# stays above the threshold for three consecutive readings (1.0 = balanced, num_experts = collapsed).
#   usage: balance_watchdog.sh <run_id> [threshold=3]
RUN=${1:?run id}; TH=${2:-3}; D=/scratch/moe8b16; F=$D/outputs/$RUN/launch.log; A=$D/logs/balance_watchdog-$RUN.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a $A; }
bad=0; last=""
while true; do
  line=$(grep -o -E '^step [0-9]+: \{[^}]*\}' $F 2>/dev/null | tail -1)
  if [ -n "$line" ] && [ "$line" != "$last" ]; then
    last=$line
    step=$(echo "$line" | grep -o -E '^step [0-9]+' | cut -d' ' -f2)
    # The trainer logs each step twice, with bare keys and with a train/ prefix; accept both.
    bal=$(echo "$line" | grep -o -E "'(train/)?moe/step_balance': [0-9.]+" | grep -o -E '[0-9.]+$')
    aux=$(echo "$line" | grep -o -E "'(train/)?moe/aux_loss': [0-9.]+" | grep -o -E '[0-9.]+$')
    if [ -n "$bal" ] && awk -v b=$bal -v t=$TH 'BEGIN{exit !(b>t)}'; then bad=$((bad+1)); else bad=0; fi
    [ "$bad" -ge 3 ] && log "ALERT step $step step_balance=$bal aux_loss=$aux (>$TH for $bad readings)"
    [ "$bad" -eq 0 ] && [ $((step % 500)) -lt 10 ] && log "ok step $step step_balance=$bal aux_loss=$aux"
  fi
  pgrep -f "specforge.cli trai[n]" >/dev/null || { log "trainer gone at step ${step:-?}; watchdog exits"; exit 0; }
  sleep 60
done
