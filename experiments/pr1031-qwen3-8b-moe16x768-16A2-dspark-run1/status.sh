#!/usr/bin/env bash
# Devbox-only: one-screen status of the run (for the 6-hourly check): progress, processes, GPUs, uploads, disk.
W=/scratch/pr1031; RUN=pr1031-qwen3-8b-moe16x768-16A2-dspark-run1; E=$W/SpecForge/experiments/$RUN
OUT=$W/outputs/$RUN; DONE=$W/uploaded/$RUN; LOG=$W/logs/train_$RUN.log; WLOG=$W/logs/watch_$RUN.log
SCHED=$OUT/control/refs.jsonl.schedule.json
echo "== $RUN @ $(date -u +%FT%TZ) on $(hostname)"
last=$(grep -o -E "^step [0-9]+: \{.*" $LOG 2>/dev/null | tail -1)
step=$(echo "$last" | grep -o -E "^step [0-9]+" | grep -o -E "[0-9]+")
total=$( [ -f $SCHED ] && python3 -c "import json;print(json.load(open('$SCHED'))['total_steps'])" )
if [ -n "$step" ]; then
  pct=$( [ -n "$total" ] && python3 -c "print(f'{100*$step/$total:.1f}%')" )
  m=$(echo "$last" | python3 -c "
import ast,sys,re
d=ast.literal_eval(re.sub(r'^step \d+: ','',sys.stdin.read().strip()))
g=lambda k:d.get('train/'+k,d.get(k))
f=lambda v,p=4:'n/a' if v is None else f'{v:.{p}f}'
print('acc',f(g('acc')),'| ce',f(g('ce_loss')),'| walk',f(g('dspark/hard_label/walk_accepted_length'),3),'| balance',f(g('moe/step_balance'),3),'| lr',f(g('lr'),6))" 2>/dev/null)
  echo "step $step / ${total:-?} (${pct:-?})  $m"
  # Rate from the (time, step) samples this script appends on every call -> ETA.
  S=$W/logs/status_samples.txt; echo "$(date +%s) $step" >> $S
  python3 - "$S" "$step" "${total:-0}" "$LOG" <<'PY'
import os,sys,time
path,step,total,log=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
now=time.time(); rows=[tuple(map(int,l.split())) for l in open(path) if l.strip()]
print(f"train log last written {(now-os.path.getmtime(log))/60:.1f} min ago")
cands=[r for r in rows if 600 <= now-r[0] <= 86400 and r[1] < step]
if not cands:
    print("rate: n/a yet (needs a status sample at least 10 min old)")
else:
    t0,s0=cands[0]; rate=(step-s0)/(now-t0)
    if rate <= 0 or not total:
        print("rate: no progress since the earlier sample")
    else:
        eta=(total-step)/rate
        print(f"rate {rate*3600:.0f} steps/h over the last {(now-t0)/3600:.1f} h; ETA ~{eta/3600:.1f} h ({eta/86400:.1f} d): "
              f"~{time.strftime('%m-%d %H:%M', time.gmtime(now+eta))} UTC = {time.strftime('%m-%d %H:%M', time.gmtime(now+eta-8*3600))} PST")
PY
else
  echo "no optimizer step logged yet; last log line: $(tail -1 $LOG 2>/dev/null | cut -c1-160)"
fi
echo "trainers alive: $(pgrep -f 'specforge.cli trai[n]' | wc -l) | watcher alive: $(pgrep -f 'watch_ckpts.s[h]' | wc -l) | capture servers: $(pgrep -f 'sglang.launch_serve[r]' | wc -l)"
echo "GPU util: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ')| mem GiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{printf "%.0f ", $1/1024}')"
echo "uploaded: $(ls $DONE 2>/dev/null | sed 's/step-//' | sort -n | tr '\n' ' ')"
echo "local checkpoints: $(ls -d $OUT/$RUN-step[0-9]* 2>/dev/null | sed 's/.*-step//' | sort -n | tr '\n' ' ')"
echo "disk: $(df -h /scratch | awk 'NR==2{print $4" free of "$2}')"
echo "watcher log: $(tail -2 $WLOG 2>/dev/null | cut -c1-160 | tr '\n' ' | ')"
grep -c -E "Traceback|CUDA out of memory|NCCL error" $LOG 2>/dev/null | sed 's/^/error signatures in train log: /'
