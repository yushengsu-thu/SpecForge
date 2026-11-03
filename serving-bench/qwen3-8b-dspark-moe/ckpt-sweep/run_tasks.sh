#!/bin/bash
# jiapingW's other four benchmarks (math500, humaneval, mbpp, mt-bench) at c=1 for his dense draft and our
# step-38000, on GPU 0 of the training box; gsm8k was measured by run_eval.sh.
W=${EVAL_DIR:-/scratch/moe8b16/eval}; RUN=qwen3-8b-moe8x1536-8A1-dspark-run1; STEP=${STEP:-38000}
LOG=/scratch/moe8b16/outputs/$RUN/launch.log
cd $W
log(){ echo "[$(date -u +%T)] $*"; }
perf(){ python3 - $LOG <<'EOF'
import re, ast, sys, statistics
rows = []
for m in re.finditer(r"step (\d+): (\{'acc'.*?\})(?=step \d+:|\n|$)", open(sys.argv[1], errors='replace').read()):
    try: d = ast.literal_eval(m.group(2))
    except Exception: continue
    rows.append((int(m.group(1)), d))
rows = rows[-30:]
k = lambda key: statistics.mean(d[key] for s, d in rows if key in d)
print(f"steps {rows[0][0]}-{rows[-1][0]}: train_compute {k('perf/train_compute_time_s'):.3f}s data_wait {k('perf/data_wait_time_s'):.3f}s samples/s {k('perf/global_samples_per_second'):.2f}")
EOF
}
export TASKS="math500:500 humaneval:164 mbpp:257 mt-bench:160" CONCS="1"
log "trainer perf before: $(perf)"
bash serve_eval.sh dense-tasks $W/models/jw-dense 2>&1 | tee serve/dense-tasks.log
sleep 30
bash serve_eval.sh moe-step$STEP-tasks $W/models/ours-step-$STEP moe 2>&1 | tee serve/moe-step$STEP-tasks.log
log "eval servers stopped; gpu0 $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 0); port 31301 listeners=$( (ss -ltn 2>/dev/null) | grep -cE ':31301 ')"
sleep 300
log "trainer perf after (5 min later): $(perf)"
log "EVAL_DONE"
