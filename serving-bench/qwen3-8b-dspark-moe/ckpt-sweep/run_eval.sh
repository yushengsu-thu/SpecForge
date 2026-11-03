#!/bin/bash
# Wait for our step-38000 export to be on HF (same samples seen as jiapingW's step 39000), fetch both drafts, then
# benchmark jiapingW's dense draft and ours back to back on GPU 0 of the training box; print the trainer's perf
# before and after so the slowdown and the recovery are on record.
W=${EVAL_DIR:-/scratch/moe8b16/eval}; RUN=qwen3-8b-moe8x1536-8A1-dspark-run1; STEP=${STEP:-38000}
LOG=/scratch/moe8b16/outputs/$RUN/launch.log
export HF_HOME=/scratch/moe8b16/hf_home HF_HUB_CACHE=/scratch/moe8b16/hf_home/hub HF_TOKEN="${HF_TOKEN:-$(cat /scratch/secrets/hf_token 2>/dev/null)}"
cd $W; mkdir -p models serve
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
log "trainer perf before: $(perf)"
until [ -f /scratch/moe8b16/exports/done-$RUN-step-$STEP ]; do sleep 120; done
log "step-$STEP is on HF"
[ -f models/jw-dense/model.safetensors ] || hf download jiapingW/Qwen3-8b-dspark-SpecForge config.json model.safetensors --local-dir models/jw-dense > serve/dl-jw.log 2>&1 || log "jw download FAILED"
hf download yushengsu/qwen3-8b-moe16x768-16A1-dspark --include "$RUN/step-$STEP/*" --local-dir models/ours > serve/dl-ours.log 2>&1 || log "ours download FAILED"
ln -sfn $W/models/ours/$RUN/step-$STEP models/ours-step-$STEP
for d in jw-dense ours-step-$STEP; do
  python3 -c "import json; c=json.load(open('models/$d/config.json')); print('$d', c['architectures'], 'experts', c.get('num_experts'), 'x', c.get('moe_intermediate_size'), 'block', c.get('block_size'))"
done
# Absolute draft paths: serve_eval.sh cds into the code checkout, and SGLang treats a relative path it cannot find
# as a Hub repo id.
bash serve_eval.sh dense $W/models/jw-dense 2>&1 | tee serve/dense.log
sleep 30
bash serve_eval.sh moe-step$STEP $W/models/ours-step-$STEP moe 2>&1 | tee serve/moe-step$STEP.log
log "eval servers stopped; gpu0 $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 0); port 31301 listeners=$( (ss -ltn 2>/dev/null) | grep -cE ':31301 ')"
sleep 300
log "trainer perf after (5 min later): $(perf)"
log "EVAL_DONE"
