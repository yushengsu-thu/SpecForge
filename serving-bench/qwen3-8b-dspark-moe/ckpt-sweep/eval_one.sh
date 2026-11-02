#!/bin/bash
# Evaluate one uploaded checkpoint of a run on jiapingW's five tasks at c=1, on GPU 0 of the training box beside the
# capture server (same bounded server as run_eval.sh), then stop the server and show the trainer's perf before/after.
# usage: eval_one.sh <run id> <checkpoint folder, e.g. step-45000>
RUN=$1; CK=$2; W=${EVAL_DIR:-/scratch/moe8b16/eval}; NAME=$RUN-$CK; REPO=yushengsu/qwen3-8b-moe16x768-16A1-dspark
LOG=/scratch/moe8b16/outputs/$RUN/launch.log
export HF_HOME=/scratch/moe8b16/hf_home HF_HUB_CACHE=/scratch/moe8b16/hf_home/hub HF_TOKEN="${HF_TOKEN:-$(cat /scratch/secrets/hf_token 2>/dev/null)}"
cd $W; mkdir -p models serve
log(){ echo "[$(date -u +%T)] $*"; }
perf(){ python3 - $LOG <<'EOF'
import re, ast, sys, statistics
rows = []
for m in re.finditer(r"step (\d+): (\{'[a-z_/]+': .*?\})(?=step \d+:|\n|$)", open(sys.argv[1], errors='replace').read()):
    try: d = ast.literal_eval(m.group(2))
    except Exception: continue
    if 'acc' in d: rows.append((int(m.group(1)), d))
rows = rows[-30:]
k = lambda key: statistics.mean(d[key] for s, d in rows if key in d)
print(f"steps {rows[0][0]}-{rows[-1][0]}: train_compute {k('perf/train_compute_time_s'):.3f}s data_wait {k('perf/data_wait_time_s'):.3f}s samples/s {k('perf/global_samples_per_second'):.2f}")
EOF
}
log "trainer perf before: $(perf)"
hf download $REPO --include "$RUN/$CK/*" --local-dir $W/models/dl-$NAME > serve/dl-$NAME.log 2>&1 || { log "download FAILED"; exit 1; }
rm -rf $W/models/$NAME; mv $W/models/dl-$NAME/$RUN/$CK $W/models/$NAME && rm -rf $W/models/dl-$NAME
python3 -c "import json; c=json.load(open('$W/models/$NAME/config.json')); print('$NAME', c['architectures'], 'experts', c.get('num_experts'), 'x', c.get('moe_intermediate_size'))"
GPU=0 PORT=31301 MAXRR=8 GRAPH_BS=8 TASKS="gsm8k:1319 math500:500 humaneval:164 mbpp:257 mt-bench:160" CONCS="1" \
  bash $W/serve_eval.sh $NAME $W/models/$NAME moe 2>&1 | tee serve/$NAME.log
rm -rf $W/models/$NAME
log "eval server stopped; gpu0 $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 0); port 31301 listeners=$( (ss -ltn 2>/dev/null) | grep -cE ':31301 ')"
sleep 300
log "trainer perf after (5 min later): $(perf)"
log "EVAL_DONE"
