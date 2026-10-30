#!/usr/bin/env bash
# Host side: for each draft, serve it at width 8 with graphs and running requests up to 256,
# then sweep client concurrency; results go to /work/logs/sweep_g8_<name>.txt
set -u
W=$HOME/claude-dflash-moe
C=claude-dflash-moe
stop() { docker exec $C bash -c "pkill -f launch_server; sleep 4; pgrep -f launch_server >/dev/null && pkill -9 -f launch_server; true"; }
for pair in "moe43000:/work/export2/step-43000"; do
  name=${pair%%:*}; draft=${pair#*:}
  stop
  docker exec -d $C bash -c "cd /sgl-workspace/sglang && DRAFT=$draft bash /work/serve.sh --speculative-dspark-block-size 8 --max-running-requests 256 --cuda-graph-max-bs-decode 256 > /work/logs/server_sweep_$name.log 2>&1"
  for i in $(seq 1 40); do sleep 10; docker exec $C curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && break; done
  echo "## $name ready; $(grep -oE 'Capture draft verify CUDA graph end. bs=\[[^]]*\]' $W/logs/server_sweep_$name.log | sed 's/.*bs=//' | tr -d '\n' | cut -c1-60)..." | tee $W/logs/sweep_g8_$name.txt
  # warm-up (kernel JIT, caches)
  docker exec $C bash -c "cd /tmp && python3 /work/bench_accept.py --concurrency 8 --num-requests 32 --max-tokens 256" > /dev/null 2>&1
  for c in 1 2 4 8 16 24 32 48 64 96 128 192 256; do
    n=$(( c * 4 > 32 ? c * 4 : 32 ))
    echo "== $name width=8 concurrency=$c requests=$n" | tee -a $W/logs/sweep_g8_$name.txt
    docker exec $C bash -c "cd /tmp && timeout 600 python3 /work/bench_accept.py --concurrency $c --num-requests $n --max-tokens 256" 2>&1 | grep -E "concurrency=|spec_accept_length|per_verify_step" | tee -a $W/logs/sweep_g8_$name.txt
  done
done
stop
echo "SWEEP DONE"
