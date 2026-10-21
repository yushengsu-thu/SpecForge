#!/bin/bash
# dense-same: end-of-segment producer reaper. After the consumer finishes, the online producer's Mooncake cleanup
# re-reads the whole refs.jsonl once per 1,024 refs (O(N^2): ~6 days for 12 epochs), which blocks the last export and
# any chained segment (both wait for the trainer to exit). Once control/refs.jsonl.consumer_done exists and the final
# checkpoint plus W&B sync are done (GRACE seconds later), SIGKILL only that segment's `--role producer` process.
# usage: reap_producer.sh <RUN_ID> [GRACE_SECONDS=300]        (run on the trainer node, rank 1)
set -u
RUN=${1:?run id}; GRACE=${2:-300}; BASE=/scratch/dense; DONE=$BASE/outputs/$RUN/control/refs.jsonl.consumer_done
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
producer_pid() { pgrep -f "specforge.cli trai[n].*recipe-${RUN}[.]yaml.*--role producer"; }
log "reaper for $RUN: waiting for $DONE"
until [ -f "$DONE" ]; do
  # trainer gone without a consumer_done marker (crash): nothing to reap
  pgrep -f "specforge.cli trai[n].*recipe-${RUN}[.]yaml" >/dev/null || { sleep 60; [ -f "$DONE" ] || { log "trainer for $RUN exited without consumer_done; reaper exits"; exit 0; }; }
  sleep 60
done
log "consumer_done for $RUN; grace ${GRACE}s"; sleep "$GRACE"
for i in 1 2 3; do
  p=$(producer_pid) || { log "producer for $RUN already exited"; exit 0; }
  log "producer for $RUN still alive after consumer_done (pid $p, stuck in Mooncake cleanup); SIGKILL"; kill -KILL $p; sleep 20
done
producer_pid >/dev/null && log "REAP_FAILED $RUN" || log "PRODUCER_REAPED $RUN"
