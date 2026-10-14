#!/bin/bash
# wait for the epoch-1 chain (GPU 0) to finish, then evaluate the official v1 drafter on GPU 0
E=/scratch/dense5l-eval
until grep -q "EVAL_ONE_DONE" $E/logs/eval_one_1.log 2>/dev/null; do sleep 60; done
sleep 20; until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -lt 2000 ]; do sleep 20; done
bash $E/eval_generic.sh official-v1 $E/drafts/official-v1 0 30205
