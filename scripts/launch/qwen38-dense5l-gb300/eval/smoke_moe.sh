#!/bin/bash
# quick load + one request for an MoE export on GPU 1, port 30399
E=/scratch/dense5l-eval; D=${1:?draft}
setsid nohup bash $E/serve_moe.sh $D 1 30399 > $E/logs/serve_smoke_moe.log 2>&1 < /dev/null &
SP=$!
for t in $(seq 1 90); do curl -sf -m 3 http://127.0.0.1:30399/health >/dev/null && break; grep -q -E "Traceback|CUDA out of memory|scheduler died" $E/logs/serve_smoke_moe.log && { echo SMOKE_SERVER_FAILED; grep -E "Error|Traceback" -A4 $E/logs/serve_smoke_moe.log | tail -20; exit 1; }; sleep 10; done
grep -a -o -E "Using MoE kernel config from [^ ]+|Using default MoE kernel config[^\n]*|Load weight end.*Qwen3MoeDSparkModel.*" $E/logs/serve_smoke_moe.log | head -3
python3 - <<"PY"
import requests, time
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/scratch/dense5l-eval/models/Qwen3.8-27B", trust_remote_code=True)
msgs = [{"role": "user", "content": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:"}]
p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
t0 = time.time()
r = requests.post("http://127.0.0.1:30399/generate", json={"text": p, "sampling_params": {"temperature": 0.0, "top_k": 1, "max_new_tokens": 256}}, timeout=600).json()
mi = r.get("meta_info", {})
print("SMOKE_OK elapsed %.1fs completion_tokens %s spec_verify_ct %s accept %.2f" % (time.time()-t0, mi.get("completion_tokens"), mi.get("spec_verify_ct"), (mi.get("completion_tokens") or 0) / max(mi.get("spec_verify_ct") or 1, 1)))
print("text:", repr(r.get("text", "")[:160]))
PY
pkill -TERM -P $SP 2>/dev/null; kill -TERM $SP 2>/dev/null; sleep 8; pkill -9 -f "launch_server.*--port 30399\$" 2>/dev/null; echo SMOKE_DONE
