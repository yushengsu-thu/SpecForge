"""Single-GPU TMA-aware fused-MoE tune for one token count M (draft M = 7 x c): runs the sep tuner BenchmarkWorker.tune with
enable_up_tma=True (round 1: down no-TMA vs TMA per config; round 2: up no-TMA vs TMA), writes the best up/down configs to JSON."""
import json, sys, time
sys.path.insert(0, "/scratch/moe8b16/serving-bench/sglang/benchmark/kernels/fused_moe_triton")
import tuning_fused_moe_triton_sep as T
from common_utils import get_model_config, get_configs_compute_bound
from sglang.srt.server_args import ServerArgs
M, out = int(sys.argv[1]), sys.argv[2]
import os
model = os.environ.get("SHAPE_CFG", "/scratch/ckpt-sweep/tune/model-cfg")
server_args = ServerArgs(model_path=model, tp_size=1, ep_size=1)
mc = get_model_config(model, 1, 1, False)
print("model config:", mc, flush=True)
w = T.BenchmarkWorker(0, server_args)
space = get_configs_compute_bound()
# Pruned to the neighbourhood of every winner of the first (non-TMA) tune: 648 instead of 1920 configs per variant.
space = space if os.environ.get("FULL_SPACE", "0") == "1" else [c for c in space if c["BLOCK_SIZE_M"] in (16, 32, 64, 128) and c["BLOCK_SIZE_N"] in (64, 128, 256) and c["BLOCK_SIZE_K"] in (64, 128, 256)
         and c["GROUP_SIZE_M"] in (1, 16, 32) and c["num_stages"] in (3, 4, 5)]
print("search space:", len(space), "configs", flush=True)
t0 = time.time()
cfg_up, cfg_down, t_up, t_down = w.tune(M, mc["num_experts"], mc["shard_intermediate_size"], mc["hidden_size"], mc["topk"], mc["dtype"],
                                        False, False, False, False, mc["block_shape"], space, os.environ.get("TOPK_DIR", "/scratch/profile/topk_ids"), 1, enable_up_tma=True)
res = {"M": M, "up": cfg_up, "down": cfg_down, "t_up_us": t_up, "t_down_us": t_down, "tune_seconds": time.time() - t0}
json.dump(res, open(out, "w"), indent=1); print("RESULT", json.dumps(res), flush=True)
