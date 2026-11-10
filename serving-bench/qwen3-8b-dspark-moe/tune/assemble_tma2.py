"""Assemble per-bucket TMA-aware tune results (M*.json from tune_tma_one.py) into the two files SGLang loads:
<OUTDIR>/E=<E>,N=<N>,device_name=<GPU>.json   (up projection, USE_TMA per bucket)
<OUTDIR>/E=<E>,N=<N>,device_name=<GPU>_down.json   (down projection)
Buckets present in BASE (an earlier config file, optional) but not re-tuned keep their BASE entry."""
import json, os, glob, torch
SHAPE = os.environ.get("SHAPE_CFG", "/scratch/ckpt-sweep/tune/model-cfg"); TUNE = os.environ.get("TUNE_DIR", "/scratch/profile/tma-tune")
OUTDIR = os.environ.get("OUTDIR", "/scratch/ckpt-sweep/moe-config-tma2/configs/triton_3_8_0"); BASE = os.environ.get("BASE", "")
cfg = json.load(open(os.path.join(SHAPE, "config.json")))
E, N = cfg["num_experts"], cfg["moe_intermediate_size"]          # N = w2.shape[2] = intermediate size (SGLang's naming convention)
gpu = torch.cuda.get_device_name(0).replace(" ", "_") if torch.cuda.is_available() else "NVIDIA_H200"
up_name = "E=%d,N=%d,device_name=%s.json" % (E, N, gpu); down_name = up_name.replace(".json", "_down.json")
os.makedirs(OUTDIR, exist_ok=True)
up = {k: dict(v) for k, v in json.load(open(BASE)).items()} if BASE and os.path.exists(BASE) else {}
down = {}
for f in sorted(glob.glob(os.path.join(TUNE, "M*.json")), key=lambda p: int(os.path.basename(p)[1:-5])):
    r = json.load(open(f)); M = str(r["M"]); up[M] = dict(r["up"]); down[M] = dict(r["down"])
up = dict(sorted(up.items(), key=lambda kv: int(kv[0]))); down = dict(sorted(down.items(), key=lambda kv: int(kv[0])))
json.dump(up, open(os.path.join(OUTDIR, up_name), "w"), indent=4); json.dump(down, open(os.path.join(OUTDIR, down_name), "w"), indent=4)
print("wrote", os.path.join(OUTDIR, up_name), "buckets", list(up)); print("wrote", os.path.join(OUTDIR, down_name), "buckets", list(down))
for M in down: print("  M=%s up TMA=%s down TMA=%s" % (M, up[M].get("USE_TMA"), down[M].get("USE_TMA")))
