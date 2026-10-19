#!/usr/bin/env python3
"""Served vs trained MoE FFN on a real checkpoint of this drafter (8 experts x 1536, top-1, no shared expert).

Loads one layer's FFN twice: from the training checkpoint into SpecForge's training MoE layer (eval mode: router
input centered by the EMA mean), and from the normalized HF export (centering folded into gate.weight / gate.bias)
into the SGLang block that serves it (QwenMoESparseBlock: TopK + FusedMoE, Triton runner). Feeds the same
RMSNorm-shaped inputs to both and compares the chosen experts and the outputs. Needs one GPU.
    check_served_moe.py --ckpt <training checkpoint dir> --export <normalized export dir> --draft-config <json>
"""
import argparse, importlib.util, os, sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--export", required=True)
ap.add_argument("--draft-config", required=True); ap.add_argument("--tokens", type=int, default=4096)
a = ap.parse_args()
repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def sglang_runtime():
    """Server config + TP1 group the way a scheduler process has them (FusedMoE / TopK need both); SGLang 0.5.18 API."""
    from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
    from sglang.srt.layers.moe.utils import initialize_moe_config
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    sa = ServerArgs(model_path="dummy", attention_backend="triton", moe_runner_backend="triton")
    publish(sa, role="scheduler")
    set_global_server_args_for_scheduler(sa)
    initialize_moe_config(sa)
    init_distributed_environment(world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:29599", local_rank=0, backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)


sglang_runtime()

from safetensors.torch import load_file
from transformers import AutoConfig
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.training.model_loading import warm_start_draft_model
from specforge.serving.sglang_models.dflash_moe import QwenMoESparseBlock

dev, dt = "cuda", torch.bfloat16
export = {}
for f in sorted(os.listdir(a.export)):
    if f.endswith(".safetensors"):
        export.update(load_file(os.path.join(a.export, f)))
serv_cfg = AutoConfig.from_pretrained(a.export)
train_cfg = AutoConfig.from_pretrained(a.draft_config)
model = DSparkDraftModel(train_cfg)
warm_start_draft_model(model, a.ckpt, draft_config=train_cfg, strategy="dspark", allow_missing_embedding=True)
model = model.to(dev, dt).eval()
ok = True
for layer in range(train_cfg.num_hidden_layers):
    prefix = f"layers.{layer}.mlp."
    prev = torch.get_default_dtype(); torch.set_default_dtype(dt)
    try:
        with torch.device(dev):
            served = QwenMoESparseBlock(serv_cfg, layer_id=layer, prefix=f"layers.{layer}.mlp")
    finally:
        torch.set_default_dtype(prev)
    served.load_weights([(k[len(prefix):], v.to(dev)) for k, v in export.items() if k.startswith(prefix)])
    served.check_loaded(f"layer {layer}")
    served.experts.quant_method.process_weights_after_loading(served.experts)
    served.eval()
    ref = model.layers[layer].mlp
    g = torch.Generator(device="cpu").manual_seed(layer)
    w = export[f"layers.{layer}.post_attention_layernorm.weight"].float()
    x = (torch.randn(1, a.tokens, serv_cfg.hidden_size, generator=g) * w).to(dev, dt)
    with torch.no_grad():
        y_ref = ref(x).float().view(-1, x.shape[-1])
        y_srv = served(x.view(-1, x.shape[-1])).float()
        r = ref.gate(x)                                   # training router, eval mode (EMA centering)
        idx_ref = r.indices.view(-1, r.indices.shape[-1])[:, 0]
        idx_srv = served.router_logits(x.view(-1, x.shape[-1])).argmax(-1)
    agree = (idx_ref == idx_srv).float().mean().item()
    rel = ((y_ref - y_srv).norm(dim=-1) / y_ref.norm(dim=-1).clamp_min(1e-6))
    same = idx_ref == idx_srv
    rel_same = rel[same].mean().item() if same.any() else float("nan")
    load = torch.bincount(idx_srv, minlength=served.num_experts).tolist()
    layer_ok = agree >= 0.995 and rel_same < 0.02
    ok &= layer_ok
    print(f"layer {layer}: top-1 agreement {agree:.4f}, mean relative |dy| on agreeing tokens {rel_same:.2e}, "
          f"expert load (served) {load} -> {'ok' if layer_ok else 'FAIL'}")
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
