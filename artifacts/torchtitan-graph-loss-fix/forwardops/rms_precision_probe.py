"""Untimed Qwen3 RMSNorm precision probe; no production code replacement."""
import json
from pathlib import Path
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from specforge.training.torchtitan.numerics import compiler_numerics

def error(a,b):
 d=a.double()-b.double()
 return dict(bitwise=bool(torch.equal(a,b)),different=int(torch.count_nonzero(d)),numel=d.numel(),max_abs=float(d.abs().max()),relative_l2=float(d.norm()/b.double().norm().clamp_min(1e-30)))
def variance(x):return x.float().square().mean(-1,keepdim=True)
def post(x,v):return (x.float()*torch.rsqrt(v+1e-6)).to(x.dtype)
records=[]
torch.manual_seed(2026)
for hidden,shape in [(2560,(1,8192,2560)),(128,(1,2048,32,128))]:
 x=torch.randn(shape,device='cuda',dtype=torch.bfloat16)
 reference_variance=x.double().square().mean(-1,keepdim=True)
 reference=x.double()*torch.rsqrt(reference_variance+1e-6)
 with compiler_numerics(),torch.no_grad():
  norm=Qwen3RMSNorm(hidden,eps=1e-6).to(device='cuda',dtype=torch.bfloat16)
  eager=norm(x);compiled=torch.compile(norm,fullgraph=True)(x)
  ev=variance(x);cv=torch.compile(variance,fullgraph=True)(x)
  epost=post(x,ev);cpost=torch.compile(post,fullgraph=True)(x,ev)
  cvar_epost=post(x,cv)
  fp32norm=Qwen3RMSNorm(hidden,eps=1e-6).cuda()
  eager32=fp32norm(x.float());compiled32=torch.compile(fp32norm,fullgraph=True)(x.float())
 row=dict(hidden_size=hidden,input_shape=list(shape),bf16_compiled_vs_eager=error(compiled,eager),bf16_eager_vs_rounded_fp64=error(eager,reference.bfloat16()),bf16_compiled_vs_rounded_fp64=error(compiled,reference.bfloat16()),variance_compiled_vs_eager=error(cv,ev),variance_eager_vs_fp64=error(ev,reference_variance),variance_compiled_vs_fp64=error(cv,reference_variance),post_compiled_vs_eager_with_same_variance=error(cpost,epost),eager_post_of_compiled_variance_vs_compiled_norm=error(cvar_epost,compiled),fp32_compiled_vs_eager=error(compiled32,eager32),fp32_eager_vs_fp64=error(eager32,reference),fp32_compiled_vs_fp64=error(compiled32,reference))
 records.append(row);print(json.dumps(row),flush=True)
 Path('/scratch/specforge-torchtitan-20261003/rms-precision-probe.json').write_text(json.dumps(records,indent=2)+'\n')
