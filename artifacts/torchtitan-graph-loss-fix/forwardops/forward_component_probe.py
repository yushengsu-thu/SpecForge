"""Untimed isolated forward arithmetic probe for native/full-graph residuals."""
import json
from pathlib import Path
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torchtitan.experiments.graph_trainer.inductor_passes import full_inductor_compilation_pass
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm,Qwen3MLP,Qwen3RotaryEmbedding
from specforge.training.torchtitan.numerics import compiler_numerics

root=Path('/scratch/specforge-torchtitan-20261003')
torch.manual_seed(2026)
config=Qwen3Config(hidden_size=2560,intermediate_size=9728,num_attention_heads=32,num_key_value_heads=8,head_dim=128,rope_theta=1000000.0,rms_norm_eps=1e-6)
rows=[]
def error(found,wanted):
 d=found.double()-wanted.double()
 return dict(bitwise=bool(torch.equal(found,wanted)),different=int(torch.count_nonzero(d)),numel=d.numel(),max_abs=float(d.abs().max()),relative_l2=float(d.norm()/wanted.double().norm().clamp_min(1e-30)))
def probe(name,fn,args):
 with compiler_numerics(),torch.no_grad():
  expected=fn(*args)
  expected=(expected,) if isinstance(expected,torch.Tensor) else expected
  native=torch.compile(fn,fullgraph=True)(*args)
  native=(native,) if isinstance(native,torch.Tensor) else native
  def forward(*values):
   return fn(*values)
  gm=make_fx(forward,tracing_mode='fake',_allow_non_fake_inputs=True)(*args)
  full=full_inductor_compilation_pass(gm,args)(*args)
  full=(full,) if isinstance(full,torch.Tensor) else full
 row=dict(component=name,native_compile_vs_eager=[error(a,b) for a,b in zip(native,expected)],full_inductor_vs_eager=[error(a,b) for a,b in zip(full,expected)],full_vs_native=[error(a,b) for a,b in zip(full,native)])
 rows.append(row);print(json.dumps(row),flush=True)
 (root/'forward-component-probe.json').write_text(json.dumps(rows,indent=2)+'\n')

for hidden,shape in [(2560,(1,8192,2560)),(128,(1,2048,32,128))]:
 x=torch.randn(shape,device='cuda',dtype=torch.bfloat16)
 norm=Qwen3RMSNorm(hidden,eps=1e-6).to(device='cuda',dtype=torch.bfloat16)
 probe('rmsnorm_'+str(hidden),norm,(x,))

x=torch.randn(1,512,2560,device='cuda',dtype=torch.bfloat16)
mlp=Qwen3MLP(config).to(device='cuda',dtype=torch.bfloat16)
probe('mlp',mlp,(x,))
rotary=Qwen3RotaryEmbedding(config=config,device='cuda')
positions=torch.arange(12288,device='cuda')[None,:]
probe('rotary_embedding',rotary,(x,positions))
y=torch.linspace(-10,10,65536,device='cuda').to(torch.bfloat16)
probe('silu',torch.nn.functional.silu,(y,))
freq=positions.float()[:,:,None]*rotary.inv_freq[None,None,:]
probe('sin_bf16',lambda t:t.sin().to(torch.bfloat16),(freq,))
probe('cos_bf16',lambda t:t.cos().to(torch.bfloat16),(freq,))
