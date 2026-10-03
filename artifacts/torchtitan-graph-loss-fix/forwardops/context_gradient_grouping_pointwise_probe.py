"""Untimed counterexample for BF16 shared-activation block backward grouping."""
import json
from pathlib import Path
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torchtitan.experiments.graph_trainer.inductor_passes import full_inductor_compilation_pass
from specforge.training.torchtitan.numerics import compiler_numerics

torch.manual_seed(2026)
x=torch.randn(16,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
weights=tuple(torch.randn(16,32,device='cuda',dtype=torch.bfloat16) for _ in range(4))
seeds=tuple(torch.randn_like(x) for _ in range(2))
def block(x,a,b):return x*a+x*b
def compute(call,x):
 a=call(x,*weights[:2]);b=call(x,*weights[2:])
 (gradient,)=torch.autograd.grad((a,b),(x,),seeds)
 return a,b,gradient
def errors(found,wanted):
 d=found.double()-wanted.double()
 return dict(bitwise=bool(torch.equal(found,wanted)),different=int(torch.count_nonzero(d)),numel=d.numel(),max_abs=float(d.abs().max()),relative_l2=float(d.norm()/wanted.double().norm()))
with compiler_numerics():
 eager=compute(block,x)
 compiled=compute(torch.compile(block,fullgraph=True),x)
 def joint(x):return compute(block,x)
 gm=make_fx(joint,tracing_mode='fake',_allow_non_fake_inputs=True)(x)
 full=full_inductor_compilation_pass(gm,(x,))(x)
 grouped=[]
 for index in range(2):
  y=block(x,*weights[2*index:2*index+2])
  grouped.append(torch.autograd.grad(y,(x,),seeds[index])[0])
 manual=grouped[0]+grouped[1]
row=dict(native_block_vs_eager=[errors(a,b) for a,b in zip(compiled,eager)],full_joint_vs_eager=[errors(a,b) for a,b in zip(full,eager)],manual_grouped_vs_native=errors(manual,compiled[-1]),manual_grouped_vs_eager=errors(manual,eager[-1]))
print(json.dumps(row,indent=2),flush=True)
Path('/scratch/specforge-torchtitan-20261003/context-gradient-grouping-pointwise-probe.json').write_text(json.dumps(row,indent=2)+'\n')
