import json, torch
from torch._dynamo.backends.registry import lookup_backend
from tests.test_modeling.test_dflash2_fused_conv import _random_conv,_prepare_finish_step
from specforge.modeling.draft import dflash2
records=[]
for dtype in (torch.float32,torch.bfloat16):
 conv=_random_conv(256,16,2,16,device='cuda',dtype=dtype,seed=11)
 torch.manual_seed(72)
 x=torch.randn(1,512,256,device='cuda',dtype=dtype);m=torch.randn_like(x)
 seeds=(torch.randn_like(x),torch.randn_like(x))
 eager=_prepare_finish_step(conv,x,m,seeds)
 conv.zero_grad(set_to_none=True);x=x.detach().clone().requires_grad_();m=m.detach().clone().requires_grad_()
 graphs=[]
 def backend(gm,inputs):
  graphs.append(str(gm));return lookup_backend('inductor')(gm,inputs)
 def forward(x,m):
  a,d=conv.prepare(x);return a,conv.finish(a*m,d)
 dflash2._load_fused_grouped_conv.cache_clear()
 fn=torch.compile(forward,backend=backend,fullgraph=True)
 a,b=fn(x,m);torch.autograd.backward((a,b),seeds)
 actual=dict(prepared=a,finished=b,grad_inputs=x.grad,grad_mixer=m.grad,grad_base_kernel=conv.base_kernel.grad,grad_kernel_projection=conv.kernel_projection.weight.grad)
 row={'dtype':str(dtype),'triton_operators':sum(s.count('triton_kernel_wrapper_mutation') for s in graphs),'errors':{}}
 for k in eager:
  diff=actual[k].double()-eager[k].double();row['errors'][k]={'bitwise':bool(torch.equal(actual[k],eager[k])),'relative_l2':(diff.norm()/eager[k].double().norm()).item(),'max_abs':diff.abs().max().item()}
 records.append(row)
 print(json.dumps(row),flush=True)
open('/scratch/specforge-torchtitan-20261003/conv-module-probe-'+torch.__version__.split('+')[0]+'.json','w').write(json.dumps(records,indent=2)+'\n')
