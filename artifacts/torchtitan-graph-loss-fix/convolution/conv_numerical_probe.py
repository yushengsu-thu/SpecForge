"""Isolate DFlash2 convolution math; no performance measurements."""
import json, os, traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from specforge.modeling.draft.dflash2_conv_triton import dflash2_grouped_conv_fused

OUTPUT=Path(os.environ.get('CONV_PROBE_OUTPUT','/scratch/specforge-torchtitan-20261003/conv-numerical-probe.json'))

def aten(x,d,b):
    n,s,h=x.shape; taps=b.shape[0]; groups=h//16
    z=x.reshape(n,s//16,16,groups,16)
    coef=b.reshape(1,1,1,taps,groups,16)+d.reshape(n,s//16,16,taps,groups).unsqueeze(-1)
    out=coef[:,:,:,0]*z
    for tap in range(1,taps):
        shifted=F.pad(z[:,:,:16-tap],(0,0,0,0,tap,0))
        out=out+coef[:,:,:,tap]*shifted
    return out.reshape_as(x)

def fused(x,d,b): return dflash2_grouped_conv_fused(x,d,b,16,16)
def step(fn,values,grad):
    values=[v.detach().clone().requires_grad_(True) for v in values]
    out=fn(*values)
    return [out.detach(), *torch.autograd.grad(out,values,grad)]
def err(a,b):
    dif=a.double()-b.double()
    return {'max_abs':dif.abs().max().item(),'relative_l2':(dif.norm()/b.double().norm().clamp_min(1e-30)).item(),'bitwise':bool(torch.equal(a,b)),'different_elements':int((a!=b).sum()),'elements':a.numel()}
result={'torch':torch.__version__,'cases':[]}
for hidden,rows in ([(256,512)] if os.environ.get("CONV_PROBE_SMALL") else [(256,512),(2560,8192)]):
    for initial in [True,False]:
        torch.manual_seed(31)
        values=[torch.randn(1,rows,hidden,device='cuda',dtype=torch.bfloat16),torch.randn(1,rows,2,hidden//16,device='cuda',dtype=torch.bfloat16)*0.02,torch.randn(2,hidden,device='cuda',dtype=torch.bfloat16)*0.02]
        if initial:
            values[1].zero_();values[2].zero_();values[2][0]=1
        grad=torch.randn_like(values[0]);row={'hidden':hidden,'rows':rows,'initial_identity':initial,'comparisons':{}}
        reference=step(aten,[v.double() for v in values],grad.double())
        eager_fused=step(fused,values,grad)
        for mode,fn in [('eager_aten',aten),('compiled_aten',torch.compile(aten,fullgraph=True)),('compiled_fused',torch.compile(fused,fullgraph=True))]:
            try:
                found=step(fn,values,grad)
                row['comparisons'][mode]={'against_fused':dict(zip(['forward','grad_x','grad_delta','grad_base'],[err(a,b) for a,b in zip(found,eager_fused)])),'against_fp64':dict(zip(['forward','grad_x','grad_delta','grad_base'],[err(a,b) for a,b in zip(found,reference)]))}
            except Exception as exc:
                row['comparisons'][mode]={'error':str(exc),'traceback':traceback.format_exc()}
        row['fused_against_fp64']=dict(zip(['forward','grad_x','grad_delta','grad_base'],[err(a,b) for a,b in zip(eager_fused,reference)]))
        result['cases'].append(row);OUTPUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row),flush=True)
