import json,re
from pathlib import Path
import torch
from safetensors.torch import load_file
root=Path('/scratch/specforge-torchtitan-20261003/graph-probes')
def state(name):
    result={}
    for p in (root/name/'draft').glob('*.safetensors'): result.update(load_file(p))
    return result
def losses(name):
    vals=re.findall(r'loss:\s*([0-9.]+)',(root/(name+'.log')).read_text())
    return [float(v) for v in vals[::2]]
report={}
for algorithm in ['dflash','dflash2','dspark']:
    a=state('graph-final-'+algorithm);b=state('graph-oracle-'+algorithm)
    assert a.keys()==b.keys()
    aa=torch.cat([v.flatten().float() for k,v in sorted(a.items())]);bb=torch.cat([v.flatten().float() for k,v in sorted(b.items())])
    report[algorithm]={'export_keys':len(a),'all_weights_bitwise_equal':bool(torch.equal(aa,bb)),'max_weight_abs_difference':(aa-bb).abs().max().item(),'relative_weight_l2_difference':((aa-bb).norm()/bb.norm()).item(),'optimized_losses':losses('graph-final-'+algorithm),'unoptimized_losses':losses('graph-oracle-'+algorithm)}
a=state('graph-full-inductor');b=state('graph-final-dflash2')
aa=torch.cat([v.flatten().float() for k,v in sorted(a.items())]);bb=torch.cat([v.flatten().float() for k,v in sorted(b.items())])
report['full_vs_regional_dflash2']={'max_weight_abs_difference':(aa-bb).abs().max().item(),'relative_weight_l2_difference':((aa-bb).norm()/bb.norm()).item(),'full_losses':losses('graph-full-inductor'),'regional_losses':losses('graph-final-dflash2')}
(root/'graph-oracle-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
