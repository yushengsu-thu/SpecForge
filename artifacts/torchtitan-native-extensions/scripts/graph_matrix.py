import copy,json,os,signal,subprocess,sys,time
from pathlib import Path
import yaml
root=Path('/scratch/specforge-torchtitan-20261003')
out=root/'graph-probes'
base=yaml.safe_load((out/'graph-fused-dp1.yaml').read_text())
base['deployment']['trainer']['nproc_per_node']=2
base['training']['num_epochs']=3
cases=[]
for algo in ['dflash2','dflash','dspark']:
    cfg=copy.deepcopy(base)
    if algo!='dflash2':
        original=yaml.safe_load((root/'native-pp-smoke'/f'{algo}.yaml').read_text())
        cfg['model']=original['model']
        cfg['data']=original['data']
        cfg['training']['strategy']=original['training']['strategy']
        cfg['training']['dflash_teacher_metrics']=original['training']['dflash_teacher_metrics']
        cfg['training']['dflash2_selector_ramp_ratio']=0.0
    cfg['run_id']=f'graph-final-{algo}'
    cases.append((cfg['run_id'],cfg))
for name,steps in [('graph-cut',1),('graph-resume',3)]:
    cfg=copy.deepcopy(base);cfg['run_id']=name;cfg['training']['max_steps']=steps;cfg['training']['save_interval']=steps
    if name=='graph-resume': cfg['training']['resume_from']=str(out/'graph-cut'/'checkpoint'/'step-1')
    cases.append((name,cfg))
status=[]
env=os.environ|{'CUDA_VISIBLE_DEVICES':'2,3','PYTHONPATH':str(root/'SpecForge-titan-extensions')}
for name,cfg in cases:
    (out/f'{name}.yaml').write_text(yaml.safe_dump(cfg))
    cmd=[str(root/'venv/bin/python'),'-m','torch.distributed.run','--standalone','--nproc-per-node=2','-m','specforge.cli','train','--config',str(out/f'{name}.yaml')]
    start=time.time()
    with (out/f'{name}.log').open('w') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=root/'SpecForge-titan-extensions',start_new_session=True)
        try: rc=p.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30);rc=124
    status.append({'case':name,'exit_code':rc,'seconds':time.time()-start})
    (out/'matrix-status.json').write_text(json.dumps(status,indent=2))
    print(status[-1],flush=True)
    if rc:
        print((out/f'{name}.log').read_text()[-7000:],flush=True)
        break
