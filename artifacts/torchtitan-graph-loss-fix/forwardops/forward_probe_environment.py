import hashlib,importlib.metadata,json,pathlib,platform,torch
import transformers.models.qwen3.modeling_qwen3 as qwen
import torchtitan.experiments.graph_trainer.inductor_passes as passes
import specforge.training.torchtitan.numerics as numerics
root=pathlib.Path('/scratch/specforge-torchtitan-20261003')
files={'qwen3_modeling':pathlib.Path(qwen.__file__),'torchtitan_inductor_passes':pathlib.Path(passes.__file__),'specforge_numerics':pathlib.Path(numerics.__file__)}
for name in ('forward_component_probe.py','rms_precision_probe.py','context_gradient_grouping_probe.py','context_gradient_grouping_pointwise_probe.py'):
 files[name]=root/name
record={'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'device':torch.cuda.get_device_name(0),'packages':{key:importlib.metadata.version(key) for key in ('transformers','torchtitan','triton')},'source_base_commit':'c70a5f413220e6ce68247afb528664986502300f','source_state':'separate graph-numerics-tests source snapshot plus new numerics.py and convolution compile-route fix; no parameter getter cache used by isolated probes','scoped_compiler_policy':{'emulate_precision_casts':True,'eager_numerics.division_rounding':True},'float32_matmul_precision':torch.get_float32_matmul_precision(),'allow_tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'files':{key:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for key,path in files.items()}}
(root/'forward-probe-environment.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
