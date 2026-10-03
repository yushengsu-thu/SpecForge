import dataclasses
import os
import sys
import torch
import yaml
from pathlib import Path
from specforge.training.torchtitan import frontend
from specforge.training.torchtitan.graph import SpecForgeGraphTrainer, SpecForgeGraphModel
from specforge.training.torchtitan.graph_parallelize import parallelize_graph_dflash
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from specforge.training.torchtitan.runtime import SpecForgeTitanTrainer
original=frontend._native_config
mode=os.environ.get('GRAPH_PROBE','graph')
def factory(*args,**kwargs):
    config=original(*args,**kwargs)
    config.dataloader.pad_to_seq_len=True
    if mode=='graph':
        values={f.name:getattr(config,f.name) for f in dataclasses.fields(config)}
        values['compile']=GraphTrainerCompileConfig(enable=True,components=['model'],mode='aot_fx_trace',enable_passes=os.environ.get('GRAPH_PASSES','0')=='1',inductor_compilation=os.environ.get('GRAPH_INDUCTOR','regional'))
        model_values={f.name:getattr(config.model_spec.model,f.name) for f in dataclasses.fields(config.model_spec.model)}
        values['model_spec']=dataclasses.replace(config.model_spec,model=SpecForgeGraphModel.Config(**model_values),parallelize_fn=parallelize_graph_dflash)
        config=SpecForgeGraphTrainer.Config(**values)
    else:
        config.training.disable_cuda_graphs=False
    return config
frontend._native_config=factory
from specforge.cli import main
raise SystemExit(main(sys.argv[1:]))
