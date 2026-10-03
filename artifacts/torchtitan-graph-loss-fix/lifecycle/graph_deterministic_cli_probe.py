"""Diagnostic-only activation of native TorchTitan's deterministic debug mode."""
import runpy
from specforge.training.torchtitan import frontend

original = frontend._native_config

def build(*args, **kwargs):
    result = original(*args, **kwargs)
    result.debug.deterministic = True
    return result

frontend._native_config = build
runpy.run_module('specforge.cli', run_name='__main__')
