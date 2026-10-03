# DFlash2 convolution correctness probes

These untimed H200 probes isolate the DFlash2 convolution path. They do not establish native FSDP2 versus GraphTrainer end-to-end equivalence or model convergence.

The production change removes the `torch.compiler.is_compiling()` fallback in `DFlashGroupedConv._fused_convolution`. Device, dtype, environment and group-size gates are unchanged. Eager, Dynamo and make_fx can therefore execute the existing fused convolution with FP32 accumulation.

## Findings

- At BF16 H2560 with 8192 rows and randomized trained-like kernel values, the previous compiled ATen route differs from eager fused convolution: forward relative L2 error 0.00277955 and input-gradient error 0.00277815. Compiling the fused route makes these two tensors bitwise equal. Direct fused parameter reductions retain small differences at this large shape (delta gradient relative L2 9.96e-6; base gradient 9.53e-9).
- In the complete prepare → multiply mixer → finish probe at H256, default Inductor produces bitwise-equal forward outputs but input/base/projection BF16 gradient errors of approximately 0.00347/0.00247/0.00297 relative L2.
- Applying both `emulate_precision_casts=True` and `eager_numerics.division_rounding=True` for first trace and backward removes all BF16 discrepancies in that module probe: both outputs and all four gradient groups are bitwise equal. Recursive Dynamo inspection finds four Triton wrapper operations. FP32 is bitwise equal except the base-kernel gradient (relative L2 6.84e-8, max absolute error 1.53e-5).
- The complete convolution test suite passes on Torch 2.14 and Torch 2.13: 12 tests plus 295 subtests on each. The new regression checks a cold lazy loader, actual nested Triton operations, exact forward outputs and dtype-bounded gradients under default compilation.

## Files and reproduction

`conv_numerical_probe.py` compares direct ATen/fused eager and compiled convolution against FP64 arithmetic over the same quantized inputs. The default script is the Torch 2.14 run; the legacy direct-kernel result records only the H256 cases.

`conv_module_probe.py` checks the full prepare/mixer/finish module under the default compiler policy. `conv_module_probe_strict.py` applies the two scoped compiler settings above. Their JSON records are also printed in the corresponding raw logs. Initial strict execution used the default output filename by mistake; the original default and strict JSON arrays were reconstructed separately from their intact respective logs, and the retained strict script now uses its own output filename. No numerical rerun or data modification was used for that correction.

Equivalent reproduction commands from the mutable remote checkout (set `PYTHONPATH` to it and use the corresponding venv interpreter):

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /scratch/specforge-torchtitan-20261003/venv/bin/python /scratch/specforge-torchtitan-20261003/conv_module_probe_strict.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /scratch/specforge-torchtitan-20261003/venv/bin/python -m pytest tests/test_modeling/test_dflash2_fused_conv.py -q
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /scratch/specforge-torchtitan-20261003/venv213/bin/python -m pytest tests/test_modeling/test_dflash2_fused_conv.py -q
```

`source-manifest.json` pins the source files used by this fix. Existing full GraphTrainer loss-trajectory differences require separate full-model verification; these module results alone do not prove their cause or resolution.
