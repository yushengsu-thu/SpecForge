# Isolated compiler arithmetic evidence

These are untimed H200 probes of the real Qwen3 modules and TorchTitan 0.3.0 full Inductor pass. They isolate arithmetic behavior; they do not prove that every full-model loss difference is explained or that two training trajectories have equivalent convergence. No production normalization or attention implementation was changed for these probes.

All probes use the production `compiler_numerics()` scope before tracing and compilation: `emulate_precision_casts=True` and `eager_numerics.division_rounding=True`. Package versions, source hashes, and the captured helper are retained alongside the scripts. The source was a separate test snapshot based on c70a5f4 with the compiler helper and convolution-route fix; these arithmetic probes do not use SimpleFSDP or its subsequently added parameter cache.

## RMSNorm: causally isolated FP32 reduction rounding

At BF16 shape `[1,8192,2560]`, Qwen3 RMSNorm eager versus compiled output differs in **60 of 20,971,520** elements (relative L2 9.62888e-6, maximum absolute difference 0.015625). Native `torch.compile` and the actual full Inductor pass produce **bitwise-identical** outputs to each other.

The H2560 variance reduction differs eager versus compiled by at most **2.38419e-7**, relative L2 **6.89973e-8**. Two counterfactual checks identify the cause in this fixture:

1. Supplying the *same* variance to eager and compiled reciprocal-square-root/multiply gives bitwise-identical BF16 results.
2. Supplying the compiled variance to eager postprocessing reproduces the compiled full RMSNorm output bitwise.

Thus all 60 BF16 output differences in this H2560 fixture arise from the FP32 mean reduction, not an altered normalization formula or the BF16 postprocessing.

An FP64 reference over the same quantized inputs shows comparable eager and compiled accuracy:

| H2560 quantity | Eager maximum absolute error vs FP64 | Compiled maximum absolute error vs FP64 |
| --- | ---: | ---: |
| Variance | 1.84684e-7 | 1.83303e-7 |
| FP32 normalized output | 7.39776e-7 | 7.26524e-7 |

Eager versus compiled FP32 normalized output differs by at most 9.53674e-7, relative L2 4.79416e-8. Casting values close to BF16 rounding boundaries accounts for a small set of larger one-ULP BF16 differences. Neither implementation is an exact FP64 oracle: after rounding the FP64 normalization to BF16, eager differs in 99 elements and compiled in 121.

For head-dimension128 at shape `[1,2048,32,128]`, 13 of 8,388,608 BF16 outputs differ (relative L2 7.63870e-6). FP32 output maximum error against FP64 is 7.00775e-7 for both modes. Its standalone variance and fused full norm use different reduction schedules, so the H2560 counterfactual identity should not be generalized to this case without qualification.

The accompanying component sweep finds bitwise equality across eager/native compilation/full Inductor for the tested MLP, RoPE cos/sin outputs, SiLU, and standalone BF16 sin/cos. This is not an exhaustive attention or whole-model equivalence test.

## BF16 shared-gradient grouping, independently minimized

`context_gradient_grouping_pointwise_probe.py` has two small pointwise blocks. Each reads one shared BF16 activation twice, with separate multiplication coefficients; there are no norms, matmuls, attention, sharding or optimizer steps.

- Both forward outputs are bitwise identical in eager, individually compiled blocks, and a fully compiled joint graph.
- The full-joint input gradient is bitwise identical to eager.
- Individually compiled blocks differ from eager in 150/512 shared-input gradient values, relative L2 0.00263265, maximum absolute difference 0.03125.
- Computing one eager BF16 gradient per block and then adding those two gives **bitwise-identical** results to block compilation.

This proves a BF16 gradient difference caused solely by backward accumulation grouping at block compilation boundaries. It does not establish the exact accumulation order in the actual FSDP2 model or explain the separate native-eager versus raw-Graph residual; those require the whole-model first-update captures collected separately.

The older `context_gradient_grouping_probe.py` uses matmuls. It additionally changes forward results under compilation, so it does **not** isolate backward grouping. It is retained as a clearly labeled observation: strict precision settings do not guarantee bitwise equality for arbitrary GEMM/add fusion. The actual fusion causing that toy's forward change was not established here, and no corresponding DFlash2 adapter defect is claimed.

## Commands and records

The four probe scripts were executed from `/scratch/specforge-torchtitan-20261003/graph-numerics-tests`, each using this exact command shape, with its corresponding script basename and log basename:

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /scratch/specforge-torchtitan-20261003/venv/bin/python /scratch/specforge-torchtitan-20261003/forward_component_probe.py > /scratch/specforge-torchtitan-20261003/forward-component-probe.log 2>&1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /scratch/specforge-torchtitan-20261003/venv/bin/python /scratch/specforge-torchtitan-20261003/rms_precision_probe.py > /scratch/specforge-torchtitan-20261003/rms-precision-probe.log 2>&1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /scratch/specforge-torchtitan-20261003/venv/bin/python /scratch/specforge-torchtitan-20261003/context_gradient_grouping_probe.py > /scratch/specforge-torchtitan-20261003/context-gradient-grouping-probe.log 2>&1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /scratch/specforge-torchtitan-20261003/venv/bin/python /scratch/specforge-torchtitan-20261003/context_gradient_grouping_pointwise_probe.py > /scratch/specforge-torchtitan-20261003/context-gradient-grouping-pointwise-probe.log 2>&1
```

All four completed with exit0. JSON metrics and intact raw logs are retained. `forward_probe_environment.py` records versions and hashes without dumping environment variables or credentials. No checkpoints or model weights are included.
