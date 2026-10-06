# Tuned fused-MoE kernel configs for MoE drafters

SGLang's fused MoE Triton kernel picks tile sizes from a per-shape JSON
(`E=<experts>,N=<expert width>,device_name=<GPU>.json`, keyed by token count
M). Without one it logs "Using default MoE kernel config. Performance might be
sub-optimal!" and, for the Qwen3.8-27B DFlash2 MoE drafter (16 experts of
width 4352), runs 1.7x slower than tuned at M = 64 and 1.6x at M = 256 on H200.

Point SGLang at this directory when serving an MoE drafter:

```bash
SGLANG_MOE_CONFIG_DIR=$(python -c "import specforge.serving.sglang_models.moe_configs as m, os; print(os.path.dirname(m.__file__))") \
SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang_models \
python -m sglang.launch_server ...
```

`SGLANG_MOE_CONFIG_DIR` replaces SGLang's own config directory, so with an MoE
*target* copy the file into SGLang's directory instead
(`sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_<version>/`) to
keep the target's configs.

Configs are produced with SGLang's `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`
(`--tune --tp-size 1 --dtype auto --batch-sizes 8 16 32 64 128 256 512`,
pointing `--model` at a config directory whose `config.json` declares
`architectures: ["Qwen3MoeForCausalLM"]`, `num_experts`, `num_experts_per_tok`,
`moe_intermediate_size` and `hidden_size` of the draft). The batch sizes are
the draft's token counts per step: block size x concurrency.

| file | drafter | tuned on |
|---|---|---|
| `configs/triton_3_7_1/E=16,N=4352,device_name=NVIDIA_H200.json` | `configs/qwen3.8-27b-dflash2-moe.json` | H200, triton 3.7.1, 762 s on 5 GPUs |
