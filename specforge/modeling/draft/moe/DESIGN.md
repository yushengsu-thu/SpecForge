# Draft MoE FFNs

This package adds an FFN to a draft model. It does not implement attention,
residual connections, target-feature capture, draft heads, or expert parallelism.
The initial supported family is Qwen3-Next / Qwen3.5-MoE / Qwen3.8, selected by
`moe_preset: "qwen3_5_moe"`. Model dimensions remain explicit overrides.

## Files and contracts

```text
draft/moe/
  __init__.py  # small public construction/lifecycle API
  config.py    # validated MoEConfig, preset resolution, build_ffn
  routing.py   # RoutingResult, router scores/top-k, auxiliary balance
  experts.py   # SwiGLU, shared expert, fused routed weights, local dispatch
  layer.py     # compose one FFN; collect/clear differentiable auxiliary state
  layout.py    # complete-checkpoint schema, independent of local FSDP shards
  DESIGN.md
```

`MoELayer(MoEConfig, hidden_size)` accepts and returns the same tensor shape
`[..., hidden_size]`. `RoutingResult` carries `[tokens, top_k]` expert indices
and weights and `[tokens, experts]` FP32 probabilities. Dispatch consumes that
contract; it does not calculate a second router. `MoEConfig` is immutable and
validated even when constructed directly. There is one implementation per
responsibility, without a registry or an abstract class for each local option.

One token follows this path:

```text
x ── gate.weight ── FP32 softmax ── top-k ── renormalize ── weights
│                                      │                     │
├── selected experts: down(silu(gate(x)) * up(x)) ──────────── sum
│                                                            │
└── shared expert(x) * sigmoid(shared_expert_gate(x)) ───────── + ── y

full softmax probabilities + detached top-k counts ── aux balance
```

The router gate, expert SwiGLU gate projection, and shared-expert gate are three
different operations. Qwen uses softmax, normalized selected weights, one
sigmoid-gated shared expert, and no SwiGLU clamp. The optional clamp seam caps
the activated gate above and the up branch on both sides; it is disabled in the
preset. Other scoring/balancing families are rejected until implemented.

## Configuration

```json
{
  "moe_preset": "qwen3_5_moe",
  "n_routed_experts": 16,
  "num_experts_per_tok": 2,
  "moe_intermediate_size": 512,
  "shared_expert_intermediate_size": 1024,
  "dflash_config": {
    "moe_aux_loss_coeff": 0.001,
    "moe_dispatch": "grouped_mm"
  }
}
```

The preset defaults are 512 experts, top-10, routed width 512 and shared width
512. Top-level fields override them; these defaults and the HF `num_experts`
alias are materialized into the saved config. Conflicting aliases, an unknown
preset, unknown `dflash_config.moe_*` controls, invalid dimensions and expert
counts without a preset fail early. Without a preset or expert count,
`build_ffn` calls the existing dense factory with the original config.

`dflash_config.moe_router_center` defaults to `"none"`. The optional `"sample"`
extension centers router inputs across tokens independently for each sample;
experts and the shared expert still receive original activations. It accepts
`[tokens, hidden]` (one sample) or `[batch, tokens, hidden]`. This changes routing
and must be preserved when loading the draft for inference. It is a SpecForge
extension, not part of the official Qwen router or a different aux-loss formula.
An external serving loader must implement this extension to serve a
sample-centered checkpoint faithfully; native weight names alone do not ensure
that behavior. The checked-in GPU example uses `"none"`.

`reference` dispatch is the default and runs on CPU or GPU. `grouped_mm` uses
PyTorch grouped matrix multiplication and requires CUDA BF16 inputs and expert
parameters. It sorts token/expert assignments, inserts zero-weight dummy rows
for empty groups, runs two grouped GEMMs, and scatters weighted expert results
back to tokens. It never silently falls back to a different dispatcher.

## Draft and trainer integration

DFlash's decoder changes its FFN construction to:

```python
self.mlp = build_ffn(config, dense=kernels.make_mlp)
```

DFlash2, DSpark and Domino inherit this decoder construction. The model's
`_init_weights` calls `MoELayer.reset_parameters(initializer_range)` so fused
expert parameters receive initialization as well as `nn.Linear` modules.
At the start of a training forward, outside decoder checkpoint regions,
`apply_pending_balance_updates(model)` begins a new auxiliary collection window.

A drafter with its own blocks can use the same three boundaries directly:

```python
self.ffn = MoELayer(resolve_moe_config(config), hidden_size)
# beginning of model training forward:
apply_pending_balance_updates(self)
# model _init_weights when isinstance(module, MoELayer):
module.reset_parameters(std)
```

The common trainer calls `finalize_step_output` once after strategy forward.
It drains every layer's coefficient-weighted auxiliary loss, sums the layers,
and adds detached `moe/*` telemetry. Strategies contain no MoE-specific code.
For additive task terms `(numerator, denominator)`, the combined numerator is
`numerator + denominator.detach() * auxiliary_loss`. Existing accumulation and
data-parallel denominator normalization therefore apply to the full objective.
This weights each local auxiliary mean by the task denominator; it does not
compute a global-batch router statistic.

The auxiliary formula is `E * sum(count_e / T * mean_probability_e)`, where
counts include all top-k selections. Its uniform baseline is `k`, matching
Transformers' Qwen convention. It currently includes every presented draft query
slot, including clean anchors and any padded anchor slots. It does not mask
queries with the task loss mask. Router probabilities retain their gradients;
assignment counts and reported metrics are detached.

`moe/*` scalar telemetry follows the existing trainer's convention: the latest
microbatch, averaged across data-parallel ranks at an optimizer boundary.
The combined objective itself is normalized over the full accumulation window.

Each layer holds one forward's auxiliary graph. Finalization drains it before
backward and disables storage during checkpoint recomputation; tensor work in
the recomputation remains identical. Draft MoE uses **non-reentrant** activation
checkpointing; DFlash-derived models reject an explicit reentrant request.
Custom block implementations must use the same non-reentrant contract.
Qwen balancing has
no selection-bias update, so the pending-update boundary currently only resets
collection. A future bias-balancing implementation must apply pending mutations
at this boundary, never inside checkpointed layer forward.

## Checkpoint layout

The supported family uses official Qwen relative names **internally**:

```text
gate.weight                         [E, H]
experts.gate_up_proj                 [E, 2I, H]
experts.down_proj                    [E, H, I]
shared_expert.gate_proj.weight       [Is, H]
shared_expert.up_proj.weight         [Is, H]
shared_expert.down_proj.weight       [H, Is]
shared_expert_gate.weight            [1, H]
```

The enclosing drafter owns its prefix, e.g. `layers.0.mlp.`. Layout conversion
is an identity for Qwen. This makes FSDP full state, resume, warm start,
`from_pretrained` (including meta loading), `materialize_draft`, and HF export
use the same tensors without shard-sensitive state-dict hooks. HF export
validates complete fused tensors after overlaying original checkpoint precision;
the export normalizer persists resolved structural config values.

## Future families

Add a family module only for genuinely family-specific behavior or external
format. Keep shared routing/dispatch/FFN operations here:

| Difference | Extension boundary |
| --- | --- |
| Qwen3 per-expert checkpoint tensors, no shared expert | Full-tensor layout adapter and preset |
| Different score, top-k scaling or selection bias | Router / RoutingResult; score used for selection can differ from mixing weights |
| Bias-based balance | Detached pending counts, applied once at model-forward boundary |
| Different SwiGLU clamp or shared gate | Expert activation / shared-expert composition |
| DeepSeek `mtp.N.ffn.*`, quantized expert weights | Explicit full-checkpoint layout and precision adapter |
| MLA, mHC, hyper-connections, Markov/confidence heads | Enclosing draft block/model, outside this FFN package |

DeepSeek-V4 DSpark's official blocks and quantization are not implemented by
this change. There is no claim that choosing another preset can load them.

## Verification commands

```bash
python -m pytest -q tests/test_modeling/test_draft_moe.py \
  tests/test_training/test_moe_objective.py \
  tests/test_runtime/test_draft_moe_checkpoint.py

torchrun --standalone --nproc-per-node=8 \
  tests/test_runtime/moe_distributed_gate.py --report /tmp/moe-fsdp.json
```

The distributed gate exercises both FSDP sharding modes with checkpointing on
and off, two microbatches per update, nonzero router/expert gradients, and
parameter updates. The checked-in offline YAML is the separate real-teacher
training entry point. These are correctness gates, not convergence or serving
acceptance benchmarks.

Official references inspected independently of other SpecForge MoE work:

- [Transformers Qwen3-Next implementation](https://github.com/huggingface/transformers/blob/3bb8b72786c7cb5abc9bd05c2e1f451e44dcc9b3/src/transformers/models/qwen3_next/modeling_qwen3_next.py)
- [Qwen3.8 Flash Next config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)
- [Qwen3.8 Flash Next weight index](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/model.safetensors.index.json)
- [DeepSeek-V4-Pro-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark/tree/7c09739fd136abfb70a49ec334157f65f45b52cd)
