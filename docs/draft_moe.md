# Draft MoE FFNs

`moe_preset: qwen3_5_moe` replaces only the DFlash-family decoder FFN.
DFlash, DFlash2, Domino, and DSpark share this boundary. Attention, target
feature projection, heads, and objectives retain their architecture ownership.
This is an FFN preset, not a complete Qwen teacher architecture or a serving
backend. Omitting the preset calls the existing dense kernel factory unchanged;
`n_routed_experts` without a preset is an error.

## Configuration and math

Declare topology at the top level of the draft JSON:

```json
{
  "moe_preset": "qwen3_5_moe",
  "n_routed_experts": 512,
  "num_experts_per_tok": 10,
  "moe_intermediate_size": 512,
  "shared_expert_intermediate_size": 2048,
  "norm_topk_prob": true,
  "dflash_config": {
    "moe_aux_loss_coeff": 0.001,
    "moe_dispatch": "grouped_mm",
    "moe_router_center": "none"
  }
}
```

All four topology dimensions are required and independent of teacher size.
`num_experts`, if also present, must agree with `n_routed_experts`. The preset
uses FP32 softmax over all experts, top-k selection, and normalized top-k weights
by default. `norm_topk_prob: false` exposes the optional Qwen3-Next/Qwen3.8
behavior; Qwen3.5 uses normalization. Selected SwiGLU expert outputs are weighted
and summed. One shared SwiGLU output is multiplied by
`sigmoid(shared_expert_gate(x))` and added to the routed result. There is no
capacity dropping or expert parallelism: every rank owns its local FFN token
dispatch and the existing backend shards/synchronizes parameters and gradients.

`moe_router_center` currently accepts only `none`, the official behavior. The
meaning of `sample` is unspecified and rejected explicitly. No routing-bias
update or centering algorithm is inferred from that name. Optional top-level
`swiglu_limit` applies `gate.clamp(max=limit)` and
`up.clamp(-limit, limit)` before `silu(gate) * up`; its default is null, matching
Qwen. It is an explicit math override, not DeepSeek model support.

`reference` dispatch is portable PyTorch and supports autograd/BF16. Explicit
`grouped_mm` uses `torch.nn.functional.grouped_mm` with sorted assignments and
int32 cumulative end offsets, including empty experts. It requires CUDA SM80+,
BF16 expert parameters, hidden/intermediate dimensions divisible by eight, and
fewer than 1024 experts. Unsupported environments fail with a reference-backend
instruction. Grouped dispatch is an execution choice, not a new checkpoint
layout or an implicit speed guarantee.

## Auxiliary objective and checkpointing

Each layer computes `E * sum_e (count_e / T) * mean_t(probability_te)`, counting
all top-k assignments; uniform routing has value `top_k`. This follows Qwen's
auxiliary formula per layer. Counts are detached; probabilities retain the
router gradient. `T` includes every executed FFN position, including anchor,
MASK, and padded draft query slots. The FFN currently receives no validity mask.
There is no claim that this equals balancing only supervised or unpadded tokens.

`finalize_step_output(model, output)` consumes each layer's differentiable
auxiliary value once and adds its configured coefficient to the real training
loss. It is called by the common trainer and evaluator, with no MoE logic in
individual strategies. With additive `loss_terms=(numerator, denominator)`, it
adds `auxiliary * denominator.detach()` to the numerator. The existing global
optimizer-window denominator therefore scales both objectives together across
accumulation and ranks. Per-layer `moe/*` entropy, activity, and raw auxiliary
metrics use the same objective-denominator weighting; `moe/aux_loss` is the sum
of coefficient-weighted layer losses.

`apply_pending_balance_updates(model)` runs at draft-model forward entry,
outside decoder checkpoint regions. Qwen has no pending bias update; the
boundary resets forward-local records. Recompute performs the same tensor math
but publishes no second record after finalization. Model checkpointing defaults
to non-reentrant mode for MoE. Reentrant mode is rejected because a side-channel
auxiliary graph from its initial no-grad forward cannot supply the required
gradient. Direct users must finalize before backward and use non-reentrant
checkpointing; the trainer handles both automatically. Resume metadata records
resolved MoE topology and math, including coefficient/clamp/normalization;
dispatch can change without invalidating optimizer state.

The FFN output also carries a zero-valued dependency on its auxiliary graph.
Backward through that graph waits for the wrapped hidden-output edge, allowing
FSDP's output pre-backward hooks to unshard parameters before auxiliary backward.
This is a lifecycle dependency; the explicit finalizer term supplies the
auxiliary value and gradient coefficient.

## Files and checkpoint layout

```text
specforge/modeling/draft/moe/
  config.py       MoEConfig and strict resolve_moe_config
  layer.py        MoELayer, Qwen _route seam, shared-path combination
  experts.py      PackedExperts, SharedExpert, swiglu and execution backends
  checkpoint.py   original Next conversion and canonical export keys
  runtime.py      forward records, finalize_step_output, resume contract
  __init__.py     build_ffn and public imports
```

Native names beneath `layers.N.mlp` match official packed Qwen3.5/Qwen3.8 FFNs:

| Name | Shape |
| --- | --- |
| `gate.weight` | `[E, H]` |
| `experts.gate_up_proj` | `[E, 2I, H]` |
| `experts.down_proj` | `[E, H, I]` |
| `shared_expert.gate_proj.weight`, `up_proj.weight` | `[S, H]` |
| `shared_expert.down_proj.weight` | `[H, S]` |
| `shared_expert_gate.weight` | `[1, H]` |

Raw packed parameters are initialized by model `_init_weights` through
`MoELayer.reset_parameters(std)`, including HF missing-key/meta materialization.
State dict loading and the pinned Transformers 5.12.1 streaming loader accept a
complete original Next `experts.N.{gate,up,down}_proj.weight` bank; mixed or
partial banks fail. The HF converter is local to one load, without a global
registry. Native save, warm start, FSDP resume, `materialize_draft`, and HF export
preserve the packed layout. Export canonicalizes a converted bank before merging
original precision tensors. Full official teacher checkpoints still require
appropriate draft architecture/config and non-FFN tensor compatibility.
The export normalizer preserves MoE fields; it does not establish SGLang MoE
serving support.

## Extension boundaries

Only `qwen3_5_moe` is implemented. A future Qwen3 preset can reuse execution
while selecting no shared path and a per-expert storage adapter in
`experts.py`/`checkpoint.py`; add its validation in `config.py` and route/merge
logic at `MoELayer._route` rather than editing decoder attention or strategies.
A future DeepSeek family should add its own concrete router module (for
sqrtsoftplus scoring, correction bias used for selection, and separate original
mixture weights), then apply routed scaling after weighted expert reduction.
Its bias buffer and detached pending counts belong to that router, with updates
at the existing forward-entry boundary and persisted state/resume contract.
Official quantized `ffn.experts.N.w1/w2/w3` plus scale tensors require a dedicated
checkpoint adapter and quantization implementation; packed Qwen parameters are
not a compatibility claim. DeepSeek custom attention/blocks belong in their
draft architecture and can call `build_ffn`; they do not belong in this FFN
package. No empty unsupported classes, registry, or ABC is needed in this change.

## Small real-teacher smoke setup

`configs/qwen3-8b-dflash-moe-smoke.json` uses Qwen3-8B's hidden/vocabulary sizes,
one draft layer, target layer 18, four experts/top-2, intermediate/shared width
256, and block size four. Capture real teacher features for that layer using
the existing offline feature workflow, then run:

```bash
specforge train --config examples/configs/offline/colocated/qwen3-8b-dflash-moe-smoke.yaml
```

The example requests eight ranks, BF16 FULL_SHARD, accumulation two, four anchors,
objective chunks of one block, and four optimizer steps. Supply at least 64
captured samples for that fixed window. Change the JSON dispatch to `grouped_mm`
to test native expert execution on H200. This is a runnable configuration,
not evidence of convergence, serving quality, speedup, or completed GPU testing.

## One H200 node: four capture servers and four DSpark MoE trainers

`configs/qwen3-8b-dspark-moe.json` and
`examples/configs/online/disaggregated/managed-local/qwen3-8b-dspark-moe-h200-disaggregated.yaml`
provide the larger requested execution pattern. The teacher remains
`Qwen/Qwen3-8B`; this is a separate five-layer DSpark configuration, not the
small offline smoke model. It uses 512 routed experts, top-10, width 512,
shared width 2048, grouped-MM, CPU optimizer offload, and FULL_SHARD across four
trainer ranks. Qwen3-8B's existing DSpark backbone geometry, target layers
`[1, 9, 17, 25, 33]`, block size seven, rank-256 Markov head, and confidence head
are retained. Routing uses the official `none` center; the undefined `sample`
option is an explicit difference from any recipe requesting that name.

The CLI owns four TP1 capture servers on GPUs 0–3, the Mooncake TCP master,
producer, and four trainer ranks on GPUs 4–7. Run after making those eight GPUs
available and providing the configured Qwen3-8B conversation dataset:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  specforge train --config examples/configs/online/disaggregated/managed-local/qwen3-8b-dspark-moe-h200-disaggregated.yaml
```

The launcher inherits this allocator setting in every managed child. Use the
clean-main SGLang capture patch matching its pinned SGLang release and install
Mooncake with `mooncake_master` available. Training uses maximum length 8192,
512 anchors, objective chunks of 128, batch two and accumulation 64: four ranks
consume 512 samples per optimizer step. The in-flight high watermark is 768,
with low watermark 512, so capture can supply a complete optimizer window plus
prefetch before durable acknowledgement. The default 256-ref cap would be too
small for this accumulation window.

At maximum length, five auxiliary layer rows plus one final row carry
`6 × 8192 × 4096 × 2` bytes = 384 MiB per sample. The configured 768-ref high
watermark represents up to 288 GiB of feature payload before staging and
metadata. Four 128 GiB Mooncake segment contributions provide capacity; this
also needs substantial host RAM alongside the offloaded FP32 Adam state. These
are capacity calculations, not measured residency or a memory-fit result.

For a bounded runtime check, override `training.max_steps=2`,
`training.save_interval=1` and the dataset path while keeping the architecture,
length, anchors, split, and accumulation settings. Supply at least 1024 samples
for those two optimizer windows. Short inputs do not validate the 8192-token
memory path. The managed-local supervisor currently rejects `resume_from`;
checkpoint/export and continuation through a compatible externally managed or
offline run must be tested separately. The bounded validation below covers
runtime correctness; convergence, throughput, and serving support are not claimed.

Math/layout references: official Transformers
[`qwen3_next`, `qwen3_5_moe`, `qwen4_exp`](https://github.com/huggingface/transformers/tree/60f49549e0db78f022ee30aa1571d9ab6e4a15bf/src/transformers/models),
Qwen checkpoint configs/indexes and safetensors headers at Next `9c7f2fbe`,
3.5 `59d61f3c`, and 3.8 `de4b8e4d`; grouped-MM/autograd source at PyTorch
[`v2.13.0`](https://github.com/pytorch/pytorch/tree/cf30153c4c131c8164ee7798e5022d810682e2cb).

## H200 validation and its boundaries

The post-correction validation used source `d921b6f8ee939121b77cf1637f1ddb3985f03842`,
PyTorch **2.13.0**, Transformers **5.12.1**, and SGLang **0.5.18** on one
eight-H200 node. The tested online model was the full
five-layer DSpark MoE above: **16,613,910,017 parameters**, 512 experts/top-10,
routed width 512, shared width 2048, and grouped-MM execution. Four TP1 teacher
capture servers used GPUs 0–3; four FULL_SHARD trainers used GPUs 4–7 with CPU
optimizer offload. Batch two and accumulation 64 produced 512 samples per
optimizer window. Actual input length was 8192, with 512 anchors and objective
chunks of 128.

The dataset contained 1280 newly authored synthetic conversations. All 1280
were actually captured through the Qwen3-8B teacher, with zero capture failures;
the unused tail of 256 samples was settled at end of stream.
These are genuine teacher features, not evidence that the authored answers are
on-policy, high quality, or suitable for convergence evaluation. The public CLI
exited successfully at 2026-10-20 21:19:53 UTC after two optimizer steps. Every
trainer rank performed 128 backward calls and two optimizer updates. The metrics
below are the reported values, without combining metrics with different weights:

| Optimizer step | Loss | CE metric | MoE auxiliary metric | Gradient norm |
| --- | --- | --- | --- | --- |
| 1 | 4.27719736099 | 14.28656196594 | 0.55796426535 | 57.79375076294 |
| 2 | 7.36499834061 | 8.90692806244 | 0.489936619997 | 6.37076759338 |

Sampled observations showed finite gradients and positive parameter updates in
the router, routed experts, shared expert, and shared gate at both steps. Sampling
does not establish a change in every individual parameter. Peak allocated CUDA
memory was 93.97 GiB; this observation is not a memory or throughput benchmark.
These short runs establish execution and update paths; they do not establish convergence,
acceptance length, serving compatibility, or a performance improvement.

A fresh isolated CUDA probe found that the earlier grouped combine's single
`index_add` over repeated top-k token IDs was nondeterministic: 31 of 31 ordinary
repeated forwards differed. PyTorch documents this CUDA behavior. Grouped output
now restores unique `(token, top-k slot)` assignments and performs a fixed top-k
sum, retaining the activation dtype and autograd paths. Router math and checkpoint
layout did not change. On the corrected source, 32 ordinary-mode K10 forwards
were bitwise identical, and output, input-gradient, and every parameter-gradient
path matched the portable reference within the explicit BF16 test tolerances.
The two native CUDA tests passed. The selected CPU regression suite passed
110 tests with 13 skips and 153 subtests. The full-scale rerun above also
exercises the combine's additional temporary buffer.

Capture traffic used `runtime.producer_lease: 4` and
`runtime.producer_concurrency: 4`; the requested reference used eight for both.
This is an explicit traffic-setting difference. The GPU topology remained four
capture servers plus four trainers on eight GPUs.

To reproduce the bounded public entry point in an isolated environment, install
the pinned dependencies and Mooncake, make `mooncake_master` available, apply
the clean-main capture patch, and provide a fresh dataset with at least 1024
valid samples:

```bash
bash scripts/apply_sglang_spec_capture_patch.sh --target v0.5.18
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  specforge train --config examples/configs/online/disaggregated/managed-local/qwen3-8b-dspark-moe-h200-disaggregated.yaml \
  data.train_data_path=/absolute/path/to/fresh-conversations.jsonl \
  training.max_steps=2 training.total_steps=2 training.save_interval=1
```

SGLang 0.5.18 requires the versioned capture patch for this run; it lacks the
observer modules required by the alternative plugin. Routing used
`moe_router_center: none`. The undefined `sample` option remains rejected.

All following acceptance checks also passed on the corrected source above.
A separate small four-rank offline model used portable `reference` dispatch and
resumed step 3 through steps 4/5 against five uninterrupted steps. Across saved
steps 3, 4, and 5, all 510 compared checkpoint tensors (516,405,768 elements) matched exactly,
including model tensors, every rank's FP32 masters, Adam moments/counters,
scheduler, CPU/CUDA RNG, and training counters/cursor. Live observations confirmed
CPU FP32 masters and CPU Adam moments. This does not establish managed-local
queue/supervisor resume, which clean main rejects, or full-model native grouped
bitwise optimizer continuation.

Separate four-rank FSDP tests explicitly enabled HF non-reentrant decoder
checkpointing before wrapping. Both portable `reference` and native `grouped_mm`
dispatch passed exact ordinary-versus-checkpointed gradient and actual updated
parameter-shard comparisons for main-plus-auxiliary and auxiliary-only objectives.
Decoder calls changed from `[2, 2]` to `[4, 4]`, confirming recomputation. Router
updates remained nonzero; the final layer's expert gradient was zero for the
auxiliary-only objective. Native acceptance used exact comparisons with no
numerical-tolerance fallback. The public online recipe separately exercises
objective chunk checkpointing.

The full 16.6B HF export/reload check exited successfully. The untouched
`AutoDraftModel.from_pretrained(dtype="auto")` result had clean loading information
and all 82 serialized tensor keys, values, and dtypes matched the finite training
checkpoint exactly. Both nonpersistent RoPE buffers had matching FP32 values
against the independent HF-policy reference. Three repeated forwards of the same
loaded model and the independent HF-policy forward were bitwise identical
(maximum difference zero), with deterministic algorithms disabled and
`CUBLAS_WORKSPACE_CONFIG` unset. No checkpoint weights were copied into the
reloaded subject to obtain this result.

The original training-builder forward comparison exposed an existing generic
RoPE precision difference, reproduced by a fresh dense control. Clean-main
`AutoDraftModel.from_config` calls `model.to(BF16)`, rounding the nonpersistent
`inv_freq` and `original_inv_freq` buffers; HF dtype-context loading creates
those buffers in FP32. Therefore parameter integrity does not imply exact
parity with the old training-buffer policy. The diagnostic failure is retained.
No change was made to eliminate this RoPE policy difference; the generic builder
and loaded subject were left intact. Any independent HF-policy comparison uses
HF's dtype context only for the separate reference model.
