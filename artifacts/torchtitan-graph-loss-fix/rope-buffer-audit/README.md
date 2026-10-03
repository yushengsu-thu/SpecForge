# Nonpersistent RoPE initialization audit

CPU-only counterfactual using the exact controlled p4 DFlash2 configuration,
Torch 2.14.0 and Transformers 5.12.1. `config.json` is extracted from the pinned
raw p4 trial identified in `provenance.json`. `probe.py`, `report.json`, and
`probe.log` preserve the executable isolated check, versions, source hash, and
complete 64-element frequency vectors. No full draft or GPU is used.

Run in this directory:

```sh
CUDA_VISIBLE_DEVICES= /scratch/specforge-torchtitan-20261003/venv/bin/python probe.py
```

Fresh `Qwen3RotaryEmbedding` stores FP32 `inv_freq` and `original_inv_freq` as
nonpersistent buffers. Neither appears in its empty state_dict. Casting the
module to BF16 and then FP32 changes 63/64 frequency values, with maximum
absolute error 0.0011547207832336426. At positions 0–4095, the largest phase
difference is 4.728581428527832 radians. The BF16 cosine and sine outputs each
have maximum absolute difference 2.0. This proves that identical state_dict
hashes did not guarantee identical initial forward models in the old p4
FSDP/Titan comparison. It does not quantify full-model loss impact or establish
that all training-trajectory differences came from RoPE.

## Existing production behavior versus benchmark control

At core `5a35ff35b0ba939242f294907892a2d8955a52b9`:

- `specforge/modeling/auto.py:24–40` explicitly calls `model.to(dtype=...)` in
  `AutoDraftModel.from_config` when a dtype is supplied.
- `specforge/algorithms/model_providers.py:152–185` passes the configured BF16
  dtype to that constructor and casts the returned draft again.
  The family training model is also cast at line 370.
- `specforge/training/backend.py:286–292` uses FSDP `buffer_dtype=torch.float32`.
  This widens previously rounded values; it does not recover discarded bits.
- `specforge/training/torchtitan/model.py:199–238` builds a fresh reference and
  restores nonpersistent buffers after empty-device initialization. Its fresh
  FP32 RoPE values are independent of the saved draft state_dict.

The rounding is an existing, explicit legacy construction path. This audit
makes no claim that its authors intentionally selected reduced-precision RoPE,
or that fresh FP32 initialization is a native Titan defect. Production code
was not changed to manufacture benchmark agreement.

The benchmark-only correction at
`8bf591727ad3b3a2287c777997f8c6da352f28cd` preserves fresh pre-cast nonpersistent
values, restores original FP32 dtype and contents after the legacy BF16 casts,
and records `nonpersistent_buffer_policy="fresh-reference-fp32-v1"`. It records
initial fingerprints and actual runtime fingerprints from every rank, validates
them after backend preparation, and validates again after training to cover
FSDP's lazy first-forward buffer policy. The matrix rejects missing fingerprints,
missing ranks, mismatched values/dtypes, or BF16 rotary buffers falsely labelled
as the FP32 policy. This is a controlled FP32-buffer variant of the original FSDP
backend, not untouched default FSDP initialization.

`cpu-tests.log` contains the final 26 passing CPU tests (4.312 seconds), including
three real tiny draft constructors proving that old and corrected construction
retain exactly equal persistent parameter/state_dict hashes for DFlash, DFlash2,
and DSpark. `cpu-tests-verification.json` records the exact command and all four
final driver/test hashes, verified against the remote tested files.
`tested-source/` preserves those sources as archival text. GPU first-forward
checks and any replacement timing matrix are separate evidence.

Old p4 results remain immutable, superseded diagnostics with this initialization
confound. Their FSDP/Titan timing or loss differences must not be presented as
matched-initialization results. Native Titan and Graph use the same fresh buffer
path, so this particular confound does not invalidate their initialization match.
