# coding=utf-8
"""SGLang draft model classes for SpecForge drafters SGLang does not ship.

Point SGLang at this package and every module here that defines ``EntryClass``
registers its architectures (overriding same-named built-ins)::

    SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang_models \\
    python -m sglang.launch_server --model <target> \\
        --speculative-algorithm DFLASH --speculative-draft-model-path <export> ...

Modules:

- ``moe_ffn``: the MoE FFN of a draft layer (router, stacked routed experts,
  shared expert) in plain PyTorch, mirroring ``specforge.modeling.draft.moe``
  so serving routes exactly as training did. No SGLang import, so it is also
  the reference used by ``scripts/gates/check_dspark_moe_sglang_equivalence.py``
  on CPU.
- ``dflash_moe``: ``DFlashMoEDraftModel``, ``DFlash2MoEDraftModel`` and
  ``Qwen3MoEDSparkModel``, the DFlash-family draft classes with the dense MLP
  replaced by ``moe_ffn.DraftMoEFFN``, loading the official per-expert
  checkpoint naming and refusing checkpoints that do not match the class.

``scripts/gates/normalize_dflash_export.py`` writes these architecture names
into an MoE export's ``config.json``.
"""
