"""Checkpoint boundaries exercised through the real HF and exporter loaders."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

from specforge.export.checkpoint_io import materialize_draft
from specforge.export.to_hf import export_to_hf
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.moe import MoELayer, resolve_moe_config


def config(**overrides):
    values = dict(
        architectures=["DFlashDraftModel"],
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        num_target_layers=1,
        block_size=4,
        moe_preset="qwen3_5_moe",
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        dflash_config={"target_layer_ids": [0], "moe_aux_loss_coeff": 0.2},
    )
    values.update(overrides)
    return Qwen3Config(**values)


def marked_model(**overrides):
    model = DFlashDraftModel(config(**overrides))
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.copy_(
                torch.arange(parameter.numel()).reshape(parameter.shape) / 1000 + index
            )
    return model


def assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(
            actual[name], expected[name], rtol=0, atol=0, msg=name
        )


def per_expert_state(packed):
    converted = dict(packed)
    for key in list(packed):
        if not key.endswith("experts.gate_up_proj"):
            continue
        prefix = key.removesuffix("experts.gate_up_proj")
        gate_up = converted.pop(key)
        down = converted.pop(prefix + "experts.down_proj")
        for expert in range(gate_up.shape[0]):
            gate, up = gate_up[expert].chunk(2, dim=0)
            for name, value in (
                ("gate_proj", gate),
                ("up_proj", up),
                ("down_proj", down[expert]),
            ):
                converted[f"{prefix}experts.{expert}.{name}.weight"] = value.clone()
    return converted


def test_native_hf_roundtrip_sharded_low_memory_and_override(tmp_path):
    model = marked_model()
    model.save_pretrained(tmp_path, max_shard_size="1KB")
    loaded, info = DFlashDraftModel.from_pretrained(
        tmp_path,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
        output_loading_info=True,
    )
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    assert all(not parameter.is_meta for parameter in loaded.parameters())
    assert_state_equal(loaded.state_dict(), model.state_dict())
    changed = {**model.config.dflash_config, "moe_aux_loss_coeff": 0.7}
    overridden = DFlashDraftModel.from_pretrained(tmp_path, dflash_config=changed)
    assert overridden.layers[0].mlp.moe_config.aux_loss_coeff == 0.7
    assert_state_equal(overridden.state_dict(), model.state_dict())


@pytest.mark.parametrize("missing", ["gate_up_proj", "down_proj"])
def test_hf_partial_packed_checkpoint_initializes_only_missing_tensor(
    tmp_path, missing
):
    model = marked_model()
    state = dict(model.state_dict())
    missing_key = f"layers.0.mlp.experts.{missing}"
    state.pop(missing_key)
    model.save_pretrained(tmp_path, state_dict=state)
    loaded, info = DFlashDraftModel.from_pretrained(
        tmp_path,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
        output_loading_info=True,
    )
    assert info["missing_keys"] == {missing_key} or info["missing_keys"] == [
        missing_key
    ]
    restored = loaded.state_dict()
    for key, value in state.items():
        torch.testing.assert_close(restored[key], value, rtol=0, atol=0, msg=key)
    initialized = restored[missing_key]
    assert not initialized.is_meta
    assert initialized.isfinite().all()
    assert 0.005 < initialized.float().std() < 0.04


def test_per_expert_load_state_dict_packs_gate_then_up():
    original = marked_model()
    loaded = DFlashDraftModel(config())
    loaded.load_state_dict(per_expert_state(original.state_dict()), strict=True)
    assert_state_equal(loaded.state_dict(), original.state_dict())


def test_hf_sharded_original_layout_loads_without_randomized_bank(tmp_path):
    # Double-digit expert ids catch lexical ordering errors; two layers catch
    # converters accidentally sharing a bank across layer prefixes.
    original = marked_model(
        n_routed_experts=12, num_hidden_layers=2, num_target_layers=8
    )
    original.save_pretrained(
        tmp_path,
        state_dict=per_expert_state(original.state_dict()),
        max_shard_size="1KB",
    )
    loaded, info = DFlashDraftModel.from_pretrained(
        tmp_path,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
        output_loading_info=True,
    )
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    assert_state_equal(loaded.state_dict(), original.state_dict())


@pytest.mark.parametrize("mutation", ["partial", "mixed"])
def test_hf_original_layout_rejects_partial_or_mixed_bank(tmp_path, mutation):
    original = marked_model()
    state = per_expert_state(original.state_dict())
    if mutation == "partial":
        state.pop("layers.0.mlp.experts.0.up_proj.weight")
    else:
        state["layers.0.mlp.experts.gate_up_proj"] = (
            original.layers[0].mlp.experts.gate_up_proj.detach().clone()
        )
    original.save_pretrained(tmp_path, state_dict=state)
    with pytest.raises(ValueError, match="complete|mixed|per-expert"):
        DFlashDraftModel.from_pretrained(tmp_path)


@pytest.mark.parametrize("original_layout", [False, True])
def test_meta_layer_assign_load_materializes_complete_bank(original_layout):
    original = MoELayer(resolve_moe_config(config()), 16)
    state = original.state_dict()
    if original_layout:
        state = per_expert_state(state)
    with torch.device("meta"):
        loaded = MoELayer(resolve_moe_config(config()), 16)
    assert all(parameter.is_meta for parameter in loaded.parameters())
    loaded.load_state_dict(state, assign=True)
    assert all(not parameter.is_meta for parameter in loaded.parameters())
    assert_state_equal(loaded.state_dict(), original.state_dict())


@pytest.mark.parametrize("mutation", ["partial", "mixed"])
def test_incomplete_or_mixed_expert_layout_fails_even_non_strict(mutation):
    layer = MoELayer(resolve_moe_config(config()), 16)
    state = per_expert_state(layer.state_dict())
    if mutation == "partial":
        state.pop("experts.0.up_proj.weight")
    else:
        state["experts.gate_up_proj"] = layer.experts.gate_up_proj.detach().clone()
    with pytest.raises(RuntimeError, match="complete packed or per-expert bank"):
        layer.load_state_dict(state, strict=False)


def test_materialize_and_export_preserve_native_bank_and_config(tmp_path):
    original = marked_model().to(dtype=torch.bfloat16)
    config_path = tmp_path / "draft.json"
    original.config.to_json_file(config_path)
    state = {"draft_state_dict": original.state_dict()}
    materialized = materialize_draft(state, str(config_path))
    assert_state_equal(materialized.state_dict(), original.state_dict())
    checkpoint_path = tmp_path / "training_state.pt"
    torch.save(state, checkpoint_path)
    destination = tmp_path / "export"
    export_to_hf(str(checkpoint_path), str(config_path), str(destination))
    loaded = AutoDraftModel.from_pretrained(destination)
    assert_state_equal(loaded.state_dict(), original.state_dict())
    exported = json.loads((destination / "config.json").read_text())
    assert exported["n_routed_experts"] == 4
    assert exported["moe_preset"] == "qwen3_5_moe"
    assert exported["dflash_config"]["moe_aux_loss_coeff"] == 0.2


def test_materialize_converts_complete_original_layout_and_rejects_missing_bank(
    tmp_path,
):
    original = marked_model()
    config_path = tmp_path / "draft.json"
    original.config.to_json_file(config_path)
    state = {"draft_state_dict": per_expert_state(original.state_dict())}
    materialized = materialize_draft(state, str(config_path))
    assert_state_equal(materialized.state_dict(), original.state_dict())
    state["draft_state_dict"].pop("layers.0.mlp.experts.0.up_proj.weight")
    with pytest.raises(RuntimeError, match="complete packed or per-expert bank"):
        materialize_draft(state, str(config_path))


def test_export_from_original_layout_writes_only_native_expert_names(tmp_path):
    original = marked_model()
    config_path = tmp_path / "draft.json"
    original.config.to_json_file(config_path)
    checkpoint_path = tmp_path / "training_state.pt"
    torch.save(
        {"draft_state_dict": per_expert_state(original.state_dict())}, checkpoint_path
    )
    destination = tmp_path / "export"
    export_to_hf(str(checkpoint_path), str(config_path), str(destination))
    loaded, info = DFlashDraftModel.from_pretrained(
        destination, output_loading_info=True
    )
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    assert_state_equal(loaded.state_dict(), original.state_dict())


def test_normalizer_preserves_moe_metadata_without_claiming_serving_support(tmp_path):
    script = (
        Path(__file__).parents[2] / "scripts" / "gates" / "normalize_dflash_export.py"
    )
    spec = importlib.util.spec_from_file_location("new_moe_normalizer", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = config().to_dict()
    payload["architectures"] = ["DFlashDraftModel"]
    payload["auto_map"] = {"AutoModel": "local.Code"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    normalized = module.normalize_export(str(path), expected_block_size=4)
    for name in (
        "moe_preset",
        "n_routed_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "dflash_config",
    ):
        assert normalized[name] == payload[name]
    assert "auto_map" not in normalized


def test_dense_constructor_keeps_qwen_mlp():
    dense = config(moe_preset=None, n_routed_experts=None)
    model = DFlashDraftModel(dense)
    assert not isinstance(model.layers[0].mlp, MoELayer)
    assert set(model.layers[0].mlp.state_dict()) == {
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    }


def test_training_block_and_layer_overrides_preserve_moe_execution_options():
    from specforge.algorithms.model_providers import apply_dflash_overrides
    from specforge.training.model_loading import _apply_draft_overrides

    draft_config = config(num_target_layers=8)
    options = dict(draft_config.dflash_config)
    options.update(moe_dispatch="reference", moe_router_center="none")
    draft_config.dflash_config = options
    run = SimpleNamespace(
        model=SimpleNamespace(draft_block_size=8, draft_num_hidden_layers=2)
    )
    _apply_draft_overrides(
        run, draft_config, SimpleNamespace(apply_overrides=apply_dflash_overrides)
    )
    assert draft_config.dflash_config["moe_aux_loss_coeff"] == 0.2
    assert draft_config.dflash_config["moe_dispatch"] == "reference"
    assert draft_config.dflash_config["moe_router_center"] == "none"
    assert draft_config.dflash_config["block_size"] == 8
    assert draft_config.moe_preset == "qwen3_5_moe"
    assert draft_config.n_routed_experts == 4


@pytest.mark.parametrize("checkpoint_format", ["pretrained", "specforge"])
def test_training_warm_start_loads_weights_and_preserves_run_aux_override(
    tmp_path, checkpoint_format
):
    from specforge.training.model_loading import warm_start_draft_model

    original = marked_model()
    if checkpoint_format == "pretrained":
        source = tmp_path / "pretrained"
        original.save_pretrained(source)
    else:
        source = tmp_path / "training_state.pt"
        torch.save(
            {
                "draft_state_dict": per_expert_state(original.state_dict()),
                "strategy": "dflash",
            },
            source,
        )
    target_config = config(
        dflash_config={"target_layer_ids": [0], "moe_aux_loss_coeff": 0.7}
    )
    target = DFlashDraftModel(target_config)
    report = warm_start_draft_model(
        target, str(source), draft_config=target_config, strategy="dflash"
    )
    assert report.checkpoint_format == checkpoint_format
    assert not report.missing_keys
    assert_state_equal(target.state_dict(), original.state_dict())
    assert target.layers[0].mlp.moe_config.aux_loss_coeff == 0.7


@pytest.mark.parametrize("saved_contract", ["different_objective", "absent"])
def test_trainer_resume_rejects_changed_or_unrecorded_moe_math(
    tmp_path, saved_contract
):
    from specforge.modeling.draft.moe.runtime import moe_resume_contract
    from specforge.training.trainer import Trainer

    saved_model = torch.nn.Module()
    saved_model.draft_model = marked_model()
    state = {"strategy": "dflash", "effective_total_steps": 2}
    if saved_contract == "different_objective":
        state.update(moe_resume_contract(saved_model))
    current = torch.nn.Module()
    current.draft_model = DFlashDraftModel(
        config(dflash_config={"target_layer_ids": [0], "moe_aux_loss_coeff": 0.7})
    )
    # Exercise the real Trainer resume comparison before any distributed wrap
    # or optimizer creation. Queue/store/controller are unused infrastructure.
    with pytest.raises(ValueError, match="specforge_moe"):
        Trainer(
            algorithm_name="dflash",
            make_step_strategy=lambda *_args, **_kwargs: None,
            controller=SimpleNamespace(register_trainer=lambda _metadata: "trainer"),
            store=SimpleNamespace(),
            ref_source={"queue": object()},
            model=current,
            target_head=None,
            optimizer_factory=None,
            run_id="moe-resume-audit",
            output_dir=str(tmp_path),
            batch_size=1,
            accumulation_steps=1,
            num_epochs=1,
            max_steps=2,
            save_interval=0,
            logger=None,
            log_interval=1,
            collate_fn=None,
            resume_from="audit-checkpoint",
            resume_state=state,
        )
