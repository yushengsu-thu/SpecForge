import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from torch.testing import assert_close
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.modeling.draft.domino import DominoDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.moe import (
    MOE_LAYOUTS,
    MOE_PRESETS,
    MoELayer,
    MoEPreset,
    MoEStateCodec,
    MoEWeightLayout,
    apply_pending_balance_updates,
    build_ffn,
    pop_moe_step_terms,
    register_moe_layout,
    register_moe_preset,
    resolve_moe_config,
    swiglu,
)
from specforge.modeling.draft.moe.balance import switch_aux_loss
from specforge.modeling.draft.moe.experts import grouped_mm_experts, loop_experts
from specforge.modeling.draft.moe.routing import center_router_logits
from specforge.training.strategies.base import StepOutput, finalize_step_output

HIDDEN, EXPERTS, TOP_K, WIDTH, SHARED = 32, 8, 2, 16, 24


class _PerExpertLayout(MoEWeightLayout):
    """Qwen3-MoE style per-expert names, to exercise non-identity I/O."""

    name = "test_per_expert"

    def to_checkpoint(self, tensors, config):
        out = {}
        for key, value in tensors.items():
            if key == "experts.gate_up_proj":
                gate, up = value.chunk(2, dim=1)
                for expert in range(value.shape[0]):
                    out[f"experts.{expert}.gate_proj.weight"] = gate[expert].clone()
                    out[f"experts.{expert}.up_proj.weight"] = up[expert].clone()
            elif key == "experts.down_proj":
                for expert in range(value.shape[0]):
                    out[f"experts.{expert}.down_proj.weight"] = value[expert].clone()
            else:
                out[key] = value
        return out

    def from_checkpoint(self, tensors, config):
        experts = range(config.n_routed_experts)
        out = {k: v for k, v in tensors.items() if not k.startswith("experts.")}
        out["experts.gate_up_proj"] = torch.stack(
            [
                torch.cat(
                    [
                        tensors[f"experts.{e}.gate_proj.weight"],
                        tensors[f"experts.{e}.up_proj.weight"],
                    ]
                )
                for e in experts
            ]
        )
        out["experts.down_proj"] = torch.stack(
            [tensors[f"experts.{e}.down_proj.weight"] for e in experts]
        )
        return out

    def config_fields(self, config):
        return {"num_experts": config.n_routed_experts}


if _PerExpertLayout.name not in MOE_LAYOUTS:
    register_moe_layout(_PerExpertLayout())
if "test_per_expert_moe" not in MOE_PRESETS:
    register_moe_preset(
        MoEPreset(
            name="test_per_expert_moe",
            layout=_PerExpertLayout.name,
            defaults={"aux_loss_coeff": 1e-3},
        )
    )
if "test_selection_bias_moe" not in MOE_PRESETS:
    register_moe_preset(
        MoEPreset(
            name="test_selection_bias_moe",
            layout="hf_fused",
            defaults={
                "score_function": "sqrtsoftplus",
                "routed_scaling_factor": 2.5,
                "swiglu_limit": 10.0,
                "selection_bias": True,
                "bias_update_rate": 1e-3,
            },
        )
    )


def _moe_keys(preset="qwen3_5_moe", **overrides):
    keys = {
        "moe_preset": preset,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "moe_intermediate_size": WIDTH,
    }
    if preset == "qwen3_5_moe":
        keys["shared_expert_intermediate_size"] = SHARED
    keys.update(overrides)
    return keys


def _layer(preset="qwen3_5_moe", dflash_config=None, **overrides):
    config = {**_moe_keys(preset, **overrides), "dflash_config": dflash_config or {}}
    return MoELayer(resolve_moe_config(config), HIDDEN)


def _draft_config(architecture="DFlashDraftModel", moe=None, dflash_config=None):
    method = {"target_layer_ids": [1, 3], "mask_token_id": 0}
    method.update(dflash_config or {})
    if architecture == "DFlash2DraftModel":
        method.update(
            conv_kernel_size=3, conv_group_size=2, selector_rank=4, selector_top_k=2
        )
    elif architecture == "DominoDraftModel":
        method.update(projector_type="domino", emb_dim=8, gru_hidden_dim=8)
    elif architecture == "DSparkDraftModel":
        method.update(projector_type="dspark", markov_rank=4)
    config = Qwen3Config(
        architectures=[architecture],
        block_size=4,
        hidden_size=HIDDEN,
        intermediate_size=48,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        num_target_layers=6,
        max_position_embeddings=64,
        vocab_size=64,
        layer_types=["full_attention"] * 2,
        dflash_config=method,
        **(moe if moe is not None else _moe_keys()),
    )
    config._attn_implementation = "sdpa"
    return config


def _draft_forward(model, batch=2, blocks=2):
    block = model.block_size
    hidden = model.config.hidden_size
    context = 5
    positions = torch.arange(context + blocks * block).expand(batch, -1)
    return model(
        position_ids=positions,
        noise_embedding=torch.randn(batch, blocks * block, hidden),
        target_hidden=torch.randn(batch, context, 2 * hidden),
    )


class _Strategy:
    def __init__(self, module):
        self.module = module

    def trainable_module(self):
        return self.module


class TestResolveMoEConfig(unittest.TestCase):
    def test_dense_config_resolves_to_none(self):
        self.assertIsNone(resolve_moe_config({"hidden_size": 8, "dflash_config": {}}))

    def test_preset_supplies_structure_and_training_defaults(self):
        config = resolve_moe_config(_moe_keys())
        self.assertEqual(config.layout, "hf_fused")
        self.assertEqual(config.score_function, "softmax")
        self.assertTrue(config.shared_expert_gate)
        self.assertTrue(config.norm_topk_prob)
        self.assertEqual(config.aux_loss_coeff, 1e-3)
        self.assertEqual(config.dispatch, "auto")
        self.assertEqual(config.router_center, "none")

    def test_top_level_keys_override_preset(self):
        config = resolve_moe_config(
            _moe_keys(
                norm_topk_prob=False,
                scoring_func="sigmoid",
                routed_scaling_factor=2.5,
                swiglu_limit=0,
            )
        )
        self.assertFalse(config.norm_topk_prob)
        self.assertEqual(config.score_function, "sigmoid")
        self.assertEqual(config.routed_scaling_factor, 2.5)
        self.assertIsNone(config.swiglu_limit)

    def test_training_keys_come_from_method_config(self):
        config = resolve_moe_config(
            {
                **_moe_keys(),
                "block_size": 16,
                "dflash_config": {
                    "moe_aux_loss_coeff": 0.01,
                    "moe_dispatch": "loop",
                    "moe_router_center": "sample",
                },
            }
        )
        self.assertEqual(config.aux_loss_coeff, 0.01)
        self.assertEqual(config.dispatch, "loop")
        self.assertEqual(config.router_center, "sample")
        self.assertEqual(config.router_center_size, 16)
        nested = resolve_moe_config(
            {
                **_moe_keys(),
                "dflash_config": {"block_size": 8, "moe_router_center": "sample"},
            }
        )
        self.assertEqual(nested.router_center_size, 8)

    def test_moe_fields_without_preset_are_rejected(self):
        for config in (
            {"n_routed_experts": 8},
            {"num_experts": 8},
            {"dflash_config": {"moe_aux_loss_coeff": 0.001}},
        ):
            with (
                self.subTest(config=config),
                self.assertRaisesRegex(ValueError, "without 'moe_preset'"),
            ):
                resolve_moe_config(config)

    def test_rejects_unknown_preset_training_key_and_missing_size(self):
        with self.assertRaisesRegex(ValueError, "unknown moe_preset"):
            resolve_moe_config(_moe_keys(preset="nope"))
        with self.assertRaisesRegex(ValueError, "unknown MoE training keys"):
            resolve_moe_config({**_moe_keys(), "dflash_config": {"moe_typo": 1}})
        keys = _moe_keys()
        del keys["shared_expert_intermediate_size"]
        with self.assertRaisesRegex(ValueError, "shared_expert_intermediate_size"):
            resolve_moe_config(keys)

    def test_num_experts_alias_must_agree(self):
        keys = _moe_keys()
        del keys["n_routed_experts"]
        self.assertEqual(
            resolve_moe_config({**keys, "num_experts": 8}).n_routed_experts, 8
        )
        with self.assertRaisesRegex(ValueError, "disagrees"):
            resolve_moe_config({**_moe_keys(), "num_experts": 4})

    def test_validates_values(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            resolve_moe_config(_moe_keys(num_experts_per_tok=EXPERTS + 1))
        with self.assertRaisesRegex(ValueError, "selection-bias preset"):
            resolve_moe_config(
                {**_moe_keys(), "dflash_config": {"moe_bias_update_rate": 0.1}}
            )
        with self.assertRaisesRegex(ValueError, "moe_dispatch"):
            resolve_moe_config(
                {**_moe_keys(), "dflash_config": {"moe_dispatch": "fast"}}
            )


class TestRouting(unittest.TestCase):
    def test_score_functions_and_topk_weights(self):
        layer = _layer(
            "test_selection_bias_moe", shared_expert_intermediate_size=0
        ).eval()
        router = layer.gate
        hidden = torch.randn(10, HIDDEN)
        routing = router(hidden)
        logits = hidden @ router.weight.t()
        expected = torch.nn.functional.softplus(logits).sqrt()
        assert_close(routing.scores, expected)
        assert_close(routing.weights.sum(-1), torch.full((10,), 2.5))

    def test_selection_bias_shifts_selection_not_weights(self):
        router = _layer("test_selection_bias_moe").gate
        hidden = torch.randn(6, HIDDEN)
        router.e_score_correction_bias[3] = 100.0
        routing = router(hidden)
        self.assertTrue((routing.indices == 3).any(dim=-1).all())
        picked = routing.scores.gather(-1, routing.indices)
        expected = picked / picked.sum(-1, keepdim=True) * 2.5
        assert_close(routing.weights, expected)

    def test_selection_bias_stays_fp32_through_dtype_casts(self):
        layer = _layer("test_selection_bias_moe")
        layer.gate.e_score_correction_bias.fill_(1e-4)
        layer.to(torch.bfloat16)
        bias = layer.gate.e_score_correction_bias
        self.assertEqual(bias.dtype, torch.float32)
        assert_close(bias, torch.full_like(bias, 1e-4))
        self.assertEqual(layer.gate.weight.dtype, torch.bfloat16)

    def test_center_router_logits_per_sample(self):
        logits = torch.randn(2, 12, EXPERTS)
        centered = center_router_logits(logits, 4)
        assert_close(
            centered.view(2, 3, 4, EXPERTS).mean(dim=2),
            torch.zeros(2, 3, EXPERTS),
            atol=1e-6,
            rtol=0,
        )
        assert_close(
            center_router_logits(logits, None).mean(dim=1),
            torch.zeros(2, EXPERTS),
            atol=1e-6,
            rtol=0,
        )
        with self.assertRaisesRegex(ValueError, "whole samples"):
            center_router_logits(logits, 5)

    def test_sample_centering_matches_one_block_at_a_time(self):
        layer = _layer(dflash_config={"moe_router_center": "sample"}, block_size=4)
        layer.eval()
        blocks = torch.randn(1, 12, HIDDEN)
        together = layer(blocks)
        separate = torch.cat([layer(blocks[:, i : i + 4]) for i in (0, 4, 8)], dim=1)
        assert_close(together, separate)


class TestExperts(unittest.TestCase):
    def test_swiglu_clamp(self):
        gate = torch.tensor([20.0, -20.0, 1.0])
        up = torch.tensor([-30.0, 30.0, 2.0])
        clamped = swiglu(gate, up, limit=10.0)
        expected = torch.nn.functional.silu(torch.tensor([10.0, -20.0, 1.0]))
        assert_close(clamped, expected * torch.tensor([-10.0, 10.0, 2.0]))
        assert_close(swiglu(gate, up), torch.nn.functional.silu(gate) * up)

    def test_grouped_mm_matches_loop_and_reference(self):
        for limit in (None, 0.5):
            layer = _layer(swiglu_limit=limit)
            experts = layer.experts
            hidden = torch.randn(9, HIDDEN, requires_grad=True)
            indices = torch.randint(0, EXPERTS, (9, TOP_K))
            weights = torch.rand(9, TOP_K)
            reference = torch.zeros(9, HIDDEN)
            for token in range(9):
                for slot in range(TOP_K):
                    expert = indices[token, slot]
                    gate, up = (experts.gate_up_proj[expert] @ hidden[token]).chunk(2)
                    reference[token] += weights[token, slot] * (
                        experts.down_proj[expert] @ swiglu(gate, up, limit)
                    )
            results = {}
            for name, impl in (("grouped", grouped_mm_experts), ("loop", loop_experts)):
                output = impl(experts, hidden, indices, weights)
                grads = torch.autograd.grad(
                    output.square().sum(), [hidden, experts.gate_up_proj]
                )
                results[name] = (output, grads)
                assert_close(output, reference.detach(), rtol=1e-5, atol=1e-5)
            for got, want in zip(results["grouped"][1], results["loop"][1]):
                assert_close(got, want, rtol=1e-5, atol=1e-5)


class TestMoELayer(unittest.TestCase):
    def test_matches_upstream_qwen3_5_moe_block(self):
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
            Qwen3_5MoeTextConfig,
        )
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeSparseMoeBlock,
        )

        upstream = Qwen3_5MoeSparseMoeBlock(
            Qwen3_5MoeTextConfig(
                hidden_size=HIDDEN,
                num_experts=EXPERTS,
                num_experts_per_tok=TOP_K,
                moe_intermediate_size=WIDTH,
                shared_expert_intermediate_size=SHARED,
            )
        )
        for parameter in upstream.parameters():
            torch.nn.init.normal_(parameter, std=0.2)
        for dispatch in ("grouped_mm", "loop"):
            layer = _layer(dflash_config={"moe_dispatch": dispatch})
            # Strict load: module names are the official fused checkpoint names.
            layer.load_state_dict(upstream.state_dict())
            hidden = torch.randn(2, 5, HIDDEN)
            assert_close(layer(hidden), upstream(hidden), rtol=1e-5, atol=1e-5)

    def test_reset_parameters(self):
        layer = _layer("test_selection_bias_moe")
        layer.gate.e_score_correction_bias.fill_(3.0)
        layer.reset_parameters(std=0.5)
        self.assertAlmostEqual(float(layer.experts.gate_up_proj.std()), 0.5, delta=0.05)
        assert_close(
            layer.gate.e_score_correction_bias,
            torch.zeros(EXPERTS),
        )


class TestLoadBalancing(unittest.TestCase):
    def test_switch_aux_loss_is_one_when_balanced(self):
        uniform = torch.full((EXPERTS, EXPERTS), 1.0 / EXPERTS)
        balanced = torch.ones(EXPERTS, dtype=torch.long)
        assert_close(switch_aux_loss(uniform, balanced), torch.tensor(1.0))
        collapsed = torch.zeros(EXPERTS, dtype=torch.long)
        collapsed[0] = EXPERTS
        peaked = torch.full((EXPERTS, EXPERTS), 0.0).index_fill_(
            1, torch.tensor([0]), 1.0
        )
        assert_close(switch_aux_loss(peaked, collapsed), torch.tensor(float(EXPERTS)))

    def test_step_terms_fold_into_step_output(self):
        model = DFlashDraftModel(
            _draft_config(dflash_config={"moe_aux_loss_coeff": 0.01})
        ).train()
        output = _draft_forward(model)
        numerator = output.float().square().sum()
        denominator = torch.tensor(4.0)
        step = finalize_step_output(
            _Strategy(model),
            StepOutput(
                loss=numerator / denominator,
                metrics={},
                loss_terms=(numerator, denominator),
            ),
        )
        aux_num, forwards = step.ratio_metrics["moe/aux_loss"]
        self.assertEqual(float(forwards), 2.0)  # two layers, one forward
        expected = numerator.detach() + 0.01 * aux_num * denominator
        assert_close(step.loss_terms[0].detach(), expected)
        self.assertIn("moe/max_load", step.ratio_metrics)
        self.assertIn("moe/unused_experts", step.ratio_metrics)
        step.loss_terms[0].backward()
        self.assertIsNotNone(model.layers[0].mlp.gate.weight.grad)
        self.assertIsNone(pop_moe_step_terms(model))  # consumed

        scalar = StepOutput(loss=torch.tensor(1.0), metrics={})
        _draft_forward(model)
        folded = finalize_step_output(_Strategy(model), scalar)
        self.assertGreater(float(folded.loss), 1.0)

    def test_dense_and_eval_steps_are_untouched(self):
        dense = DFlashDraftModel(_draft_config(moe={})).train()
        _draft_forward(dense)
        out = StepOutput(loss=torch.tensor(1.0), metrics={})
        self.assertIs(finalize_step_output(_Strategy(dense), out), out)
        model = DFlashDraftModel(_draft_config()).eval()
        _draft_forward(model)
        self.assertIsNone(pop_moe_step_terms(model))

    def test_pending_selection_bias_update(self):
        config = _draft_config(moe=_moe_keys("test_selection_bias_moe"))
        model = DFlashDraftModel(config).train()
        _draft_forward(model)
        layer = model.layers[0].mlp
        load = layer.balance.pending_load.clone().float()
        self.assertEqual(len(layer.balance.records), 1)
        apply_pending_balance_updates(model)
        expected = torch.sign(load.mean() - load) * 1e-3
        assert_close(layer.gate.e_score_correction_bias, expected)
        self.assertIsNone(layer.balance.pending_load)
        self.assertEqual(layer.balance.records, [])  # stale records dropped
        model.eval()
        _draft_forward(model)
        self.assertIsNone(layer.balance.pending_load)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_training_step_has_no_host_syncs(self):
        layer = _layer(dflash_config={"moe_dispatch": "grouped_mm"})
        layer = layer.to("cuda", torch.bfloat16).train()
        hidden = torch.randn(2, 64, HIDDEN, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            output = layer(hidden)
            terms = pop_moe_step_terms(layer)
            (output.float().square().mean() + terms.objective).backward()
        finally:
            torch.cuda.set_sync_debug_mode("default")
        self.assertIsNotNone(layer.experts.gate_up_proj.grad)

    def test_checkpoint_recompute_records_once(self):
        layer = _layer(dflash_config={"moe_aux_loss_coeff": 0.1}).train()
        hidden = torch.randn(2, 4, HIDDEN, requires_grad=True)
        output = checkpoint(layer, hidden, use_reentrant=False)
        self.assertEqual(len(layer.balance.records), 1)
        terms = pop_moe_step_terms(layer)
        (output.square().sum() + terms.objective).backward()
        self.assertEqual(layer.balance.records, [])
        self.assertIsNotNone(layer.gate.weight.grad)


class TestDraftIntegration(unittest.TestCase):
    def test_build_ffn_dense_and_moe(self):
        dense = Qwen3Config(hidden_size=HIDDEN, intermediate_size=48)
        sentinel = torch.nn.Identity()
        self.assertIs(build_ffn(dense, dense=lambda _config: sentinel), sentinel)
        self.assertIsInstance(
            build_ffn(_draft_config(), dense=lambda _config: sentinel), MoELayer
        )
        with self.assertRaisesRegex(ValueError, "without 'moe_preset'"):
            build_ffn(
                Qwen3Config(hidden_size=HIDDEN, n_routed_experts=8),
                dense=lambda _config: sentinel,
            )

    def test_dflash_family_drafts_adopt_moe_ffn(self):
        for architecture, model_cls in (
            ("DFlashDraftModel", DFlashDraftModel),
            ("DFlash2DraftModel", DFlash2DraftModel),
            ("DominoDraftModel", DominoDraftModel),
            ("DSparkDraftModel", DSparkDraftModel),
        ):
            with self.subTest(architecture=architecture):
                model = model_cls(_draft_config(architecture))
                for layer in model.layers:
                    self.assertIsInstance(layer.mlp, MoELayer)
                self.assertAlmostEqual(
                    float(model.layers[0].mlp.experts.down_proj.std()),
                    model.config.initializer_range,
                    delta=0.005,
                )
                model.train()
                self.assertEqual(_draft_forward(model).shape, (2, 8, HIDDEN))
                self.assertIsNotNone(pop_moe_step_terms(model))


class TestWeightLayout(unittest.TestCase):
    def _model(self):
        return DFlashDraftModel(_draft_config(moe=_moe_keys("test_per_expert_moe")))

    def test_identity_layout_passes_state_through(self):
        model = DFlashDraftModel(_draft_config())
        state = model.state_dict()
        self.assertIs(MoEStateCodec.for_model(model).decode(state), state)
        self.assertIn("layers.0.mlp.experts.gate_up_proj", state)
        self.assertIn("layers.0.mlp.shared_expert_gate.weight", state)

    def test_codec_round_trip_and_wrapped_names(self):
        model = self._model()
        codec = MoEStateCodec.for_model(model)
        state = model.state_dict()
        encoded = codec.encode(state)
        self.assertIn("layers.1.mlp.experts.7.down_proj.weight", encoded)
        self.assertNotIn("layers.1.mlp.experts.down_proj", encoded)
        self.assertIn("fc.weight", encoded)
        decoded = codec.decode(encoded)
        self.assertEqual(set(decoded), set(state))
        for key, value in state.items():
            assert_close(decoded[key], value)
        wrapped = torch.nn.Module()
        wrapped._fsdp_wrapped_module = torch.nn.Module()
        wrapped._fsdp_wrapped_module.draft = model
        prefixes = MoEStateCodec.for_model(wrapped)._layers
        self.assertIn("draft.layers.0.mlp", prefixes)

    def test_file_io_sites_convert_names(self):
        from specforge.export.checkpoint_io import materialize_draft
        from specforge.export.to_hf import export_to_hf
        from specforge.modeling.auto import AutoDraftModel
        from specforge.training.model_loading import warm_start_draft_model

        source = self._model()
        config = source.config
        codec = MoEStateCodec.for_model(source)
        state = {
            "draft_state_dict": dict(codec.encode(source.state_dict())),
            "strategy": "dflash",
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "draft.json"
            payload = config.to_dict()
            payload.pop("_attn_implementation_autoset", None)
            config_path.write_text(json.dumps(payload))
            ckpt = Path(tmp) / "run-step1"
            ckpt.mkdir()
            torch.save(state, ckpt / "training_state.pt")

            def assert_same(model):
                for key, value in source.state_dict().items():
                    assert_close(model.state_dict()[key], value.to(model.dtype))

            assert_same(materialize_draft(state, str(config_path)))

            warm = DFlashDraftModel(config)
            warm_start_draft_model(
                warm, str(ckpt), draft_config=config, strategy="dflash"
            )
            assert_same(warm)

            export_dir = Path(tmp) / "hf"
            export_to_hf(str(ckpt), str(config_path), str(export_dir))
            from safetensors.torch import load_file

            exported = load_file(str(export_dir / "model.safetensors"))
            self.assertIn("layers.0.mlp.experts.3.gate_proj.weight", exported)
            self.assertNotIn("layers.0.mlp.experts.gate_up_proj", exported)
            reloaded, info = AutoDraftModel.from_pretrained(
                str(export_dir), output_loading_info=True
            )
            self.assertEqual(info["unexpected_keys"], [])
            assert_same(reloaded)

    def test_controller_saves_checkpoint_names(self):
        from specforge.training.controller import TrainerController

        model = self._model()
        codec = MoEStateCodec.for_model(model)

        class Backend:
            def state_dict(self):
                return {
                    "model": {
                        f"draft_model.{k}": v for k, v in model.state_dict().items()
                    },
                    "optimizer": None,
                    "rng": None,
                }

        class Strategy(_Strategy):
            name = "dflash"

            def checkpoint_state_filter(self, state_dict):
                return {k.replace("draft_model.", ""): v for k, v in state_dict.items()}

        class Core:
            backend = Backend()
            strategy = Strategy(model)

        with tempfile.TemporaryDirectory() as tmp:
            controller = TrainerController(
                Core(), run_id="run", output_dir=tmp, draft_state_codec=codec
            )
            controller.save_checkpoint(1)
            saved = torch.load(
                os.path.join(tmp, "run-step1", "training_state.pt"),
                weights_only=False,
            )["draft_state_dict"]
        self.assertIn("layers.0.mlp.experts.0.up_proj.weight", saved)
        self.assertNotIn("layers.0.mlp.experts.gate_up_proj", saved)

    def test_export_normalizer_writes_upstream_fields(self):
        import importlib.util

        script = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "gates"
            / "normalize_dflash_export.py"
        )
        spec = importlib.util.spec_from_file_location("normalize_dflash_export", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "architectures": ["DFlashDraftModel"],
                        "block_size": 16,
                        **_moe_keys(),
                        "dflash_config": {"moe_aux_loss_coeff": 0.002},
                    }
                )
            )
            normalized = module.normalize_export(str(path), 16)
        self.assertEqual(normalized["num_experts"], EXPERTS)
        self.assertEqual(normalized["router_aux_loss_coef"], 0.002)
        self.assertEqual(normalized["shared_expert_intermediate_size"], SHARED)
        self.assertEqual(resolve_moe_config(normalized).n_routed_experts, EXPERTS)


if __name__ == "__main__":
    unittest.main()
