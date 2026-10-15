# coding=utf-8
"""Qwen3-8B DSpark MoE drafter (``configs/qwen3-8b-dspark-moe.json``).

The drafter keeps Kan's Qwen3.8-27B MoE structure (``qwen3_5_moe`` preset,
one sigmoid-gated SwiGLU shared expert, sample router centering, auxiliary
balance loss, grouped_mm dispatch) at the Qwen3-8B target width: 8 routed
experts of width 4096, top-1 routing. Top-1 is only sound without top-k
renormalization, which is what the tests below pin down.
"""

import json
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.moe import (
    MoELayer,
    collect_moe_aux_loss,
    collect_moe_metrics,
    iter_moe_layers,
    resolve_moe_config,
    to_checkpoint_state_dict,
)
from specforge.modeling.draft.moe.aux_loss import AuxLossController

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "qwen3-8b-dspark-moe.json"
DENSE_CONFIG_PATH = REPO_ROOT / "configs" / "qwen3-8b-dspark.json"
HIDDEN = 32


def _json(**overrides):
    payload = dict(
        moe_preset="qwen3_5_moe",
        n_routed_experts=8,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=24,
        norm_topk_prob=False,
        dflash_config={"moe_aux_loss_coeff": 0.01},
    )
    payload.update(overrides)
    return payload


def _layer(**overrides) -> MoELayer:
    torch.manual_seed(0)
    layer = MoELayer(resolve_moe_config(_json(**overrides)), HIDDEN)
    layer.reset_parameters(std=0.05)
    if layer.shared_experts is not None:
        for p in layer.shared_experts.parameters():
            nn.init.normal_(p, std=0.05)
    return layer


def _draft_config(**overrides):
    """A tiny DSpark config with the Qwen3-8B drafter's MoE recipe."""
    dflash_config = {
        "attention_mode": "gqa",
        "projector_type": "dspark",
        "markov_rank": 4,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "target_layer_ids": [1, 3],
        "moe_aux_loss_coeff": 0.001,
        "moe_dispatch": "grouped_mm",
        "moe_router_center": "sample",
        "moe_router_center_momentum": 0.99,
    }
    fields = dict(
        architectures=["DSparkDraftModel"],
        block_size=3,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=2,
        num_target_layers=6,
        head_dim=16,
        max_position_embeddings=64,
        vocab_size=32,
        layer_types=["full_attention", "full_attention"],
        initializer_range=0.02,
        **_json(dflash_config=dflash_config),
    )
    fields.update(overrides)
    config = Qwen3Config(**fields)
    config._attn_implementation = "sdpa"
    return config


class TestCheckedInConfig(unittest.TestCase):
    def test_resolves_to_kans_structure_at_qwen3_8b_width(self):
        payload = json.loads(CONFIG_PATH.read_text())
        cfg = resolve_moe_config(payload)
        self.assertEqual(cfg.preset, "qwen3_5_moe")
        self.assertEqual(cfg.n_routed_experts, 8)
        self.assertEqual(cfg.num_experts_per_tok, 1)
        self.assertEqual(cfg.moe_intermediate_size, 4096)
        self.assertEqual(cfg.n_shared_experts, 1)
        self.assertEqual(cfg.shared_expert_intermediate_size, 2048)
        self.assertFalse(cfg.norm_topk_prob)
        # routing recipe inherited from the preset
        self.assertEqual(cfg.scoring_func, "softmax")
        self.assertEqual(cfg.balance, "aux_loss")
        self.assertEqual(cfg.shared_expert_gate, "sigmoid")
        # Kan's training-only knobs
        self.assertEqual(cfg.aux_loss_coeff, 0.001)
        self.assertEqual(cfg.dispatch, "grouped_mm")
        self.assertEqual(cfg.router_center, "sample")
        self.assertEqual(cfg.router_center_momentum, 0.99)

    def test_everything_but_the_ffn_matches_the_dense_qwen3_8b_drafter(self):
        moe = json.loads(CONFIG_PATH.read_text())
        dense = json.loads(DENSE_CONFIG_PATH.read_text())
        moe_only = {
            "moe_preset",
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "n_shared_experts",
            "shared_expert_intermediate_size",
            "norm_topk_prob",
        }
        self.assertEqual(set(moe) - set(dense), moe_only)
        for key in dense:
            if key == "dflash_config":
                continue
            self.assertEqual(moe[key], dense[key], key)
        dense_dflash = dense["dflash_config"]
        moe_dflash = moe["dflash_config"]
        for key, value in dense_dflash.items():
            self.assertEqual(moe_dflash[key], value, key)
        self.assertEqual(
            set(moe_dflash) - set(dense_dflash),
            {
                "moe_aux_loss_coeff",
                "moe_dispatch",
                "moe_router_center",
                "moe_router_center_momentum",
            },
        )
        self.assertEqual(moe["hidden_size"], 4096)
        self.assertEqual(moe["dflash_config"]["mask_token_id"], 151669)
        self.assertEqual(moe["dflash_config"]["target_layer_ids"], [1, 9, 17, 25, 33])

    def test_serving_fields_carry_the_top1_recipe(self):
        fields = resolve_moe_config(json.loads(CONFIG_PATH.read_text())).serving_fields()
        self.assertEqual(fields["n_routed_experts"], 8)
        self.assertEqual(fields["num_experts_per_tok"], 1)
        self.assertEqual(fields["moe_intermediate_size"], 4096)
        self.assertIs(fields["norm_topk_prob"], False)
        self.assertEqual(fields["topk_method"], "greedy")
        self.assertNotIn("swiglu_limit", fields)

    def test_builds_on_meta_at_the_expected_scale(self):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(str(CONFIG_PATH))
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = DSparkDraftModel(config)
        layers = list(iter_moe_layers(model))
        self.assertEqual(len(layers), 5)
        per_expert = 3 * 4096 * 4096
        routed = sum(p.numel() for l in layers for p in l.experts.parameters())
        self.assertEqual(routed, 5 * 8 * per_expert)
        shared = sum(p.numel() for l in layers for p in l.shared_experts.parameters())
        # SwiGLU shared expert of width 2048 plus its sigmoid gate (4096 -> 1)
        self.assertEqual(shared, 5 * (3 * 4096 * 2048 + 4096))
        total = sum(p.numel() for p in model.parameters())
        active = total - routed + 5 * 1 * per_expert
        # 2.0B routed + ~0.5B everything else; top-1 keeps ~0.7B active
        self.assertAlmostEqual(routed / 1e9, 2.01, delta=0.02)
        self.assertGreater(total / 1e9, 2.3)
        self.assertLess(total / 1e9, 3.0)
        self.assertLess(active / 1e9, 1.0)
        official = to_checkpoint_state_dict(model.state_dict())
        self.assertIn("layers.4.mlp.experts.7.gate_proj.weight", official)
        self.assertIn("layers.4.mlp.shared_expert.gate_proj.weight", official)
        self.assertIn("layers.4.mlp.shared_expert_gate.weight", official)
        self.assertNotIn("layers.4.mlp.experts.8.gate_proj.weight", official)


class TestTop1Routing(unittest.TestCase):
    def test_top1_with_renormalization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "norm_topk_prob"):
            resolve_moe_config(_json(norm_topk_prob=True))

    def test_top1_combine_weight_is_the_softmax_probability(self):
        layer = _layer().train()
        x = torch.randn(64, HIDDEN)
        routing = layer.gate(x)
        self.assertEqual(routing.indices.shape, (64, 1))
        weights = routing.weights
        self.assertTrue(torch.all(weights > 0))
        self.assertTrue(torch.all(weights < 1))
        probs = (x.float() @ layer.gate.weight.float().t()).softmax(-1)
        expected = probs.gather(1, routing.indices)
        self.assertTrue(torch.allclose(weights.float(), expected, atol=1e-5))

    def test_top1_router_gets_a_task_gradient_without_the_balance_loss(self):
        layer = _layer(dflash_config={"moe_aux_loss_coeff": 0.0}).train()
        x = torch.randn(64, HIDDEN)
        out = layer(x)
        self.assertIsNone(collect_moe_aux_loss(layer))
        out.float().square().mean().backward()
        grad = layer.gate.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_every_token_uses_exactly_one_routed_expert(self):
        layer = _layer().train()
        x = torch.randn(96, HIDDEN)
        routing = layer.gate(x)
        self.assertEqual(tuple(routing.counts.shape), (8,))
        self.assertEqual(int(routing.counts.sum()), 96)
        layer(x)
        metrics = collect_moe_metrics(layer)
        self.assertIn("moe/load_frac_max", metrics)


class TestDSparkIntegration(unittest.TestCase):
    def _forward(self, model, batch=2):
        hidden = model.config.hidden_size
        block = model.config.block_size
        anchors = 5
        noise = torch.randn(batch, block, hidden, requires_grad=True)
        target_hidden = torch.randn(batch, anchors, model.fc.in_features)
        out = model(
            position_ids=torch.arange(anchors + block).expand(batch, -1),
            noise_embedding=noise,
            target_hidden=target_hidden,
            attention_mask=torch.ones(
                batch, 1, block, anchors + block, dtype=torch.bool
            ),
        )
        return out, noise

    def test_forward_backward_with_top1_experts(self):
        torch.manual_seed(7)
        model = DSparkDraftModel(_draft_config()).train()
        layers = list(iter_moe_layers(model))
        self.assertEqual(len(layers), 2)
        for layer in layers:
            self.assertEqual(layer.cfg.num_experts_per_tok, 1)
            self.assertFalse(layer.cfg.norm_topk_prob)
            self.assertIsInstance(layer.balance, AuxLossController)
            self.assertTrue(layer.shared_experts.gated)
        self.assertIsNotNone(model.markov_head)
        self.assertIsNotNone(model.confidence_head)
        out, noise = self._forward(model)
        self.assertTrue(torch.isfinite(out).all())
        aux = collect_moe_aux_loss(model)
        self.assertIsNotNone(aux)
        self.assertTrue(torch.isfinite(aux))
        loss = out.float().square().mean() + aux
        loss.backward()
        self.assertTrue(torch.isfinite(noise.grad).all())
        for layer in layers:
            params = dict(layer.named_parameters())
            for name in (
                "gate.weight",
                "experts.w1",
                "experts.w2",
                "shared_experts.gate.weight",
            ):
                grad = params[name].grad
                self.assertIsNotNone(grad, name)
                self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(float(params["gate.weight"].grad.abs().sum()), 0.0)
        # a second step is a plain forward (aux-loss policy keeps no deferred state)
        self._forward(model)
        self.assertIsNotNone(collect_moe_aux_loss(model))


if __name__ == "__main__":
    unittest.main()
