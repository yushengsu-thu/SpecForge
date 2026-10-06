# coding=utf-8
"""Serving-side MoE FFN (``specforge.serving.sglang_models``) vs the trainer.

The SGLang draft classes themselves need SGLang and are exercised on GPU by
``scripts/gates/check_dspark_moe_sglang_equivalence.py`` and the serving
recipes; here the plain-PyTorch FFN, the expert stacking and the strict
checkpoint check are covered on CPU.
"""

import unittest

import torch
from torch import nn

from specforge.modeling.draft.moe import MoEConfig, MoELayer, to_checkpoint_state_dict
from specforge.serving.sglang_models.moe_ffn import (
    DraftMoEFFN,
    routed_expert_count,
    stack_expert_weights,
    verify_moe_weights,
)

RECIPES = {
    "deepseek_v4": dict(
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        n_shared_experts=1,
        swiglu_limit=10.0,
    ),
    "qwen3": dict(
        scoring_func="softmax",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        n_shared_experts=0,
        swiglu_limit=0.0,
    ),
}


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _pair(preset, e=8, k=2, inter=32, hidden=64, bias_scale=0.3):
    recipe = RECIPES[preset]
    ref = MoELayer(
        MoEConfig(
            preset=preset,
            n_routed_experts=e,
            num_experts_per_tok=k,
            moe_intermediate_size=inter,
            router="topk",
            balance="noaux_tc",
            experts_backend="grouped",
            shared_expert="swiglu",
            shared_expert_gate="none",
            dispatch="sorted_loop",
            **recipe,
        ),
        hidden,
    )
    ref.reset_parameters(0.02)
    ref.gate.balance.bias.normal_(0, bias_scale)
    ref = ref.to(torch.bfloat16).eval()
    ref.gate.balance.bias.data = ref.gate.balance.bias.data.float()
    state = to_checkpoint_state_dict(
        {n: v.detach() for n, v in ref.state_dict().items()}
    )
    # The keys MoEConfig.serving_fields() writes into config.json.
    cfg = _Cfg(
        hidden_size=hidden,
        n_routed_experts=e,
        num_experts_per_tok=k,
        moe_intermediate_size=inter,
        n_group=1,
        topk_group=1,
        topk_method="noaux_tc",
        hidden_act="silu",
        moe_preset=preset,
        **recipe,
    )
    sg = DraftMoEFFN(cfg).to(torch.bfloat16).eval()
    sg.gate.bias.data = sg.gate.bias.data.float()
    stacked = dict(stack_expert_weights(list(state.items())))
    params = dict(sg.named_parameters())
    assert set(stacked) == set(params), set(stacked) ^ set(params)
    with torch.no_grad():
        for name, tensor in stacked.items():
            params[name].copy_(tensor.to(params[name].dtype))
    return ref, sg, hidden


class TestServingMoEFFN(unittest.TestCase):
    def test_matches_trainer_layer_for_both_presets(self):
        torch.manual_seed(0)
        for preset in RECIPES:
            with self.subTest(preset=preset):
                ref, sg, hidden = _pair(preset)
                x = torch.randn(23, hidden, dtype=torch.bfloat16)
                with torch.no_grad():
                    y_ref, y_sg = ref(x), sg(x)
                    routing = ref.gate(x)
                    w_sg, i_sg, counts = sg.route(x)
                self.assertTrue(
                    torch.equal(routing.indices.sort(-1).values, i_sg.sort(-1).values)
                )
                torch.testing.assert_close(
                    routing.weights.sort(-1).values.float(),
                    w_sg.sort(-1).values,
                    atol=1e-5,
                    rtol=0,
                )
                self.assertEqual(int(counts.sum()), 23 * sg.topk)
                torch.testing.assert_close(y_sg.float(), y_ref.float(), atol=0, rtol=0)
                self.assertEqual(sg.shared_experts is not None, preset == "deepseek_v4")

    def test_qwen3_preset_defaults_apply_when_config_omits_recipe_keys(self):
        cfg = _Cfg(
            hidden_size=16,
            num_experts=4,  # Qwen spelling of the expert count
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            moe_preset="qwen3",
        )
        ffn = DraftMoEFFN(cfg)
        self.assertEqual(ffn.n_experts, 4)
        self.assertEqual(ffn.scoring_func, "softmax")
        self.assertEqual(ffn.routed_scaling_factor, 1.0)
        self.assertIsNone(ffn.shared_experts)
        self.assertIsNotNone(ffn.gate.bias)
        self.assertEqual(routed_expert_count(cfg), 4)
        self.assertEqual(routed_expert_count(_Cfg(n_routed_experts=0)), 0)

    def test_stacking_rejects_a_truncated_expert_set(self):
        with self.assertRaisesRegex(ValueError, "missing expert indices"):
            stack_expert_weights(
                [
                    ("layers.0.mlp.experts.0.w1.weight", torch.zeros(2, 2)),
                    ("layers.0.mlp.experts.2.w1.weight", torch.zeros(2, 2)),
                ]
            )
        stacked = dict(
            stack_expert_weights(
                [
                    ("layers.0.mlp.experts.1.w2.weight", torch.ones(2, 3)),
                    ("layers.0.mlp.experts.0.w2.weight", torch.zeros(2, 3)),
                    ("layers.0.mlp.gate.weight", torch.zeros(2, 2)),
                ]
            )
        )
        self.assertEqual(tuple(stacked["layers.0.mlp.experts.w2"].shape), (2, 2, 3))
        self.assertEqual(float(stacked["layers.0.mlp.experts.w2"][1].sum()), 6.0)
        self.assertIn("layers.0.mlp.gate.weight", stacked)

    def test_verify_rejects_mismatched_ffn_entries(self):
        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Linear(2, 2, bias=False)
                self.self_attn = nn.Linear(2, 2, bias=False)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Layer()])

        model = Model()
        verify_moe_weights(
            model, {"layers.0.mlp.weight", "layers.0.self_attn.weight"}, "T"
        )
        with self.assertRaisesRegex(ValueError, "not in the checkpoint"):
            verify_moe_weights(model, {"layers.0.self_attn.weight"}, "T")
        with self.assertRaisesRegex(ValueError, "do not map to any parameter"):
            verify_moe_weights(
                model,
                {"layers.0.mlp.weight", "layers.0.mlp.experts.w1"},
                "T",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
