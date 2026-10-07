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
    DraftSharedExpert,
    merge_gate_up,
    routed_expert_count,
    shared_expert_as_experts,
    stack_expert_weights,
    to_native_names,
    to_sglang_module_entries,
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

    def test_qwen3_5_moe_recipe_matches_a_plain_reference(self):
        # Kan's Qwen3.8-27B DSpark MoE export: softmax over (x W^T + folded
        # centering bias), top-k renormalised, no scaling, sigmoid-gated shared
        # expert, Qwen checkpoint naming.
        torch.manual_seed(1)
        e, k, inter, hidden, shared = 8, 3, 16, 32, 24
        cfg = _Cfg(
            hidden_size=hidden,
            num_experts=e,
            num_experts_per_tok=k,
            moe_intermediate_size=inter,
            shared_expert_intermediate_size=shared,
            n_shared_experts=1,
            moe_preset="qwen3_5_moe",
            scoring_func="softmax",
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
            moe_router_bias=True,
        )
        ffn = DraftMoEFFN(cfg).float().eval()
        self.assertEqual(ffn.gate.bias_mode, "logit")
        self.assertEqual(ffn.shared_expert_gate, "sigmoid")
        # Qwen-named checkpoint entries, as the export writes them.
        w = {
            "layers.0.mlp.gate.weight": torch.randn(e, hidden) * 0.2,
            "layers.0.mlp.gate.bias": torch.randn(e),
            "layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(shared, hidden)
            * 0.1,
            "layers.0.mlp.shared_expert.up_proj.weight": torch.randn(shared, hidden)
            * 0.1,
            "layers.0.mlp.shared_expert.down_proj.weight": torch.randn(hidden, shared)
            * 0.1,
            "layers.0.mlp.shared_expert_gate.weight": torch.randn(1, hidden) * 0.1,
        }
        for i in range(e):
            w[f"layers.0.mlp.experts.{i}.gate_proj.weight"] = (
                torch.randn(inter, hidden) * 0.1
            )
            w[f"layers.0.mlp.experts.{i}.up_proj.weight"] = (
                torch.randn(inter, hidden) * 0.1
            )
            w[f"layers.0.mlp.experts.{i}.down_proj.weight"] = (
                torch.randn(hidden, inter) * 0.1
            )
        stacked = dict(stack_expert_weights(to_native_names(w.items())))
        params = dict(ffn.named_parameters())
        self.assertEqual(
            {n.removeprefix("layers.0.mlp.") for n in stacked}, set(params)
        )
        with torch.no_grad():
            for name, tensor in stacked.items():
                params[name.removeprefix("layers.0.mlp.")].copy_(tensor)
        x = torch.randn(11, hidden)
        with torch.no_grad():
            y = ffn(x)
            # Plain reference.
            logits = x @ w["layers.0.mlp.gate.weight"].T + w["layers.0.mlp.gate.bias"]
            probs = logits.softmax(-1)
            top_w, top_i = probs.topk(k, dim=-1)
            top_w = top_w / top_w.sum(-1, keepdim=True)
            ref = torch.zeros_like(x)
            for t in range(x.shape[0]):
                for j in range(k):
                    i = int(top_i[t, j])
                    h = torch.nn.functional.silu(
                        w[f"layers.0.mlp.experts.{i}.gate_proj.weight"] @ x[t]
                    ) * (w[f"layers.0.mlp.experts.{i}.up_proj.weight"] @ x[t])
                    ref[t] += top_w[t, j] * (
                        w[f"layers.0.mlp.experts.{i}.down_proj.weight"] @ h
                    )
            hs = torch.nn.functional.silu(
                x @ w["layers.0.mlp.shared_expert.gate_proj.weight"].T
            ) * (x @ w["layers.0.mlp.shared_expert.up_proj.weight"].T)
            ys = hs @ w["layers.0.mlp.shared_expert.down_proj.weight"].T
            ref += torch.sigmoid(x @ w["layers.0.mlp.shared_expert_gate.weight"].T) * ys
        torch.testing.assert_close(y, ref, atol=1e-4, rtol=1e-4)

    def test_to_native_names_maps_qwen_layout(self):
        out = dict(
            to_native_names(
                [
                    ("layers.2.mlp.experts.5.up_proj.weight", torch.zeros(1)),
                    ("layers.2.mlp.shared_expert.down_proj.weight", torch.zeros(1)),
                    ("layers.2.mlp.shared_expert_gate.weight", torch.zeros(1)),
                    ("layers.2.mlp.gate.weight", torch.zeros(1)),
                    ("layers.2.self_attn.q_proj.weight", torch.zeros(1)),
                ]
            )
        )
        self.assertEqual(
            set(out),
            {
                "layers.2.mlp.experts.5.w3.weight",
                "layers.2.mlp.shared_experts.w2.weight",
                "layers.2.mlp.shared_experts.gate.weight",
                "layers.2.mlp.gate.weight",
                "layers.2.self_attn.q_proj.weight",
            },
        )

    def test_shared_expert_splits_into_exact_kernel_experts(self):
        torch.manual_seed(2)
        hidden, width, chunk = 24, 32, 8  # 4 pieces
        shared = DraftSharedExpert(hidden, width, 0.0, gated=True).float()
        for p in shared.parameters():
            p.data.normal_(0, 0.3)
        w13, w2 = shared_expert_as_experts(shared, chunk)
        self.assertEqual(tuple(w13.shape), (4, 2 * chunk, hidden))
        self.assertEqual(tuple(w2.shape), (4, hidden, chunk))
        x = torch.randn(5, hidden)
        with torch.no_grad():
            ref = shared(x)
            gate = torch.sigmoid(shared.gate(x))
            out = torch.zeros_like(x)
            for c in range(4):
                g, u = (x @ w13[c].T).chunk(2, -1)
                out += gate * ((torch.nn.functional.silu(g) * u) @ w2[c].T)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
        with self.assertRaisesRegex(ValueError, "not a multiple"):
            shared_expert_as_experts(shared, 7)

    def test_merge_gate_up_builds_the_fused_layout(self):
        w1 = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
        w3 = -w1
        merged = dict(
            merge_gate_up(
                [
                    ("layers.0.mlp.experts.w1", w1),
                    ("layers.0.mlp.experts.w3", w3),
                    ("layers.0.mlp.experts.w2", torch.zeros(2, 4, 3)),
                    ("layers.1.mlp.experts.w1", w1),  # no partner: left as is
                ]
            )
        )
        self.assertEqual(tuple(merged["layers.0.mlp.experts.w13"].shape), (2, 6, 4))
        torch.testing.assert_close(merged["layers.0.mlp.experts.w13"][:, :3], w1)
        torch.testing.assert_close(merged["layers.0.mlp.experts.w13"][:, 3:], w3)
        self.assertIn("layers.0.mlp.experts.w2", merged)
        self.assertIn("layers.1.mlp.experts.w1", merged)
        self.assertNotIn("layers.1.mlp.experts.w13", merged)

    def test_sglang_module_entries_rename_experts_and_merge_shared(self):
        w1 = torch.arange(24, dtype=torch.float32).view(4, 6)  # shared gate [S, H]
        w3 = -w1
        w2 = torch.ones(6, 4)  # shared down [H, S]
        entries = {
            "gate.weight": torch.zeros(2, 6),
            "gate.bias": torch.zeros(2),
            "experts.w13": torch.zeros(2, 4, 6),
            "experts.w2": torch.zeros(2, 6, 2),
            "shared_experts.w1.weight": w1,
            "shared_experts.w3.weight": w3,
            "shared_experts.w2.weight": w2,
            "shared_experts.gate.weight": torch.zeros(1, 6),
        }
        mapped = to_sglang_module_entries(entries)
        self.assertEqual(
            set(mapped),
            {
                "gate.weight",
                "gate.bias",
                "experts.w13_weight",
                "experts.w2_weight",
                "shared_experts.gate_up_proj.weight",
                "shared_experts.down_proj.weight",
                "shared_experts.gate.weight",
            },
        )
        gate_up = mapped["shared_experts.gate_up_proj.weight"]
        self.assertEqual(tuple(gate_up.shape), (8, 6))
        torch.testing.assert_close(gate_up[:4], w1)
        torch.testing.assert_close(gate_up[4:], w3)
        self.assertIs(mapped["shared_experts.down_proj.weight"], w2)
        self.assertIs(mapped["experts.w13_weight"], entries["experts.w13"])
        # A dense-shared-free layer passes through untouched.
        self.assertEqual(
            set(to_sglang_module_entries({"gate.weight": w1, "experts.w13": w1})),
            {"gate.weight", "experts.w13_weight"},
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            to_sglang_module_entries({"shared_experts.w1.weight": w1})

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
