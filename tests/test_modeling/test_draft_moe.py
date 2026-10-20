"""Numerical contracts for the independently implemented Qwen draft FFN."""

import copy
import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from specforge.modeling.draft.moe import (
    MoEConfig,
    MoELayer,
    apply_pending_balance_updates,
    build_ffn,
    collect_moe_auxiliary_loss,
    resolve_moe_config,
)


def _config(**overrides):
    return replace(
        MoEConfig(
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=12,
        ),
        **overrides,
    )


def _manual_ffn(layer, hidden):
    """Token-by-token formula independent of the dispatch implementation."""
    flat = hidden.reshape(-1, hidden.shape[-1])
    logits = F.linear(flat, layer.gate.weight)
    probabilities = logits.float().softmax(-1)
    weights, indices = probabilities.topk(layer.config.num_experts_per_tok, -1)
    weights = weights / weights.sum(-1, keepdim=True)
    weights = weights.to(hidden.dtype) * layer.config.routed_scaling_factor
    rows = []
    for token, expert_ids in enumerate(indices):
        row = flat[token].new_zeros(flat.shape[-1])
        for slot, expert in enumerate(expert_ids):
            gate, up = F.linear(flat[token], layer.experts.gate_up_proj[expert]).chunk(
                2, -1
            )
            value = F.linear(F.silu(gate) * up, layer.experts.down_proj[expert])
            row = row + weights[token, slot] * value
        if layer.shared_expert is not None:
            shared = layer.shared_expert
            value = F.linear(
                F.silu(F.linear(flat[token], shared.gate_proj.weight))
                * F.linear(flat[token], shared.up_proj.weight),
                shared.down_proj.weight,
            )
            if layer.shared_expert_gate is not None:
                value = value * torch.sigmoid(
                    F.linear(flat[token], layer.shared_expert_gate.weight)
                )
            row = row + value
        rows.append(row)
    return torch.stack(rows).reshape_as(hidden)


class DraftMoEConfigTest(unittest.TestCase):
    def test_dense_factory_is_untouched_and_explicit_opt_in_required(self):
        config = SimpleNamespace(hidden_size=16)
        dense = nn.Identity()
        self.assertIs(build_ffn(config, dense=lambda supplied: dense), dense)
        self.assertIsNone(resolve_moe_config(config))
        for key in ("n_routed_experts", "num_experts"):
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "moe_preset"),
            ):
                resolve_moe_config(SimpleNamespace(**{key: 4}))

    def test_preset_overrides_are_persisted_in_source_config(self):
        config = {
            "moe_preset": "qwen3_5_moe",
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 12,
            "dflash_config": {"moe_aux_loss_coeff": 0.02, "moe_dispatch": "grouped_mm"},
        }
        resolved = resolve_moe_config(config)
        self.assertEqual(resolved, _config(aux_loss_coeff=0.02, dispatch="grouped_mm"))
        self.assertEqual(config["num_experts"], 4)
        self.assertEqual(config["moe_layout"], "qwen3_5_moe")
        self.assertNotIn("aux_loss_coeff", config)
        self.assertIsInstance(
            build_ffn({**config, "hidden_size": 16}, dense=lambda _: nn.Identity()),
            MoELayer,
        )

    def test_invalid_combinations_fail_early(self):
        cases = [
            ({"moe_preset": "deepseek_v4"}, "Unsupported"),
            ({"n_routed_experts": 2, "num_experts_per_tok": 3}, "cannot exceed"),
            ({"n_routed_experts": True}, "positive integer"),
            ({"num_experts": 8, "n_routed_experts": 4}, "must agree"),
            ({"moe_layout": "future"}, "layout"),
            ({"norm_topk_prob": False}, "norm_topk_prob"),
            ({"routed_scaling_factor": float("nan")}, "finite"),
            ({"swiglu_limit": float("inf")}, "finite"),
            ({"dflash_config": {"moe_aux_loss_coeff": True}}, "finite"),
            ({"scoring_func": "sqrtsoftplus"}, "softmax"),
            ({"dflash_config": {"moe_aux_loss_coeff": -1}}, "nonnegative"),
            ({"dflash_config": {"moe_dispatch": "typo"}}, "moe_dispatch"),
            ({"dflash_config": {"moe_unknown": True}}, "Unknown"),
        ]
        for override, message in cases:
            with (
                self.subTest(override=override),
                self.assertRaisesRegex(ValueError, message),
            ):
                resolve_moe_config({"moe_preset": "qwen3_5_moe", **override})


class DraftMoENumericsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_initialization_preserves_hf_loaded_parameters(self):
        layer = MoELayer(_config(), 16)
        loaded = layer.experts.gate_up_proj
        loaded._is_hf_initialized = True
        before = loaded.detach().clone()
        layer.reset_parameters(0.3)
        self.assertTrue(torch.equal(loaded, before))
        self.assertGreater(layer.experts.down_proj.std().item(), 0.2)

    def test_reference_forward_and_gradients_match_explicit_token_formula(self):
        layer = MoELayer(_config(), 16)
        other = copy.deepcopy(layer)
        hidden = torch.randn(2, 3, 16, requires_grad=True)
        other_hidden = hidden.detach().clone().requires_grad_()
        output = layer(hidden)
        expected = _manual_ffn(other, other_hidden)
        torch.testing.assert_close(output, expected)
        output.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(hidden.grad, other_hidden.grad)
        for (name, actual), (_, wanted) in zip(
            layer.named_parameters(), other.named_parameters()
        ):
            torch.testing.assert_close(actual.grad, wanted.grad, msg=name)

    def test_sample_centering_only_changes_router_inputs(self):
        layer = MoELayer(_config(router_center="sample"), 16)
        hidden = torch.randn(2, 5, 16)
        shift = torch.randn(2, 1, 16) * 10
        routes = []
        expert_inputs = []
        shared_inputs = []
        layer.gate.register_forward_hook(lambda _m, _i, output: routes.append(output))
        layer.experts.register_forward_pre_hook(
            lambda _m, inputs: expert_inputs.append(inputs[0])
        )
        layer.shared_expert.register_forward_pre_hook(
            lambda _m, inputs: shared_inputs.append(inputs[0])
        )
        layer(hidden)
        layer(hidden + shift)
        torch.testing.assert_close(
            routes[0].probabilities,
            routes[1].probabilities,
            rtol=0.00001,
            atol=0.000001,
        )
        self.assertTrue(torch.equal(routes[0].expert_indices, routes[1].expert_indices))
        torch.testing.assert_close(expert_inputs[0], hidden.reshape(-1, 16))
        torch.testing.assert_close(expert_inputs[1], (hidden + shift).reshape(-1, 16))
        torch.testing.assert_close(shared_inputs[0], hidden.reshape(-1, 16))
        torch.testing.assert_close(shared_inputs[1], (hidden + shift).reshape(-1, 16))
        layer(torch.randn(5, 16))

    def test_shared_sigmoid_gate_and_optional_shared_expert(self):
        layer = MoELayer(_config(), 16)
        with torch.no_grad():
            layer.experts.gate_up_proj.zero_()
            layer.experts.down_proj.zero_()
            layer.shared_expert_gate.weight.zero_()
        hidden = torch.randn(2, 3, 16)
        torch.testing.assert_close(layer(hidden), 0.5 * layer.shared_expert(hidden))
        layer = MoELayer(_config(shared_expert_intermediate_size=0), 16)
        self.assertIsNone(layer.shared_expert)
        self.assertIsNone(layer.shared_expert_gate)
        self.assertEqual(
            set(layer.state_dict()),
            {"gate.weight", "experts.gate_up_proj", "experts.down_proj"},
        )

    def test_auxiliary_matches_qwen_topk_convention_and_has_router_gradient(self):
        layer = MoELayer(_config(aux_loss_coeff=0.17), 16)
        hidden = torch.randn(2, 5, 16)
        layer(hidden)
        weighted, metrics = collect_moe_auxiliary_loss(layer)
        logits = F.linear(hidden.reshape(-1, 16), layer.gate.weight)
        probs = logits.float().softmax(-1)
        indices = probs.topk(2, -1).indices
        # Match Transformers' load_balancing_loss_func without relying on it.
        assignment_frequency = F.one_hot(indices, 4).float().mean(0)
        expected = (assignment_frequency * probs.mean(0).unsqueeze(0)).sum() * 4
        torch.testing.assert_close(weighted, expected * 0.17)
        torch.testing.assert_close(metrics["moe/aux_loss"], expected.detach())
        actual_grad = torch.autograd.grad(
            weighted, layer.gate.weight, retain_graph=True
        )[0]
        expected_grad = torch.autograd.grad(expected * 0.17, layer.gate.weight)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        self.assertGreater(actual_grad.abs().sum().item(), 0)
        self.assertEqual(collect_moe_auxiliary_loss(layer), (None, {}))

    def test_checkpointing_auxiliary_and_gradients_match_and_are_drained_once(self):
        eager = MoELayer(_config(aux_loss_coeff=0.3), 16)
        rematerialized = copy.deepcopy(eager)
        hidden = torch.randn(2, 5, 16, requires_grad=True)
        checkpoint_hidden = hidden.detach().clone().requires_grad_()
        apply_pending_balance_updates(eager)
        apply_pending_balance_updates(rematerialized)
        eager_output = eager(hidden)
        checkpoint_output = checkpoint(
            rematerialized, checkpoint_hidden, use_reentrant=False
        )
        eager_aux, eager_metrics = collect_moe_auxiliary_loss(eager)
        checkpoint_aux, checkpoint_metrics = collect_moe_auxiliary_loss(rematerialized)
        torch.testing.assert_close(eager_aux, checkpoint_aux)
        (eager_output.square().mean() + eager_aux).backward()
        (checkpoint_output.square().mean() + checkpoint_aux).backward()
        torch.testing.assert_close(hidden.grad, checkpoint_hidden.grad)
        for (name, actual), (_, wanted) in zip(
            eager.named_parameters(), rematerialized.named_parameters()
        ):
            torch.testing.assert_close(actual.grad, wanted.grad, msg=name)
        self.assertEqual(collect_moe_auxiliary_loss(rematerialized), (None, {}))
        for name in eager_metrics:
            torch.testing.assert_close(eager_metrics[name], checkpoint_metrics[name])
        apply_pending_balance_updates(rematerialized)
        rematerialized(torch.randn(1, 2, 16))
        self.assertIsNotNone(collect_moe_auxiliary_loss(rematerialized)[0])

    def test_multiple_layers_sum_weighted_aux_and_eval_has_no_objective(self):
        model = nn.Sequential(
            MoELayer(_config(aux_loss_coeff=0.1), 16),
            MoELayer(_config(aux_loss_coeff=0.2), 16),
        )
        apply_pending_balance_updates(model)
        model(torch.randn(2, 3, 16))
        expected = sum(layer._auxiliary_loss for layer in model)
        auxiliary, metrics = collect_moe_auxiliary_loss(model)
        torch.testing.assert_close(auxiliary, expected)
        self.assertIn("moe/router_entropy", metrics)
        model.eval()
        apply_pending_balance_updates(model)
        model(torch.randn(2, 3, 16))
        self.assertEqual(collect_moe_auxiliary_loss(model), (None, {}))

    def test_grouped_mm_does_not_silently_fallback_on_cpu(self):
        layer = MoELayer(_config(dispatch="grouped_mm"), 16)
        with self.assertRaisesRegex(RuntimeError, "grouped_mm"):
            layer(torch.randn(1, 2, 16))

    @unittest.skipUnless(
        torch.cuda.is_available() and hasattr(F, "grouped_mm"),
        "CUDA grouped_mm unavailable",
    )
    def test_grouped_mm_cuda_output_and_gradient_parity_with_empty_experts(self):
        reference = (
            MoELayer(
                _config(
                    n_routed_experts=8,
                    num_experts_per_tok=2,
                    moe_intermediate_size=32,
                    shared_expert_intermediate_size=32,
                ),
                32,
            )
            .cuda()
            .bfloat16()
        )
        reference.reset_parameters(0.1)
        grouped = copy.deepcopy(reference)
        grouped.config = replace(grouped.config, dispatch="grouped_mm")
        grouped.experts.config = grouped.config
        hidden = torch.randn(
            1, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        grouped_hidden = hidden.detach().clone().requires_grad_()
        output = reference(hidden)
        actual = grouped(grouped_hidden)
        torch.testing.assert_close(actual, output, rtol=0.03, atol=0.0002)
        output.float().square().sum().backward()
        actual.float().square().sum().backward()
        torch.testing.assert_close(
            grouped_hidden.grad, hidden.grad, rtol=0.05, atol=0.00002
        )
        for (name, wanted), (_, value) in zip(
            reference.named_parameters(), grouped.named_parameters()
        ):
            torch.testing.assert_close(
                value.grad, wanted.grad, rtol=0.06, atol=0.00002, msg=name
            )


if __name__ == "__main__":
    unittest.main()
