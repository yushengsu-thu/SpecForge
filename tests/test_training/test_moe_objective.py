from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.testing import assert_close
from transformers import Qwen3Config

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.modeling.draft.domino import DominoDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.moe import (
    MoELayer,
    apply_pending_balance_updates,
    collect_moe_auxiliary_loss,
    resolve_moe_config,
)
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import StepOutput, finalize_step_output


def _config():
    return Qwen3Config(
        architectures=["DFlashDraftModel"],
        block_size=3,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        num_target_layers=4,
        head_dim=8,
        max_position_embeddings=64,
        vocab_size=32,
        layer_types=["full_attention"],
        moe_preset="qwen3_5_moe",
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        dflash_config={
            "target_layer_ids": [1],
            "mask_token_id": 0,
            "moe_aux_loss_coeff": 0.1,
            "conv_kernel_size": 3,
            "conv_group_size": 4,
            "selector_rank": 4,
            "selector_top_k": 4,
            "emb_dim": 8,
            "gru_hidden_dim": 8,
            "markov_rank": 4,
            "enable_confidence_head": True,
        },
    )


class _Strategy:
    def __init__(self, model):
        self.model = model

    def trainable_module(self):
        return self.model

    def forward_loss(self, batch, ctx):
        inputs, denominator = batch
        apply_pending_balance_updates(self.model)
        task = self.model(inputs).square().mean()
        return StepOutput(task, {}, loss_terms=(task * denominator, denominator))


class _Backend:
    def __init__(self, model):
        self.model = model
        self.parallel_config = SimpleNamespace(fsdp_process_group=None)
        self.gradients = None

    def backward(self, loss, *, is_boundary):
        loss.backward()

    def scale_gradients(self, factor):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.mul_(factor)

    def step(self):
        self.gradients = [param.grad.clone() for param in self.model.parameters()]


class TestMoEObjective(unittest.TestCase):
    def test_auxiliary_mean_preserves_accumulation_normalization(self):
        torch.manual_seed(7)
        model = MoELayer(resolve_moe_config(_config()), 16)
        reference = copy.deepcopy(model)
        backend = _Backend(model)
        core = TrainerCore(_Strategy(model), backend, accumulation_steps=2)
        batches = [
            (torch.randn(2, 3, 16), torch.tensor(2.0)),
            (torch.randn(2, 3, 16), torch.tensor(7.0)),
        ]
        expected_numerator = 0
        for inputs, denominator in batches:
            apply_pending_balance_updates(reference)
            task = reference(inputs).square().mean()
            auxiliary, _ = collect_moe_auxiliary_loss(reference)
            expected_numerator = expected_numerator + (task + auxiliary) * denominator
        expected_loss = expected_numerator / 9
        expected_loss.backward()
        self.assertFalse(core.train_step(batches[0]).optimizer_stepped)
        result = core.train_step(batches[1])
        self.assertTrue(result.optimizer_stepped)
        self.assertIn("moe/aux_loss", result.metrics)
        self.assertAlmostEqual(result.loss, expected_loss.item(), places=6)
        for actual, param in zip(backend.gradients, reference.parameters()):
            assert_close(actual, param.grad, atol=1e-7, rtol=1e-5)
        self.assertGreater(backend.gradients[0].abs().sum().item(), 0)

    def test_dense_output_is_unchanged_and_moe_is_consumed_once(self):
        dense = nn.Linear(16, 16)
        original = StepOutput(dense(torch.randn(2, 16)).sum(), {"existing": 1})
        self.assertIs(finalize_step_output(original, dense), original)
        moe = MoELayer(resolve_moe_config(_config()), 16)
        apply_pending_balance_updates(moe)
        original = StepOutput(moe(torch.randn(2, 3, 16)).square().mean(), {})
        finalized = finalize_step_output(original, moe)
        self.assertIn("moe/aux_loss", finalized.metrics)
        self.assertIs(finalize_step_output(finalized, moe), finalized)

    def test_family_inherits_moe_and_checkpoint_mode(self):
        for model_cls in (
            DFlashDraftModel,
            DFlash2DraftModel,
            DSparkDraftModel,
            DominoDraftModel,
        ):
            with self.subTest(model=model_cls.__name__):
                config = _config()
                config._attn_implementation = "sdpa"
                model = model_cls(config)
                self.assertIsInstance(model.layers[0].mlp, MoELayer)
                model.gradient_checkpointing_enable()
                self.assertFalse(
                    model.layers[0]._gradient_checkpointing_func.keywords[
                        "use_reentrant"
                    ]
                )
                with self.assertRaisesRegex(ValueError, "non-reentrant"):
                    model.gradient_checkpointing_enable({"use_reentrant": True})

    def test_checkpoint_recompute_preserves_full_model_gradients(self):
        torch.manual_seed(13)
        config = _config()
        config._attn_implementation = "sdpa"
        model = DFlashDraftModel(config).train()
        checkpointed = copy.deepcopy(model)
        checkpointed.gradient_checkpointing_enable()
        inputs = dict(
            target_hidden=torch.randn(2, 4, 16),
            noise_embedding=torch.randn(2, 3, 16),
            position_ids=torch.arange(7).expand(2, -1),
        )
        for current in (model, checkpointed):
            output = current(**inputs)
            finalized = finalize_step_output(
                StepOutput(output.square().mean(), {}), current
            )
            finalized.loss.backward()
            self.assertIsNone(collect_moe_auxiliary_loss(current)[0])
        for (_, left), (_, right) in zip(
            model.named_parameters(), checkpointed.named_parameters()
        ):
            assert_close(left.grad, right.grad, atol=1e-7, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
