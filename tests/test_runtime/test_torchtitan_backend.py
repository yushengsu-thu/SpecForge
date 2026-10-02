"""Focused adapter contracts; distributed numerical gates live separately."""

import unittest
from importlib import metadata
from unittest import mock

import torch
from torch import nn

from specforge.training.backend import ParallelConfig
from specforge.training.torchtitan_backend import (
    TorchTitanTrainingBackend,
    _decoder_layout,
    _load_torchtitan_components,
)


class _Draft(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.norm = nn.LayerNorm(4)
        self.fc = nn.Linear(8, 4)
        self.candidate_selector = nn.Linear(4, 8)


class TestTorchTitanDecoderLayout(unittest.TestCase):
    def test_layout_keeps_parameter_paths_and_object_identity(self):
        draft = _Draft()
        layers = draft.layers
        before = {name: id(parameter) for name, parameter in draft.named_parameters()}

        with _decoder_layout(draft) as decoder:
            self.assertIs(decoder, draft)
            self.assertIs(decoder.layers, layers)
            self.assertEqual(
                list(decoder.layers.items()), list(layers._modules.items())
            )
            self.assertIsNone(decoder.tok_embeddings)
            self.assertIsNone(decoder.lm_head)
            self.assertFalse(decoder.enable_weight_tying)
            self.assertEqual(
                {name: id(parameter) for name, parameter in decoder.named_parameters()},
                before,
            )

        self.assertIs(draft.layers, layers)
        self.assertFalse(hasattr(layers, "items"))
        for name in ("tok_embeddings", "lm_head", "enable_weight_tying"):
            self.assertFalse(hasattr(draft, name))
        self.assertEqual(
            {name: id(parameter) for name, parameter in draft.named_parameters()},
            before,
        )

    def test_failed_sharding_restores_existing_and_absent_attributes(self):
        draft = _Draft()
        draft.enable_weight_tying = True
        draft.tok_embeddings = None

        def previous_items():
            return "custom accessor"

        draft.layers.items = previous_items
        original_norm = draft.norm
        before = set(draft.state_dict())

        with self.assertRaisesRegex(RuntimeError, "sharding failed"):
            with _decoder_layout(draft):
                self.assertFalse(draft.enable_weight_tying)
                raise RuntimeError("sharding failed")

        self.assertTrue(draft.enable_weight_tying)
        self.assertIsNone(draft.tok_embeddings)
        self.assertFalse(hasattr(draft, "lm_head"))
        self.assertIs(draft.layers.items, previous_items)
        self.assertIs(draft.norm, original_norm)
        self.assertEqual(set(draft.state_dict()), before)

    def test_frozen_teacher_table_inside_draft_is_rejected(self):
        draft = _Draft()
        draft.lm_head = nn.Linear(4, 8).requires_grad_(False)
        before = set(draft.state_dict())
        with self.assertRaisesRegex(ValueError, "outside the trainable draft"):
            with _decoder_layout(draft):
                self.fail("must reject before modifying decoder layout")
        self.assertEqual(set(draft.state_dict()), before)
        self.assertFalse(hasattr(draft, "enable_weight_tying"))


class TestTorchTitanBackendContracts(unittest.TestCase):
    def test_unsupported_topologies_fail_without_importing_titan(self):
        cases = [
            (ParallelConfig(tp_size=2), "DP only"),
            (ParallelConfig(sp_ring_size=2), "DP only"),
            (ParallelConfig(sharding_strategy="NO_SHARD"), "FULL_SHARD"),
            (ParallelConfig(param_dtype=torch.float32), "BF16"),
        ]
        with mock.patch(
            "specforge.training.torchtitan_backend._load_torchtitan_components"
        ) as load:
            for config, message in cases:
                with self.subTest(config=config):
                    with self.assertRaisesRegex(ValueError, message):
                        TorchTitanTrainingBackend(config)
            load.assert_not_called()

    def test_missing_dependency_and_wrong_release_are_actionable(self):
        with mock.patch(
            "specforge.training.torchtitan_backend.metadata.version",
            side_effect=metadata.PackageNotFoundError("torchtitan"),
        ):
            with self.assertRaisesRegex(ImportError, "requires torchtitan==0.3.0"):
                _load_torchtitan_components()
        with mock.patch(
            "specforge.training.torchtitan_backend.metadata.version",
            return_value="0.2.0",
        ):
            with self.assertRaisesRegex(ImportError, "found '0.2.0'"):
                _load_torchtitan_components()

    def test_unwrapped_seam_preserves_frozen_tables_and_scales_trainable_grads(self):
        model = nn.Module()
        model.draft_model = _Draft()
        model.lm_head = nn.Linear(4, 8).requires_grad_(False)
        model.embed_tokens = nn.Embedding(8, 4).requires_grad_(False)
        before = set(model.state_dict())
        factory = mock.Mock(return_value=mock.Mock())
        backend = TorchTitanTrainingBackend(ParallelConfig(), optimizer_factory=factory)
        with mock.patch(
            "specforge.training.torchtitan_backend._load_torchtitan_components"
        ) as load:
            returned = backend.prepare_model(
                model, wrap=False, optimizer_target=model.draft_model
            )
            load.assert_not_called()
        factory.assert_called_once_with(model.draft_model)
        self.assertIs(returned, model)
        self.assertEqual(set(backend._module_state_dict()), before)
        parameter = model.draft_model.fc.weight
        parameter.grad = torch.full_like(parameter, 8.0)
        backend.scale_gradients(torch.tensor(0.25))
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 2.0))
        self.assertIsNone(model.lm_head.weight.grad)
        self.assertIsNone(model.embed_tokens.weight.grad)

    def test_accumulation_flags_cover_forward_and_restore_after_failure(self):
        backend = TorchTitanTrainingBackend(ParallelConfig())
        draft = mock.Mock()
        backend._draft = draft
        backend._wrapped = True
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            with backend.forward_context(is_boundary=False):
                draft.set_requires_gradient_sync.assert_called_once_with(False)
                draft.set_reshard_after_backward.assert_called_once_with(False)
                raise RuntimeError("forward failed")
        self.assertEqual(
            draft.set_requires_gradient_sync.call_args_list,
            [mock.call(False), mock.call(True)],
        )
        self.assertEqual(
            draft.set_reshard_after_backward.call_args_list,
            [mock.call(False), mock.call(True)],
        )

    def test_boundary_backward_reduces_the_accumulated_window(self):
        backend = TorchTitanTrainingBackend(ParallelConfig())
        draft = mock.Mock()
        backend._draft = draft
        backend._wrapped = True
        weight = torch.tensor(2.0, requires_grad=True)
        backend.backward(weight * 3.0, is_boundary=False)
        backend.backward(weight * 5.0, is_boundary=True)
        torch.testing.assert_close(weight.grad, torch.tensor(8.0))
        self.assertEqual(
            draft.set_requires_gradient_sync.call_args_list,
            [mock.call(False), mock.call(True)],
        )


if __name__ == "__main__":
    unittest.main()
