"""Native Qwen MoE weights survive the existing draft checkpoint boundaries."""

import json
import os
import tempfile
import unittest

import torch
from safetensors.torch import load_file

from specforge.export.checkpoint_io import materialize_draft
from specforge.export.to_hf import export_to_hf
from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
from specforge.modeling.draft.moe.layout import canonicalize_moe_state_dict
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.model_loading import warm_start_draft_model


def _payload():
    return {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "num_target_layers": 4,
        "head_dim": 4,
        "max_position_embeddings": 64,
        "vocab_size": 32,
        "layer_types": ["full_attention"],
        "moe_preset": "qwen3_5_moe",
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 12,
        "block_size": 4,
        "dflash_config": {
            "target_layer_ids": [1],
            "mask_token_id": 31,
            "moe_dispatch": "reference",
        },
    }


class DraftMoECheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="draft_moe_checkpoint_")
        self.addCleanup(self.tempdir.cleanup)
        self.config_path = os.path.join(self.tempdir.name, "draft.json")
        with open(self.config_path, "w", encoding="utf-8") as handle:
            json.dump(_payload(), handle)
        self.config = AutoDraftModelConfig.from_file(self.config_path)
        self.model = AutoDraftModel.from_config(self.config, torch_dtype=torch.float32)
        self.weights = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }
        # A value that is not representable in BF16 detects accidental export
        # materialization casts, not just key-set agreement.
        self.weights["layers.0.mlp.experts.gate_up_proj"][0, 0, 0] = 0.1234567
        self.model.load_state_dict(self.weights)
        self.state = {"strategy": "dflash", "draft_state_dict": self.weights}
        self.checkpoint_path = os.path.join(self.tempdir.name, "training_state.pt")
        torch.save(self.state, self.checkpoint_path)

    def assert_weights_equal(self, actual):
        self.assertEqual(set(actual), set(self.weights))
        for key, expected in self.weights.items():
            self.assertEqual(actual[key].dtype, expected.dtype, key)
            self.assertTrue(torch.equal(actual[key], expected), key)

    def test_native_keys_and_layout_identity(self):
        prefix = "layers.0.mlp."
        expected = {
            "gate.weight",
            "experts.gate_up_proj",
            "experts.down_proj",
            "shared_expert.gate_proj.weight",
            "shared_expert.up_proj.weight",
            "shared_expert.down_proj.weight",
            "shared_expert_gate.weight",
        }
        self.assertEqual(
            {key[len(prefix) :] for key in self.weights if key.startswith(prefix)},
            expected,
        )
        canonical = canonicalize_moe_state_dict(self.weights)
        for key, value in self.weights.items():
            self.assertIs(canonical[key], value)
        with self.assertRaisesRegex(ValueError, "unsupported draft MoE weight layout"):
            canonicalize_moe_state_dict(self.weights, layout="future_family")

    def test_backend_and_materialize_preserve_weights(self):
        # CPU verifies the same pre/post-gather native schema. The real FSDP
        # gather/load is exercised separately by the multi-GPU smoke test.
        backend = FSDPTrainingBackend(ParallelConfig())
        backend.prepare_model(self.model, wrap=False)
        self.assert_weights_equal(backend.state_dict()["model"])
        restored = materialize_draft(self.state, self.config_path)
        self.assert_weights_equal(restored.state_dict())
        backend.load_state_dict({"model": restored.state_dict()})
        self.assert_weights_equal(self.model.state_dict())

    def test_specforge_and_hf_warm_start(self):
        hf_dir = os.path.join(self.tempdir.name, "pretrained")
        self.model.save_pretrained(hf_dir)
        for source, expected_format in (
            (self.checkpoint_path, "specforge"),
            (hf_dir, "pretrained"),
        ):
            with self.subTest(format=expected_format):
                restored = AutoDraftModel.from_config(self.config)
                report = warm_start_draft_model(
                    restored,
                    source,
                    draft_config=self.config,
                    strategy="dflash",
                )
                self.assertEqual(report.checkpoint_format, expected_format)
                self.assert_weights_equal(restored.state_dict())

    def test_export_from_pretrained_and_meta_loading(self):
        output_dir = os.path.join(self.tempdir.name, "export")
        export_to_hf(self.checkpoint_path, self.config_path, output_dir)
        self.assert_weights_equal(
            load_file(os.path.join(output_dir, "model.safetensors"))
        )
        restored, info = AutoDraftModel.from_pretrained(
            output_dir,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        self.assertFalse(info["missing_keys"])
        self.assertFalse(info["unexpected_keys"])
        self.assertFalse(any(parameter.is_meta for parameter in restored.parameters()))
        self.assert_weights_equal(restored.state_dict())
        with open(os.path.join(output_dir, "config.json"), encoding="utf-8") as handle:
            exported_config = json.load(handle)
        self.assertEqual(exported_config["num_experts"], 4)
        self.assertTrue(exported_config["norm_topk_prob"])
        self.assertEqual(exported_config["moe_layout"], "qwen3_5_moe")

    def test_sharded_hf_meta_loading_preserves_native_fused_weights(self):
        output_dir = os.path.join(self.tempdir.name, "sharded")
        self.model.save_pretrained(output_dir, max_shard_size="2KB")
        with open(
            os.path.join(output_dir, "model.safetensors.index.json"), encoding="utf-8"
        ) as handle:
            index = json.load(handle)
        self.assertGreater(len(set(index["weight_map"].values())), 1)
        self.assertIn("layers.0.mlp.experts.gate_up_proj", index["weight_map"])
        restored, info = AutoDraftModel.from_pretrained(
            output_dir,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        self.assertFalse(info["missing_keys"])
        self.assertFalse(info["unexpected_keys"])
        self.assertFalse(any(parameter.is_meta for parameter in restored.parameters()))
        self.assert_weights_equal(restored.state_dict())

    def test_missing_and_wrong_shape_expert_weights_fail(self):
        key = "layers.0.mlp.experts.down_proj"
        missing = dict(self.weights)
        missing.pop(key)
        with self.assertRaisesRegex(ValueError, "missing non-embedding"):
            materialize_draft({"draft_state_dict": missing}, self.config_path)
        wrong_shape = {**self.weights, key: self.weights[key][:, :, :-1]}
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            materialize_draft({"draft_state_dict": wrong_shape}, self.config_path)
        with self.assertRaisesRegex(ValueError, "must have shape"):
            canonicalize_moe_state_dict(wrong_shape)
        with self.assertRaisesRegex(ValueError, "requires fused"):
            canonicalize_moe_state_dict(
                {"layers.0.mlp.experts.0.gate_proj.weight": torch.zeros(8, 16)}
            )

    def test_hf_missing_expert_init_preserves_loaded_parameters(self):
        output_dir = os.path.join(self.tempdir.name, "incomplete")
        missing_key = "layers.0.mlp.experts.down_proj"
        incomplete = {
            key: value for key, value in self.weights.items() if key != missing_key
        }
        self.model.save_pretrained(output_dir, state_dict=incomplete)
        restored, info = AutoDraftModel.from_pretrained(
            output_dir, torch_dtype=torch.float32, output_loading_info=True
        )
        self.assertEqual(set(info["missing_keys"]), {missing_key})
        restored_weights = restored.state_dict()
        for key, expected in incomplete.items():
            self.assertTrue(torch.equal(restored_weights[key], expected), key)
        self.assertTrue(torch.isfinite(restored_weights[missing_key]).all())
        self.assertGreater(restored_weights[missing_key].std().item(), 0)
        # A weights-only training warm start is stricter than HF's permissive
        # loader and must reject its newly initialized missing expert tensor.
        with self.assertRaisesRegex(ValueError, "missing draft weights"):
            warm_start_draft_model(
                AutoDraftModel.from_config(self.config),
                output_dir,
                draft_config=self.config,
                strategy="dflash",
            )

    def test_export_normalizer_persists_preset_defaults(self):
        from scripts.gates.normalize_dflash_export import normalize_export

        payload = _payload()
        payload.pop("shared_expert_intermediate_size")
        with open(self.config_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        normalized = normalize_export(self.config_path, expected_block_size=4)
        self.assertEqual(normalized["num_experts"], 4)
        self.assertTrue(normalized["norm_topk_prob"])
        self.assertGreater(normalized["shared_expert_intermediate_size"], 0)
        self.assertEqual(normalized["moe_layout"], "qwen3_5_moe")
        payload.pop("moe_preset")
        with open(self.config_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        with self.assertRaisesRegex(ValueError, "moe_preset"):
            normalize_export(self.config_path, expected_block_size=4)
        payload["num_experts"] = payload.pop("n_routed_experts")
        with open(self.config_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        with self.assertRaisesRegex(ValueError, "moe_preset"):
            normalize_export(self.config_path, expected_block_size=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
