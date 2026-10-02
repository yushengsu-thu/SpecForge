"""Backend selection, constructor wiring, and checkpoint compatibility gates."""

import copy
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from pydantic import ValidationError

from specforge.config import Config
from specforge.training.backend import ParallelConfig


def _payload(**training):
    return {
        "model": {"target_model_path": "target", "draft_model_config": "draft.json"},
        "data": {"hidden_states_path": "/features"},
        "training": {"strategy": "dflash", **training},
    }


class TestTrainingBackendConfig(unittest.TestCase):
    def test_default_stays_fsdp_and_titan_round_trips(self):
        self.assertEqual(Config.model_validate(_payload()).training.backend, "fsdp")
        for strategy in ("dflash", "dspark"):
            with self.subTest(strategy=strategy):
                cfg = Config.model_validate(
                    _payload(strategy=strategy, backend="torchtitan")
                )
                restored = Config.model_validate(cfg.model_dump())
                self.assertEqual(restored.training.backend, "torchtitan")

    def test_unsupported_algorithm_topology_and_sharding_fail_early(self):
        invalid = (
            {"strategy": "eagle3"},
            {"tp_size": 2},
            {"sp_ulysses_size": 2},
            {"sp_ring_size": 2},
            {"fsdp_sharding": "NO_SHARD"},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValidationError):
                Config.model_validate(_payload(backend="torchtitan", **overrides))

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValidationError):
            Config.model_validate(_payload(backend="typo"))

    def test_non_bf16_or_non_text_models_are_rejected(self):
        for field, value in (("torch_dtype", "float32"), ("input_modality", "vlm")):
            payload = _payload(backend="torchtitan")
            payload["model"][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                Config.model_validate(payload)

    def test_common_launch_arguments_preserve_backend(self):
        from specforge.training.assembly import _common_launch_kwargs

        cfg = Config.model_validate(
            _payload(backend="torchtitan", fsdp_sharding="FULL_SHARD")
        )
        with (
            mock.patch(
                "specforge.training.assembly._dataloader_num_workers", return_value=0
            ),
            mock.patch(
                "specforge.training.assembly._profiling_options", return_value=None
            ),
        ):
            kwargs = _common_launch_kwargs(
                cfg, SimpleNamespace(strategy_kwargs={}), SimpleNamespace(name="dflash")
            )
        self.assertEqual(kwargs["backend_name"], "torchtitan")
        self.assertEqual(kwargs["sharding_strategy"], "FULL_SHARD")

    def test_typed_sharding_wins_over_environment_and_direct_default_keeps_it(self):
        with (
            mock.patch.dict("os.environ", {"FSDP_SHARDING": "NO_SHARD"}),
            mock.patch("torch.distributed.is_initialized", return_value=False),
        ):
            explicit = ParallelConfig.from_distributed(sharding_strategy="FULL_SHARD")
            legacy = ParallelConfig.from_distributed()
        self.assertEqual(explicit.sharding_strategy, "FULL_SHARD")
        self.assertEqual(legacy.sharding_strategy, "NO_SHARD")

    def test_launch_seam_forwards_selected_backend_to_trainer(self):
        from specforge.launch import _assemble_trainer

        algorithm = SimpleNamespace(
            name="dflash",
            providers=SimpleNamespace(step=SimpleNamespace(build=mock.Mock())),
        )
        with mock.patch("specforge.training.Trainer") as constructor:
            result = _assemble_trainer(
                algorithm=algorithm,
                controller=mock.Mock(),
                store=mock.Mock(),
                ref_source={"refs": []},
                model=mock.Mock(),
                target_head=None,
                optimizer_factory=mock.Mock(),
                backend_name="torchtitan",
                sharding_strategy="FULL_SHARD",
                run_id="backend-gate",
                output_dir="/unused",
                batch_size=1,
                accumulation_steps=1,
                num_epochs=1,
                max_steps=1,
                save_interval=0,
                logger=None,
                log_interval=1,
                collate_fn=mock.Mock(),
            )
        self.assertIs(result, constructor.return_value)
        self.assertEqual(constructor.call_args.kwargs["backend_name"], "torchtitan")
        self.assertEqual(
            constructor.call_args.kwargs["sharding_strategy"], "FULL_SHARD"
        )


class TestTrainerBackendResumeContract(unittest.TestCase):
    def _build(self, *, backend_name="fsdp", resume_state=None):
        from specforge.training.trainer import Trainer
        from tests.test_runtime.test_checkpoint_resume import _fake_seam

        Composite, Strategy, _ = _fake_seam()
        model = Composite()
        backend = mock.Mock()
        backend.prepare_model.return_value = model
        self.addCleanup(backend.reset_mock)
        directory = tempfile.TemporaryDirectory(prefix="backend_contract_")
        self.addCleanup(directory.cleanup)
        parallel = ParallelConfig(sharding_strategy="SHARD_GRAD_OP")
        with (
            mock.patch("specforge.training.trainer.FeatureDataLoader"),
            mock.patch(
                "specforge.training.trainer.ParallelConfig.from_distributed",
                return_value=parallel,
            ),
            mock.patch(
                "specforge.training.trainer.FSDPTrainingBackend", return_value=backend
            ) as fsdp,
            mock.patch(
                "specforge.training.torchtitan_backend.TorchTitanTrainingBackend",
                return_value=backend,
            ) as titan,
            mock.patch("specforge.training.trainer.TrainerController") as controller,
        ):
            trainer = Trainer(
                algorithm_name="dflash",
                make_step_strategy=lambda wrapped, *, target_head: Strategy(wrapped),
                controller=mock.Mock(),
                store=mock.Mock(),
                ref_source={"refs": ["sample-0", "sample-1"]},
                model=model,
                target_head=None,
                optimizer_factory=lambda draft: None,
                backend_name=backend_name,
                run_id="backend-gate",
                output_dir=directory.name,
                batch_size=1,
                accumulation_steps=1,
                num_epochs=1,
                max_steps=1,
                total_steps=100,
                save_interval=0,
                logger=None,
                log_interval=1,
                collate_fn=mock.Mock(),
                durable_ack=False,
                resume_from="/provided-state" if resume_state is not None else None,
                resume_state=resume_state,
            )
        return trainer, backend, fsdp, titan, controller.call_args.kwargs

    @staticmethod
    def _resume_state(**extra):
        return {
            "strategy": "dflash",
            "world_size": 1,
            "effective_total_steps": 100,
            "draft_state_dict": {"w": torch.tensor([3.0])},
            "backend": {"optimizer": None, "rng": {}},
            "global_step": 0,
            "epoch": 0,
            "epoch_batch": 0,
            "epoch_samples": 0,
            **extra,
        }

    def test_constructor_selects_backend_and_persists_identity(self):
        for name in ("fsdp", "torchtitan"):
            with self.subTest(backend=name):
                trainer, backend, fsdp, titan, options = self._build(backend_name=name)
                self.assertIs(trainer.backend, backend)
                self.assertEqual(fsdp.call_count, int(name == "fsdp"))
                self.assertEqual(titan.call_count, int(name == "torchtitan"))
                self.assertEqual(options["checkpoint_extra"]["training_backend"], name)
                self.assertEqual(
                    options["checkpoint_extra"]["backend_sharding"], "SHARD_GRAD_OP"
                )

    def test_legacy_checkpoints_keep_fsdp_compatibility(self):
        _trainer, backend, fsdp, titan, _options = self._build(
            resume_state=self._resume_state()
        )
        fsdp.assert_called_once()
        titan.assert_not_called()
        backend.load_state_dict.assert_called_once_with({"optimizer": None, "rng": {}})

    def test_backend_changes_are_rejected_before_wrapping(self):
        for saved, requested in (("fsdp", "torchtitan"), ("torchtitan", "fsdp")):
            with self.subTest(saved=saved, requested=requested):
                with self.assertRaisesRegex(ValueError, "training_backend"):
                    self._build(
                        backend_name=requested,
                        resume_state=self._resume_state(
                            training_backend=saved, backend_sharding="SHARD_GRAD_OP"
                        ),
                    )
        with self.assertRaisesRegex(ValueError, "training_backend"):
            self._build(backend_name="torchtitan", resume_state=self._resume_state())

    def test_titan_requires_matching_recorded_sharding(self):
        for extra in ({}, {"backend_sharding": "FULL_SHARD"}):
            with (
                self.subTest(extra=extra),
                self.assertRaisesRegex(ValueError, "backend_sharding"),
            ):
                self._build(
                    backend_name="torchtitan",
                    resume_state=self._resume_state(
                        training_backend="torchtitan", **extra
                    ),
                )

    def test_matching_titan_checkpoint_restores_rank_state(self):
        state = self._resume_state(
            training_backend="torchtitan", backend_sharding="SHARD_GRAD_OP"
        )
        _trainer, backend, fsdp, titan, _options = self._build(
            backend_name="torchtitan", resume_state=copy.deepcopy(state)
        )
        fsdp.assert_not_called()
        titan.assert_called_once()
        backend.load_state_dict.assert_called_once_with(state["backend"])


if __name__ == "__main__":
    unittest.main()
