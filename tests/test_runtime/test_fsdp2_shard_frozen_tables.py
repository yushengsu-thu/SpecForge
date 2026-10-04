"""``training.shard_frozen_tables``: frozen target tables in the FSDP2 root group."""

import json
import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from torch.distributed.tensor import DTensor

from specforge.training.backend import (
    BackendOptions,
    ParallelConfig,
    create_training_backend,
)


class TestShardFrozenSelection(unittest.TestCase):
    def test_fsdp1_rejects_option(self):
        from tests.test_runtime.test_fsdp2_backend import TinyComposite

        pc = ParallelConfig(world_size=1)
        backend = create_training_backend(
            "fsdp", pc, options=BackendOptions(shard_frozen_tables=True)
        )
        with self.assertRaisesRegex(ValueError, "shard_frozen_tables"):
            backend.prepare_model(TinyComposite(), optimizer_target=None)

    def test_config_requires_fsdp2(self):
        from specforge.config.schema import TrainingConfig

        with self.assertRaisesRegex(ValueError, "shard_frozen_tables"):
            TrainingConfig(shard_frozen_tables=True)
        self.assertTrue(
            TrainingConfig(backend="fsdp2", shard_frozen_tables=True).shard_frozen_tables
        )


def _worker(rank, world_size, port, results_dir):
    from tests.test_runtime import _fixtures as fx
    from tests.test_runtime.test_fsdp2_backend import TinyComposite, _optimizer

    fx.init_rank_distributed(rank, world_size, port=str(port))
    torch.manual_seed(0)
    replicated = TinyComposite().cuda()
    sharded = TinyComposite().cuda()
    sharded.load_state_dict(replicated.state_dict())
    x = torch.randn(4, 7, device="cuda")
    out = {}
    for name, model, opts in (
        ("replicated", replicated, BackendOptions()),
        ("sharded", sharded, BackendOptions(shard_frozen_tables=True)),
    ):
        # fp32 test models: match the backend's compute dtype like PR 915's gates.
        pc = ParallelConfig.from_distributed(param_dtype=torch.float32)
        backend = create_training_backend(
            "fsdp2", pc, optimizer_factory=_optimizer, options=opts
        )
        wrapped = backend.prepare_model(model, optimizer_target=model.draft_model)
        table_is_dtensor = isinstance(wrapped.lm_head.weight, DTensor)
        losses = []
        for _ in range(3):
            loss = wrapped(x).float().pow(2).mean()
            backend.backward(loss, is_boundary=True)
            backend.step()
            losses.append(float(loss.detach()))
        full = backend.state_dict()["model"]
        out[name] = {
            "losses": losses,
            "table_is_dtensor": table_is_dtensor,
            "has_lm_head_key": ("lm_head.weight" in full) if full else None,
        }
    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as f:
        json.dump(out, f)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestShardFrozenDistributed(unittest.TestCase):
    def test_sharded_tables_match_replicated(self):
        import torch.multiprocessing as mp

        results_dir = tempfile.mkdtemp(prefix="shard_frozen_")
        mp.spawn(_worker, args=(2, 29619, results_dir), nprocs=2, join=True)
        for rank in range(2):
            with open(os.path.join(results_dir, f"rank{rank}.json")) as f:
                out = json.load(f)
            self.assertFalse(out["replicated"]["table_is_dtensor"])
            self.assertTrue(out["sharded"]["table_is_dtensor"])
            for a, b in zip(out["replicated"]["losses"], out["sharded"]["losses"]):
                self.assertAlmostEqual(a, b, places=5)
            if rank == 0:
                self.assertTrue(out["sharded"]["has_lm_head_key"])


class TestDisaggregatedLaunchForwardsShardFrozenTables(unittest.TestCase):
    """The disaggregated runtime builds its trainers through its own call sites
    (offline consumer and online consumer); ``training.shard_frozen_tables`` must
    reach them exactly like the single-process path."""

    @staticmethod
    def _training():
        return {
            "strategy": "dflash",
            "role": "consumer",
            "max_steps": 1,
            "backend": "fsdp2",
            "shard_frozen_tables": True,
        }

    def test_online_consumer_receives_the_option(self):
        from specforge.algorithms.builtin import builtin_algorithm_registry
        from specforge.config import Config
        from specforge.training.disaggregated import _build_online

        cfg = Config.model_validate(
            {
                "model": {"target_model_path": "t", "draft_model_config": "d"},
                "data": {"prompts_path": "prompts.jsonl"},
                "training": self._training(),
                "deployment": {
                    "mode": "disaggregated",
                    "disaggregated": {
                        "control_dir": "/shared/shard_frozen_tables",
                        "backend": "mooncake",
                        "server_urls": ["http://capture:30000"],
                    },
                },
            }
        )
        bundle = types.SimpleNamespace(model=object(), target_head=None, strategy_kwargs={})
        with (
            mock.patch.dict(os.environ, {"DISAGG_REF_CHANNEL": "/shared/refs"}),
            mock.patch("specforge.runtime.data_plane.streaming_ref_channel.StreamingRefChannel"),
            mock.patch("specforge.training.disaggregated._mooncake_store", return_value=mock.Mock()),
            mock.patch("specforge.launch.build_disagg_online_consumer", return_value=_FakeFitTrainer()) as build,
        ):
            _build_online(
                cfg,
                algorithm=builtin_algorithm_registry().resolve("dflash"),
                build_model_bundle=lambda _cfg: bundle,
                prepare_prompts=mock.Mock(),
                optimizer_factory=mock.Mock(),
                logger=None,
            )
        options = build.call_args.kwargs["backend_options"]
        self.assertIsInstance(options, BackendOptions)
        self.assertTrue(options.shard_frozen_tables)

    def test_offline_consumer_receives_the_option(self):
        from specforge.algorithms.builtin import builtin_algorithm_registry
        from specforge.config import Config
        from specforge.training.disaggregated import _build_offline

        cfg = Config.model_validate(
            {
                "model": {"target_model_path": "t", "draft_model_config": "d"},
                "data": {"hidden_states_path": "features"},
                "training": self._training(),
                "deployment": {
                    "mode": "disaggregated",
                    "disaggregated": {"control_dir": "/shared/shard_frozen_tables", "backend": "mooncake"},
                },
            }
        )
        bundle = types.SimpleNamespace(model=object(), target_head=None, strategy_kwargs={})
        with (
            mock.patch.dict(os.environ, {"DISAGG_MANIFEST": "/shared/manifest.json"}),
            mock.patch("specforge.training.disaggregated._wait_for"),
            mock.patch("specforge.training.disaggregated._offline_store", return_value=mock.Mock()),
            mock.patch("specforge.runtime.data_plane.disagg_ingest.read_ref_manifest", return_value=[]),
            mock.patch("specforge.launch.build_disagg_offline_runtime", return_value=_FakeFitTrainer()) as build,
        ):
            _build_offline(
                cfg,
                algorithm=builtin_algorithm_registry().resolve("dflash"),
                build_model_bundle=lambda _cfg: bundle,
                optimizer_factory=mock.Mock(),
                logger=None,
            )
        options = build.call_args.kwargs["backend_options"]
        self.assertIsInstance(options, BackendOptions)
        self.assertTrue(options.shard_frozen_tables)


class _FakeFitTrainer:
    def fit(self):
        return 1


if __name__ == "__main__":
    unittest.main()
