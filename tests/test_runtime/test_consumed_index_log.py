# coding=utf-8
"""The plan position of every online prompt rides into its sample ref, and the
durable ack writes one JSON line per optimizer step naming what it consumed."""

import json
import os
import tempfile
import unittest

from specforge.launch import (
    _consumed_log_path,
    _epoch_online_prompts,
    _epoch_prompt_indices,
)
from specforge.runtime.contracts import FeatureSpec, SampleRef
from specforge.runtime.control_plane.controller import DataFlowController
from specforge.runtime.control_plane.metadata_store import InMemoryMetadataStore


def _ref(sample_id: str, prompt_index: int, epoch: int = 0) -> SampleRef:
    return SampleRef(
        sample_id=sample_id,
        run_id="run0",
        source_task_id=f"task-{sample_id}",
        feature_store_uri=f"mooncake://run0/{sample_id}",
        feature_keys={"hidden_state": f"{sample_id}/hidden_state"},
        feature_specs={
            "hidden_state": FeatureSpec(
                name="hidden_state", shape=(2, 4), dtype="float32"
            )
        },
        strategy="eagle3",
        metadata={
            "target_repr": "hidden_state",
            "prompt_index": prompt_index,
            "epoch": epoch,
        },
    )


class ConsumedIndexLogTest(unittest.TestCase):
    def test_single_epoch_prompts_carry_their_plan_position(self):
        prompts = [{"task_id": f"t{i}", "payload": {"text": str(i)}} for i in range(6)]
        items = _epoch_online_prompts(prompts, 0, 1, seed=7)
        order = _epoch_prompt_indices(prompts, 0, seed=7)
        self.assertEqual([item["metadata"]["prompt_index"] for item in items], order)
        self.assertEqual({item["metadata"]["epoch"] for item in items}, {0})
        # One epoch keeps the task ids as they were; the source prompts are untouched.
        self.assertEqual([item["task_id"] for item in items], [f"t{i}" for i in order])
        self.assertNotIn("metadata", prompts[0])
        self.assertNotIn("prompt_epochs", items[0]["metadata"])
        multi = _epoch_online_prompts(prompts, 1, 3, seed=7)
        self.assertEqual(multi[0]["metadata"]["prompt_epochs"], 3)
        self.assertEqual(multi[0]["metadata"]["epoch"], 1)

    def test_ack_appends_the_step_and_its_prompt_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "consumed_steps.jsonl")
            dfc = DataFlowController(
                "run0", metadata_store=InMemoryMetadataStore(), consumed_log=log
            )
            dfc.commit_samples("producer", [_ref(f"s{i}", 100 + i) for i in range(3)])
            dfc.ack_train_refs("t0", ["s2", "s0"], global_step=7, optimizer_durable=True)
            dfc.ack_train_refs("t0", ["s1"], global_step=None)  # no step: not logged
            dfc.ack_train_refs("t0", ["s1", "unknown"], global_step=8)
            lines = [json.loads(line) for line in open(log, encoding="utf-8")]
            self.assertEqual([line["global_step"] for line in lines], [7, 8])
            self.assertEqual(lines[0]["prompt_index"], [102, 100])
            self.assertEqual(lines[0]["epoch"], [0, 0])
            self.assertEqual(lines[0]["sample_ids"], ["s2", "s0"])
            self.assertTrue(lines[0]["optimizer_durable"])
            self.assertEqual(lines[1]["prompt_index"], [101, None])
            self.assertEqual(lines[1]["count"], 2)

    def test_no_log_without_a_path(self):
        dfc = DataFlowController("run0", metadata_store=InMemoryMetadataStore())
        dfc.commit_samples("producer", [_ref("s0", 5)])
        dfc.ack_train_refs("t0", ["s0"], global_step=1)  # nothing to write to
        self.assertIsNone(_consumed_log_path(None))
        self.assertEqual(
            _consumed_log_path("/x/consumer-state/ledger.sqlite"),
            "/x/consumer-state/consumed_steps.jsonl",
        )
