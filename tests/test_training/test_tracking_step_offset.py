"""SPECFORGE_TRACKING_STEP_OFFSET shifts only the tracker step."""

import os
import unittest
from unittest import mock

from specforge.training.tracking import TrackerLogger


class _FakeTracker:
    def __init__(self):
        self.calls = []

    def log(self, values, step=None):
        self.calls.append((dict(values), step))

    def close(self):
        pass


class TestTrackingStepOffset(unittest.TestCase):
    def test_default_offset_is_zero(self):
        tracker, console = _FakeTracker(), []
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPECFORGE_TRACKING_STEP_OFFSET", None)
            logger = TrackerLogger(tracker, console_logger=lambda v, s: console.append(s))
            logger({"acc": 0.5}, step=10)
        values, step = tracker.calls[0]
        self.assertEqual(step, 10)
        self.assertEqual(values["train/optimizer_step"], 10.0)
        self.assertNotIn("train/segment_optimizer_step", values)
        self.assertEqual(console, [10])

    def test_offset_shifts_tracker_step_only(self):
        tracker, console = _FakeTracker(), []
        with mock.patch.dict(os.environ, {"SPECFORGE_TRACKING_STEP_OFFSET": "129168"}):
            logger = TrackerLogger(tracker, console_logger=lambda v, s: console.append(s))
            logger({"acc": 0.5}, step=10)
        values, step = tracker.calls[0]
        self.assertEqual(step, 129178)
        self.assertEqual(values["train/optimizer_step"], 129178.0)
        self.assertEqual(values["train/segment_optimizer_step"], 10.0)
        self.assertEqual(console, [10])  # console / checkpoints keep the run-local step


if __name__ == "__main__":
    unittest.main()
