import os
import unittest
from unittest import mock

from specforge.training.tracking import TrackerLogger


class _Recorder:
    def __init__(self):
        self.calls = []

    def log(self, values, step=None):
        self.calls.append((dict(values), step))

    def close(self):
        pass


class TestTrackerStepOffset(unittest.TestCase):
    def test_default_offset_is_zero(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPECFORGE_TRACKER_STEP_OFFSET", None)
            rec = _Recorder()
            TrackerLogger(rec)({"acc": 0.5}, 10)
        values, step = rec.calls[0]
        self.assertEqual(step, 10)
        self.assertEqual(values["train/optimizer_step"], 10.0)
        self.assertNotIn("train/run_optimizer_step", values)

    def test_offset_shifts_tracker_step_only(self):
        console = []
        with mock.patch.dict(os.environ, {"SPECFORGE_TRACKER_STEP_OFFSET": "88800"}):
            rec = _Recorder()
            TrackerLogger(rec, console_logger=lambda v, s: console.append(s))({"acc": 0.5}, 10)
        values, step = rec.calls[0]
        self.assertEqual(step, 88810)
        self.assertEqual(values["train/optimizer_step"], 88810.0)
        self.assertEqual(values["train/run_optimizer_step"], 10.0)
        self.assertEqual(console, [10])

    def test_rejects_invalid_offsets(self):
        for raw in ("abc", "-5"):
            with mock.patch.dict(os.environ, {"SPECFORGE_TRACKER_STEP_OFFSET": raw}):
                with self.assertRaises(ValueError):
                    TrackerLogger(_Recorder())


if __name__ == "__main__":
    unittest.main()
