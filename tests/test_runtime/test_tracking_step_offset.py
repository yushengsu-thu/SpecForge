"""SPECFORGE_LOG_STEP_OFFSET shifts logged optimizer steps for a continuation run."""

from specforge.training.tracking import TrackerLogger


class _Tracker:
    def __init__(self):
        self.calls = []

    def log(self, values, step=None):
        self.calls.append((step, values["train/optimizer_step"]))

    def close(self):
        pass


def test_no_offset_by_default(monkeypatch):
    monkeypatch.delenv("SPECFORGE_LOG_STEP_OFFSET", raising=False)
    tracker = _Tracker()
    TrackerLogger(tracker)({"acc": 0.5}, 10)
    assert tracker.calls == [(10, 10.0)]


def test_offset_shifts_tracker_and_console_steps(monkeypatch):
    monkeypatch.setenv("SPECFORGE_LOG_STEP_OFFSET", "129173")
    tracker, console = _Tracker(), []
    logger = TrackerLogger(tracker, console_logger=lambda values, step: console.append(step))
    logger({"acc": 0.5}, 1)
    logger({"acc": 0.6}, 10)
    assert tracker.calls == [(129174, 129174.0), (129183, 129183.0)]
    assert console == [129174, 129183]
