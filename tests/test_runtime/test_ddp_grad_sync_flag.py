"""The trainer core arms DDP gradient reduction only on boundary micro-steps."""

import torch

from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import StepOutput


class _RecordingBackend:
    """Minimal backend: records the boundary flag seen before every forward."""

    def __init__(self):
        self.prepared = []
        self.backward_flags = []
        self.steps = 0

    def prepare_micro_step(self, is_boundary):
        self.prepared.append(bool(is_boundary))

    def backward(self, loss, *, is_boundary=True):
        self.backward_flags.append(bool(is_boundary))
        loss.backward()

    def step(self, **kwargs):
        self.steps += 1
        return torch.tensor(0.0)

    def scale_gradients(self, factor):
        return None


class _Strategy:
    def __init__(self):
        self.param = torch.nn.Parameter(torch.ones(()))
        self.backend_seen = []

    def forward_loss(self, batch, ctx):
        loss = self.param * 2.0
        return StepOutput(loss=loss, metrics={}, ratio_metrics={}, sum_metrics={})


def test_boundary_flag_is_set_before_each_forward():
    backend = _RecordingBackend()
    core = TrainerCore(_Strategy(), backend, accumulation_steps=3)
    for _ in range(6):
        core.train_step(batch=None)
    assert backend.prepared == [False, False, True, False, False, True]
    assert backend.backward_flags == backend.prepared
    assert backend.steps == 2
