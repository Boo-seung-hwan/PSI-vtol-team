"""Generic fixed-step FIFO delay line.

Same pattern as ``landing_rl/envs/action_latency.py`` /
``landing_rl/perception/obs_latency.py`` (a plain Python list FIFO), but
this one operates at PHYSICS-substep granularity and is used for the
identified physical inner-loop delay (``T_delay``) -- a distinct concept
from action-path latency (``ActionLatency``, control-step granularity) and
observation/perception latency (``ObsLatency``, control-step granularity).
See task spec section 11: these three delays must never be conflated.
"""

from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


class DelayBuffer(Generic[T]):
    """Delays a scalar or ``np.ndarray`` signal by a fixed number of
    ``push_and_get`` calls (physics substeps)."""

    def __init__(self, delay_steps: int, initial_value: T):
        self.delay_steps = max(0, int(delay_steps))
        self._buffer: list[T] = [
            _copy(initial_value) for _ in range(self.delay_steps)
        ]

    def push_and_get(self, value: T) -> T:
        if self.delay_steps <= 0:
            return value
        self._buffer.append(_copy(value))
        return self._buffer.pop(0)

    def reset(self, initial_value: T) -> None:
        self._buffer = [_copy(initial_value) for _ in range(self.delay_steps)]


def _copy(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def steps_from_seconds(delay_s: float, physics_dt: float) -> int:
    """Convert a continuous-time delay to a whole number of physics
    substeps. Rounds to nearest; a delay smaller than half a physics step
    rounds to zero (no delay representable at this resolution)."""
    return int(round(float(delay_s) / float(physics_dt)))
