"""Action-path latency.

Phase 5 extraction. This helper holds ONLY the command-path delay between the
current policy action and the action actually handed to the residual
controller. Three pieces of behavior move here verbatim from ``LandingEnv``:

    1. per-episode sampling of ``action_delay_steps``
    2. reset-time initialization of the zero-filled action FIFO
    3. the ``_apply_action_delay(raw_action)`` FIFO itself

Everything is copied character-for-character from
``mujoco_rl/envs/env_prototype.py`` (the immutable reference):

    action_delay_steps = int(np_random.integers(
        cfg.action_delay_steps_min, cfg.action_delay_steps_max + 1))

    action_queue = [np.zeros(3, dtype=np.float64)
                    for _ in range(max(0, action_delay_steps))]

    def _apply_action_delay(raw_action):
        if action_delay_steps <= 0:
            return raw_action.copy()
        action_queue.append(raw_action.copy())
        return action_queue.pop(0)

Nothing is vectorized, pre-sampled, or restructured; ``integers`` bounds and
inclusive/exclusive behavior are unchanged; the FIFO stays a plain ``list``
(not a ``deque`` / ring buffer / history array).

This is command-path latency, not perception, so it lives under
``landing_rl/envs/`` rather than ``landing_rl/perception/``.

Ownership
---------
Owns ONLY ``cfg``, ``action_delay_steps``, and ``action_queue``. Does NOT own
an RNG, ``obs_delay_steps``, the target queue / target measurement / target
dropout-stale-outlier, observation noise, ``prev_action``, ``action_delta``,
the residual target-valid gate, position, velocity, perception, the
controller, or the episode clock. The constructor consumes zero random
numbers. ``LandingEnv`` passes its own ``np_random`` in during ``reset`` only,
and ``reset`` consumes exactly one ``integers(...)`` draw. ``LandingEnv`` still
performs the ``obs_delay_steps`` draw itself, immediately after
``ActionLatency.reset``, so the relative RNG-draw order (action-delay before
obs-delay) is preserved.
"""

from __future__ import annotations

import numpy as np


class ActionLatency:
    """Fixed-step FIFO delay on the policy action (command-path latency)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.action_delay_steps = 0
        self.action_queue: list[np.ndarray] = []

    def reset(self, rng) -> int:
        """Sample this episode's action delay and (re)initialize the FIFO.

        ``rng`` must be ``LandingEnv.np_random``. Consumes exactly one
        ``rng.integers(...)`` call. Returns ``action_delay_steps`` so the env
        can keep its compatibility mirror. ``self.action_queue`` is rebound to
        a fresh list; the caller should alias ``env.action_queue`` to it.
        """
        self.action_delay_steps = int(
            rng.integers(
                self.cfg.action_delay_steps_min,
                self.cfg.action_delay_steps_max + 1,
            )
        )
        self.action_queue = [
            np.zeros(3, dtype=np.float64)
            for _ in range(max(0, self.action_delay_steps))
        ]
        return self.action_delay_steps

    def apply(self, raw_action: np.ndarray) -> np.ndarray:
        """Verbatim copy of the legacy ``_apply_action_delay``."""
        if self.action_delay_steps <= 0:
            return raw_action.copy()

        self.action_queue.append(raw_action.copy())
        return self.action_queue.pop(0)
