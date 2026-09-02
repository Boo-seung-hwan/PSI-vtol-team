"""Target-observation latency (perception delay queue).

Phase 7 extraction. This helper holds ONLY the target observation-delay state
and FIFO that previously lived on ``LandingEnv``:

    * per-episode ``obs_delay_steps`` sampling (one ``rng.integers`` draw)
    * the ``target_queue`` list
    * the append / trim / ``[0]`` FIFO inside ``_update_target_pipeline``

The raw target-measurement stochastic model is NOT part of this component --
it is owned by ``TargetMeasurementModel`` (Phase 6). ``LandingEnv`` still owns
``target_true`` / ``target_measured`` / ``obs_target`` / ``obs_target_valid`` /
``obs_target_mode`` and still decides how many times to prime the pipeline
during ``reset``.

Verbatim copy
------------
The sampling expression and the queue logic are copied character-for-character
from ``mujoco_rl/envs/env_prototype.py`` (the immutable reference), with only
``self.np_random -> rng``:

    obs_delay_steps = int(rng.integers(
        cfg.obs_delay_steps_min, cfg.obs_delay_steps_max + 1))

    target_queue.append((raw_target.copy(), bool(valid), mode))
    max_len = max(1, int(obs_delay_steps) + 1)
    while len(target_queue) > max_len:
        target_queue.pop(0)
    delayed_target, delayed_valid, delayed_mode = target_queue[0]

The FIFO stays a plain Python ``list`` (no ``deque`` / ``deque(maxlen=...)`` /
ring buffer / numpy array). ``integers`` bounds and half-open behavior are
unchanged. The ``obs_delay_steps`` draw is the *second* ``integers`` draw in
``LandingEnv.reset`` -- it must stay AFTER ``ActionLatency.reset``'s
``action_delay_steps`` draw and BEFORE the response-alpha / wind / target
draws. The two integer draws are never merged or vectorized.

Ownership
---------
Owns ONLY ``cfg``, ``obs_delay_steps``, and ``target_queue``. Does NOT own an
RNG, the measurement model, ``target_true`` / ``target_measured`` /
``obs_target*``, position, velocity, action, dynamics, contact, or reward. The
constructor consumes zero random numbers; ``reset`` consumes exactly one
``integers(...)`` draw from the RNG it is handed. ``LandingEnv`` remains the
sole RNG owner.
"""

from __future__ import annotations

import numpy as np


class ObsLatency:
    """Fixed-step FIFO delay on the target measurement (observation latency)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.obs_delay_steps = 0
        self.target_queue: list[tuple[np.ndarray, bool, str]] = []

    def reset(self, rng) -> int:
        """Sample this episode's observation delay and clear the FIFO.

        ``rng`` must be ``LandingEnv.np_random``. Consumes exactly one
        ``rng.integers(...)`` call. Returns ``obs_delay_steps`` so the env can
        keep its compatibility mirror. ``self.target_queue`` is rebound to a
        fresh empty list; the caller should alias ``env.target_queue`` to it.
        """
        self.obs_delay_steps = int(
            rng.integers(
                self.cfg.obs_delay_steps_min,
                self.cfg.obs_delay_steps_max + 1,
            )
        )
        self.target_queue = []
        return self.obs_delay_steps

    def push_and_get(self, raw_target, valid, mode):
        """Append one raw measurement and return the delayed one.

        Verbatim copy of the queue block from the legacy
        ``LandingEnv._update_target_pipeline``. Returns the
        ``(delayed_target, delayed_valid, delayed_mode)`` tuple stored at
        ``target_queue[0]`` (the same tuple object the queue holds; the caller
        copies as the legacy code does).
        """
        self.target_queue.append((raw_target.copy(), bool(valid), mode))

        max_len = max(1, int(self.obs_delay_steps) + 1)
        while len(self.target_queue) > max_len:
            self.target_queue.pop(0)

        return self.target_queue[0]
