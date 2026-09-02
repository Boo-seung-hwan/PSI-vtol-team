"""Control-loop timing.

Phase 4 extraction. This helper holds ONLY the per-step control period sampling
that previously lived on ``LandingEnv``:

    LandingEnv._sample_dt()  ->  LoopTiming.sample_dt(rng)

The expression is copied verbatim from ``mujoco_rl/envs/env_prototype.py`` (the
immutable reference): one ``rng.normal(0.0, cfg.dt_jitter_std)`` draw added to
``cfg.dt``, then ``float(np.clip(dt, cfg.dt_min, cfg.dt_max))``. Nothing is
vectorized, pre-sampled, cached, or moved; ``normal`` is not swapped for
another API and the clip / float conversion is unchanged.

This is the simulation/control-loop clock behavior, not a disturbance, so it
lives under ``landing_rl/envs/`` rather than ``landing_rl/disturbances/``.

Ownership
---------
Owns ONLY ``cfg``. Does NOT own an RNG, ``step_count``, wall/sim time, episode
state, or dynamics state. The constructor consumes zero random numbers. The
caller (``LandingEnv.step``) passes its own ``np_random`` in and decides when
``sample_dt`` is called.

Not in scope: ``step_count``, ``max_steps``, truncation, the episode clock,
physics integration, and every dt-dependent dynamics calculation all stay in
``LandingEnv`` unchanged.
"""

from __future__ import annotations

import numpy as np


class LoopTiming:
    """Samples the control-step period dt [s] with per-step timing jitter."""

    def __init__(self, cfg):
        self.cfg = cfg

    def sample_dt(self, rng) -> float:
        """Return the control period for one step. ``rng`` must be
        ``LandingEnv.np_random``. Verbatim copy of the legacy ``_sample_dt``."""
        dt = self.cfg.dt + rng.normal(0.0, self.cfg.dt_jitter_std)
        return float(np.clip(dt, self.cfg.dt_min, self.cfg.dt_max))
