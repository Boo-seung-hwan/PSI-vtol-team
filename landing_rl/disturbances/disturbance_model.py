"""Per-episode environmental disturbance model.

Phase 3 extraction. This module holds ONLY the per-episode constant wind
acceleration that previously lived on ``LandingEnv``:

    LandingEnv.reset()  ->  wind_accel sampling  ->  DisturbanceModel.reset(rng)

The sampling expression is copied verbatim from
``mujoco_rl/envs/env_prototype.py`` (the immutable reference): three scalar
``rng.uniform(...)`` calls in x, y, z order with the legacy bounds, assembled
into a ``np.float64`` array of shape ``(3,)``. Nothing is vectorized to
``size=3`` and no bound, order, dtype, or call count is changed.

Ownership
---------
Owns ONLY ``cfg`` and the current ``wind_accel`` vector. Does NOT own an RNG,
does NOT create one, and the constructor consumes zero random numbers. The
caller (``LandingEnv``) passes its own ``np_random`` into ``reset``.

This phase covers per-episode constant wind ONLY. dt jitter, body-rate /
thrust / translational process noise, target noise, observation noise, and
bounce restitution stay exactly where they are in ``LandingEnv``.
"""

from __future__ import annotations

import numpy as np


class DisturbanceModel:
    """Per-episode constant wind acceleration [m/s^2], NED."""

    def __init__(self, cfg):
        self.cfg = cfg
        # Matches the legacy LandingEnv.__init__ default; replaced (same object
        # semantics) by reset(). There is only ever one wind vector.
        self.wind_accel = np.zeros(3, dtype=np.float64)

    def reset(self, rng):
        """Sample the per-episode wind. ``rng`` must be ``LandingEnv.np_random``.

        Verbatim copy of the legacy reset() block. Returns the new vector so the
        env can keep ``self.wind_accel`` pointing at the same object.
        """
        self.wind_accel = np.array(
            [
                rng.uniform(
                    -self.cfg.wind_accel_xy_max_mps2,
                    self.cfg.wind_accel_xy_max_mps2,
                ),
                rng.uniform(
                    -self.cfg.wind_accel_xy_max_mps2,
                    self.cfg.wind_accel_xy_max_mps2,
                ),
                rng.uniform(
                    -self.cfg.wind_accel_z_max_mps2,
                    self.cfg.wind_accel_z_max_mps2,
                ),
            ],
            dtype=np.float64,
        )
        return self.wind_accel
