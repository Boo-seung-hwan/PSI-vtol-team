"""Per-episode response-alpha sampling.

Phase 8 extraction. This helper holds ONLY the four per-episode response
parameters sampled in ``LandingEnv.reset()``:

    vel_response_alpha        (3,)  float64   -- info-only, dynamics-inactive
    attitude_response_alpha   (2,)  float64   -- info-only, dynamics-inactive
    body_rate_response_alpha  (3,)  float64   -- used by the PX4-cascade proxy
    thrust_response_alpha     scalar float    -- used by the PX4-cascade proxy

Verbatim copy
------------
The four draws are copied character-for-character from
``mujoco_rl/envs/env_prototype.py`` (the immutable reference), with only
``self.np_random -> rng``:

    vel_response_alpha       = rng.uniform(vel_min,  vel_max,  size=3).astype(np.float64)
    attitude_response_alpha  = rng.uniform(att_min,  att_max,  size=2).astype(np.float64)
    body_rate_response_alpha = rng.uniform(br_min,   br_max,   size=3).astype(np.float64)
    thrust_response_alpha    = float(rng.uniform(thr_min, thr_max))

NOTE: the actual legacy source uses the *environment* RNG
(``self.np_random.uniform(...)``), not ``np.random`` global state. The Phase-7
NEXT STEP pseudocode wrote ``np.random.uniform(...)``; that was inaccurate and
is NOT followed here -- ``reset`` takes ``LandingEnv.np_random`` and the env
remains the sole RNG owner.

Exactly four ``uniform`` calls, in this order, same bounds, same ``size``
arguments, same ``.astype(np.float64)`` / ``float(...)`` conversions. The
draws are never merged, vectorized across parameters (no ``size=9``), or
reordered. The two info-only fields are still sampled -- they are part of the
frozen RNG stream and the frozen info contract.

Ownership
---------
Owns ONLY ``cfg`` and the four alpha fields. Does NOT own an RNG, position,
velocity, attitude, body rates, thrust state, action, target, contact,
``step_count``, or any environment reference. The constructor consumes zero
random numbers; ``reset`` consumes exactly four ``uniform`` draws from the RNG
it is handed.
"""

from __future__ import annotations

import numpy as np


class ResponseAlphas:
    """The four per-episode response-alpha parameters."""

    def __init__(self, cfg):
        self.cfg = cfg
        # Placeholders identical to the legacy LandingEnv.__init__ values;
        # overwritten by reset() before any read.
        self.vel_response_alpha = np.ones(3, dtype=np.float64) * 0.35
        self.attitude_response_alpha = np.ones(2, dtype=np.float64) * 0.40
        self.body_rate_response_alpha = np.ones(3, dtype=np.float64) * 0.40
        self.thrust_response_alpha = 0.30

    def reset(self, rng) -> None:
        """Sample this episode's response alphas. ``rng`` must be
        ``LandingEnv.np_random``. Consumes exactly four ``uniform`` draws
        (vel, attitude, body_rate, thrust -- in that order)."""
        self.vel_response_alpha = rng.uniform(
            self.cfg.vel_response_alpha_min,
            self.cfg.vel_response_alpha_max,
            size=3,
        ).astype(np.float64)

        self.attitude_response_alpha = rng.uniform(
            self.cfg.attitude_response_alpha_min,
            self.cfg.attitude_response_alpha_max,
            size=2,
        ).astype(np.float64)

        self.body_rate_response_alpha = rng.uniform(
            self.cfg.body_rate_response_alpha_min,
            self.cfg.body_rate_response_alpha_max,
            size=3,
        ).astype(np.float64)
        self.thrust_response_alpha = float(rng.uniform(
            self.cfg.thrust_response_alpha_min,
            self.cfg.thrust_response_alpha_max,
        ))
