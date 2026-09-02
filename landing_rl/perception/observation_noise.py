"""Vehicle-state observation-noise sampling.

Phase 10 extraction. This helper holds ONLY the three per-``_get_obs()`` RNG
draws that add Gaussian sensor noise to the velocity, acceleration, and
attitude observation channels of ``LandingEnv``:

    velocity_noise     = rng.normal(0.0, cfg.velocity_obs_noise_std_mps,      size=3)
    acceleration_noise = rng.normal(0.0, cfg.acceleration_obs_noise_std_mps2, size=3)
    attitude_noise     = rng.normal(0.0, cfg.attitude_obs_noise_std_rad,      size=3)

Copied character-for-character from ``mujoco_rl/envs/env_prototype.py ::
LandingEnv._get_obs`` (the immutable reference), with only ``self.np_random ->
rng``. Three separate ``normal(size=3)`` calls, in the order velocity ->
acceleration -> attitude. Never merged into ``normal(..., size=9)``, never a
vector of standard deviations, never reordered, no ``np.random`` global state,
no component RNG, no caching / pre-sampling.

This phase moves ONLY RNG-sampling ownership. ``LandingEnv._get_obs`` stays the
assembly seam and keeps every deterministic transform: the ``vel + noise`` /
``accel + noise`` / ``attitude_obs += noise`` additions, the acceleration
``np.clip``, the ``attitude_obs`` construction (``self.attitude[0]``,
``self.attitude[1]``, ``self._yaw_error()``), the ``wrap_pi`` on the yaw
channel, the target-error and ``target_valid`` encoding, the
``np.concatenate``, and the final ``.astype(np.float32)``.

Zero-noise configs (``*_std == 0``) still execute ``rng.normal(0.0, 0.0,
size=3)`` three times and still consume RNG -- this is NOT optimized away.

Ownership
---------
Owns ONLY ``cfg``. Does NOT retain sampled noise, an RNG, velocity,
acceleration, attitude, target, previous action, or any episode/step/env
state -- ``sample`` returns three fresh arrays and keeps nothing. The
constructor consumes zero random numbers. ``LandingEnv`` passes its own
``np_random`` into ``sample`` (only from ``_get_obs``) and remains the sole
RNG owner.
"""

from __future__ import annotations

import numpy as np


class ObservationNoiseSampler:
    """Samples additive Gaussian noise for the vel / accel / attitude channels."""

    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(velocity_noise, acceleration_noise, attitude_noise)``,
        each a ``(3,)`` array. ``rng`` must be ``LandingEnv.np_random``.
        Consumes exactly three ``normal(0.0, std, size=3)`` draws in the order
        velocity -> acceleration -> attitude. Verbatim copy of the legacy
        ``_get_obs`` noise draws."""
        velocity_noise = rng.normal(
            0.0,
            self.cfg.velocity_obs_noise_std_mps,
            size=3,
        )
        acceleration_noise = rng.normal(
            0.0,
            self.cfg.acceleration_obs_noise_std_mps2,
            size=3,
        )
        attitude_noise = rng.normal(
            0.0,
            self.cfg.attitude_obs_noise_std_rad,
            size=3,
        )
        return velocity_noise, acceleration_noise, attitude_noise
