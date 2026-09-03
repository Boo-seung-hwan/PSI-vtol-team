"""Rigid-body dynamics process-noise sampling.

Phase 11 extraction. This helper holds ONLY the three stochastic process-noise
draws performed inside ``LandingEnv._update_rigid_body_dynamics()``:

    1. body-rate         rng.normal(0.0, body_rate_process_noise_std_radps * noise_scale, size=3)
    2. thrust  (scalar)  rng.normal(0.0, thrust_process_noise_std_mps2 * noise_scale)
    3. translational      rng.normal(0.0, process_noise_vel_std_mps * noise_scale / max(dt_safe, 1e-6), size=3)

Copied character-for-character from ``mujoco_rl/envs/env_prototype.py`` (the
immutable reference), with only ``self.np_random -> rng``. Three separate
``normal`` calls, in the order body-rate -> thrust -> translational. The
deterministic physics between the draws (attitude integration, thrust-response
proxy, rotation / drag / gravity assembly) stays in ``LandingEnv``, so this is
deliberately NOT a single ``sample(rng, dt)`` call -- it is three explicit
methods called at their original, separated call sites.

Kept in LandingEnv (NOT here)
-----------------------------
``noise_scale = math.sqrt(dt_safe / max(cfg.dt, 1e-6))`` -- the dynamics
scaling expression stays at its current location; the sampler receives
``noise_scale`` (and, for the translational draw, ``dt_safe``) as arguments.

Exact semantics
---------------
* body-rate / translational: ``np.ndarray`` shape ``(3,)`` (whatever
  ``rng.normal(..., size=3)`` returns -- no extra cast).
* thrust: a SCALAR ``rng.normal(...)`` draw -- not ``size=1``, not an ndarray.
* translational std: ``process_noise_vel_std_mps * noise_scale / max(dt_safe,
  1e-6)`` -- exact left-to-right multiplication then division, verbatim. The
  config field name ``process_noise_vel_std_mps`` is NOT renamed even though
  the sampled quantity is used as acceleration noise.
* ``cfg.attitude_process_noise_std_rad`` is deliberately UNUSED -- the legacy
  dynamics performs exactly three process-noise draws, not four. Do not add a
  fourth.

Zero-noise configs (``*_std == 0``) still execute all three ``rng.normal``
calls and still consume RNG -- this is NOT optimized away.

Ownership
---------
Owns ONLY ``cfg``. Does NOT retain sampled noise, an RNG, ``dt``,
``noise_scale``, body rates, thrust, acceleration, velocity, position,
attitude, contact, or any episode/step/env state. The constructor consumes
zero random numbers. ``LandingEnv`` passes its own ``np_random`` into each
method and remains the sole RNG owner.
"""

from __future__ import annotations

import numpy as np


class ProcessNoiseSampler:
    """Samples the three rigid-body process-noise terms (body-rate / thrust /
    translational). Stateless beyond ``cfg``."""

    def __init__(self, cfg):
        self.cfg = cfg

    def sample_body_rate(self, rng, noise_scale) -> np.ndarray:
        """Body-rate process noise. ``rng`` must be ``LandingEnv.np_random``.
        Verbatim copy of the legacy body-rate ``normal(size=3)`` draw."""
        return rng.normal(
            0.0,
            self.cfg.body_rate_process_noise_std_radps * noise_scale,
            size=3,
        )

    def sample_thrust(self, rng, noise_scale):
        """Thrust process noise -- a SCALAR ``normal`` draw. Verbatim copy of
        the legacy thrust draw."""
        return rng.normal(
            0.0,
            self.cfg.thrust_process_noise_std_mps2 * noise_scale,
        )

    def sample_translational_accel(self, rng, noise_scale, dt_safe) -> np.ndarray:
        """Translational-acceleration process noise. Verbatim copy of the
        legacy draw, including ``/ max(dt_safe, 1e-6)``."""
        return rng.normal(
            0.0,
            self.cfg.process_noise_vel_std_mps * noise_scale / max(dt_safe, 1e-6),
            size=3,
        )
