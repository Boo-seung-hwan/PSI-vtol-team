"""Randomized initial kinematic-state sampling.

Phase 9 extraction. This helper holds ONLY the randomized initial-condition
draws performed near the TOP of ``LandingEnv.reset()`` (right after
``super().reset(seed=seed)`` / ``self.step_count = 0``):

    x initial position        rng.uniform(-init_xy_range_m, init_xy_range_m)
    y initial position        rng.uniform(-init_xy_range_m, init_xy_range_m)
    initial altitude          rng.uniform(init_altitude_min_m, init_altitude_max_m)
    z = -altitude             (NED: altitude above ground -> negative z)
    pos = np.array([x, y, z], dtype=np.float64)
    initial velocity          rng.uniform(-init_vel_range_mps, init_vel_range_mps, size=3).astype(np.float64)
    initial roll              rng.uniform(-init_roll_pitch_range_rad, init_roll_pitch_range_rad)
    initial pitch             rng.uniform(-init_roll_pitch_range_rad, init_roll_pitch_range_rad)
    initial yaw               rng.uniform(-init_yaw_range_rad, init_yaw_range_rad)
    attitude = np.array([roll, pitch, yaw], dtype=np.float64)

Copied character-for-character from ``mujoco_rl/envs/env_prototype.py`` (the
immutable reference), with only ``self.np_random -> rng``. Seven ``uniform``
draws in this order: x, y, altitude (scalars), velocity (``size=3``), roll,
pitch, yaw (scalars). Never merged into a vector draw, never vectorized across
the attitude scalars, never one combined call; no ``np.random`` global state.

In the legacy source the velocity draw and the attitude draws are separated by
deterministic zero/const assignments (``self.accel``, ``self.prev_accel``,
``self.prev_action``, ``self.target_yaw``). None of those consume RNG or feed
the sampled values, so folding all seven draws into one ``sample`` call leaves
the RNG stream byte-identical; ``LandingEnv`` performs those deterministic
assignments after the call. The authoritative OLD-vs-NEW regression proves no
observable change.

Ownership
---------
Owns ONLY ``cfg``. Does NOT retain ``pos`` / ``vel`` / ``attitude``, an RNG,
episode state, target, contact, controller, or dynamics state -- ``sample``
returns fresh arrays and keeps nothing. The constructor consumes zero random
numbers. ``LandingEnv`` continues to own ``self.pos`` / ``self.vel`` /
``self.attitude``.
"""

from __future__ import annotations

import numpy as np


class InitialStateSampler:
    """Samples a randomized initial (pos, vel, attitude). Stateless beyond cfg."""

    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(pos, vel, attitude)``, each ``float64`` shape ``(3,)``.

        ``rng`` must be ``LandingEnv.np_random``. Consumes exactly seven
        ``uniform`` draws in the legacy order (x, y, altitude, velocity, roll,
        pitch, yaw). Verbatim copy of the legacy initial-state block.
        """
        x = rng.uniform(-self.cfg.init_xy_range_m, self.cfg.init_xy_range_m)
        y = rng.uniform(-self.cfg.init_xy_range_m, self.cfg.init_xy_range_m)
        altitude = rng.uniform(
            self.cfg.init_altitude_min_m,
            self.cfg.init_altitude_max_m,
        )
        z = -altitude

        pos = np.array([x, y, z], dtype=np.float64)
        vel = rng.uniform(
            -self.cfg.init_vel_range_mps,
            self.cfg.init_vel_range_mps,
            size=3,
        ).astype(np.float64)

        attitude = np.array(
            [
                rng.uniform(
                    -self.cfg.init_roll_pitch_range_rad,
                    self.cfg.init_roll_pitch_range_rad,
                ),
                rng.uniform(
                    -self.cfg.init_roll_pitch_range_rad,
                    self.cfg.init_roll_pitch_range_rad,
                ),
                rng.uniform(
                    -self.cfg.init_yaw_range_rad,
                    self.cfg.init_yaw_range_rad,
                ),
            ],
            dtype=np.float64,
        )

        return pos, vel, attitude
