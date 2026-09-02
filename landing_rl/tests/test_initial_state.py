"""Focused tests for the Phase 9 InitialStateSampler (randomized reset state).

Secondary to the OLD-vs-NEW environment regression in
``test_legacy_regression_contract.py`` and the checkpoint gate in
``test_checkpoint_compatibility.py``, both of which remain authoritative.

These tests localize an initial-state regression by comparing
``InitialStateSampler.sample`` against a verbatim copy of the legacy
initial-state block, using independent generators started from the same state.
A reset-order test pins the corrected sequence: initial-state draws come
FIRST in ``reset()`` (before action-delay / obs-delay / response-alpha / wind).

Comparison is exact (``np.array_equal`` / ``==`` / identical RNG state).
No tolerance.
"""

from __future__ import annotations

import unittest

import numpy as np

from test_legacy_regression_contract import (
    CONFIG_SPECS,
    NewLandingConfig,
    NewLandingEnv,
    OldLandingConfig,
    OldLandingEnv,
    _rng_state_equal,
)

from landing_rl.envs.action_latency import ActionLatency
from landing_rl.envs.initial_state import InitialStateSampler
from landing_rl.dynamics import ResponseAlphas
from landing_rl.disturbances import DisturbanceModel
from landing_rl.perception import ObsLatency


def _legacy_initial_state(cfg, rng):
    """Verbatim copy of the initial-state block from
    mujoco_rl/envs/env_prototype.py :: LandingEnv.reset() (lines ~865-901)."""
    x = rng.uniform(-cfg.init_xy_range_m, cfg.init_xy_range_m)
    y = rng.uniform(-cfg.init_xy_range_m, cfg.init_xy_range_m)
    altitude = rng.uniform(cfg.init_altitude_min_m, cfg.init_altitude_max_m)
    z = -altitude
    pos = np.array([x, y, z], dtype=np.float64)
    vel = rng.uniform(
        -cfg.init_vel_range_mps, cfg.init_vel_range_mps, size=3
    ).astype(np.float64)
    attitude = np.array(
        [
            rng.uniform(-cfg.init_roll_pitch_range_rad, cfg.init_roll_pitch_range_rad),
            rng.uniform(-cfg.init_roll_pitch_range_rad, cfg.init_roll_pitch_range_rad),
            rng.uniform(-cfg.init_yaw_range_rad, cfg.init_yaw_range_rad),
        ],
        dtype=np.float64,
    )
    return pos, vel, attitude


def _overrides(name):
    for cfg_name, overrides, _doc in CONFIG_SPECS:
        if cfg_name == name:
            return overrides
    raise KeyError(name)


_CONFIG_NAMES = ("default", "stage0_eval", "stage2_eval", "stage2_train")


def _cfg(name, **extra):
    base = {} if name == "default" else dict(_overrides(name))
    base.update(extra)
    return NewLandingConfig(**base)


class InitialStateSamplerTest(unittest.TestCase):
    maxDiff = None

    # -- A. constructor consumes zero RNG -------------------------

    def test_constructor_consumes_no_rng(self):
        rng = np.random.default_rng(41)
        before = rng.bit_generator.state
        s = InitialStateSampler(NewLandingConfig())
        self.assertTrue(_rng_state_equal(before, rng.bit_generator.state))
        self.assertEqual(set(vars(s)), {"cfg"})
        self.assertFalse(hasattr(s, "np_random"))
        self.assertFalse(hasattr(s, "pos"))
        self.assertFalse(hasattr(s, "vel"))
        self.assertFalse(hasattr(s, "attitude"))

    def test_sample_keeps_no_state(self):
        s = InitialStateSampler(_cfg("stage2_train"))
        s.sample(np.random.default_rng(0))
        self.assertEqual(set(vars(s)), {"cfg"})  # nothing retained

    # -- B. exact parity vs verbatim legacy replica -------------

    def test_sample_matches_legacy_block_exactly(self):
        for cfg_name in _CONFIG_NAMES:
            cfg = _cfg(cfg_name)
            for seed in (0, 1, 5000, 999, 424242):
                with self.subTest(config=cfg_name, seed=seed):
                    rng_legacy = np.random.default_rng(seed)
                    rng_new = np.random.default_rng(seed)

                    pos_l, vel_l, att_l = _legacy_initial_state(cfg, rng_legacy)
                    pos_n, vel_n, att_n = InitialStateSampler(cfg).sample(rng_new)

                    self.assertTrue(np.array_equal(pos_n, pos_l),
                                    f"[{cfg_name} seed={seed}] pos "
                                    f"{pos_n!r} != {pos_l!r}")
                    self.assertTrue(np.array_equal(vel_n, vel_l),
                                    f"[{cfg_name} seed={seed}] vel")
                    self.assertTrue(np.array_equal(att_n, att_l),
                                    f"[{cfg_name} seed={seed}] attitude")
                    self.assertTrue(
                        _rng_state_equal(
                            rng_legacy.bit_generator.state,
                            rng_new.bit_generator.state,
                        ),
                        f"[{cfg_name} seed={seed}] RNG state diverged after "
                        f"initial-state block",
                    )

    def test_consumes_exactly_seven_uniform_draws_in_order(self):
        cfg = _cfg("stage2_train")
        for seed in (0, 1, 5000):
            with self.subTest(seed=seed):
                rng_new = np.random.default_rng(seed)
                InitialStateSampler(cfg).sample(rng_new)

                rng_ref = np.random.default_rng(seed)
                rng_ref.uniform(-cfg.init_xy_range_m, cfg.init_xy_range_m)
                rng_ref.uniform(-cfg.init_xy_range_m, cfg.init_xy_range_m)
                rng_ref.uniform(cfg.init_altitude_min_m, cfg.init_altitude_max_m)
                rng_ref.uniform(-cfg.init_vel_range_mps, cfg.init_vel_range_mps, size=3)
                rng_ref.uniform(-cfg.init_roll_pitch_range_rad,
                                cfg.init_roll_pitch_range_rad)
                rng_ref.uniform(-cfg.init_roll_pitch_range_rad,
                                cfg.init_roll_pitch_range_rad)
                rng_ref.uniform(-cfg.init_yaw_range_rad, cfg.init_yaw_range_rad)

                self.assertTrue(
                    _rng_state_equal(
                        rng_new.bit_generator.state, rng_ref.bit_generator.state
                    )
                )

    # -- C. shapes / dtypes ------------------------------------

    def test_shapes_and_dtypes(self):
        for cfg_name in _CONFIG_NAMES:
            pos, vel, att = InitialStateSampler(_cfg(cfg_name)).sample(
                np.random.default_rng(3)
            )
            for name, arr in (("pos", pos), ("vel", vel), ("attitude", att)):
                self.assertEqual(arr.shape, (3,), name)
                self.assertEqual(arr.dtype, np.dtype(np.float64), name)

    # -- D. NED altitude convention --------------------------

    def test_ned_altitude_convention(self):
        cfg = _cfg("default", init_altitude_min_m=3.0, init_altitude_max_m=8.0)
        for seed in range(30):
            pos, _, _ = InitialStateSampler(cfg).sample(np.random.default_rng(seed))
            self.assertLess(pos[2], 0.0)                       # NED: above ground -> z < 0
            self.assertGreaterEqual(-pos[2], cfg.init_altitude_min_m)
            self.assertLessEqual(-pos[2], cfg.init_altitude_max_m)

    # -- E/F/G/H. bounds -----------------------------------

    def test_xy_bounds(self):
        cfg = _cfg("default", init_xy_range_m=6.0)
        for seed in range(50):
            pos, _, _ = InitialStateSampler(cfg).sample(np.random.default_rng(seed))
            self.assertGreaterEqual(pos[0], -cfg.init_xy_range_m)
            self.assertLessEqual(pos[0], cfg.init_xy_range_m)
            self.assertGreaterEqual(pos[1], -cfg.init_xy_range_m)
            self.assertLessEqual(pos[1], cfg.init_xy_range_m)

    def test_altitude_bounds(self):
        cfg = _cfg("default", init_altitude_min_m=2.0, init_altitude_max_m=7.0)
        alts = []
        for seed in range(80):
            pos, _, _ = InitialStateSampler(cfg).sample(np.random.default_rng(seed))
            alt = -pos[2]
            self.assertGreaterEqual(alt, cfg.init_altitude_min_m)
            self.assertLessEqual(alt, cfg.init_altitude_max_m)
            alts.append(alt)
        self.assertGreater(max(alts) - min(alts), 0.0)

    def test_velocity_bounds(self):
        cfg = _cfg("default", init_vel_range_mps=0.60)
        for seed in range(50):
            _, vel, _ = InitialStateSampler(cfg).sample(np.random.default_rng(seed))
            self.assertTrue(np.all(vel >= -cfg.init_vel_range_mps))
            self.assertTrue(np.all(vel <= cfg.init_vel_range_mps))

    def test_attitude_bounds(self):
        cfg = _cfg("default", init_roll_pitch_range_rad=0.3, init_yaw_range_rad=0.5)
        for seed in range(50):
            _, _, att = InitialStateSampler(cfg).sample(np.random.default_rng(seed))
            self.assertGreaterEqual(att[0], -cfg.init_roll_pitch_range_rad)
            self.assertLessEqual(att[0], cfg.init_roll_pitch_range_rad)
            self.assertGreaterEqual(att[1], -cfg.init_roll_pitch_range_rad)
            self.assertLessEqual(att[1], cfg.init_roll_pitch_range_rad)
            self.assertGreaterEqual(att[2], -cfg.init_yaw_range_rad)
            self.assertLessEqual(att[2], cfg.init_yaw_range_rad)

    # -- I. degenerate / zero-randomization config ----------

    def test_degenerate_zero_ranges(self):
        cfg = _cfg("default",
                   init_xy_range_m=0.0,
                   init_altitude_min_m=4.0, init_altitude_max_m=4.0,
                   init_vel_range_mps=0.0,
                   init_roll_pitch_range_rad=0.0, init_yaw_range_rad=0.0)
        pos, vel, att = InitialStateSampler(cfg).sample(np.random.default_rng(1))
        self.assertTrue(np.array_equal(pos, np.array([0.0, 0.0, -4.0])))
        self.assertTrue(np.array_equal(vel, np.zeros(3)))
        self.assertTrue(np.array_equal(att, np.zeros(3)))
        # still consumes seven uniform draws even when all ranges are zero
        rng = np.random.default_rng(1)
        InitialStateSampler(cfg).sample(rng)
        rng_ref = np.random.default_rng(1)
        for _ in range(3):
            rng_ref.uniform(0.0, 0.0)
        rng_ref.uniform(0.0, 0.0, size=3)
        for _ in range(3):
            rng_ref.uniform(0.0, 0.0)
        self.assertTrue(
            _rng_state_equal(rng.bit_generator.state, rng_ref.bit_generator.state)
        )

    # -- J. guard against draw reordering ------------------

    def test_swapped_xy_reference_does_not_match(self):
        # x and y share bounds, so a swap won't be caught by values alone;
        # instead assert the full ordered reference matches and a reference
        # that puts the velocity draw first does NOT.
        cfg = _cfg("stage2_train")
        rng_new = np.random.default_rng(9)
        pos, vel, att = InitialStateSampler(cfg).sample(rng_new)

        rng_bad = np.random.default_rng(9)
        vel_first = rng_bad.uniform(
            -cfg.init_vel_range_mps, cfg.init_vel_range_mps, size=3
        ).astype(np.float64)
        self.assertFalse(np.array_equal(vel, vel_first))  # vel is NOT drawn first

    def test_swapped_roll_yaw_reference_does_not_match(self):
        # yaw range differs from roll/pitch range in stage2_train (0.5 vs radians(5))
        cfg = _cfg("stage2_train")
        self.assertNotEqual(cfg.init_yaw_range_rad, cfg.init_roll_pitch_range_rad)
        rng_new = np.random.default_rng(11)
        _, _, att = InitialStateSampler(cfg).sample(rng_new)

        # reference that draws yaw BEFORE the two roll/pitch values
        rng_bad = np.random.default_rng(11)
        rng_bad.uniform(-cfg.init_xy_range_m, cfg.init_xy_range_m)
        rng_bad.uniform(-cfg.init_xy_range_m, cfg.init_xy_range_m)
        rng_bad.uniform(cfg.init_altitude_min_m, cfg.init_altitude_max_m)
        rng_bad.uniform(-cfg.init_vel_range_mps, cfg.init_vel_range_mps, size=3)
        yaw_first = rng_bad.uniform(-cfg.init_yaw_range_rad, cfg.init_yaw_range_rad)
        rp1 = rng_bad.uniform(-cfg.init_roll_pitch_range_rad, cfg.init_roll_pitch_range_rad)
        rp2 = rng_bad.uniform(-cfg.init_roll_pitch_range_rad, cfg.init_roll_pitch_range_rad)
        self.assertFalse(np.array_equal(att, np.array([rp1, rp2, yaw_first])))

    # -- env compatibility: env owns pos/vel/attitude ---------

    def test_env_owns_state_sampler_does_not(self):
        for cfg_name in _CONFIG_NAMES:
            for seed in (0, 1, 5000):
                with self.subTest(config=cfg_name, seed=seed):
                    env = NewLandingEnv(_cfg(cfg_name))
                    try:
                        env.reset(seed=seed)
                        self.assertEqual(set(vars(env.initial_state_sampler)), {"cfg"})
                        self.assertEqual(env.pos.shape, (3,))
                        self.assertEqual(env.vel.shape, (3,))
                        self.assertEqual(env.attitude.shape, (3,))
                        self.assertEqual(env.pos.dtype, np.dtype(np.float64))
                        # attitude_setpoint is a copy of the sampled attitude
                        self.assertTrue(
                            np.array_equal(env.attitude_setpoint, env.attitude)
                        )
                        self.assertIsNot(env.attitude_setpoint, env.attitude)
                    finally:
                        env.close()


class ResetDrawOrderTest(unittest.TestCase):
    """Pins the corrected reset RNG order at the top of reset():
    initial-state -> action-delay -> obs-delay -> response-alpha -> wind."""

    maxDiff = None

    def test_reset_prefix_rng_order_matches_verbatim_sequence(self):
        for cfg_name in _CONFIG_NAMES:
            overrides = {} if cfg_name == "default" else dict(_overrides(cfg_name))
            cfg = NewLandingConfig(**overrides)
            for seed in (0, 1, 5000, 7):
                with self.subTest(config=cfg_name, seed=seed):
                    # component path, one shared generator, in reset() order
                    from gymnasium.utils import seeding

                    rng_comp, _ = seeding.np_random(seed)
                    InitialStateSampler(cfg).sample(rng_comp)
                    ActionLatency(cfg).reset(rng_comp)
                    ObsLatency(cfg).reset(rng_comp)
                    ResponseAlphas(cfg).reset(rng_comp)
                    DisturbanceModel(cfg).reset(rng_comp)

                    # verbatim legacy draw sequence on an identically-seeded gen
                    rng_ref, _ = seeding.np_random(seed)
                    _legacy_initial_state(cfg, rng_ref)
                    rng_ref.integers(cfg.action_delay_steps_min,
                                     cfg.action_delay_steps_max + 1)
                    rng_ref.integers(cfg.obs_delay_steps_min,
                                     cfg.obs_delay_steps_max + 1)
                    rng_ref.uniform(cfg.vel_response_alpha_min,
                                    cfg.vel_response_alpha_max, size=3)
                    rng_ref.uniform(cfg.attitude_response_alpha_min,
                                    cfg.attitude_response_alpha_max, size=2)
                    rng_ref.uniform(cfg.body_rate_response_alpha_min,
                                    cfg.body_rate_response_alpha_max, size=3)
                    rng_ref.uniform(cfg.thrust_response_alpha_min,
                                    cfg.thrust_response_alpha_max)
                    rng_ref.uniform(-cfg.wind_accel_xy_max_mps2,
                                    cfg.wind_accel_xy_max_mps2)
                    rng_ref.uniform(-cfg.wind_accel_xy_max_mps2,
                                    cfg.wind_accel_xy_max_mps2)
                    rng_ref.uniform(-cfg.wind_accel_z_max_mps2,
                                    cfg.wind_accel_z_max_mps2)

                    self.assertTrue(
                        _rng_state_equal(
                            rng_comp.bit_generator.state,
                            rng_ref.bit_generator.state,
                        ),
                        f"[{cfg_name} seed={seed}] reset-prefix RNG order diverged",
                    )

    def test_initial_state_drawn_before_delays_old_vs_new(self):
        """OLD env vs NEW env: identically-seeded reset yields identical
        pos/vel/attitude and identical action/obs delay -- a reorder of the
        initial-state block relative to the delay draws would diverge here."""
        for cfg_name in _CONFIG_NAMES:
            overrides = {} if cfg_name == "default" else dict(_overrides(cfg_name))
            for seed in (0, 1, 5000):
                with self.subTest(config=cfg_name, seed=seed):
                    old_env = OldLandingEnv(OldLandingConfig(**overrides))
                    new_env = NewLandingEnv(NewLandingConfig(**overrides))
                    try:
                        old_env.reset(seed=seed)
                        new_env.reset(seed=seed)
                        self.assertTrue(np.array_equal(old_env.pos, new_env.pos))
                        self.assertTrue(np.array_equal(old_env.vel, new_env.vel))
                        self.assertTrue(
                            np.array_equal(old_env.attitude, new_env.attitude)
                        )
                        self.assertEqual(
                            (old_env.action_delay_steps, old_env.obs_delay_steps),
                            (new_env.action_delay_steps, new_env.obs_delay_steps),
                        )
                        self.assertTrue(
                            _rng_state_equal(
                                old_env.np_random.bit_generator.state,
                                new_env.np_random.bit_generator.state,
                            )
                        )
                    finally:
                        old_env.close()
                        new_env.close()


if __name__ == "__main__":
    unittest.main()
