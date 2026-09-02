"""Focused tests for the Phase 10 ObservationNoiseSampler.

Secondary to the OLD-vs-NEW environment regression in
``test_legacy_regression_contract.py`` and the checkpoint gate in
``test_checkpoint_compatibility.py``, both of which remain authoritative.

These tests localize an observation-noise regression by comparing
``ObservationNoiseSampler.sample`` against a verbatim copy of the three legacy
``_get_obs`` noise draws, using independent generators started from the same
state. Observation-noise feeds the policy input directly, so a regression here
would also break the checkpoint gate.

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
    _rng_state_equal,
)

from landing_rl.perception import ObservationNoiseSampler


def _legacy_obs_noise_draws(cfg, rng):
    """Verbatim copy of the three noise draws from
    mujoco_rl/envs/env_prototype.py :: LandingEnv._get_obs()."""
    velocity_noise = rng.normal(0.0, cfg.velocity_obs_noise_std_mps, size=3)
    acceleration_noise = rng.normal(0.0, cfg.acceleration_obs_noise_std_mps2, size=3)
    attitude_noise = rng.normal(0.0, cfg.attitude_obs_noise_std_rad, size=3)
    return velocity_noise, acceleration_noise, attitude_noise


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


class ObservationNoiseSamplerTest(unittest.TestCase):
    maxDiff = None

    # -- A. constructor consumes zero RNG / H. no extra state ------

    def test_constructor_consumes_no_rng_and_owns_only_cfg(self):
        rng = np.random.default_rng(51)
        before = rng.bit_generator.state
        s = ObservationNoiseSampler(NewLandingConfig())
        self.assertTrue(_rng_state_equal(before, rng.bit_generator.state))
        self.assertEqual(set(vars(s)), {"cfg"})
        self.assertFalse(hasattr(s, "np_random"))
        self.assertFalse(hasattr(s, "rng"))

    # -- B. sample() retains no state -------------------------

    def test_sample_retains_no_state(self):
        s = ObservationNoiseSampler(_cfg("default"))
        s.sample(np.random.default_rng(0))
        self.assertEqual(set(vars(s)), {"cfg"})

    # -- C. exact parity vs verbatim legacy replica ----------

    def test_sample_matches_legacy_draws_exactly(self):
        for cfg_name in _CONFIG_NAMES:
            cfg = _cfg(cfg_name)
            for seed in (0, 1, 5000, 999, 424242):
                with self.subTest(config=cfg_name, seed=seed):
                    rng_legacy = np.random.default_rng(seed)
                    rng_new = np.random.default_rng(seed)

                    vn_l, an_l, tn_l = _legacy_obs_noise_draws(cfg, rng_legacy)
                    vn_n, an_n, tn_n = ObservationNoiseSampler(cfg).sample(rng_new)

                    self.assertTrue(np.array_equal(vn_n, vn_l),
                                    f"[{cfg_name} seed={seed}] velocity noise")
                    self.assertTrue(np.array_equal(an_n, an_l),
                                    f"[{cfg_name} seed={seed}] acceleration noise")
                    self.assertTrue(np.array_equal(tn_n, tn_l),
                                    f"[{cfg_name} seed={seed}] attitude noise")
                    self.assertTrue(
                        _rng_state_equal(
                            rng_legacy.bit_generator.state,
                            rng_new.bit_generator.state,
                        ),
                        f"[{cfg_name} seed={seed}] RNG state diverged after "
                        f"observation-noise block",
                    )

    def test_consumes_exactly_three_normal_draws_in_order(self):
        cfg = _cfg("default")
        for seed in (0, 1, 5000):
            with self.subTest(seed=seed):
                rng_new = np.random.default_rng(seed)
                ObservationNoiseSampler(cfg).sample(rng_new)

                rng_ref = np.random.default_rng(seed)
                rng_ref.normal(0.0, cfg.velocity_obs_noise_std_mps, size=3)
                rng_ref.normal(0.0, cfg.acceleration_obs_noise_std_mps2, size=3)
                rng_ref.normal(0.0, cfg.attitude_obs_noise_std_rad, size=3)

                self.assertTrue(
                    _rng_state_equal(
                        rng_new.bit_generator.state, rng_ref.bit_generator.state
                    )
                )

    # -- D. shapes -------------------------------------------

    def test_shapes(self):
        vn, an, tn = ObservationNoiseSampler(_cfg("default")).sample(
            np.random.default_rng(2)
        )
        for name, arr in (("vel", vn), ("accel", an), ("attitude", tn)):
            self.assertEqual(arr.shape, (3,), name)

    # -- E. draw-order guard --------------------------------

    def test_reordered_reference_does_not_match(self):
        """The three std devs differ (0.015 / 0.05 / radians(1)); a reference
        that draws attitude -> acceleration -> velocity must NOT match."""
        cfg = _cfg("default")
        self.assertNotEqual(cfg.velocity_obs_noise_std_mps,
                            cfg.acceleration_obs_noise_std_mps2)
        self.assertNotEqual(cfg.acceleration_obs_noise_std_mps2,
                            cfg.attitude_obs_noise_std_rad)

        rng_new = np.random.default_rng(13)
        vn, an, tn = ObservationNoiseSampler(cfg).sample(rng_new)

        rng_bad = np.random.default_rng(13)
        tn_first = rng_bad.normal(0.0, cfg.attitude_obs_noise_std_rad, size=3)
        an_second = rng_bad.normal(0.0, cfg.acceleration_obs_noise_std_mps2, size=3)
        vn_third = rng_bad.normal(0.0, cfg.velocity_obs_noise_std_mps, size=3)

        self.assertFalse(np.array_equal(vn, vn_third))
        self.assertFalse(np.array_equal(tn, tn_first))

    # -- F. zero-noise config still consumes RNG ------------

    def test_zero_noise_still_consumes_rng(self):
        cfg = _cfg("default",
                   velocity_obs_noise_std_mps=0.0,
                   acceleration_obs_noise_std_mps2=0.0,
                   attitude_obs_noise_std_rad=0.0)
        for seed in (0, 1, 5000):
            with self.subTest(seed=seed):
                rng_new = np.random.default_rng(seed)
                state_before = np.random.default_rng(seed).bit_generator.state
                vn, an, tn = ObservationNoiseSampler(cfg).sample(rng_new)

                self.assertTrue(np.array_equal(vn, np.zeros(3)))
                self.assertTrue(np.array_equal(an, np.zeros(3)))
                self.assertTrue(np.array_equal(tn, np.zeros(3)))
                # RNG stream still advanced by three normal(0,0,size=3) draws
                self.assertFalse(
                    _rng_state_equal(state_before, rng_new.bit_generator.state)
                )
                rng_ref = np.random.default_rng(seed)
                rng_ref.normal(0.0, 0.0, size=3)
                rng_ref.normal(0.0, 0.0, size=3)
                rng_ref.normal(0.0, 0.0, size=3)
                self.assertTrue(
                    _rng_state_equal(
                        rng_new.bit_generator.state, rng_ref.bit_generator.state
                    ),
                    f"[seed={seed}] zero-noise path must still do 3 normal draws",
                )

    # -- G. non-zero config exercises all three channels -----

    def test_nonzero_config_all_channels_active(self):
        # default has all three std devs > 0
        cfg = NewLandingConfig()
        self.assertGreater(cfg.velocity_obs_noise_std_mps, 0.0)
        self.assertGreater(cfg.acceleration_obs_noise_std_mps2, 0.0)
        self.assertGreater(cfg.attitude_obs_noise_std_rad, 0.0)
        seen = {"vel": set(), "accel": set(), "attitude": set()}
        for seed in range(30):
            vn, an, tn = ObservationNoiseSampler(cfg).sample(
                np.random.default_rng(seed)
            )
            self.assertFalse(np.array_equal(vn, np.zeros(3)))
            self.assertFalse(np.array_equal(an, np.zeros(3)))
            self.assertFalse(np.array_equal(tn, np.zeros(3)))
            seen["vel"].add(tuple(vn.tolist()))
            seen["accel"].add(tuple(an.tolist()))
            seen["attitude"].add(tuple(tn.tolist()))
        for k, v in seen.items():
            self.assertGreater(len(v), 1, k)
        # std magnitudes actually differ between channels
        big = ObservationNoiseSampler(cfg).sample(np.random.default_rng(0))
        self.assertGreater(
            float(np.std(big[2])) + 1e-12, 0.0
        )

    # -- env-level: still the only _get_obs() RNG-consuming noise ---

    def test_env_get_obs_uses_component(self):
        env = NewLandingEnv(NewLandingConfig())
        try:
            env.reset(seed=5000)
            self.assertEqual(set(vars(env.observation_noise)), {"cfg"})
            self.assertIs(env.observation_noise.cfg, env.cfg)
            # driving steps still works and observations stay float32 (16,)
            for _ in range(10):
                obs, _, term, trunc, _ = env.step(np.zeros(3, dtype=np.float32))
                self.assertEqual(obs.shape, (16,))
                self.assertEqual(obs.dtype, np.dtype(np.float32))
                if term or trunc:
                    break
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
