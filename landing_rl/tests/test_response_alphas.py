"""Focused tests for the Phase 8 ResponseAlphas helper (per-episode alphas).

Secondary to the OLD-vs-NEW environment regression in
``test_legacy_regression_contract.py`` and the checkpoint gate in
``test_checkpoint_compatibility.py``, both of which remain authoritative.

These tests localize a response-alpha regression by comparing
``ResponseAlphas.reset`` against a verbatim copy of the four legacy
``self.np_random.uniform(...)`` draws, using independent generators started
from the same state.

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

from landing_rl.dynamics import ResponseAlphas


def _legacy_response_alpha_draws(cfg, rng):
    """Verbatim copy of the four response-alpha draws from
    mujoco_rl/envs/env_prototype.py :: LandingEnv.reset()."""
    vel = rng.uniform(
        cfg.vel_response_alpha_min,
        cfg.vel_response_alpha_max,
        size=3,
    ).astype(np.float64)
    att = rng.uniform(
        cfg.attitude_response_alpha_min,
        cfg.attitude_response_alpha_max,
        size=2,
    ).astype(np.float64)
    br = rng.uniform(
        cfg.body_rate_response_alpha_min,
        cfg.body_rate_response_alpha_max,
        size=3,
    ).astype(np.float64)
    thr = float(rng.uniform(
        cfg.thrust_response_alpha_min,
        cfg.thrust_response_alpha_max,
    ))
    return vel, att, br, thr


def _overrides(name):
    for cfg_name, overrides, _doc in CONFIG_SPECS:
        if cfg_name == name:
            return overrides
    raise KeyError(name)


_CONFIG_NAMES = ("default", "stage0_eval", "stage2_eval", "stage2_train")


def _cfg(name):
    return NewLandingConfig(**({} if name == "default" else _overrides(name)))


class ResponseAlphasTest(unittest.TestCase):
    maxDiff = None

    # -- A. constructor consumes zero RNG --------------------------

    def test_constructor_consumes_no_rng(self):
        rng = np.random.default_rng(31)
        before = rng.bit_generator.state
        ra = ResponseAlphas(NewLandingConfig())
        self.assertTrue(_rng_state_equal(before, rng.bit_generator.state))
        self.assertEqual(
            set(vars(ra)),
            {
                "cfg",
                "vel_response_alpha",
                "attitude_response_alpha",
                "body_rate_response_alpha",
                "thrust_response_alpha",
            },
        )
        self.assertFalse(hasattr(ra, "np_random"))
        self.assertFalse(hasattr(ra, "rng"))

    # -- B. exact parity vs verbatim legacy replica --------------

    def test_reset_matches_legacy_draws_exactly(self):
        for cfg_name in _CONFIG_NAMES:
            cfg = _cfg(cfg_name)
            for seed in (0, 1, 5000, 999, 424242):
                with self.subTest(config=cfg_name, seed=seed):
                    rng_legacy = np.random.default_rng(seed)
                    rng_new = np.random.default_rng(seed)

                    vel_l, att_l, br_l, thr_l = _legacy_response_alpha_draws(
                        cfg, rng_legacy
                    )
                    ra = ResponseAlphas(cfg)
                    ra.reset(rng_new)

                    self.assertTrue(
                        np.array_equal(ra.vel_response_alpha, vel_l),
                        f"[{cfg_name} seed={seed}] vel: "
                        f"{ra.vel_response_alpha!r} != {vel_l!r}",
                    )
                    self.assertTrue(
                        np.array_equal(ra.attitude_response_alpha, att_l),
                        f"[{cfg_name} seed={seed}] attitude",
                    )
                    self.assertTrue(
                        np.array_equal(ra.body_rate_response_alpha, br_l),
                        f"[{cfg_name} seed={seed}] body_rate",
                    )
                    self.assertEqual(
                        ra.thrust_response_alpha, thr_l,
                        f"[{cfg_name} seed={seed}] thrust",
                    )
                    self.assertTrue(
                        _rng_state_equal(
                            rng_legacy.bit_generator.state,
                            rng_new.bit_generator.state,
                        ),
                        f"[{cfg_name} seed={seed}] RNG state diverged after "
                        f"response-alpha block",
                    )

    # -- C. shapes / dtypes / scalar semantics -----------------

    def test_shapes_and_types(self):
        ra = ResponseAlphas(_cfg("stage2_train"))
        ra.reset(np.random.default_rng(0))
        self.assertEqual(ra.vel_response_alpha.shape, (3,))
        self.assertEqual(ra.attitude_response_alpha.shape, (2,))
        self.assertEqual(ra.body_rate_response_alpha.shape, (3,))
        self.assertEqual(ra.vel_response_alpha.dtype, np.dtype(np.float64))
        self.assertEqual(ra.attitude_response_alpha.dtype, np.dtype(np.float64))
        self.assertEqual(ra.body_rate_response_alpha.dtype, np.dtype(np.float64))
        self.assertIsInstance(ra.thrust_response_alpha, float)
        self.assertNotIsInstance(ra.thrust_response_alpha, np.ndarray)

    # -- D. exact draw ordering + count with a reference generator ---

    def test_consumes_exactly_four_uniform_draws_in_order(self):
        cfg = _cfg("stage2_train")
        for seed in (0, 1, 5000):
            with self.subTest(seed=seed):
                rng_new = np.random.default_rng(seed)
                ResponseAlphas(cfg).reset(rng_new)

                rng_ref = np.random.default_rng(seed)
                rng_ref.uniform(cfg.vel_response_alpha_min,
                                cfg.vel_response_alpha_max, size=3)
                rng_ref.uniform(cfg.attitude_response_alpha_min,
                                cfg.attitude_response_alpha_max, size=2)
                rng_ref.uniform(cfg.body_rate_response_alpha_min,
                                cfg.body_rate_response_alpha_max, size=3)
                rng_ref.uniform(cfg.thrust_response_alpha_min,
                                cfg.thrust_response_alpha_max)

                self.assertTrue(
                    _rng_state_equal(
                        rng_new.bit_generator.state, rng_ref.bit_generator.state
                    )
                )

    def test_draw_order_is_vel_attitude_bodyrate_thrust(self):
        """A deliberately swapped reference (attitude before vel) must NOT
        match -> guards against reordering."""
        cfg = _cfg("stage2_train")
        rng_new = np.random.default_rng(7)
        ra = ResponseAlphas(cfg)
        ra.reset(rng_new)

        rng_swapped = np.random.default_rng(7)
        att_first = rng_swapped.uniform(
            cfg.attitude_response_alpha_min,
            cfg.attitude_response_alpha_max, size=2,
        ).astype(np.float64)
        vel_second = rng_swapped.uniform(
            cfg.vel_response_alpha_min,
            cfg.vel_response_alpha_max, size=3,
        ).astype(np.float64)
        self.assertFalse(np.array_equal(ra.vel_response_alpha, vel_second))
        self.assertFalse(np.array_equal(ra.attitude_response_alpha, att_first))

    # -- E. info-only fields are still sampled / consume RNG ----

    def test_info_only_fields_still_consume_rng(self):
        """Dropping the vel/attitude draws would shift the stream; prove they
        are drawn by checking the post-block RNG state advances past them."""
        cfg = _cfg("stage2_train")
        rng = np.random.default_rng(3)
        state_before = np.random.default_rng(3).bit_generator.state
        ra = ResponseAlphas(cfg)
        ra.reset(rng)
        self.assertFalse(_rng_state_equal(state_before, rng.bit_generator.state))

        # exactly four draws: a generator that does only body_rate + thrust
        # (skipping the two info-only draws) must end in a DIFFERENT state
        rng_missing = np.random.default_rng(3)
        rng_missing.uniform(cfg.body_rate_response_alpha_min,
                            cfg.body_rate_response_alpha_max, size=3)
        rng_missing.uniform(cfg.thrust_response_alpha_min,
                            cfg.thrust_response_alpha_max)
        self.assertFalse(
            _rng_state_equal(
                rng.bit_generator.state, rng_missing.bit_generator.state
            ),
            "info-only vel/attitude draws appear to have been dropped",
        )

    def test_info_only_fields_are_nondegenerate(self):
        cfg = _cfg("stage2_train")
        seen_vel = set()
        for seed in range(20):
            ra = ResponseAlphas(cfg)
            ra.reset(np.random.default_rng(seed))
            self.assertTrue(
                np.all(ra.vel_response_alpha >= cfg.vel_response_alpha_min)
                and np.all(ra.vel_response_alpha <= cfg.vel_response_alpha_max)
            )
            self.assertTrue(
                np.all(ra.attitude_response_alpha >= cfg.attitude_response_alpha_min)
                and np.all(ra.attitude_response_alpha <= cfg.attitude_response_alpha_max)
            )
            seen_vel.add(tuple(ra.vel_response_alpha.tolist()))
        self.assertGreater(len(seen_vel), 1)

    # -- F. LandingEnv compatibility aliases -------------------

    def test_env_alpha_attributes_alias_component(self):
        for cfg_name in _CONFIG_NAMES:
            for seed in (0, 1, 5000):
                with self.subTest(config=cfg_name, seed=seed):
                    env = NewLandingEnv(_cfg(cfg_name))
                    try:
                        env.reset(seed=seed)
                        ra = env.response_alphas
                        # ndarray fields: same canonical object
                        self.assertIs(env.vel_response_alpha, ra.vel_response_alpha)
                        self.assertIs(
                            env.attitude_response_alpha, ra.attitude_response_alpha
                        )
                        self.assertIs(
                            env.body_rate_response_alpha, ra.body_rate_response_alpha
                        )
                        # scalar float: value equality
                        self.assertEqual(
                            env.thrust_response_alpha, ra.thrust_response_alpha
                        )
                        self.assertIsInstance(env.thrust_response_alpha, float)

                        # still holds after several steps (step() does not
                        # resample these; they are per-episode)
                        for _ in range(20):
                            env.step(np.zeros(3, dtype=np.float32))
                            self.assertIs(
                                env.vel_response_alpha, ra.vel_response_alpha
                            )
                            self.assertIs(
                                env.body_rate_response_alpha,
                                ra.body_rate_response_alpha,
                            )
                            self.assertEqual(
                                env.thrust_response_alpha, ra.thrust_response_alpha
                            )

                        # a second reset re-aliases the new draws
                        old_vel = env.vel_response_alpha
                        env.reset(seed=seed + 1)
                        self.assertIs(
                            env.vel_response_alpha,
                            env.response_alphas.vel_response_alpha,
                        )
                        self.assertIsNot(env.vel_response_alpha, old_vel)
                    finally:
                        env.close()

    def test_env_info_channels_match_component(self):
        env = NewLandingEnv(_cfg("stage2_train"))
        try:
            env.reset(seed=5000)
            _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
            ra = env.response_alphas
            self.assertTrue(
                np.array_equal(info["vel_response_alpha"], ra.vel_response_alpha)
            )
            self.assertTrue(
                np.array_equal(
                    info["attitude_response_alpha"], ra.attitude_response_alpha
                )
            )
            self.assertTrue(
                np.array_equal(
                    info["body_rate_response_alpha"], ra.body_rate_response_alpha
                )
            )
            self.assertEqual(
                info["thrust_response_alpha"], ra.thrust_response_alpha
            )
            # info arrays are copies (legacy uses .copy())
            self.assertIsNot(info["vel_response_alpha"], ra.vel_response_alpha)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
