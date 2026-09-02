"""Focused tests for the Phase 7 ObsLatency helper (target observation delay).

Secondary to the OLD-vs-NEW environment regression in
``test_legacy_regression_contract.py`` and the checkpoint gate in
``test_checkpoint_compatibility.py``, both of which remain authoritative.

These tests localize an obs-latency regression by comparing ``ObsLatency``
against verbatim copies of the legacy ``obs_delay_steps`` draw and the legacy
``_update_target_pipeline`` queue block, using independent generators started
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
    OldLandingConfig,
    OldLandingEnv,
    _rng_state_equal,
)

from landing_rl.envs.action_latency import ActionLatency
from landing_rl.perception import ObsLatency


def _legacy_obs_delay_sample(cfg, rng) -> int:
    """Verbatim copy of the obs_delay_steps draw from
    mujoco_rl/envs/env_prototype.py :: LandingEnv.reset()."""
    return int(
        rng.integers(
            cfg.obs_delay_steps_min,
            cfg.obs_delay_steps_max + 1,
        )
    )


class _LegacyObsQueue:
    """Verbatim standalone copy of the queue block from
    LandingEnv._update_target_pipeline + its reset clear. Not used by
    production code."""

    def __init__(self, obs_delay_steps):
        self.obs_delay_steps = int(obs_delay_steps)
        self.target_queue: list[tuple[np.ndarray, bool, str]] = []

    def push_and_get(self, raw_target, valid, mode):
        self.target_queue.append((raw_target.copy(), bool(valid), mode))
        max_len = max(1, int(self.obs_delay_steps) + 1)
        while len(self.target_queue) > max_len:
            self.target_queue.pop(0)
        return self.target_queue[0]


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


def _tgt(*xyz):
    return np.asarray(xyz, dtype=np.float64)


class ObsLatencyTest(unittest.TestCase):
    maxDiff = None

    # -- A. constructor consumes zero RNG ---------------------------

    def test_constructor_consumes_no_rng(self):
        rng = np.random.default_rng(21)
        before = rng.bit_generator.state
        lat = ObsLatency(NewLandingConfig())
        self.assertTrue(_rng_state_equal(before, rng.bit_generator.state))
        self.assertEqual(set(vars(lat)), {"cfg", "obs_delay_steps", "target_queue"})
        self.assertEqual(lat.obs_delay_steps, 0)
        self.assertEqual(lat.target_queue, [])
        self.assertFalse(hasattr(lat, "np_random"))
        self.assertFalse(hasattr(lat, "rng"))

    # -- B. reset(rng) == legacy obs_delay_steps draw, same RNG after --

    def test_reset_matches_legacy_integers_draw_exactly(self):
        for cfg_name in _CONFIG_NAMES:
            cfg = _cfg(cfg_name)
            for seed in (0, 1, 5000, 999, 424242):
                with self.subTest(config=cfg_name, seed=seed):
                    rng_legacy = np.random.default_rng(seed)
                    rng_new = np.random.default_rng(seed)

                    n_legacy = _legacy_obs_delay_sample(cfg, rng_legacy)
                    n_new = ObsLatency(cfg).reset(rng_new)

                    self.assertIsInstance(n_new, int)
                    self.assertEqual(
                        n_legacy, n_new,
                        f"[{cfg_name} seed={seed}] obs_delay_steps mismatch: "
                        f"OLD={n_legacy} NEW={n_new}",
                    )
                    self.assertTrue(
                        _rng_state_equal(
                            rng_legacy.bit_generator.state,
                            rng_new.bit_generator.state,
                        ),
                        f"[{cfg_name} seed={seed}] RNG state diverged after "
                        f"obs-delay draw",
                    )

    def test_reset_consumes_exactly_one_integers_draw_and_clears_queue(self):
        cfg = _cfg("stage2_train")
        for seed in (0, 1, 5000):
            with self.subTest(seed=seed):
                lat = ObsLatency(cfg)
                lat.reset(np.random.default_rng(seed))
                lat.push_and_get(_tgt(1, 2, 3), True, "normal")
                self.assertTrue(len(lat.target_queue) >= 1)

                rng_new = np.random.default_rng(seed)
                old_q = lat.target_queue
                lat.reset(rng_new)
                self.assertEqual(lat.target_queue, [])       # cleared
                self.assertIsNot(lat.target_queue, old_q)     # fresh list

                rng_ref = np.random.default_rng(seed)
                rng_ref.integers(
                    cfg.obs_delay_steps_min, cfg.obs_delay_steps_max + 1
                )
                self.assertTrue(
                    _rng_state_equal(
                        rng_new.bit_generator.state, rng_ref.bit_generator.state
                    )
                )

    # -- C/D/E. delay == 0 / 1 / max behavior --------------------

    def _fifo_reference_check(self, delay, n_targets=10):
        cfg = _cfg("default", obs_delay_steps_min=delay, obs_delay_steps_max=delay)
        lat = ObsLatency(cfg)
        lat.reset(np.random.default_rng(0))
        self.assertEqual(lat.obs_delay_steps, delay)
        self.assertEqual(lat.target_queue, [])

        ref = _LegacyObsQueue(delay)
        seq = [(_tgt(i, -i, 2 * i), bool(i % 3 != 0), ("stale" if i % 4 == 0 else "normal"))
               for i in range(1, n_targets + 1)]
        for k, (t, v, m) in enumerate(seq):
            g_t, g_v, g_m = lat.push_and_get(t.copy(), v, m)
            e_t, e_v, e_m = ref.push_and_get(t.copy(), v, m)
            self.assertTrue(np.array_equal(g_t, e_t), f"delay={delay} step {k}")
            self.assertEqual((g_v, g_m), (e_v, e_m), f"delay={delay} step {k}")
            self.assertEqual(len(lat.target_queue), len(ref.target_queue))
            self.assertEqual(len(lat.target_queue), min(k + 1, max(1, delay + 1)))
        # with delay D, the delayed output lags the input by exactly D once the
        # queue has filled (queue holds D+1 entries, [0] is D behind the newest)
        return seq, lat

    def test_delay_zero(self):
        seq, lat = self._fifo_reference_check(0)
        # queue length is always 1; [0] is always the newest push
        lat2 = ObsLatency(_cfg("default", obs_delay_steps_min=0, obs_delay_steps_max=0))
        lat2.reset(np.random.default_rng(0))
        for t, v, m in seq:
            out_t, out_v, out_m = lat2.push_and_get(t.copy(), v, m)
            self.assertTrue(np.array_equal(out_t, t))   # no delay: newest back
            self.assertEqual((out_v, out_m), (bool(v), m))
            self.assertEqual(len(lat2.target_queue), 1)

    def test_delay_one(self):
        seq, _ = self._fifo_reference_check(1)
        lat = ObsLatency(_cfg("default", obs_delay_steps_min=1, obs_delay_steps_max=1))
        lat.reset(np.random.default_rng(0))
        outs = [lat.push_and_get(t.copy(), v, m)[0] for t, v, m in seq]
        # first output is the first push (queue was [A] -> [A]); then lags by 1
        self.assertTrue(np.array_equal(outs[0], seq[0][0]))
        for k in range(1, len(seq)):
            self.assertTrue(np.array_equal(outs[k], seq[k - 1][0]), f"step {k}")

    def test_delay_max_configured(self):
        # default dataclass obs_delay_steps_max == 4
        dmax = NewLandingConfig().obs_delay_steps_max
        seq, _ = self._fifo_reference_check(dmax, n_targets=12)
        lat = ObsLatency(_cfg("default",
                              obs_delay_steps_min=dmax, obs_delay_steps_max=dmax))
        lat.reset(np.random.default_rng(0))
        outs = [lat.push_and_get(t.copy(), v, m)[0] for t, v, m in seq]
        for k in range(dmax, len(seq)):
            self.assertTrue(np.array_equal(outs[k], seq[k - dmax][0]), f"step {k}")
        self.assertEqual(len(lat.target_queue), dmax + 1)

    # -- F. deterministic sequence vs legacy replica ------------

    def test_long_sequence_matches_legacy_replica_exactly(self):
        for cfg_name in _CONFIG_NAMES:
            cfg = _cfg(cfg_name)
            for seed in (0, 1, 5000, 999):
                with self.subTest(config=cfg_name, seed=seed):
                    rng_new = np.random.default_rng(seed)
                    rng_leg = np.random.default_rng(seed)
                    n_new = ObsLatency(cfg).reset(rng_new)
                    n_leg = _legacy_obs_delay_sample(cfg, rng_leg)
                    self.assertEqual(n_new, n_leg)

                    lat = ObsLatency(cfg)
                    lat.reset(np.random.default_rng(seed))
                    ref = _LegacyObsQueue(lat.obs_delay_steps)

                    for i in range(300):
                        t = _tgt(0.01 * i, -0.02 * (i % 11), 0.5 + 0.001 * i)
                        v = bool((i * 7 + 1) % 5)
                        m = ("dropout" if i % 13 == 0
                             else "stale" if i % 9 == 0 else "normal")
                        g_t, g_v, g_m = lat.push_and_get(t.copy(), v, m)
                        e_t, e_v, e_m = ref.push_and_get(t.copy(), v, m)
                        self.assertTrue(np.array_equal(g_t, e_t), f"step {i}")
                        self.assertEqual((g_v, g_m), (e_v, e_m), f"step {i}")
                        self.assertEqual(
                            len(lat.target_queue), len(ref.target_queue), f"step {i}"
                        )

    # -- G. copy semantics -------------------------------------

    def test_push_copies_raw_target(self):
        lat = ObsLatency(_cfg("default", obs_delay_steps_min=2, obs_delay_steps_max=2))
        lat.reset(np.random.default_rng(0))
        t = _tgt(1.0, 2.0, 3.0)
        lat.push_and_get(t, True, "normal")
        t += 100.0  # mutate the caller's array afterwards
        stored = lat.target_queue[-1][0]
        self.assertTrue(np.array_equal(stored, _tgt(1.0, 2.0, 3.0)))
        self.assertIsNot(stored, t)

    # -- H. valid bool conversion + mode string preserved --------

    def test_valid_bool_conversion_and_mode_preserved(self):
        lat = ObsLatency(_cfg("default", obs_delay_steps_min=0, obs_delay_steps_max=0))
        lat.reset(np.random.default_rng(0))
        # non-bool truthy/falsy inputs become real bools
        out_t, out_v, out_m = lat.push_and_get(_tgt(0, 0, 0), 1, "normal")
        self.assertIs(type(out_v), bool)
        self.assertTrue(out_v)
        out_t, out_v, out_m = lat.push_and_get(_tgt(0, 0, 0), 0, "dropout")
        self.assertIs(type(out_v), bool)
        self.assertFalse(out_v)
        self.assertEqual(out_m, "dropout")
        _, _, out_m = lat.push_and_get(_tgt(0, 0, 0), True, "stale")
        self.assertEqual(out_m, "stale")

    # -- I. env compatibility aliases -------------------------

    def test_env_aliases_after_reset_and_during_rollout(self):
        for cfg_name in _CONFIG_NAMES:
            for seed in (0, 1, 5000):
                with self.subTest(config=cfg_name, seed=seed):
                    env = NewLandingEnv(_cfg(cfg_name))
                    try:
                        env.reset(seed=seed)
                        self.assertIs(env.target_queue, env.obs_latency.target_queue)
                        self.assertEqual(
                            env.obs_delay_steps, env.obs_latency.obs_delay_steps
                        )
                        for _ in range(40):
                            env.step(np.zeros(3, dtype=np.float32))
                            self.assertIs(
                                env.target_queue, env.obs_latency.target_queue,
                                f"[{cfg_name} seed={seed}] target_queue identity lost",
                            )
                            self.assertEqual(
                                env.obs_delay_steps,
                                env.obs_latency.obs_delay_steps,
                            )
                        # a second reset re-aliases the (new) queue object
                        old_q = env.target_queue
                        env.reset(seed=seed + 1)
                        self.assertIs(env.target_queue, env.obs_latency.target_queue)
                        self.assertIsNot(env.target_queue, old_q)
                    finally:
                        env.close()

    # -- reset draw order: action_delay BEFORE obs_delay ---------

    def test_action_delay_sampled_before_obs_delay(self):
        """Introducing ObsLatency must not swap the two reset integer draws.

        Two independent checks:
        1. OLD env vs NEW env: identically-seeded reset yields the same
           (action_delay_steps, obs_delay_steps) pair -- OLD samples action
           first then obs, so a swap in NEW would diverge here.
        2. pure-generator: from one shared starting state, ActionLatency.reset
           then ObsLatency.reset consume the two draws in exactly the order a
           verbatim action-then-obs reference does, leaving identical RNG state.
        """
        for cfg_name in ("default", "stage2_train", "stage2_eval", "stage0_eval"):
            for seed in (0, 1, 5000, 7):
                with self.subTest(config=cfg_name, seed=seed):
                    overrides = {} if cfg_name == "default" else dict(_overrides(cfg_name))

                    old_env = OldLandingEnv(OldLandingConfig(**overrides))
                    new_env = NewLandingEnv(NewLandingConfig(**overrides))
                    try:
                        old_env.reset(seed=seed)
                        new_env.reset(seed=seed)
                        self.assertEqual(
                            (old_env.action_delay_steps, old_env.obs_delay_steps),
                            (new_env.action_delay_steps, new_env.obs_delay_steps),
                            f"[{cfg_name} seed={seed}] delay pair diverged OLD vs NEW",
                        )
                    finally:
                        old_env.close()
                        new_env.close()

                    cfg = NewLandingConfig(**overrides)
                    # helpers in sequence, shared starting generator state
                    rng_helpers = np.random.default_rng(seed)
                    a_h = ActionLatency(cfg).reset(rng_helpers)
                    o_h = ObsLatency(cfg).reset(rng_helpers)

                    # verbatim action-then-obs reference from the same state
                    rng_ref = np.random.default_rng(seed)
                    a_ref = int(
                        rng_ref.integers(
                            cfg.action_delay_steps_min,
                            cfg.action_delay_steps_max + 1,
                        )
                    )
                    o_ref = int(
                        rng_ref.integers(
                            cfg.obs_delay_steps_min,
                            cfg.obs_delay_steps_max + 1,
                        )
                    )
                    self.assertEqual((a_h, o_h), (a_ref, o_ref))
                    self.assertTrue(
                        _rng_state_equal(
                            rng_helpers.bit_generator.state,
                            rng_ref.bit_generator.state,
                        )
                    )

    def test_regression_matrix_covers_delay_regimes(self):
        self.assertEqual(_overrides("stage2_eval")["obs_delay_steps_min"], 1)
        self.assertEqual(_overrides("stage2_eval")["obs_delay_steps_max"], 1)
        self.assertEqual(_overrides("stage2_train")["obs_delay_steps_min"], 1)
        self.assertEqual(_overrides("stage2_train")["obs_delay_steps_max"], 5)
        self.assertEqual(_overrides("stage0_eval")["obs_delay_steps_min"], 0)
        self.assertEqual(_overrides("stage0_eval")["obs_delay_steps_max"], 2)
        self.assertEqual(NewLandingConfig().obs_delay_steps_min, 1)
        self.assertEqual(NewLandingConfig().obs_delay_steps_max, 4)


if __name__ == "__main__":
    unittest.main()
