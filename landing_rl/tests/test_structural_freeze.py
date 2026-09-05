"""Phase 13D -- expanded structural freeze gate.

Purpose
-------
This is the "last known-good legacy-equivalent modular simulator" checkpoint.
It does not introduce any new behavior verification technique beyond what
Phase 1-13C already established (exact OLD-vs-NEW parity, exact RNG-state
equality, no tolerance); it widens the matrix (full episodes, more seeds,
more configs, dedicated contact/bounce/ground-effect/cutoff regimes) and adds
a small number of structural-invariant checks (component graph, ownership,
frozen field sets, compatibility-mirror identity) that no prior test file
owns as its primary responsibility.

This file is deliberately test-only. It imports the already-verified
comparison machinery from ``test_legacy_regression_contract.py`` (the
``LandingEnvParityContractTest`` instance-method trick below) rather than
re-implementing OLD-vs-NEW comparison logic, to avoid drifting two exact
comparators out of sync.

Class names are prefixed ``FreezeNN_`` so that alphabetic test discovery
order (unittest's default) matches the intended reporting order: component
graph and ownership first, negative controls near the end, and a
final report-printing test last.

No production source is touched by this file. If any test here needs a
production change to pass, that is a Phase 13D STOP condition, not something
to route around from this file.
"""

from __future__ import annotations

import unittest

import numpy as np

import test_legacy_regression_contract as _trc
from test_legacy_regression_contract import (
    CONFIG_SPECS,
    EXPECTED_INFO_KEYS,
    LEGACY_IMPL,
    NEW_IMPL,
    NewLandingConfig,
    NewLandingEnv,
    OldLandingConfig,
    OldLandingEnv,
    PATTERNS,
    _rng_state_equal,
    make_action_sequence,
)

# NOTE: ``LandingEnvParityContractTest`` (a unittest.TestCase subclass) is
# deliberately accessed only as ``_trc.LandingEnvParityContractTest`` inside
# ``_new_helper()`` below, NEVER imported by name into this module's
# namespace. unittest's module-level test discovery (``loadTestsFromModule``)
# walks ``dir(module)`` and re-collects any TestCase subclass it finds there
# -- importing the class by name would make it run a SECOND time under
# ``test_structural_freeze`` (silently inflating / duplicating the suite).
# ``_trc`` itself is a module, not a class, so it is never picked up.

from landing_rl.contact import ContactModel, ContactResult, ContactState
from landing_rl.controllers import BaselineController
from landing_rl.disturbances import DisturbanceModel
from landing_rl.dynamics import (
    LegacyVehicleDynamics,
    PlantModel,
    ProcessNoiseSampler,
    ResponseAlphas,
    VehicleState,
)
from landing_rl.envs.action_latency import ActionLatency
from landing_rl.envs.initial_state import InitialStateSampler
from landing_rl.envs.loop_timing import LoopTiming
from landing_rl.perception import ObsLatency, ObservationNoiseSampler, TargetMeasurementModel


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_EXPANDED_SEEDS = tuple(range(10))  # section 10: matched seeds 0..9

_BOUNCE_OVERRIDES = dict(
    contact_enabled=True,
    init_altitude_min_m=2.0,
    init_altitude_max_m=2.5,
    bounce_vz_threshold_mps=0.05,
    touchdown_vz_soft_mps=0.01,
)
_GROUND_EFFECT_OVERRIDES = dict(ground_effect_gain=0.18)

_STATS = {"paired_episodes": 0, "new_only_episodes": 0, "total_steps": 0}
_REPORT: dict = {}


def _new_helper():
    """A ``LandingEnvParityContractTest`` instance used purely as a holder of
    its already-verified ``_assert_reset_equal`` / ``_assert_states_equal``
    methods -- never registered with the test loader, never run as a test
    itself. Any assertion failure raised through it is attributed by
    unittest to whichever FreezeNN test is currently executing, exactly as
    if the code were written inline there."""
    return _trc.LandingEnvParityContractTest("test_new_package_import")


def _rng_state(env):
    return env.np_random.bit_generator.state


def _run_full_episode_paired(
    overrides, seed, pattern, max_steps, left_impl=LEGACY_IMPL, right_impl=NEW_IMPL,
):
    """Run two envs in lockstep to termination/truncation (or ``max_steps``),
    with full exact-parity assertion every step via the Phase 1 comparator.
    Returns a coverage dict describing what happened during the episode."""
    helper = _new_helper()
    left_name, left_env_cls, left_cfg_cls = left_impl
    right_name, right_env_cls, right_cfg_cls = right_impl

    left_env = left_env_cls(left_cfg_cls(**overrides))
    right_env = right_env_cls(right_cfg_cls(**overrides))
    label = f"{left_name}-vs-{right_name} {pattern} seed={seed}"

    cov = dict(
        steps=0, contact_event=False, soft_contact=False, hard_contact=False,
        bounced=False, ground_effect_active=False, motor_cutoff_multistep=False,
        dropout=False, stale=False, outlier=False,
        terminated=False, truncated=False, success=False, failure_reason="none",
    )
    try:
        left_reset = left_env.reset(seed=seed)
        right_reset = right_env.reset(seed=seed)
        helper._assert_reset_equal(
            left_reset, right_reset, left_env, right_env, f"{label} @reset"
        )

        actions = make_action_sequence(pattern, max_steps)
        rng_equal_before = _rng_state_equal(_rng_state(left_env), _rng_state(right_env))
        cutoff_run = 0
        for i in range(max_steps):
            act = actions[i].copy()
            left_out = left_env.step(act.copy())
            right_out = right_env.step(act.copy())
            helper._assert_states_equal(
                left_out, right_out, left_env, right_env, f"{label} step={i}",
                rng_equal_before,
            )
            rng_equal_before = _rng_state_equal(_rng_state(left_env), _rng_state(right_env))

            info = left_out[4]
            cov["steps"] += 1
            if info.get("contact_event"):
                cov["contact_event"] = True
            if info.get("soft_contact"):
                cov["soft_contact"] = True
            if info.get("hard_contact"):
                cov["hard_contact"] = True
            if info.get("bounced"):
                cov["bounced"] = True
            ge = info.get("ground_effect_factor")
            if ge is not None and float(ge) != 1.0:
                cov["ground_effect_active"] = True
            mode = info.get("target_mode")
            if mode == "dropout":
                cov["dropout"] = True
            elif mode == "stale":
                cov["stale"] = True
            elif mode == "outlier":
                cov["outlier"] = True

            term, trunc = left_out[2], left_out[3]
            if info.get("motor_cutoff") and not (term or trunc):
                cutoff_run += 1
                if cutoff_run >= 2:
                    cov["motor_cutoff_multistep"] = True
            else:
                cutoff_run = 0

            if term or trunc:
                cov["terminated"], cov["truncated"] = bool(term), bool(trunc)
                cov["success"] = bool(info.get("success"))
                cov["failure_reason"] = info.get("failure_reason", "none")
                break

        _STATS["total_steps"] += cov["steps"]
        _STATS["new_only_episodes" if left_impl is right_impl else "paired_episodes"] += 1
    finally:
        left_env.close()
        right_env.close()
    return cov


_VEHICLE_ARR_FIELDS = ("pos", "vel", "accel", "prev_accel", "attitude", "body_rates",
                       "attitude_setpoint", "accel_cmd")
_VEHICLE_SCALAR_FIELDS = ("yaw_rate", "thrust_accel", "thrust_accel_setpoint",
                          "ground_effect_factor")
_CONTACT_PERSIST_FIELDS = ("ground_contact", "contact_count", "bounce_count", "motor_cutoff")
_CONTACT_TRANSIENT_FIELDS = ("contact_event", "soft_contact", "hard_contact", "bounced",
                            "touchdown_quality", "last_impact_vz", "last_touchdown_vxy",
                            "last_bounce_speed")


# ---------------------------------------------------------------------------
# Section 3/4/5/6: component graph, ownership, frozen field sets
# ---------------------------------------------------------------------------

class Freeze00_ComponentGraphAndOwnershipTest(unittest.TestCase):

    def test_component_graph_present_and_no_duplicate_collaborators(self):
        env = NewLandingEnv(NewLandingConfig())
        try:
            self.assertIsInstance(env.controller, BaselineController)
            self.assertIsInstance(env.disturbance, DisturbanceModel)
            self.assertIsInstance(env.loop_timing, LoopTiming)
            self.assertIsInstance(env.action_latency, ActionLatency)
            self.assertIsInstance(env.initial_state_sampler, InitialStateSampler)
            self.assertIsInstance(env.target_measurement_model, TargetMeasurementModel)
            self.assertIsInstance(env.obs_latency, ObsLatency)
            self.assertIsInstance(env.observation_noise, ObservationNoiseSampler)
            self.assertIsInstance(env.response_alphas, ResponseAlphas)
            self.assertIsInstance(env.process_noise, ProcessNoiseSampler)
            self.assertIsInstance(env._vehicle_state, VehicleState)
            self.assertIsInstance(env.legacy_dynamics, LegacyVehicleDynamics)
            self.assertIsInstance(env.contact, ContactModel)
            self.assertIsInstance(env.contact.state, ContactState)
            self.assertIsInstance(env.contact.result, ContactResult)
            self.assertIsInstance(env.plant, PlantModel)

            # PlantModel must coordinate the SAME collaborator instances the
            # env already constructed -- no duplicate ContactModel /
            # LegacyVehicleDynamics.
            self.assertIs(env.plant.dynamics, env.legacy_dynamics)
            self.assertIs(env.plant.contact, env.contact)
        finally:
            env.close()

    def test_ownership_sets_frozen_and_no_component_owns_rng_or_env(self):
        env = NewLandingEnv(NewLandingConfig())
        try:
            self.assertEqual(set(vars(env.plant)), {"cfg", "dynamics", "contact"})
            self.assertEqual(set(vars(env.legacy_dynamics)), {"cfg", "process_noise"})

            components = (
                ("controller", env.controller),
                ("disturbance", env.disturbance),
                ("loop_timing", env.loop_timing),
                ("action_latency", env.action_latency),
                ("initial_state_sampler", env.initial_state_sampler),
                ("target_measurement_model", env.target_measurement_model),
                ("obs_latency", env.obs_latency),
                ("observation_noise", env.observation_noise),
                ("response_alphas", env.response_alphas),
                ("process_noise", env.process_noise),
                ("legacy_dynamics", env.legacy_dynamics),
                ("contact", env.contact),
                ("plant", env.plant),
            )
            for name, comp in components:
                for banned in ("rng", "np_random", "env", "_env"):
                    self.assertFalse(
                        hasattr(comp, banned), f"{name}.{banned} must not exist"
                    )
        finally:
            env.close()

    def test_vehicle_state_field_set_frozen(self):
        self.assertEqual(
            tuple(VehicleState.__dataclass_fields__),
            ("pos", "vel", "accel", "prev_accel", "attitude", "yaw_rate",
             "body_rates", "thrust_accel", "attitude_setpoint",
             "thrust_accel_setpoint", "accel_cmd", "ground_effect_factor"),
        )

    def test_contact_state_and_result_field_sets_frozen(self):
        self.assertEqual(
            tuple(ContactState.__dataclass_fields__),
            ("ground_contact", "contact_count", "bounce_count", "motor_cutoff"),
        )
        self.assertEqual(
            tuple(ContactResult.__dataclass_fields__),
            ("contact_event", "soft_contact", "hard_contact", "bounced",
             "touchdown_quality", "last_impact_vz", "last_touchdown_vxy",
             "last_bounce_speed"),
        )


# ---------------------------------------------------------------------------
# Section 7: compatibility mirror freeze
# ---------------------------------------------------------------------------

class Freeze01_CompatibilityMirrorFreezeTest(unittest.TestCase):

    def _check(self, env, where):
        st = env._vehicle_state
        for f in _VEHICLE_ARR_FIELDS:
            self.assertIs(getattr(env, f), getattr(st, f), f"{where}: {f} identity broken")
        for f in _VEHICLE_SCALAR_FIELDS:
            self.assertEqual(getattr(env, f), getattr(st, f), f"{where}: {f}")
        cs = env.contact.state
        for f in _CONTACT_PERSIST_FIELDS:
            self.assertEqual(getattr(env, f), getattr(cs, f), f"{where}: {f}")
        cr = env.contact.result
        for f in _CONTACT_TRANSIENT_FIELDS:
            self.assertEqual(getattr(env, f), getattr(cr, f), f"{where}: {f}")

    def test_mirrors_hold_across_reset_and_full_episode(self):
        for pattern in ("zero", "pseudo_random", "saturation", "descend_z"):
            for seed in (0, 5000):
                with self.subTest(pattern=pattern, seed=seed):
                    env = NewLandingEnv(NewLandingConfig())
                    try:
                        env.reset(seed=seed)
                        self._check(env, f"reset[{pattern},{seed}]")
                        actions = make_action_sequence(pattern, env.cfg.max_steps)
                        for i in range(env.cfg.max_steps):
                            _, _, term, trunc, _ = env.step(actions[i].copy())
                            self._check(env, f"step {i}[{pattern},{seed}]")
                            if term or trunc:
                                break
                    finally:
                        env.close()


# ---------------------------------------------------------------------------
# Section 8: external interface freeze
# ---------------------------------------------------------------------------

class Freeze02_ExternalInterfaceFreezeTest(unittest.TestCase):

    def test_observation_and_action_space_contract_each_impl(self):
        for impl_name, env_cls, cfg_cls in (
            ("legacy", OldLandingEnv, OldLandingConfig),
            ("landing_rl", NewLandingEnv, NewLandingConfig),
        ):
            with self.subTest(impl=impl_name):
                env = env_cls(cfg_cls())
                try:
                    self.assertEqual(env.observation_space.shape, (16,))
                    self.assertEqual(env.observation_space.dtype, np.dtype(np.float32))
                    self.assertEqual(env.action_space.shape, (3,))
                    self.assertEqual(env.action_space.dtype, np.dtype(np.float32))
                    self.assertTrue(
                        np.array_equal(env.action_space.low, -np.ones(3, np.float32))
                    )
                    self.assertTrue(
                        np.array_equal(env.action_space.high, np.ones(3, np.float32))
                    )
                finally:
                    env.close()

    def test_info_key_set_is_frozen_58_keys(self):
        self.assertEqual(len(EXPECTED_INFO_KEYS), 58)
        env = NewLandingEnv(NewLandingConfig())
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
            self.assertEqual(set(info), EXPECTED_INFO_KEYS)
        finally:
            env.close()

    def test_alpha_eff_nan_sentinel(self):
        env = NewLandingEnv(NewLandingConfig())
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
            ae = info["alpha_eff"]
            self.assertEqual(ae.shape, (3,))
            self.assertTrue(np.all(np.isnan(ae)))
        finally:
            env.close()

    def test_residual_action_is_bounded_velocity_not_attitude_or_thrust(self):
        """The PPO action remains a bounded RESIDUAL VELOCITY command, never a
        direct attitude / thrust / PWM command. v_cmd = v_pid + v_residual is
        the env<->plant boundary."""
        env = NewLandingEnv(NewLandingConfig())
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
            self.assertTrue(np.array_equal(info["v_residual"], np.zeros(3)))

            env.reset(seed=0)
            _, _, _, _, info = env.step(np.array([1.0, -1.0, 1.0], dtype=np.float32))
            self.assertLessEqual(abs(float(info["v_residual"][0])), env.cfg.residual_xy_mps + 1e-12)
            self.assertLessEqual(abs(float(info["v_residual"][1])), env.cfg.residual_xy_mps + 1e-12)
            self.assertLessEqual(abs(float(info["v_residual"][2])), env.cfg.residual_z_mps + 1e-12)

            self.assertIn("v_pid", info)
            self.assertIn("v_residual", info)
            self.assertIn("v_cmd", info)
            for banned in ("attitude_cmd", "thrust_cmd", "pwm", "motor_cmd"):
                self.assertNotIn(banned, info)
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Section 9: non-obvious semantics freeze
# ---------------------------------------------------------------------------

class Freeze03_NonObviousSemanticsFreezeTest(unittest.TestCase):
    """A/B/C/D/G/H/I directly, at the running-env level. E (motor_cutoff is
    persistent/latched) is protected by Freeze07's multi-step feedback gate.
    F (ground effect evaluated before this-step contact resolution) is
    protected by test_legacy_dynamics.py's LegacyDynamicsReplicaParityTest,
    which varies the ``ground_contact`` input explicitly, plus every
    OLD-vs-NEW full-episode run in this file (OLD embeds the identical
    ordering, so any divergence there would also break parity)."""

    def test_A_target_measured_is_independent_copy_of_obs_target(self):
        env = NewLandingEnv(NewLandingConfig())
        try:
            env.reset(seed=0)
            env.step(np.zeros(3, dtype=np.float32))
            self.assertIsNot(env.target_measured, env.obs_target)
            self.assertTrue(np.array_equal(env.target_measured, env.obs_target))
        finally:
            env.close()

    def test_B_prev_action_stores_raw_clipped_not_delayed(self):
        cfg = NewLandingConfig(action_delay_steps_min=3, action_delay_steps_max=3)
        env = NewLandingEnv(cfg)
        try:
            env.reset(seed=0)
            action = np.array([0.8, -0.6, 0.4], dtype=np.float32)
            _, _, _, _, info = env.step(action.copy())
            self.assertTrue(np.array_equal(env.prev_action, action.astype(np.float64)))
            # with a 3-step primer, the APPLIED action this step is still a
            # zero-filled FIFO entry -- proof prev_action is NOT the delayed one.
            self.assertTrue(np.array_equal(info["applied_action"], np.zeros(3)))
        finally:
            env.close()

    def test_C_final_accel_matches_post_contact_velocity_formula(self):
        env = NewLandingEnv(NewLandingConfig())
        try:
            env.reset(seed=0)
            actions = make_action_sequence("descend_z", 60)
            for i in range(60):
                vel_before = env.vel.copy()
                _, _, term, trunc, info = env.step(actions[i].copy())
                dt_safe = max(float(info["dt"]), 1e-6)
                expected = (info["vel"] - vel_before) / dt_safe
                self.assertTrue(
                    np.array_equal(info["accel"], expected),
                    f"step {i}: accel={info['accel']!r} expected={expected!r}",
                )
                if term or trunc:
                    break
        finally:
            env.close()

    def test_D_prev_accel_equals_previous_steps_finalized_accel(self):
        env = NewLandingEnv(NewLandingConfig())
        try:
            env.reset(seed=0)
            actions = make_action_sequence("pseudo_random", 40)
            _, _, _, _, info_prev = env.step(actions[0].copy())
            for i in range(1, 40):
                _, _, term, trunc, info = env.step(actions[i].copy())
                self.assertTrue(np.array_equal(info["prev_accel"], info_prev["accel"]))
                info_prev = info
                if term or trunc:
                    break
        finally:
            env.close()

    def test_G_attitude_process_noise_std_rad_is_behaviorally_unused(self):
        actions = make_action_sequence("pseudo_random", 60)
        env_a = NewLandingEnv(NewLandingConfig())
        env_b = NewLandingEnv(NewLandingConfig(attitude_process_noise_std_rad=999.0))
        try:
            obs_a, _ = env_a.reset(seed=7)
            obs_b, _ = env_b.reset(seed=7)
            self.assertTrue(np.array_equal(obs_a, obs_b))
            self.assertTrue(_rng_state_equal(_rng_state(env_a), _rng_state(env_b)))
            for i in range(60):
                out_a = env_a.step(actions[i].copy())
                out_b = env_b.step(actions[i].copy())
                self.assertTrue(np.array_equal(out_a[0], out_b[0]), f"step {i} obs")
                self.assertEqual(float(out_a[1]), float(out_b[1]), f"step {i} reward")
                self.assertTrue(
                    _rng_state_equal(_rng_state(env_a), _rng_state(env_b)), f"step {i} rng"
                )
                if out_a[2] or out_a[3]:
                    break
        finally:
            env_a.close()
            env_b.close()

    def test_H_vel_and_attitude_response_alpha_consume_rng_but_stay_dynamics_inactive(self):
        actions = make_action_sequence("pseudo_random", 60)
        env_a = NewLandingEnv(NewLandingConfig())
        env_b = NewLandingEnv(NewLandingConfig(
            vel_response_alpha_min=0.90, vel_response_alpha_max=0.999,
            attitude_response_alpha_min=0.90, attitude_response_alpha_max=0.999,
        ))
        try:
            obs_a, _ = env_a.reset(seed=11)
            obs_b, _ = env_b.reset(seed=11)
            # same RNG position after reset: the uniform() draw shape/count is
            # unaffected by different (min, max) bounds.
            self.assertTrue(_rng_state_equal(_rng_state(env_a), _rng_state(env_b)))
            # the sampled info-only values themselves legitimately differ
            self.assertFalse(np.array_equal(env_a.vel_response_alpha, env_b.vel_response_alpha))
            # every OTHER observable channel matches exactly (dynamics-inactive)
            self.assertTrue(np.array_equal(obs_a, obs_b))
            for i in range(60):
                out_a = env_a.step(actions[i].copy())
                out_b = env_b.step(actions[i].copy())
                self.assertTrue(np.array_equal(out_a[0], out_b[0]), f"step {i} obs")
                self.assertEqual(float(out_a[1]), float(out_b[1]), f"step {i} reward")
                self.assertTrue(
                    _rng_state_equal(_rng_state(env_a), _rng_state(env_b)), f"step {i} rng"
                )
                if out_a[2] or out_a[3]:
                    break
        finally:
            env_a.close()
            env_b.close()

    def test_I_alpha_eff_nan_sentinel_every_step(self):
        env = NewLandingEnv(NewLandingConfig())
        try:
            env.reset(seed=0)
            actions = make_action_sequence("saturation", 40)
            for i in range(40):
                _, _, term, trunc, info = env.step(actions[i].copy())
                self.assertTrue(np.all(np.isnan(info["alpha_eff"])))
                if term or trunc:
                    break
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Section 10/11/16: expanded full-episode OLD-vs-NEW matrix (also the
# randomized-uncertainty / shared-RNG-topology freeze, via stage2_train)
# ---------------------------------------------------------------------------

class Freeze04_ExpandedFullEpisodeMatrixTest(unittest.TestCase):
    maxDiff = None

    def test_expanded_matrix_all_configs_all_seeds_all_patterns(self):
        summary = {}
        for cfg_name, overrides, _doc in CONFIG_SPECS:
            max_steps = NewLandingConfig(**overrides).max_steps
            agg = dict(episodes=0, total_steps=0, bounce=0, soft=0, hard=0,
                      ground_effect=0, dropout=0, stale=0, outlier=0,
                      success=0, failed=0, truncated=0)
            for seed in _EXPANDED_SEEDS:
                for pattern in PATTERNS:
                    with self.subTest(config=cfg_name, seed=seed, pattern=pattern):
                        cov = _run_full_episode_paired(overrides, seed, pattern, max_steps)
                        agg["episodes"] += 1
                        agg["total_steps"] += cov["steps"]
                        agg["bounce"] += int(cov["bounced"])
                        agg["soft"] += int(cov["soft_contact"])
                        agg["hard"] += int(cov["hard_contact"])
                        agg["ground_effect"] += int(cov["ground_effect_active"])
                        agg["dropout"] += int(cov["dropout"])
                        agg["stale"] += int(cov["stale"])
                        agg["outlier"] += int(cov["outlier"])
                        agg["success"] += int(cov["success"])
                        agg["failed"] += int(cov["failure_reason"] != "none")
                        agg["truncated"] += int(cov["truncated"])
            summary[cfg_name] = agg
            print(f"[structural-freeze matrix] config={cfg_name} {agg}")
        _REPORT["expanded_matrix"] = summary

        # Section 16: stage2_train is the widest-randomization config -- the
        # shared-RNG-topology freeze requires dropout/stale/outlier to
        # actually be exercised (not merely configured) somewhere in it.
        st = summary.get("stage2_train")
        if st is not None:
            self.assertGreater(
                st["dropout"] + st["stale"] + st["outlier"], 0,
                "stage2_train never exercised dropout/stale/outlier",
            )


# ---------------------------------------------------------------------------
# Section 12: contact-disabled full-episode gate
# ---------------------------------------------------------------------------

class Freeze05_ContactDisabledInvariantTest(unittest.TestCase):
    """The contact_enabled=False branch must be a TRUE no-op on persistent
    contact state. OLD-vs-NEW parity alone cannot catch a regression that
    unconditionally rewrites ground_contact=False / contact_count=0, because
    both sides already start at those defaults -- this test pre-seeds
    non-default persistent state and proves it survives untouched."""

    def test_disabled_branch_never_writes_persistent_state_new(self):
        env = NewLandingEnv(NewLandingConfig(contact_enabled=False))
        try:
            env.reset(seed=0)
            env.contact.state.ground_contact = True
            env.contact.state.contact_count = 7
            env.contact.state.bounce_count = 3
            actions = make_action_sequence("zero", 60)
            for i in range(60):
                _, _, term, trunc, _ = env.step(actions[i].copy())
                self.assertTrue(env.contact.state.ground_contact)
                self.assertEqual(env.contact.state.contact_count, 7)
                self.assertEqual(env.contact.state.bounce_count, 3)
                if term or trunc:
                    break
        finally:
            env.close()

    def test_disabled_branch_never_writes_persistent_state_old(self):
        env = OldLandingEnv(OldLandingConfig(contact_enabled=False))
        try:
            env.reset(seed=0)
            env.ground_contact = True
            env.contact_count = 7
            env.bounce_count = 3
            actions = make_action_sequence("zero", 60)
            for i in range(60):
                _, _, term, trunc, _ = env.step(actions[i].copy())
                self.assertTrue(env.ground_contact)
                self.assertEqual(env.contact_count, 7)
                self.assertEqual(env.bounce_count, 3)
                if term or trunc:
                    break
        finally:
            env.close()

    def test_contact_disabled_full_episode_old_vs_new(self):
        overrides = dict(next(o for n, o, _ in CONFIG_SPECS if n == "stage0_eval"))
        self.assertEqual(overrides.get("contact_enabled"), False)
        max_steps = NewLandingConfig(**overrides).max_steps
        for seed in (0, 1, 2, 3, 4):
            with self.subTest(seed=seed):
                _run_full_episode_paired(overrides, seed, "zero", max_steps)


# ---------------------------------------------------------------------------
# Section 13: forced-bounce full-episode gate
# ---------------------------------------------------------------------------

class Freeze06_ForcedBounceFullEpisodeTest(unittest.TestCase):

    def test_forced_bounce_full_episode_old_vs_new(self):
        max_steps = NewLandingConfig(**_BOUNCE_OVERRIDES).max_steps
        any_bounce = False
        for seed in (0, 1, 2, 3, 4):
            with self.subTest(seed=seed):
                cov = _run_full_episode_paired(_BOUNCE_OVERRIDES, seed, "descend_z", max_steps)
                if cov["bounced"]:
                    any_bounce = True
        self.assertTrue(any_bounce, "forced-bounce config never bounced across seeds 0-4")


# ---------------------------------------------------------------------------
# Section 14: motor-cutoff feedback full gate
# ---------------------------------------------------------------------------

class Freeze07_MotorCutoffFeedbackFullEpisodeTest(unittest.TestCase):

    def test_motor_cutoff_multistep_full_episode_old_vs_new(self):
        max_steps = NewLandingConfig().max_steps
        any_multistep = False
        for seed in (0, 1, 2, 3, 5):
            with self.subTest(seed=seed):
                cov = _run_full_episode_paired({}, seed, "descend_z", max_steps)
                if cov["motor_cutoff_multistep"]:
                    any_multistep = True
        self.assertTrue(
            any_multistep,
            "no seed produced a >=2-step motor_cutoff window with the episode still running",
        )


# ---------------------------------------------------------------------------
# Section 15: ground-effect path gate
# ---------------------------------------------------------------------------

class Freeze08_GroundEffectPathTest(unittest.TestCase):

    def test_ground_effect_active_full_episode_old_vs_new(self):
        max_steps = NewLandingConfig(**_GROUND_EFFECT_OVERRIDES).max_steps
        any_active = False
        for seed in (0, 1, 2, 3, 4):
            with self.subTest(seed=seed):
                cov = _run_full_episode_paired(
                    _GROUND_EFFECT_OVERRIDES, seed, "descend_z", max_steps
                )
                if cov["ground_effect_active"]:
                    any_active = True
        self.assertTrue(
            any_active, "ground_effect_factor never left 1.0 across seeds 0-4"
        )


# ---------------------------------------------------------------------------
# Section 17: NEW-only determinism
# ---------------------------------------------------------------------------

class Freeze09_NewOnlyDeterminismTest(unittest.TestCase):

    def test_new_only_determinism_stage2_train(self):
        overrides = dict(next(o for n, o, _ in CONFIG_SPECS if n == "stage2_train"))
        max_steps = NewLandingConfig(**overrides).max_steps
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                _run_full_episode_paired(
                    overrides, seed, "pseudo_random", max_steps,
                    left_impl=NEW_IMPL, right_impl=NEW_IMPL,
                )

    def test_new_only_determinism_forced_bounce(self):
        max_steps = NewLandingConfig(**_BOUNCE_OVERRIDES).max_steps
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                _run_full_episode_paired(
                    _BOUNCE_OVERRIDES, seed, "descend_z", max_steps,
                    left_impl=NEW_IMPL, right_impl=NEW_IMPL,
                )


# ---------------------------------------------------------------------------
# Section 22: negative control -- dynamics path
# ---------------------------------------------------------------------------

class Freeze10_NegativeControlDynamicsTest(unittest.TestCase):

    def test_plant_final_accel_perturbation_is_caught(self):
        from landing_rl.dynamics import plant_model as pm

        orig_step = pm.PlantModel.step

        def perturbed_step(self, state, v_cmd, dt, rng, wind_accel, target_yaw,
                           body_rate_response_alpha, thrust_response_alpha):
            result = orig_step(
                self, state, v_cmd, dt, rng, wind_accel, target_yaw,
                body_rate_response_alpha, thrust_response_alpha,
            )
            state.accel = state.accel + 1e-3
            return result

        pm.PlantModel.step = perturbed_step
        try:
            with self.assertRaises(AssertionError) as ctx:
                _run_full_episode_paired({}, 0, "zero", 10)
            self.assertIn("parity broken", str(ctx.exception))
        finally:
            pm.PlantModel.step = orig_step
            self.assertIs(pm.PlantModel.step, orig_step)
            self.assertEqual(pm.PlantModel.step.__module__, "landing_rl.dynamics.plant_model")


# ---------------------------------------------------------------------------
# Section 23: negative control -- contact path
# ---------------------------------------------------------------------------

class Freeze11_NegativeControlContactTest(unittest.TestCase):

    def test_post_contact_velocity_perturbation_is_caught(self):
        from landing_rl.contact import contact_model as cm

        orig_apply = cm.ContactModel.apply

        def perturbed_apply(self, **kwargs):
            result, thrust_accel, thrust_accel_setpoint = orig_apply(self, **kwargs)
            if result.contact_event:
                kwargs["vel"][2] += 1e-3
            return result, thrust_accel, thrust_accel_setpoint

        cm.ContactModel.apply = perturbed_apply
        try:
            caught = False
            first_exc = None
            for seed in (0, 1, 2, 3, 4):
                try:
                    _run_full_episode_paired(_BOUNCE_OVERRIDES, seed, "descend_z", 420)
                except AssertionError as exc:
                    caught = True
                    first_exc = exc
                    break
            self.assertTrue(caught, "contact-path perturbation was never detected")
            self.assertIn("parity broken", str(first_exc))
        finally:
            cm.ContactModel.apply = orig_apply
            self.assertIs(cm.ContactModel.apply, orig_apply)


# ---------------------------------------------------------------------------
# Final report (runs last by class-name sort order)
# ---------------------------------------------------------------------------

class Freeze12_FinalReportTest(unittest.TestCase):

    def test_print_final_structural_freeze_report(self):
        print("=== STRUCTURAL FREEZE STATS ===")
        print(_STATS)
        print("=== STRUCTURAL FREEZE EXPANDED MATRIX SUMMARY ===")
        for k, v in _REPORT.items():
            print(k, "=", v)


if __name__ == "__main__":
    unittest.main()
