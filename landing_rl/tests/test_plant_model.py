"""Focused tests for the Phase 13C-2 PlantModel orchestration extraction.

Secondary to the OLD-vs-NEW environment regression in
``test_legacy_regression_contract.py`` and the checkpoint gate in
``test_checkpoint_compatibility.py``, both of which remain authoritative (and
which now exercise ``PlantModel`` end-to-end through ``NewLandingEnv``, since
``LandingEnv._update_rigid_body_dynamics`` calls ``self.plant.step(...)``).
``test_contact_model.py`` and ``test_legacy_dynamics.py`` also continue to
run OLD-vs-NEW through the PlantModel-based environment unchanged.

``PlantModel`` coordinates the already-extracted ``LegacyVehicleDynamics``
(free flight) and ``ContactModel`` (ground contact) collaborators in the
exact D2/D3/D26/D27 order that used to live inline in
``LandingEnv._update_rigid_body_dynamics``. These tests compare
``PlantModel.step()`` against an independent, manually-sequenced reference
that calls ``contact.begin_step()`` / ``dynamics.advance_free_flight()`` /
``contact.apply()`` / the accel recompute directly in that order -- NOT
through ``PlantModel`` -- so a parity failure cannot be masked by a bug
shared between the two call sites.

Comparison is exact (``np.array_equal`` / ``==`` / identical RNG state). No
tolerance.
"""

from __future__ import annotations

import unittest

import numpy as np

from test_legacy_dynamics import _ARR_FIELDS, _SCALAR_FIELDS, _INIT_STATES, _make_state
from test_legacy_regression_contract import NewLandingConfig, _rng_state_equal

from landing_rl.contact import ContactModel
from landing_rl.dynamics import LegacyVehicleDynamics, PlantModel, ProcessNoiseSampler

_PERSIST = ("ground_contact", "contact_count", "bounce_count", "motor_cutoff")
_TRANSIENT = (
    "contact_event", "soft_contact", "hard_contact", "bounced",
    "touchdown_quality", "last_impact_vz", "last_touchdown_vxy",
    "last_bounce_speed",
)


def _manual_orchestration(cfg, state, v_cmd, dt, rng, wind_accel, target_yaw,
                          bra, ta, contact):
    """Independent D2/D3/D26/D27 reference -- calls the same collaborators
    directly, NOT through PlantModel."""
    dt_safe = max(float(dt), 1e-6)
    contact.begin_step()
    vel_before = state.vel.copy()
    dyn = LegacyVehicleDynamics(cfg, ProcessNoiseSampler(cfg))
    dyn.advance_free_flight(
        state, v_cmd, dt_safe, rng, wind_accel, target_yaw, bra, ta,
        contact.state.motor_cutoff, contact.state.ground_contact,
    )
    result, thrust_accel, thrust_accel_setpoint = contact.apply(
        pos=state.pos,
        vel=state.vel,
        attitude=state.attitude,
        vel_before_step=vel_before,
        thrust_accel=state.thrust_accel,
        thrust_accel_setpoint=state.thrust_accel_setpoint,
        rng=rng,
    )
    state.thrust_accel = thrust_accel
    state.thrust_accel_setpoint = thrust_accel_setpoint
    state.accel = (state.vel - vel_before) / dt_safe
    return result, vel_before


def _run_plant(cfg, seed_fields, motor_cutoff_pre, ground_contact_pre,
              v_cmd, dt, seed, wind, target_yaw, bra, ta):
    state = _make_state(seed_fields)
    contact = ContactModel(cfg)
    contact.state.motor_cutoff = motor_cutoff_pre
    contact.state.ground_contact = ground_contact_pre
    dyn = LegacyVehicleDynamics(cfg, ProcessNoiseSampler(cfg))
    plant = PlantModel(cfg, dyn, contact)
    rng = np.random.default_rng(seed)
    plant.step(
        state, v_cmd, dt, rng, np.array(wind, dtype=np.float64), target_yaw,
        np.array(bra, dtype=np.float64), ta,
    )
    return state, contact, rng


def _run_manual(cfg, seed_fields, motor_cutoff_pre, ground_contact_pre,
                v_cmd, dt, seed, wind, target_yaw, bra, ta):
    state = _make_state(seed_fields)
    contact = ContactModel(cfg)
    contact.state.motor_cutoff = motor_cutoff_pre
    contact.state.ground_contact = ground_contact_pre
    rng = np.random.default_rng(seed)
    _manual_orchestration(
        cfg, state, v_cmd, dt, rng, np.array(wind, dtype=np.float64),
        target_yaw, np.array(bra, dtype=np.float64), ta, contact,
    )
    return state, contact, rng


def _assert_full_parity(tc, state_a, contact_a, rng_a, state_b, contact_b, rng_b):
    for f in _ARR_FIELDS:
        tc.assertTrue(
            np.array_equal(getattr(state_a, f), getattr(state_b, f)),
            f"{f}: plant={getattr(state_a, f)!r} manual={getattr(state_b, f)!r}",
        )
    for f in _SCALAR_FIELDS:
        tc.assertEqual(getattr(state_a, f), getattr(state_b, f), f)
    for k in _PERSIST:
        tc.assertEqual(getattr(contact_a.state, k), getattr(contact_b.state, k), k)
    for k in _TRANSIENT:
        tc.assertEqual(getattr(contact_a.result, k), getattr(contact_b.result, k), k)
    tc.assertTrue(
        _rng_state_equal(rng_a.bit_generator.state, rng_b.bit_generator.state),
        "RNG state diverged between PlantModel.step() and the manual reference",
    )


# ---------------------------------------------------------------------------
# Scenario matrix: (name, cfg_overrides, seed_fields, motor_cutoff_pre,
#                    ground_contact_pre, v_cmd, dt)
# ---------------------------------------------------------------------------
_V_CMD = np.array([0.15, -0.20, 0.35], dtype=np.float64)
_WIND = (0.05, -0.02, 0.01)
_BRA = (0.4, 0.4, 0.4)
_TA = 0.30

_SCENARIOS = {
    # E: airborne well above ground -- contact never resolves this step.
    "no_contact": (
        {}, _INIT_STATES["descending"], False, False, _V_CMD, 0.05,
    ),
    # F: already at/below ground, small velocities/tilt -> soft touchdown.
    "soft_contact": (
        {}, dict(pos=[0.0, 0.0, 0.02], vel=[0.02, 0.01, 0.05]), False, False,
        np.array([0.0, 0.0, 0.05]), 0.05,
    ),
    # G: already at/below ground, large downward velocity -> hard touchdown.
    "hard_contact": (
        {}, dict(pos=[0.0, 0.0, 0.02], vel=[0.1, 0.0, 1.2]), False, False,
        np.array([0.0, 0.0, 0.5]), 0.05,
    ),
    # H: forced-bounce config (low bounce threshold) from an at-ground state.
    "forced_bounce": (
        dict(bounce_vz_threshold_mps=0.05, touchdown_vz_soft_mps=0.01),
        dict(pos=[0.0, 0.0, 0.02], vel=[0.05, 0.0, 0.15]), False, False,
        np.array([0.0, 0.0, 0.3]), 0.05,
    ),
    # M: contact disabled entirely -- ContactModel.apply() is a no-op.
    "contact_disabled": (
        dict(contact_enabled=False),
        dict(pos=[0.0, 0.0, 0.05], vel=[0.1, 0.0, 1.2]), False, False,
        np.array([0.0, 0.0, 0.5]), 0.05,
    ),
    # motor_cutoff / ground_contact latched from a PREVIOUS step feed into
    # this step's free-flight thrust clamp (persistent-state input).
    "prior_cutoff_and_contact": (
        {}, _INIT_STATES["near_ground"], True, True, _V_CMD, 0.05,
    ),
}


class PlantModelStructuralTest(unittest.TestCase):
    """A/B/C: ownership constraints."""

    def _make_plant(self, cfg=None):
        cfg = cfg or NewLandingConfig()
        contact = ContactModel(cfg)
        dyn = LegacyVehicleDynamics(cfg, ProcessNoiseSampler(cfg))
        return PlantModel(cfg, dyn, contact), dyn, contact

    def test_owns_only_cfg_dynamics_contact(self):
        plant, _, _ = self._make_plant()
        self.assertEqual(set(vars(plant)), {"cfg", "dynamics", "contact"})

    def test_holds_no_rng(self):
        plant, _, _ = self._make_plant()
        for banned in ("np_random", "rng", "_rng"):
            self.assertFalse(hasattr(plant, banned), banned)

    def test_holds_no_vehicle_state_or_env_reference(self):
        plant, _, _ = self._make_plant()
        for banned in ("state", "vehicle_state", "_vehicle_state", "env", "_env"):
            self.assertFalse(hasattr(plant, banned), banned)

    def test_constructor_consumes_no_rng(self):
        rng = np.random.default_rng(123)
        before = rng.bit_generator.state
        self._make_plant()
        self.assertTrue(_rng_state_equal(before, rng.bit_generator.state))


class PlantModelOrchestrationOrderTest(unittest.TestCase):
    """D: canonical order begin_step -> free-flight -> contact -> final accel,
    using recording fakes so the order/data-flow is asserted directly rather
    than inferred from a numeric match."""

    class _RecordingDynamics:
        def __init__(self):
            self.calls = []

        def advance_free_flight(self, state, v_cmd, dt, rng, wind_accel,
                                 target_yaw, bra, ta, motor_cutoff,
                                 ground_contact):
            self.calls.append(("free_flight", motor_cutoff, ground_contact))
            # Mutate vel to prove the D3 snapshot was taken BEFORE this call.
            state.vel = state.vel + np.array([1.0, 0.0, 0.0])

    class _RecordingContact:
        def __init__(self, motor_cutoff, ground_contact):
            self.calls = []
            self.result = object()  # sentinel: PlantModel must return THIS object

            class _S:
                pass

            self.state = _S()
            self.state.motor_cutoff = motor_cutoff
            self.state.ground_contact = ground_contact

        def begin_step(self):
            self.calls.append("begin_step")

        def apply(self, *, pos, vel, attitude, vel_before_step, thrust_accel,
                  thrust_accel_setpoint, rng):
            self.calls.append(("apply", vel_before_step.copy(), vel.copy()))
            return self.result, thrust_accel, thrust_accel_setpoint

    def test_order_vel_before_snapshot_and_persistent_state_feed(self):
        dyn = self._RecordingDynamics()
        contact = self._RecordingContact(motor_cutoff=True, ground_contact=True)
        plant = PlantModel(NewLandingConfig(), dyn, contact)
        state = _make_state({})
        state.vel = np.array([0.1, 0.2, 0.3])

        result = plant.step(
            state, np.zeros(3), 0.05, np.random.default_rng(0),
            np.zeros(3), 0.0, np.array([0.4, 0.4, 0.4]), 0.3,
        )

        # exact call order
        self.assertEqual(
            [c if isinstance(c, str) else c[0] for c in
             ([contact.calls[0]] + [dyn.calls[0]] + [contact.calls[1]])],
            ["begin_step", "free_flight", "apply"],
        )
        # motor_cutoff / ground_contact fed from the PERSISTENT contact.state
        self.assertEqual(dyn.calls[0][1], True)
        self.assertEqual(dyn.calls[0][2], True)
        # D3 vel_before handed to contact.apply is the PRE-free-flight vel
        np.testing.assert_array_equal(contact.calls[1][1], [0.1, 0.2, 0.3])
        # post-free-flight vel handed to contact.apply reflects the mutation
        np.testing.assert_array_equal(contact.calls[1][2], [1.1, 0.2, 0.3])
        # PlantModel returns contact.result verbatim -- no new result object
        self.assertIs(result, contact.result)
        # D27: final accel uses (post-fake-contact vel - PRE-free-flight vel)
        np.testing.assert_array_equal(
            state.accel, np.array([1.0, 0.0, 0.0]) / 0.05
        )


class PlantModelParityTest(unittest.TestCase):
    """E/F/G/H/I/M/J/K: PlantModel.step() vs. the independent manual
    orchestration, across contact regimes, including exact final-accel
    semantics and RNG-state equality."""

    maxDiff = None

    def test_scenarios_match_manual_orchestration(self):
        seen = {name: False for name in _SCENARIOS}
        for name, (overrides, seed_fields, mc_pre, gc_pre, v_cmd, dt) in _SCENARIOS.items():
            for seed in (0, 1, 7, 5000):
                with self.subTest(scenario=name, seed=seed):
                    cfg = NewLandingConfig(**overrides)
                    state_a, contact_a, rng_a = _run_plant(
                        cfg, seed_fields, mc_pre, gc_pre, v_cmd, dt, seed,
                        _WIND, 0.0, _BRA, _TA,
                    )
                    state_b, contact_b, rng_b = _run_manual(
                        cfg, seed_fields, mc_pre, gc_pre, v_cmd, dt, seed,
                        _WIND, 0.0, _BRA, _TA,
                    )
                    _assert_full_parity(
                        self, state_a, contact_a, rng_a, state_b, contact_b, rng_b
                    )
                    if name == "no_contact" and not contact_a.result.contact_event:
                        seen[name] = True
                    if name == "soft_contact" and contact_a.result.soft_contact:
                        seen[name] = True
                    if name == "hard_contact" and contact_a.result.hard_contact:
                        seen[name] = True
                    if name == "forced_bounce" and contact_a.result.bounced:
                        seen[name] = True
                    if name == "contact_disabled" and not contact_a.result.contact_event:
                        seen[name] = True
                    if name == "prior_cutoff_and_contact":
                        seen[name] = True

        for name, ok in seen.items():
            self.assertTrue(ok, f"scenario {name!r} never exercised its named regime")

    def test_final_accel_uses_post_contact_velocity(self):
        """J: state.accel == (post-contact vel - pre-free-flight vel) / dt_safe,
        computed independently of PlantModel's internals."""
        cfg = NewLandingConfig()
        seed_fields = dict(pos=[0.0, 0.0, 0.02], vel=[0.02, 0.01, 0.05])
        vel_before_expected = np.array(seed_fields["vel"], dtype=np.float64)
        dt = 0.05
        state, contact, _ = _run_plant(
            cfg, seed_fields, False, False, np.array([0.0, 0.0, 0.05]), dt,
            0, _WIND, 0.0, _BRA, _TA,
        )
        expected_accel = (state.vel - vel_before_expected) / max(dt, 1e-6)
        np.testing.assert_array_equal(state.accel, expected_accel)
        # sanity: contact actually ran and changed velocity this step, so this
        # is not a vacuous check against the free-flight provisional accel.
        self.assertTrue(contact.result.contact_event)

    def test_motor_cutoff_feed_forward_across_two_steps(self):
        """I: motor_cutoff latched by step N's contact.apply() is read (via
        contact.state, persistent) by step N+1's free-flight call. Two
        chained PlantModel.step() calls (shared state + contact) must match
        two chained manual-orchestration calls exactly, and the scenario must
        actually exercise a step-N cutoff latch feeding step N+1."""
        cfg = NewLandingConfig()
        seed_fields = dict(pos=[0.0, 0.0, 0.02], vel=[0.02, 0.01, 0.05])
        v_cmd = np.array([0.0, 0.0, 0.05], dtype=np.float64)
        dt = 0.05

        state_a = _make_state(seed_fields)
        contact_a = ContactModel(cfg)
        dyn_a = LegacyVehicleDynamics(cfg, ProcessNoiseSampler(cfg))
        plant = PlantModel(cfg, dyn_a, contact_a)
        rng_a = np.random.default_rng(0)

        state_b = _make_state(seed_fields)
        contact_b = ContactModel(cfg)
        rng_b = np.random.default_rng(0)

        for step_i in range(2):
            plant.step(state_a, v_cmd, dt, rng_a, np.array(_WIND), 0.0,
                       np.array(_BRA), _TA)
            _manual_orchestration(cfg, state_b, v_cmd, dt, rng_b,
                                  np.array(_WIND), 0.0, np.array(_BRA), _TA,
                                  contact_b)
            with self.subTest(step=step_i):
                _assert_full_parity(
                    self, state_a, contact_a, rng_a, state_b, contact_b, rng_b
                )

        self.assertTrue(contact_a.state.motor_cutoff, "cutoff never latched")


class PlantModelRngBudgetTest(unittest.TestCase):
    """K/L: RNG-state equality vs. a hand-executed reference, and no extra
    bounce draw is introduced by PlantModel/ContactModel on a no-bounce step."""

    def test_no_bounce_step_consumes_exactly_the_three_free_flight_draws(self):
        cfg = NewLandingConfig()
        seed_fields = _INIT_STATES["descending"]  # well above ground: no contact at all
        v_cmd = np.array([0.15, -0.20, 0.35], dtype=np.float64)
        dt = 0.05
        dt_safe = max(dt, 1e-6)

        state, contact, rng_after = _run_plant(
            cfg, seed_fields, False, False, v_cmd, dt, 42, _WIND, 0.0, _BRA, _TA,
        )
        self.assertFalse(contact.result.contact_event, "scenario must stay airborne")

        # Reference: ONLY the three canonical free-flight normal draws, on an
        # identically-seeded RNG, via LegacyVehicleDynamics directly (no
        # ContactModel / PlantModel involvement at all).
        ref_state = _make_state(seed_fields)
        ref_rng = np.random.default_rng(42)
        LegacyVehicleDynamics(cfg, ProcessNoiseSampler(cfg)).advance_free_flight(
            ref_state, v_cmd, dt_safe, ref_rng, np.array(_WIND), 0.0,
            np.array(_BRA), _TA, False, False,
        )
        self.assertTrue(
            _rng_state_equal(rng_after.bit_generator.state, ref_rng.bit_generator.state),
            "PlantModel/ContactModel consumed RNG beyond the 3 free-flight draws "
            "on a no-bounce step",
        )

    def test_bounce_step_consumes_exactly_one_extra_uniform_draw(self):
        cfg = NewLandingConfig(bounce_vz_threshold_mps=0.05, touchdown_vz_soft_mps=0.01)
        seed_fields = dict(pos=[0.0, 0.0, 0.02], vel=[0.05, 0.0, 0.15])
        v_cmd = np.array([0.0, 0.0, 0.3], dtype=np.float64)
        dt = 0.05
        dt_safe = max(dt, 1e-6)

        state, contact, rng_after = _run_plant(
            cfg, seed_fields, False, False, v_cmd, dt, 3, _WIND, 0.0, _BRA, _TA,
        )
        self.assertTrue(contact.result.bounced, "scenario must actually bounce")

        ref_state = _make_state(seed_fields)
        ref_rng = np.random.default_rng(3)
        LegacyVehicleDynamics(cfg, ProcessNoiseSampler(cfg)).advance_free_flight(
            ref_state, v_cmd, dt_safe, ref_rng, np.array(_WIND), 0.0,
            np.array(_BRA), _TA, False, False,
        )
        ref_rng.uniform(cfg.bounce_restitution_min, cfg.bounce_restitution_max)
        self.assertTrue(
            _rng_state_equal(rng_after.bit_generator.state, ref_rng.bit_generator.state),
            "bounce step did not consume exactly free-flight(3) + restitution(1)",
        )


if __name__ == "__main__":
    unittest.main()
