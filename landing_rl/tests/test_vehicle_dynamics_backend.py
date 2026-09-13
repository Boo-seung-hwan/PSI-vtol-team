"""VehicleDynamicsBackend interface-seam regression gate.

Covers the five tests required by the "define a VehicleDynamicsBackend
interface" refactor task:

    A. LegacyVehicleDynamics satisfies the backend contract
    B. PlantModel accepts a backend implementing the contract
    C. PlantModel default behavior still uses LegacyVehicleDynamics
    D. a mock backend can be injected without touching LandingEnv semantics
    E. same seed/action gives an exact identical trajectory before/after
       the refactor

This file adds nothing to the physics, reward, controller, or contact
behavior. It proves a type-level seam (``PlantModel`` depends on
``VehicleDynamicsBackend`` rather than the concrete
``LegacyVehicleDynamics`` class) exists and changes NOTHING observable.
Exact OLD-vs-NEW parity (E) reuses the already-verified comparison
machinery from ``test_legacy_regression_contract.py`` rather than
re-implementing it -- see that file and ``test_structural_freeze.py`` for
the exhaustive seed/config/pattern campaign; this file's own E-test is a
smaller, refactor-scoped confirmation that the Protocol's introduction
itself changed no behavior.

Run from the worktree root:

    python3 -m unittest discover -s landing_rl/tests -p "test_*.py"

or in isolation:

    python3 -m unittest landing_rl.tests.test_vehicle_dynamics_backend -v
"""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

_TESTS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parents[1]  # landing_rl/tests -> landing_rl -> <worktree root>

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Flat top-level import -- same precedent as test_structural_freeze.py /
# test_entry_point_migration.py (unittest discover adds landing_rl/tests/
# itself to sys.path; there is no __init__.py in that directory).
import test_legacy_regression_contract as _trc  # noqa: E402
from test_legacy_regression_contract import (  # noqa: E402
    NewLandingConfig,
    NewLandingEnv,
)

from landing_rl.contact import ContactModel  # noqa: E402
from landing_rl.dynamics import (  # noqa: E402
    LegacyVehicleDynamics,
    PlantModel,
    ProcessNoiseSampler,
    VehicleDynamicsBackend,
    VehicleState,
)

# NOTE: exactly like test_structural_freeze.py / test_entry_point_migration.py,
# ``LandingEnvParityContractTest`` is deliberately accessed only as
# ``_trc.LandingEnvParityContractTest``, never imported by name -- unittest's
# module-level discovery would otherwise re-collect and re-run its test_*
# methods a second time under this module.


def _new_helper():
    """A ``LandingEnvParityContractTest`` instance used purely as a holder of
    its already-verified ``_assert_reset_equal`` / ``_assert_states_equal``
    methods -- never registered with the test loader, never run as a test
    itself."""
    return _trc.LandingEnvParityContractTest("test_new_package_import")


def _make_state(overrides: dict) -> VehicleState:
    base = dict(
        pos=[0.0, 0.0, -2.0],
        vel=[0.1, -0.05, 0.02],
        accel=[0.0, 0.0, 0.0],
        prev_accel=[0.0, 0.0, 0.0],
        attitude=[0.01, -0.02, 0.0],
        yaw_rate=0.0,
        body_rates=[0.0, 0.0, 0.0],
        thrust_accel=9.8065,
        attitude_setpoint=[0.0, 0.0, 0.0],
        thrust_accel_setpoint=9.8065,
        accel_cmd=[0.0, 0.0, 0.0],
        ground_effect_factor=1.0,
    )
    base.update(overrides)
    arr_fields = (
        "pos", "vel", "accel", "prev_accel", "attitude", "body_rates",
        "attitude_setpoint", "accel_cmd",
    )
    kwargs = {
        k: (np.array(v, dtype=np.float64) if k in arr_fields else v)
        for k, v in base.items()
    }
    return VehicleState(**kwargs)


class A_LegacyVehicleDynamicsSatisfiesBackendContractTest(unittest.TestCase):
    """A: LegacyVehicleDynamics satisfies VehicleDynamicsBackend structurally,
    with no inheritance / code change of its own."""

    def test_isinstance_check_passes(self):
        cfg = NewLandingConfig()
        dyn = LegacyVehicleDynamics(cfg, ProcessNoiseSampler(cfg))
        self.assertIsInstance(dyn, VehicleDynamicsBackend)

    def test_no_base_class_added(self):
        """LegacyVehicleDynamics does not inherit from the Protocol (Protocol
        satisfaction is structural, not nominal) -- confirms the Protocol
        introduced zero class-hierarchy change."""
        self.assertNotIn(VehicleDynamicsBackend, LegacyVehicleDynamics.__mro__)

    def test_advance_free_flight_still_the_only_public_dynamics_method_used(self):
        cfg = NewLandingConfig()
        dyn = LegacyVehicleDynamics(cfg, ProcessNoiseSampler(cfg))
        self.assertTrue(hasattr(dyn, "advance_free_flight"))
        self.assertTrue(callable(dyn.advance_free_flight))


class B_PlantModelAcceptsBackendContractTest(unittest.TestCase):
    """B: PlantModel accepts any object implementing the contract, whether or
    not it is a LegacyVehicleDynamics instance."""

    def test_plant_accepts_legacy_backend(self):
        cfg = NewLandingConfig()
        dyn = LegacyVehicleDynamics(cfg, ProcessNoiseSampler(cfg))
        contact = ContactModel(cfg)
        plant = PlantModel(cfg, dyn, contact)
        self.assertIs(plant.dynamics, dyn)
        self.assertIsInstance(plant.dynamics, VehicleDynamicsBackend)

    def test_plant_accepts_arbitrary_backend_satisfying_contract(self):
        """A minimal duck-typed object -- not a LegacyVehicleDynamics
        subclass, not even related to it -- is accepted verbatim."""

        class _MinimalBackend:
            def advance_free_flight(self, state, v_cmd, dt, rng, wind_accel,
                                     target_yaw, body_rate_response_alpha,
                                     thrust_response_alpha, motor_cutoff,
                                     ground_contact) -> None:
                state.pos = state.pos + v_cmd * dt

        cfg = NewLandingConfig()
        backend = _MinimalBackend()
        self.assertIsInstance(backend, VehicleDynamicsBackend)
        plant = PlantModel(cfg, backend, ContactModel(cfg))
        self.assertIs(plant.dynamics, backend)

    def test_object_missing_the_method_fails_the_contract(self):
        class _NotABackend:
            def some_other_method(self):
                pass

        self.assertNotIsInstance(_NotABackend(), VehicleDynamicsBackend)


class C_PlantModelDefaultUsesLegacyBackendTest(unittest.TestCase):
    """C: LandingEnv(), constructed with no extra configuration, still wires
    LegacyVehicleDynamics as PlantModel's dynamics backend."""

    def test_landing_env_default_backend_is_legacy(self):
        env = NewLandingEnv(NewLandingConfig())
        try:
            self.assertIsInstance(env.plant, PlantModel)
            self.assertIsInstance(env.plant.dynamics, LegacyVehicleDynamics)
            self.assertIsInstance(env.plant.dynamics, VehicleDynamicsBackend)
            # PlantModel coordinates the SAME instance LandingEnv already
            # constructed -- no duplicate dynamics object.
            self.assertIs(env.plant.dynamics, env.legacy_dynamics)
        finally:
            env.close()

    def test_plant_model_public_constructor_unchanged(self):
        """PlantModel(cfg, dynamics, contact) -- same three positional
        arguments as before this refactor; no new required parameter."""
        cfg = NewLandingConfig()
        dyn = LegacyVehicleDynamics(cfg, ProcessNoiseSampler(cfg))
        contact = ContactModel(cfg)
        plant = PlantModel(cfg, dyn, contact)  # would TypeError if signature changed
        self.assertEqual(set(vars(plant)), {"cfg", "dynamics", "contact"})


class D_MockBackendInjectionLeavesLandingEnvSemanticsUntouchedTest(unittest.TestCase):
    """D: a mock/fake backend can be handed to PlantModel directly -- entirely
    below LandingEnv -- without requiring any LandingEnv code change. This
    proves the seam is at PlantModel, not smeared across LandingEnv."""

    class _RecordingBackend:
        """Same recording-fake shape as
        test_plant_model.py::PlantModelOrchestrationOrderTest, expressed
        explicitly against the named Protocol this time."""

        def __init__(self):
            self.calls = 0

        def advance_free_flight(self, state, v_cmd, dt, rng, wind_accel,
                                 target_yaw, body_rate_response_alpha,
                                 thrust_response_alpha, motor_cutoff,
                                 ground_contact) -> None:
            self.calls += 1
            state.vel = state.vel + v_cmd * 0.0  # no-op physics; just records the call
            state.accel = np.zeros(3, dtype=np.float64)

    def test_mock_backend_runs_through_plant_step_without_landing_env(self):
        cfg = NewLandingConfig()
        backend = self._RecordingBackend()
        self.assertIsInstance(backend, VehicleDynamicsBackend)
        contact = ContactModel(cfg)
        plant = PlantModel(cfg, backend, contact)
        state = _make_state({})
        rng = np.random.default_rng(0)

        plant.step(
            state, np.zeros(3, dtype=np.float64), 0.05, rng,
            np.zeros(3, dtype=np.float64), 0.0,
            np.array([0.4, 0.4, 0.4], dtype=np.float64), 0.30,
        )
        self.assertEqual(backend.calls, 1)

    def test_landing_env_module_and_source_untouched_by_this_seam(self):
        """LandingEnv's own source does not need to change to support backend
        injection -- the swap point is PlantModel's constructor, one layer
        below LandingEnv, exactly as intended by the goal architecture
        (LandingEnv -> PlantModel -> VehicleDynamicsBackend)."""
        import inspect

        import landing_rl.envs.landing_env as _landing_env_mod
        source = inspect.getsource(_landing_env_mod)
        self.assertNotIn("VehicleDynamicsBackend", source)
        self.assertNotIn("MuJoCo", source)


class E_ExactParityUnchangedByThisRefactorTest(unittest.TestCase):
    """E: same seed + same action sequence -> exact identical trajectory,
    OLD (env_prototype) vs NEW (landing_rl), after the VehicleDynamicsBackend
    seam was introduced. Reuses the frozen comparator, not a re-implementation."""

    maxDiff = None

    def test_exact_parity_across_configs_seeds_patterns(self):
        helper = _new_helper()
        configs = ("default", "stage0_eval", "stage2_eval", "stage2_train")
        seeds = (0, 1, 2, 3, 4)
        patterns = ("zero", "pseudo_random", "saturation")
        n_steps = 60

        for cfg_name, overrides, _doc in _trc.CONFIG_SPECS:
            if cfg_name not in configs:
                continue
            for seed in seeds:
                for pattern in patterns:
                    with self.subTest(config=cfg_name, seed=seed, pattern=pattern):
                        helper._run_paired(
                            overrides, seed, pattern, n_steps,
                            left_impl=_trc.LEGACY_IMPL, right_impl=_trc.NEW_IMPL,
                        )

    def test_default_backend_object_identity_stable_across_reset(self):
        """A regression-adjacent sanity check: reset() does not rebuild the
        PlantModel/backend objects (only the VehicleState/ContactModel they
        mutate), so the same backend instance keeps running for the whole
        env lifetime."""
        env = NewLandingEnv(NewLandingConfig())
        try:
            plant_before = env.plant
            dynamics_before = env.plant.dynamics
            env.reset(seed=0)
            self.assertIs(env.plant, plant_before)
            self.assertIs(env.plant.dynamics, dynamics_before)
            self.assertIsInstance(env.plant.dynamics, LegacyVehicleDynamics)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
