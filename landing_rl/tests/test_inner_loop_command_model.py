"""InnerLoopCommandModel extraction regression gate.

Covers the five tests required by the "separate controller approximation
from physical dynamics approximation" refactor task:

    A. InnerLoopCommandModel produces the same output as the former
       LegacyVehicleDynamics._velocity_command_to_inner_loop_setpoints
       across a range of states/v_cmd/dt/target_yaw.
    B. accel_cmd exact parity.
    C. attitude_setpoint exact parity.
    D. thrust_accel_setpoint exact parity (including the motor-cutoff
       override, which stays in LegacyVehicleDynamics.advance_free_flight,
       applied to the InnerLoopCommand returned by compute()).
    E. full environment-trajectory exact parity, OLD (env_prototype) vs NEW
       (landing_rl), unaffected by this refactor.

This file adds nothing to the physics, reward, controller, or contact
behavior. It proves a type-level/structural extraction (the former private
method now lives on its own ``InnerLoopCommandModel`` component, called by
``LegacyVehicleDynamics.advance_free_flight`` at the same call site) changed
NOTHING observable. A/B/C/D reuse ``test_legacy_dynamics.py``'s already
independent ``_LegacyFreeFlightReplica`` reference (never
``LegacyVehicleDynamics`` itself) so the comparison cannot be masked by a bug
shared between the extracted code and its own reference. E reuses the
already-verified OLD-vs-NEW comparison machinery from
``test_legacy_regression_contract.py`` rather than re-implementing it -- see
that file, ``test_structural_freeze.py``, and ``test_entry_point_migration.py``
for the exhaustive seed/config/pattern campaign; this file's own E-test is a
smaller, refactor-scoped confirmation.

Run from the worktree root:

    python3 -m unittest discover -s landing_rl/tests -p "test_*.py"

or in isolation:

    python3 -m unittest landing_rl.tests.test_inner_loop_command_model -v
"""

from __future__ import annotations

import inspect
import itertools
import pathlib
import sys
import unittest

import numpy as np

_TESTS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parents[1]  # landing_rl/tests -> landing_rl -> <worktree root>

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Flat top-level import -- same precedent as test_structural_freeze.py /
# test_vehicle_dynamics_backend.py (unittest discover adds landing_rl/tests/
# itself to sys.path; there is no __init__.py in that directory).
import test_legacy_regression_contract as _trc  # noqa: E402
from test_legacy_regression_contract import (  # noqa: E402
    NewLandingConfig,
)

# ``_LegacyFreeFlightReplica`` and ``_INIT_STATES`` / ``_make_state`` are the
# already-independent reference machinery from the Phase 13C-1 gate -- reused
# here, never re-implemented.
from test_legacy_dynamics import (  # noqa: E402
    _INIT_STATES,
    _LegacyFreeFlightReplica,
    _make_state,
)

from landing_rl.dynamics import (  # noqa: E402
    InnerLoopCommand,
    InnerLoopCommandModel,
    LegacyVehicleDynamics,
    ProcessNoiseSampler,
)

# NOTE: exactly like the other Phase 13-family gate files,
# ``LandingEnvParityContractTest`` is deliberately accessed only as
# ``_trc.LandingEnvParityContractTest``, never imported by name -- unittest's
# module-level discovery would otherwise re-collect and re-run its test_*
# methods a SECOND time under this module.


def _new_helper():
    return _trc.LandingEnvParityContractTest("test_new_package_import")


def _reference_setpoints(cfg, vel, target_yaw, v_cmd, dt):
    """Independent reference: the verbatim legacy free-flight replica's own
    setpoint sub-step, never routed through InnerLoopCommandModel or
    LegacyVehicleDynamics."""
    rep = _LegacyFreeFlightReplica(cfg)
    rep.vel = np.array(vel, dtype=np.float64)
    rep.target_yaw = float(target_yaw)
    return rep._velocity_command_to_inner_loop_setpoints(v_cmd, dt)


_V_CMDS = (
    np.array([0.0, 0.0, 0.0], dtype=np.float64),
    np.array([0.15, -0.20, 0.35], dtype=np.float64),
    np.array([-1.5, 1.5, -0.9], dtype=np.float64),
    np.array([3.0, -3.0, 2.0], dtype=np.float64),  # saturating
)
_DTS = (0.03, 0.05, 0.08)
_TARGET_YAWS = (0.0, 0.4, -1.2)


class ABCD_InnerLoopCommandModelParityTest(unittest.TestCase):
    """A/B/C/D: InnerLoopCommandModel.compute() output matches the
    independent legacy replica exactly, across states/v_cmd/dt/target_yaw."""

    maxDiff = None

    def test_compute_matches_legacy_replica_setpoints(self):
        cfg = NewLandingConfig()
        model = InnerLoopCommandModel(cfg)

        for sname, seed_fields in _INIT_STATES.items():
            for v_cmd in _V_CMDS:
                for dt in _DTS:
                    for target_yaw in _TARGET_YAWS:
                        with self.subTest(
                            state=sname, v_cmd=tuple(v_cmd), dt=dt,
                            target_yaw=target_yaw,
                        ):
                            state = _make_state(seed_fields)
                            cmd = model.compute(state, v_cmd, dt, target_yaw)
                            self.assertIsInstance(cmd, InnerLoopCommand)

                            ref_attitude_sp, ref_thrust_sp, ref_accel_cmd = (
                                _reference_setpoints(
                                    cfg, state.vel, target_yaw, v_cmd, dt
                                )
                            )

                            # B: accel_cmd exact parity
                            self.assertTrue(
                                np.array_equal(cmd.accel_cmd, ref_accel_cmd),
                                f"accel_cmd: new={cmd.accel_cmd!r} "
                                f"ref={ref_accel_cmd!r}",
                            )
                            # C: attitude_setpoint exact parity
                            self.assertTrue(
                                np.array_equal(
                                    cmd.attitude_setpoint, ref_attitude_sp
                                ),
                                f"attitude_setpoint: new={cmd.attitude_setpoint!r} "
                                f"ref={ref_attitude_sp!r}",
                            )
                            # D: thrust_accel_setpoint exact parity (pre
                            # motor-cutoff override -- that override is
                            # LegacyVehicleDynamics's own responsibility,
                            # verified separately below)
                            self.assertEqual(
                                cmd.thrust_accel_setpoint, ref_thrust_sp,
                            )

    def test_compute_consumes_no_rng_and_mutates_no_input(self):
        """compute() takes no rng parameter at all (zero draws by
        construction), and does not mutate the VehicleState handed in."""
        sig = inspect.signature(InnerLoopCommandModel.compute)
        self.assertEqual(
            list(sig.parameters), ["self", "state", "v_cmd", "dt", "target_yaw"]
        )

        cfg = NewLandingConfig()
        model = InnerLoopCommandModel(cfg)
        state = _make_state(_INIT_STATES["tilted_rolling"])
        pos_before = state.pos.copy()
        vel_before = state.vel.copy()
        attitude_before = state.attitude.copy()
        model.compute(state, np.array([0.2, -0.1, 0.3]), 0.05, 0.1)
        self.assertTrue(np.array_equal(state.pos, pos_before))
        self.assertTrue(np.array_equal(state.vel, vel_before))
        self.assertTrue(np.array_equal(state.attitude, attitude_before))

    def test_inner_loop_command_model_owns_only_cfg(self):
        cfg = NewLandingConfig()
        model = InnerLoopCommandModel(cfg)
        self.assertEqual(set(vars(model)), {"cfg"})
        for banned in ("np_random", "rng", "state", "env", "_env", "contact"):
            self.assertFalse(hasattr(model, banned), banned)

    def test_motor_cutoff_override_stays_in_legacy_vehicle_dynamics(self):
        """The motor-cutoff override on thrust_accel_setpoint is NOT part of
        InnerLoopCommandModel's output -- it is applied by
        LegacyVehicleDynamics.advance_free_flight immediately afterward,
        exactly where it sat before this extraction. This test proves
        compute() itself is motor-cutoff-agnostic (same output regardless of
        cutoff), and that advance_free_flight's actual written setpoint
        still reflects the override."""
        cfg = NewLandingConfig()
        model = InnerLoopCommandModel(cfg)
        state_a = _make_state(_INIT_STATES["near_ground"])
        state_b = _make_state(_INIT_STATES["near_ground"])
        v_cmd = np.array([0.0, 0.0, 0.5], dtype=np.float64)

        cmd_a = model.compute(state_a, v_cmd, 0.05, 0.0)
        cmd_b = model.compute(state_b, v_cmd, 0.05, 0.0)
        self.assertEqual(cmd_a.thrust_accel_setpoint, cmd_b.thrust_accel_setpoint)

        lvd = LegacyVehicleDynamics(cfg, ProcessNoiseSampler(cfg))
        st_cutoff = _make_state(_INIT_STATES["near_ground"])
        rng = np.random.default_rng(0)
        lvd.advance_free_flight(
            st_cutoff, v_cmd, 0.05, rng, np.zeros(3), 0.0,
            np.array([0.4, 0.4, 0.4]), 0.30, True, False,
        )
        self.assertEqual(
            st_cutoff.thrust_accel_setpoint,
            float(cfg.motor_cutoff_thrust_accel_mps2),
        )
        self.assertNotEqual(
            st_cutoff.thrust_accel_setpoint, cmd_a.thrust_accel_setpoint,
        )


# ---------------------------------------------------------------------------
# E: full environment-trajectory exact parity, unaffected by this refactor.
# ---------------------------------------------------------------------------

def _fixed_axis_actions(sign: float, n_steps: int) -> np.ndarray:
    return np.full((n_steps, 3), float(sign), dtype=np.float32)


class E_FullTrajectoryExactParityTest(unittest.TestCase):
    maxDiff = None

    def test_exact_parity_main_patterns(self):
        helper = _new_helper()
        configs = ("default", "stage0_eval", "stage2_eval", "stage2_train")
        seeds = tuple(range(10))
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

    def test_exact_parity_fixed_axis_patterns(self):
        """+axis / -axis sustained commands, not in the frozen PATTERNS
        table -- exercises accel_cmd saturation clamp on all three axes."""
        from test_entry_point_migration import _run_paired_actions

        helper = _new_helper()
        configs = ("default", "stage0_eval", "stage2_eval", "stage2_train")
        seeds = tuple(range(10))
        n_steps = 40

        for cfg_name, overrides, _doc in _trc.CONFIG_SPECS:
            if cfg_name not in configs:
                continue
            for seed, sign in itertools.product(seeds, (+1.0, -1.0)):
                with self.subTest(config=cfg_name, seed=seed, sign=sign):
                    actions = _fixed_axis_actions(sign, n_steps)
                    _run_paired_actions(
                        helper, overrides, seed, actions,
                        f"{cfg_name} axis={sign} seed={seed}",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
