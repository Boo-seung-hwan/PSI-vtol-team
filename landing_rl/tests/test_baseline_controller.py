"""Controller-level localization tests for the Phase 2 extraction.

These are SECONDARY. The authoritative gate is the OLD-vs-NEW environment
regression in ``test_legacy_regression_contract.py``. This module only makes a
controller-arithmetic regression easier to localize, by comparing
``BaselineController`` outputs directly against the legacy reference methods
(``LandingEnv._pid_velocity`` / ``_scale_action`` / ``_limit_velocity_command``)
and a verbatim copy of the legacy inline descent gate #2, for the specific
regimes named in the Phase 2 brief.

Comparison is exact (``np.array_equal`` / ``==``). No tolerance.
"""

from __future__ import annotations

import unittest

import numpy as np

# Reuse the legacy loader / handles from the primary regression module.
from test_legacy_regression_contract import OldLandingConfig, OldLandingEnv

from landing_rl.controllers import BaselineController


def _legacy_descent_gate2(cfg, v_cmd, control_target_valid, control_xy_error,
                          altitude_agl_before, ground_contact):
    """Verbatim copy of the descent gate #2 block that was inline in
    ``LandingEnv.step`` in mujoco_rl/envs/env_prototype.py."""
    if (
        control_target_valid
        and control_xy_error < cfg.descent_xy_gate_m
        and altitude_agl_before < cfg.descent_start_height_m
        and not ground_contact
    ):
        if altitude_agl_before < 0.10:
            v_cmd[2] = max(v_cmd[2], 0.06)
        else:
            v_cmd[2] = max(v_cmd[2], cfg.landing_descent_bias_mps)
    return v_cmd


class BaselineControllerParityTest(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        # Share the exact same cfg object between the legacy env and the
        # extracted controller so any difference is arithmetic, not config.
        self.old = OldLandingEnv(OldLandingConfig())
        self.cfg = self.old.cfg
        self.ctrl = BaselineController(self.cfg)

    def tearDown(self):
        self.old.close()

    # -- helpers --------------------------------------------------------

    def _legacy_pid(self, target, valid, pos, vel):
        self.old.pos = np.asarray(pos, dtype=np.float64)
        self.old.vel = np.asarray(vel, dtype=np.float64)
        return self.old._pid_velocity(np.asarray(target, dtype=np.float64), valid)

    def _new_pid(self, target, valid, pos, vel):
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)
        altitude_agl_before = max(0.0, float(self.cfg.ground_z_m - pos[2]))
        return self.ctrl.pid_velocity(
            np.asarray(target, dtype=np.float64), valid, pos, vel,
            altitude_agl_before,
        )

    def _assert_pid_equal(self, target, valid, pos, vel, msg):
        old_v = self._legacy_pid(target, valid, pos, vel)
        new_v = self._new_pid(target, valid, pos, vel)
        self.assertEqual(old_v.dtype, new_v.dtype, msg)
        self.assertEqual(old_v.shape, new_v.shape, msg)
        self.assertTrue(
            np.array_equal(old_v, new_v),
            f"{msg}: OLD={old_v!r} NEW={new_v!r}",
        )
        return new_v

    # -- PID XY norm saturation -------------------------------------

    def test_pid_xy_norm_saturation(self):
        # Large lateral error -> raw [vx, vy] norm far above max_pid_xy_mps.
        v = self._assert_pid_equal(
            target=[0.0, 0.0, 0.0], valid=True,
            pos=[-20.0, -20.0, -3.0], vel=[0.0, 0.0, 0.0],
            msg="pid xy norm saturation",
        )
        # legacy uses `norm / (norm + 1e-9)` so the result sits just under the
        # limit by ~1e-9; this descriptive check is not the parity gate.
        self.assertAlmostEqual(
            float(np.linalg.norm(v[:2])), self.cfg.max_pid_xy_mps, places=6
        )

    # -- PID Z saturation -----------------------------------------

    def test_pid_z_saturation(self):
        # Large altitude error, lateral error big enough that descent gate #1
        # does not apply -> vz clipped to max_pid_z_mps.
        v = self._assert_pid_equal(
            target=[0.0, 0.0, 0.0], valid=True,
            pos=[-20.0, -20.0, -3.0], vel=[0.0, 0.0, 0.0],
            msg="pid z saturation",
        )
        self.assertEqual(float(v[2]), self.cfg.max_pid_z_mps)

    # -- descent gate #1 -----------------------------------------

    def test_pid_descent_gate1_raises_vz(self):
        # Inside the xy gate, altitude in (success_altitude_m, descent_start),
        # descending fast enough that raw vz < landing_descent_bias_mps.
        v = self._assert_pid_equal(
            target=[0.0, 0.0, 0.0], valid=True,
            pos=[0.1, 0.1, -0.5], vel=[0.0, 0.0, 2.0],
            msg="descent gate #1 active",
        )
        self.assertEqual(float(v[2]), self.cfg.landing_descent_bias_mps)

    def test_pid_descent_gate1_inactive_below_success_altitude(self):
        # Altitude below success_altitude_m -> gate #1 must NOT fire.
        self._assert_pid_equal(
            target=[0.0, 0.0, 0.0], valid=True,
            pos=[0.1, 0.1, -0.02], vel=[0.0, 0.0, 2.0],
            msg="descent gate #1 inactive (too low)",
        )

    def test_pid_frozen_when_target_invalid(self):
        old_v = self._legacy_pid([0.0, 0.0, 0.0], False, [1.0, 1.0, -2.0], [0.1, 0.0, 0.0])
        new_v = self._new_pid([0.0, 0.0, 0.0], False, [1.0, 1.0, -2.0], [0.1, 0.0, 0.0])
        self.assertTrue(np.array_equal(old_v, new_v))
        self.assertTrue(np.array_equal(new_v, np.zeros(3)))

    # -- residual scaling / clipping ------------------------------

    def test_residual_scaling(self):
        a = np.array([0.5, -0.5, 1.0], dtype=np.float32)
        old_v = self.old._scale_action(a)
        new_v = self.ctrl.scale_action(a)
        self.assertEqual(old_v.dtype, new_v.dtype)
        self.assertTrue(np.array_equal(old_v, new_v), f"OLD={old_v!r} NEW={new_v!r}")
        self.assertTrue(np.array_equal(
            new_v,
            np.array([0.5 * self.cfg.residual_xy_mps,
                      -0.5 * self.cfg.residual_xy_mps,
                      1.0 * self.cfg.residual_z_mps], dtype=np.float64),
        ))

    def test_residual_clipping(self):
        a = np.array([2.0, -3.0, 5.0], dtype=np.float64)
        old_v = self.old._scale_action(a)
        new_v = self.ctrl.scale_action(a)
        self.assertTrue(np.array_equal(old_v, new_v), f"OLD={old_v!r} NEW={new_v!r}")
        self.assertTrue(np.array_equal(
            new_v,
            np.array([self.cfg.residual_xy_mps,
                      -self.cfg.residual_xy_mps,
                      self.cfg.residual_z_mps], dtype=np.float64),
        ))

    # -- final command saturation (combine_and_limit) -------------

    def test_final_xy_command_norm_saturation(self):
        v_pid = np.array([0.7, 0.7, 0.1], dtype=np.float64)
        v_res = np.array([0.25, 0.25, 0.05], dtype=np.float64)
        old_v = self.old._limit_velocity_command(v_pid + v_res)
        new_v = self.ctrl.combine_and_limit(v_pid, v_res)
        self.assertEqual(old_v.dtype, new_v.dtype)
        self.assertTrue(np.array_equal(old_v, new_v), f"OLD={old_v!r} NEW={new_v!r}")
        # legacy `norm / (norm + 1e-9)` -> result sits ~1e-9 under the limit.
        self.assertAlmostEqual(
            float(np.linalg.norm(new_v[:2])), self.cfg.max_cmd_xy_mps, places=6
        )

    def test_final_z_command_saturation(self):
        v_pid = np.array([0.0, 0.0, 0.35], dtype=np.float64)
        v_res = np.array([0.0, 0.0, 0.05], dtype=np.float64)
        old_v = self.old._limit_velocity_command(v_pid + v_res)
        new_v = self.ctrl.combine_and_limit(v_pid, v_res)
        self.assertTrue(np.array_equal(old_v, new_v), f"OLD={old_v!r} NEW={new_v!r}")
        self.assertEqual(float(new_v[2]), self.cfg.max_cmd_z_mps)

    # -- descent gate #2 ----------------------------------------

    def _assert_gate2_equal(self, v_cmd, valid, xy_err, alt, gc, msg):
        old_v = _legacy_descent_gate2(
            self.cfg, np.array(v_cmd, dtype=np.float64), valid, xy_err, alt, gc
        )
        new_v = self.ctrl.apply_descent_gate(
            np.array(v_cmd, dtype=np.float64), valid, xy_err, alt, gc
        )
        self.assertTrue(np.array_equal(old_v, new_v), f"{msg}: OLD={old_v!r} NEW={new_v!r}")
        return new_v

    def test_descent_gate2_active_above_10cm(self):
        v = self._assert_gate2_equal(
            [0.0, 0.0, -0.2], True, 0.2, 0.5, False, "gate #2 active, alt>=0.10"
        )
        self.assertEqual(float(v[2]), self.cfg.landing_descent_bias_mps)

    def test_descent_gate2_active_below_10cm(self):
        v = self._assert_gate2_equal(
            [0.0, 0.0, -0.2], True, 0.2, 0.05, False, "gate #2 active, alt<0.10"
        )
        self.assertEqual(float(v[2]), 0.06)

    def test_descent_gate2_inactive_on_ground_contact(self):
        v = self._assert_gate2_equal(
            [0.0, 0.0, -0.2], True, 0.2, 0.5, True, "gate #2 inactive (ground contact)"
        )
        self.assertEqual(float(v[2]), -0.2)

    def test_descent_gate2_inactive_when_target_invalid(self):
        v = self._assert_gate2_equal(
            [0.0, 0.0, -0.2], False, 0.2, 0.5, False, "gate #2 inactive (invalid target)"
        )
        self.assertEqual(float(v[2]), -0.2)

    def test_descent_gate2_does_not_lower_vz(self):
        # v_cmd[2] already above the bias -> gate must not reduce it.
        v = self._assert_gate2_equal(
            [0.0, 0.0, 0.3], True, 0.2, 0.5, False, "gate #2 keeps larger vz"
        )
        self.assertEqual(float(v[2]), 0.3)


if __name__ == "__main__":
    unittest.main()
