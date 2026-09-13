"""Task spec section 27: required physical validation (A, B, C, G, H)."""

import unittest

import mujoco
import numpy as np

from landing_mujoco.dynamics.identified_inner_loop import (
    body_torque_from_rate_response,
    force_and_point_torque,
    thrust_response_step,
)
from landing_mujoco.tests._helpers import fresh_state, load_reference_params, make_dynamics

GRAVITY = 9.80665


class FreeFallTest(unittest.TestCase):
    """A. thrust = 0 -> accel_z ~= +g (NED, downward)."""

    def test_free_fall_acceleration(self):
        # Linear drag is disabled for this specific test: with thrust=0 the
        # vehicle falls long enough to build up real velocity, at which
        # point cfg.linear_drag_z (an existing, unmodified control-tuning
        # parameter) legitimately opposes the motion -- this test isolates
        # pure gravity, per the task spec's literal "thrust=0 -> a~=g" ask.
        dyn = make_dynamics(linear_drag_xy=0.0, linear_drag_z=0.0)
        # Start well above the ground -- otherwise the CG-height-vs-leg-
        # standoff geometry (see InertiaResponseTest) would put the legs
        # into the ground plane and this would measure a contact force,
        # not free fall.
        # Start far above the ground: free fall covers a lot of distance
        # once the thrust response settles, and this test must measure
        # acceleration BEFORE touchdown, not after.
        dyn.reset(pos_ned=np.array([0.0, 0.0, -200.0]), vel_ned=np.zeros(3), attitude_rpy=np.zeros(3))
        dyn.actuation.thrust_accel = 0.0

        state = fresh_state(pos=np.array([0.0, 0.0, -200.0]), thrust_accel=0.0)
        rng = np.random.default_rng(0)
        dt = 0.01
        # motor_cutoff=True pins the thrust SETPOINT to zero, but the
        # actuator's identified first-order response (tau_thrust) plus its
        # T_delay buffer (pre-filled with the hover value at reset) both
        # take real time to flush out -- run enough steps (~6*tau_thrust)
        # for the thrust response to actually settle near zero before
        # checking free fall, rather than expecting an instantaneous drop.
        for _ in range(120):
            dyn.step(
                state, v_cmd=np.zeros(3), dt=dt, rng=rng, wind_accel_ned=np.zeros(3),
                target_yaw=0.0, motor_cutoff=True, ground_contact_prev=False,
            )
        self.assertLess(state.thrust_accel, 0.1, "thrust response did not settle near zero")
        np.testing.assert_allclose(state.accel, [0.0, 0.0, GRAVITY], atol=0.1)


class HoverEquilibriumTest(unittest.TestCase):
    """B. roll=pitch=0, collective thrust = m*g -> vertical accel ~= 0.
    Also verifies m*g < T_max and reports the thrust margin."""

    def test_hover_vertical_accel_near_zero(self):
        dyn = make_dynamics()
        dyn.reset(pos_ned=np.array([0.0, 0.0, -3.0]), vel_ned=np.zeros(3), attitude_rpy=np.zeros(3))
        state = fresh_state(pos=np.array([0.0, 0.0, -3.0]), thrust_accel=GRAVITY)

        rng = np.random.default_rng(0)
        dt = 0.01
        dyn.step(
            state, v_cmd=np.zeros(3), dt=dt, rng=rng, wind_accel_ned=np.zeros(3),
            target_yaw=0.0, motor_cutoff=False, ground_contact_prev=False,
        )
        np.testing.assert_allclose(state.accel, [0.0, 0.0, 0.0], atol=5e-2)

    def test_thrust_margin_positive(self):
        params = load_reference_params()
        hover_n = params.hover_thrust_n(GRAVITY)
        self.assertLess(hover_n, params.thrust.max_collective_thrust_n)
        margin = params.thrust_margin(GRAVITY)
        print(f"[thrust margin] T_max/(m*g) = {margin:.3f}")
        self.assertGreater(margin, 1.0)


class ThrustToWeightLimitTest(unittest.TestCase):
    """C. commanded collective thrust cannot exceed T_max."""

    def test_actuation_model_reports_correct_physical_bound(self):
        dyn = make_dynamics()
        params = load_reference_params()
        expected = params.thrust.max_collective_thrust_n / params.mass_properties.mass_kg
        self.assertAlmostEqual(dyn.actuation.max_thrust_accel_physical, expected, places=9)

    def test_thrust_response_step_saturates(self):
        clipped = thrust_response_step(
            thrust_accel=GRAVITY,
            thrust_sp_delayed=1000.0,  # absurdly large demand
            tau_thrust=0.2,
            k_thrust=1.0,
            physics_dt=0.002,
            min_thrust_accel=0.0,
            max_thrust_accel=20.0,  # the enforced physical bound in this call
        )
        self.assertLessEqual(clipped, 20.0)


class CGForceMomentTest(unittest.TestCase):
    """H. tau = (r_i - r_CG) x F, with r_i already expressed relative to CG."""

    def test_cross_product_torque(self):
        r = np.array([0.2606, 0.2606, 0.0])
        f = np.array([0.0, 0.0, -5.0])
        force_out, torque_out = force_and_point_torque(r, f)
        np.testing.assert_allclose(force_out, f)
        np.testing.assert_allclose(torque_out, np.cross(r, f))
        # A downward force at +x,+y should produce a torque with the sign
        # that would roll/pitch the vehicle away from that corner.
        expected = np.cross(r, f)
        self.assertAlmostEqual(torque_out[0], expected[0])
        self.assertAlmostEqual(torque_out[1], expected[1])


class InertiaResponseTest(unittest.TestCase):
    """G. Apply a known RAW body torque (bypassing the identified-response
    layer entirely) and verify alpha ~= J^-1 * tau, confirming the MJCF
    inertia is wired correctly into MuJoCo's own rigid-body equations."""

    def test_raw_torque_matches_inertia(self):
        dyn = make_dynamics()
        # Well clear of the ground -- at pos_ned=[0,0,0] the CG would be
        # exactly at pad height, which (given a nonzero landing-gear
        # standoff) penetrates the ground plane and introduces a spurious
        # contact-constraint force that this test is not meant to exercise.
        dyn.reset(pos_ned=np.array([0.0, 0.0, -3.0]), vel_ned=np.zeros(3), attitude_rpy=np.zeros(3))
        j_diag = np.array(
            [
                dyn.params.mass_properties.ixx,
                dyn.params.mass_properties.iyy,
                dyn.params.mass_properties.izz,
            ]
        )

        for axis in range(3):
            tau = np.zeros(3)
            tau[axis] = 0.5
            dyn.data.xfrc_applied[dyn.body_id, 0:3] = 0.0
            dyn.data.xfrc_applied[dyn.body_id, 3:6] = tau
            mujoco.mj_forward(dyn.model, dyn.data)
            expected_alpha = tau[axis] / j_diag[axis]
            self.assertAlmostEqual(dyn.data.qacc[3 + axis], expected_alpha, delta=1e-5)

    def test_body_torque_helper_matches_zero_omega_case(self):
        j_diag = np.array([0.05, 0.05, 0.06])
        omega = np.zeros(3)
        omega_dot_des = np.array([1.0, -2.0, 0.5])
        tau = body_torque_from_rate_response(j_diag, omega, omega_dot_des)
        np.testing.assert_allclose(tau, j_diag * omega_dot_des)


if __name__ == "__main__":
    unittest.main()
