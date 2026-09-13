"""Task spec section 27 I/J: verify the identified rate/thrust response math
itself reproduces a first-order-plus-delay step response with the supplied
K/tau/T_delay, within a documented tolerance. This tests the ODE in
isolation (``identified_inner_loop.py``) -- ``test_physical_validation
.py::InertiaResponseTest`` separately confirms MuJoCo's rigid-body
integration is consistent with whatever torque this math produces."""

import unittest

import numpy as np

from landing_mujoco.dynamics.delay_buffer import DelayBuffer, steps_from_seconds
from landing_mujoco.dynamics.identified_inner_loop import (
    rate_response_omega_dot,
    thrust_response_step,
)


def simulate_rate_step_response(k, tau, delay_s, physics_dt, step_value, n_steps):
    delay_steps = steps_from_seconds(delay_s, physics_dt)
    buf = DelayBuffer(delay_steps, 0.0)
    omega = 0.0
    trace = []
    for _ in range(n_steps):
        delayed_cmd = buf.push_and_get(step_value)
        omega_dot = rate_response_omega_dot(
            np.array([omega, 0.0, 0.0]),
            np.array([delayed_cmd, 0.0, 0.0]),
            np.array([tau, tau, tau]),
            np.array([k, k, k]),
        )[0]
        omega += omega_dot * physics_dt
        trace.append(omega)
    return np.array(trace)


class RateStepResponseTest(unittest.TestCase):
    def _check_axis(self, k, tau, delay_s):
        physics_dt = 0.002
        step_value = 1.0
        n_steps = int(6 * tau / physics_dt) + steps_from_seconds(delay_s, physics_dt) + 50
        trace = simulate_rate_step_response(k, tau, delay_s, physics_dt, step_value, n_steps)
        t = np.arange(n_steps) * physics_dt

        steady_state = trace[-1]
        self.assertAlmostEqual(steady_state, k * step_value, delta=0.01 * k * step_value + 1e-4)

        # Observed delay: first time index the response exceeds 5% of its
        # final value.
        threshold = 0.05 * steady_state
        idx_above = np.argmax(trace > threshold) if np.any(trace > threshold) else -1
        observed_delay = t[idx_above] if idx_above >= 0 else None
        self.assertIsNotNone(observed_delay)
        self.assertAlmostEqual(observed_delay, delay_s, delta=4 * physics_dt)

        # Rise time 10% -> 90% of steady state, compared to the analytic
        # first-order value tau * ln(9) ~= 2.197 * tau, generous tolerance
        # since this is a discretized Euler integration, not the exact
        # continuous solution.
        v10, v90 = 0.10 * steady_state, 0.90 * steady_state
        idx10 = np.argmax(trace > v10)
        idx90 = np.argmax(trace > v90)
        rise_time = t[idx90] - t[idx10]
        expected_rise_time = tau * np.log(9)
        self.assertAlmostEqual(rise_time, expected_rise_time, delta=0.3 * expected_rise_time + 0.01)

    def test_roll_axis_reference_parameters(self):
        self._check_axis(k=1.0, tau=0.15, delay_s=0.02)

    def test_pitch_axis_reference_parameters(self):
        self._check_axis(k=1.0, tau=0.15, delay_s=0.02)


class ThrustStepResponseTest(unittest.TestCase):
    def test_thrust_first_order_step_response(self):
        physics_dt = 0.002
        tau, k, delay_s = 0.20, 1.0, 0.02
        delay_steps = steps_from_seconds(delay_s, physics_dt)
        buf = DelayBuffer(delay_steps, 9.80665)

        thrust_accel = 9.80665
        step_sp = 14.0  # step up from hover
        n_steps = int(6 * tau / physics_dt) + delay_steps + 50
        trace = []
        for _ in range(n_steps):
            delayed_sp = buf.push_and_get(step_sp)
            thrust_accel = thrust_response_step(
                thrust_accel, delayed_sp, tau, k, physics_dt,
                min_thrust_accel=0.0, max_thrust_accel=100.0,
            )
            trace.append(thrust_accel)
        trace = np.array(trace)
        self.assertAlmostEqual(trace[-1], k * step_sp, delta=0.02 * step_sp)
        self.assertLess(trace[0], trace[-1])


if __name__ == "__main__":
    unittest.main()
