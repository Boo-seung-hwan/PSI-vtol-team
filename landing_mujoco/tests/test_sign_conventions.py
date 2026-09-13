"""Task spec section 27 D/E/F + yaw sign, run through the full dynamics
stack (v_cmd -> InnerLoopCommandModel -> identified response -> MuJoCo)."""

import unittest

import numpy as np

from landing_mujoco.tests._helpers import fresh_state, make_dynamics


def run_constant_command(dyn, v_cmd, target_yaw=0.0, n_steps=40, dt=0.05):
    state = fresh_state()
    dyn.reset(pos_ned=np.zeros(3), vel_ned=np.zeros(3), attitude_rpy=np.zeros(3))
    rng = np.random.default_rng(0)
    rolls, pitches = [], []
    for _ in range(n_steps):
        dyn.step(
            state, v_cmd=np.asarray(v_cmd, dtype=np.float64), dt=dt, rng=rng,
            wind_accel_ned=np.zeros(3), target_yaw=target_yaw,
            motor_cutoff=False, ground_contact_prev=False,
        )
        rolls.append(float(state.attitude[0]))
        pitches.append(float(state.attitude[1]))
    return state, rolls, pitches


class RollSignTest(unittest.TestCase):
    def test_positive_east_velocity_command_produces_positive_roll_and_east_velocity(self):
        dyn = make_dynamics()
        state, rolls, _ = run_constant_command(dyn, v_cmd=[0.0, 0.3, 0.0])
        self.assertGreater(max(rolls), 0.0, "expected transient positive roll for +East demand")
        self.assertGreater(state.vel[1], 0.0, "expected East velocity to increase")


class PitchSignTest(unittest.TestCase):
    def test_positive_north_velocity_command_produces_negative_pitch_and_north_velocity(self):
        dyn = make_dynamics()
        state, _, pitches = run_constant_command(dyn, v_cmd=[0.3, 0.0, 0.0])
        self.assertLess(min(pitches), 0.0, "expected transient negative pitch for +North demand")
        self.assertGreater(state.vel[0], 0.0, "expected North velocity to increase")


class VerticalSignTest(unittest.TestCase):
    def test_positive_z_velocity_command_means_descending(self):
        dyn = make_dynamics()
        state = fresh_state()
        dyn.reset(pos_ned=np.array([0.0, 0.0, -5.0]), vel_ned=np.zeros(3), attitude_rpy=np.zeros(3))
        rng = np.random.default_rng(0)
        pos_z_trace = []
        for _ in range(40):
            dyn.step(
                state, v_cmd=np.array([0.0, 0.0, 0.2]), dt=0.05, rng=rng,
                wind_accel_ned=np.zeros(3), target_yaw=0.0,
                motor_cutoff=False, ground_contact_prev=False,
            )
            pos_z_trace.append(float(state.pos[2]))
        # NED: pos_z increasing (less negative / approaching 0) == descending.
        self.assertGreater(pos_z_trace[-1], pos_z_trace[0])
        self.assertGreater(state.vel[2], 0.0, "+NED z velocity must mean descending")


class YawSignTest(unittest.TestCase):
    def test_yaw_tracks_positive_target_yaw(self):
        dyn = make_dynamics()
        state, _, _ = run_constant_command(dyn, v_cmd=[0.0, 0.0, 0.0], target_yaw=0.5, n_steps=60)
        self.assertGreater(state.attitude[2], 0.0)
        self.assertLess(abs(state.attitude[2] - 0.5), 0.3)


if __name__ == "__main__":
    unittest.main()
