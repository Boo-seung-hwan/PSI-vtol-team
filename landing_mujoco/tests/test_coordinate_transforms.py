import unittest

import numpy as np

from landing_mujoco.coordinates import transforms as T


class CoordinateTransformTest(unittest.TestCase):
    def test_position_round_trip(self):
        p = np.array([1.0, -2.5, 3.7])
        p_mj = T.ned_position_to_mujoco(p)
        p_back = T.mujoco_position_to_ned(p_mj)
        np.testing.assert_allclose(p_back, p, atol=1e-12)

    def test_velocity_round_trip(self):
        v = np.array([0.3, -0.1, 0.9])
        v_mj = T.ned_velocity_to_mujoco(v)
        v_back = T.mujoco_velocity_to_ned(v_mj)
        np.testing.assert_allclose(v_back, v, atol=1e-12)

    def test_body_vector_round_trip(self):
        r = np.array([0.26, -0.26, 0.12])
        r_mj = T.frd_vector_to_mujoco_body(r)
        r_back = T.mujoco_body_vector_to_frd(r_mj)
        np.testing.assert_allclose(r_back, r, atol=1e-12)

    def test_gravity_sign(self):
        # NED gravity is [0, 0, +g] (down); MuJoCo world is Z-up, so the
        # same physical vector must be [0, 0, -g] in MuJoCo coordinates.
        g = 9.80665
        g_mj = T.ned_velocity_to_mujoco(np.array([0.0, 0.0, g]))
        np.testing.assert_allclose(g_mj, [0.0, 0.0, -g], atol=1e-12)

    def test_euler_quaternion_round_trip(self):
        cases = [
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.0, 0.15, 0.0),
            (0.0, 0.0, 0.3),
            (0.2, -0.1, 0.4),
            (-0.3, 0.25, -0.5),
        ]
        for roll, pitch, yaw in cases:
            quat = T.policy_euler_to_mujoco_quaternion(roll, pitch, yaw)
            r2, p2, y2 = T.mujoco_quaternion_to_policy_euler(quat)
            np.testing.assert_allclose([roll, pitch, yaw], [r2, p2, y2], atol=1e-9)

    def test_quaternion_is_unit(self):
        quat = T.policy_euler_to_mujoco_quaternion(0.2, -0.3, 0.6)
        self.assertAlmostEqual(float(np.linalg.norm(quat)), 1.0, places=9)

    def test_reflect_is_involutory(self):
        np.testing.assert_allclose(T.REFLECT @ T.REFLECT, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(T.REFLECT)), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
