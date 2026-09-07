import math
import unittest

import numpy as np

from system_id.preprocessing import frames


class QuaternionEulerTest(unittest.TestCase):
    def test_identity(self):
        r, p, y = frames.euler_xyz_columns(np.array([1.0, 0.0, 0.0, 0.0]))
        self.assertAlmostEqual(float(r), 0.0)
        self.assertAlmostEqual(float(p), 0.0)
        self.assertAlmostEqual(float(y), 0.0)

    def test_roundtrip_small_angles(self):
        rng = np.random.default_rng(1)
        rpy = rng.uniform(-0.4, 0.4, size=(200, 3))
        cr, sr = np.cos(rpy[:, 0] / 2), np.sin(rpy[:, 0] / 2)
        cp, sp = np.cos(rpy[:, 1] / 2), np.sin(rpy[:, 1] / 2)
        cy, sy = np.cos(rpy[:, 2] / 2), np.sin(rpy[:, 2] / 2)
        q = np.stack([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ], axis=-1)
        out = frames.quat_to_euler_xyz(q)
        np.testing.assert_allclose(out, rpy, atol=1e-6)

    def test_batched_shape(self):
        q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (7, 1))
        self.assertEqual(frames.quat_to_euler_xyz(q).shape, (7, 3))

    def test_degenerate_quat_maps_to_zero_not_nan(self):
        out = frames.quat_to_euler_xyz(np.array([0.0, 0.0, 0.0, 0.0]))
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out, np.zeros(3))

    def test_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            frames.quat_to_euler_xyz(np.array([1.0, 0.0, 0.0]))


class TiltTest(unittest.TestCase):
    def test_level_is_zero(self):
        self.assertAlmostEqual(float(frames.tilt_from_quat(np.array([1.0, 0.0, 0.0, 0.0]))), 0.0)

    def test_30_deg_roll(self):
        a = math.radians(30.0)
        q = np.array([math.cos(a / 2), math.sin(a / 2), 0.0, 0.0])
        self.assertAlmostEqual(float(frames.tilt_from_quat(q)), a, places=6)

    def test_yaw_only_no_tilt(self):
        a = 2.0
        q = np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])
        self.assertAlmostEqual(float(frames.tilt_from_quat(q)), 0.0, places=6)


class MiscTest(unittest.TestCase):
    def test_thrust_magnitude(self):
        m = frames.thrust_vector_magnitude([0.0, 3.0], [0.0, 4.0], [-1.0, 0.0])
        np.testing.assert_allclose(m, [1.0, 5.0])

    def test_wrap_pi(self):
        np.testing.assert_allclose(frames.wrap_pi([0.0, math.pi + 0.1, -math.pi - 0.1]),
                                   [0.0, -math.pi + 0.1, math.pi - 0.1], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
