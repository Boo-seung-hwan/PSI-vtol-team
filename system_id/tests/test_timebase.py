import unittest

import numpy as np

from system_id.preprocessing import timebase
from system_id.preprocessing.schema import LOOP_ATTITUDE, LOOP_VELOCITY
from system_id.tests._synth import build_synthetic_loaded, make_topic


class StampChoiceTest(unittest.TestCase):
    def test_measurement_topic_prefers_timestamp_sample(self):
        td = make_topic("vehicle_attitude", {
            "timestamp": np.array([10.0, 20.0]),
            "timestamp_sample": np.array([9.0, 19.0]),
            "q[0]": np.array([1.0, 1.0]),
        })
        t, kind = timebase.choose_stamp_us(td)
        self.assertEqual(kind, "timestamp_sample")
        np.testing.assert_array_equal(t, [9.0, 19.0])

    def test_setpoint_topic_uses_timestamp(self):
        td = make_topic("vehicle_attitude_setpoint", {
            "timestamp": np.array([10.0, 20.0]),
            "q_d[0]": np.array([1.0, 1.0]),
        })
        t, kind = timebase.choose_stamp_us(td)
        self.assertEqual(kind, "timestamp")

    def test_measurement_topic_without_sample_falls_back(self):
        td = make_topic("vehicle_local_position", {
            "timestamp": np.array([10.0, 20.0]),
            "vx": np.array([0.0, 0.0]),
        })
        _, kind = timebase.choose_stamp_us(td)
        self.assertEqual(kind, "timestamp")


class ZohTest(unittest.TestCase):
    def test_hold_and_pre_first(self):
        src_t = np.array([0.0, 100.0, 200.0])
        src_v = np.array([10.0, 20.0, 30.0])
        q = np.array([-50.0, 0.0, 50.0, 100.0, 150.0, 999.0])
        out = timebase.zoh(src_t, src_v, q)
        np.testing.assert_array_equal(out, [10.0, 10.0, 10.0, 20.0, 20.0, 30.0])

    def test_empty_source_returns_nan(self):
        out = timebase.zoh(np.array([]), np.array([]), np.array([1.0, 2.0]))
        self.assertTrue(np.all(np.isnan(out)))


class NearestMatchTest(unittest.TestCase):
    def test_pairs_within_tolerance(self):
        ref = np.array([100.0, 200.0, 300.0])
        other = np.array([100.5, 199.0, 400.0])
        idx = timebase.nearest_match(ref, other, tol_us=5.0)
        np.testing.assert_array_equal(idx, [0, 1, -1])

    def test_sample_aligned_fills_nan_when_unmatched(self):
        ref = np.array([0.0, 1000.0])
        other_t = np.array([1.0])
        other_v = np.array([42.0])
        out = timebase.sample_aligned(ref, other_t, other_v, tol_us=10.0)
        self.assertAlmostEqual(out[0], 42.0)
        self.assertTrue(np.isnan(out[1]))

    def test_setpoint_aligned_to_same_cycle_not_previous(self):
        # local_position_setpoint stamped ~0.5 us after the local_position sample
        lp_t = np.array([0.0, 100000.0, 200000.0])
        lps_t = lp_t + 0.5
        lps_v = np.array([1.0, 2.0, 3.0])
        out = timebase.sample_aligned(lp_t, lps_t, lps_v, tol_us=15000.0)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


class LoopTimebaseTest(unittest.TestCase):
    def test_velocity_and_attitude_have_distinct_rates(self):
        loaded = build_synthetic_loaded()
        t0 = timebase.log_t0_us(loaded)
        vel = timebase.build_loop_timebase(loaded, LOOP_VELOCITY, t0)
        att = timebase.build_loop_timebase(loaded, LOOP_ATTITUDE, t0)
        self.assertAlmostEqual(vel.effective_hz, 10.0, delta=0.5)
        self.assertAlmostEqual(att.effective_hz, 20.0, delta=0.5)
        self.assertEqual(vel.nonmonotonic_count, 0)
        self.assertNotEqual(vel.n, att.n)  # pipeline keeps separate grids

    def test_t_starts_near_zero(self):
        loaded = build_synthetic_loaded()
        t0 = timebase.log_t0_us(loaded)
        vel = timebase.build_loop_timebase(loaded, LOOP_VELOCITY, t0)
        self.assertLess(abs(vel.t_s[0]), 0.2)


if __name__ == "__main__":
    unittest.main()
