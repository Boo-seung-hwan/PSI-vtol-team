import unittest

import numpy as np

from system_id.preprocessing import segments


def _fr(n):
    a = np.empty(n, dtype=object)
    a[:] = ""
    return a


class SegmentSplitTest(unittest.TestCase):
    def test_single_run_above_min_duration(self):
        t = np.arange(0, 5.0, 0.05)            # 20 Hz, 5 s
        valid = np.ones(t.size, dtype=bool)
        segs = segments.split_segments("L", "attitude", t, valid, _fr(t.size), 0.05)
        self.assertEqual(len(segs), 1)
        self.assertAlmostEqual(segs[0].duration_s, t[-1], places=3)
        self.assertEqual(segs[0].valid_fraction, 1.0)

    def test_short_run_discarded(self):
        t = np.arange(0, 1.0, 0.05)            # only 1 s < MIN_SEGMENT_S
        valid = np.ones(t.size, dtype=bool)
        segs = segments.split_segments("L", "attitude", t, valid, _fr(t.size), 0.05)
        self.assertEqual(segs, [])

    def test_two_disconnected_runs_not_merged(self):
        t = np.arange(0, 10.0, 0.05)
        valid = np.ones(t.size, dtype=bool)
        valid[(t > 3.0) & (t < 5.0)] = False   # invalid gap
        segs = segments.split_segments("L", "attitude", t, valid, _fr(t.size), 0.05)
        self.assertEqual(len(segs), 2)
        self.assertLess(segs[0].end_s, 3.01)
        self.assertGreater(segs[1].start_s, 4.99)

    def test_internal_time_gap_splits_a_run(self):
        t = np.concatenate([np.arange(0, 3.0, 0.05), np.arange(6.0, 9.0, 0.05)])
        valid = np.ones(t.size, dtype=bool)
        segs = segments.split_segments("L", "rate", t, valid, _fr(t.size), 0.05)
        self.assertEqual(len(segs), 2)

    def test_preceding_gap_rejection_summary(self):
        t = np.arange(0, 10.0, 0.05)
        valid = np.ones(t.size, dtype=bool)
        fr = _fr(t.size)
        gap = (t > 3.0) & (t < 5.0)
        valid[gap] = False
        fr[gap] = "motor_saturated"
        segs = segments.split_segments("L", "attitude", t, valid, fr, 0.05)
        self.assertEqual(len(segs), 2)
        self.assertIn("motor_saturated", segs[1].rejection_summary)
        self.assertAlmostEqual(segs[1].rejection_summary["motor_saturated"], 1.0, places=3)

    def test_table_aggregates(self):
        t = np.arange(0, 8.0, 0.05)
        valid = np.ones(t.size, dtype=bool)
        segs = segments.split_segments("L", "attitude", t, valid, _fr(t.size), 0.05)
        tab = segments.segments_table(segs)
        self.assertEqual(tab["n_segments"], 1)
        self.assertGreater(tab["longest_valid_s"], 7.0)


if __name__ == "__main__":
    unittest.main()
