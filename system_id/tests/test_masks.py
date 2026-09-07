import unittest

import numpy as np

from system_id.preprocessing import masks
from system_id.preprocessing.schema import LOOP_VELOCITY
from system_id.preprocessing.timebase import choose_stamp_us
from system_id.tests._synth import build_synthetic_loaded


class MaskTest(unittest.TestCase):
    def setUp(self):
        self.loaded = build_synthetic_loaded(dur_s=20.0)
        self.t_us, _ = choose_stamp_us(self.loaded.topic("vehicle_local_position"))
        self.m = masks.compute_masks(self.loaded, self.t_us, LOOP_VELOCITY)
        self.ts = (self.t_us - self.t_us[0]) / 1e6

    def test_ground_phase_rejected_as_not_armed_or_landed(self):
        early = self.ts < 2.5
        self.assertFalse(self.m.valid_strict[early].any())

    def test_airborne_offboard_phase_has_valid_samples(self):
        mid = (self.ts > 9.5) & (self.ts < 9.9)
        self.assertTrue(self.m.valid_core[mid].any())

    def test_saturation_burst_rejected(self):
        # synthetic burst is t in [8.0, 8.4)
        burst = (self.ts >= 8.05) & (self.ts < 8.35)
        self.assertTrue(self.m.reason_masks["motor_saturated"][burst].all())
        # well clear of the burst -> not flagged (hover thrust, no rail)
        clear = (self.ts >= 9.0) & (self.ts < 9.3)
        self.assertFalse(self.m.reason_masks["motor_saturated"][clear].any())

    def test_dilation_expands_a_single_saturated_sample(self):
        # direct check of the +/- MOTOR_SAT_DILATION_S widening on a fine grid
        src_t = np.arange(0.0, 1.0, 0.005) * 1e6           # 200 Hz source
        flag = np.zeros(src_t.size, dtype=bool)
        flag[100] = True                                    # single spike at t=0.5 s
        query = np.arange(0.0, 1.0, 0.005) * 1e6
        out = masks._dilate_bool(src_t, flag, query, masks.MOTOR_SAT_DILATION_S)
        widened_s = float(np.sum(out)) * 0.005
        self.assertGreater(widened_s, 2 * masks.MOTOR_SAT_DILATION_S * 0.8)
        self.assertLess(widened_s, 2 * masks.MOTOR_SAT_DILATION_S * 1.6)

    def test_accel_sentinel_rejected_on_ground(self):
        onground = self.ts < 2.0
        self.assertTrue(self.m.reason_masks["accel_sentinel"][onground].all())

    def test_first_reason_tracks_flight_base_not_ground_effect(self):
        # first_reason names the first failing BASE reason; "" iff flight_base clear
        self.assertTrue((self.m.first_reason[~self.m.flight_base] != "").all())
        self.assertTrue((self.m.first_reason[self.m.flight_base] == "").all())

    def test_summary_fractions_between_0_and_1(self):
        for r, f in self.m.summary.items():
            self.assertGreaterEqual(f, 0.0)
            self.assertLessEqual(f, 1.0)

    def test_named_constants_exist(self):
        self.assertEqual(masks.MOTOR_SAT_HIGH, 0.98)
        self.assertEqual(masks.MOTOR_SAT_LOW, 0.02)
        self.assertGreater(masks.MOTOR_SAT_DILATION_S, 0.0)
        self.assertEqual(masks.ACCEL_SENTINEL_ABS_MPS2, 30.0)
        self.assertGreater(masks.NEAR_LEVEL_MAX_RAD, 0.0)

    def test_flight_base_excludes_ground_effect_only_via_strict_alias(self):
        # flight_base keeps in-ground-effect samples; valid_strict alias drops them
        self.assertTrue((self.m.valid_strict <= self.m.valid_core).all())
        self.assertTrue((self.m.valid_strict == (self.m.flight_base & ~self.m.ground_effect)).all())
        # if the fixture has GE samples inside flight_base, the two differ
        ge_in_base = self.m.flight_base & self.m.ground_effect
        if ge_in_base.any():
            self.assertTrue(self.m.flight_base.sum() > self.m.valid_strict.sum())

    def test_block_base_helper(self):
        for b in ("attitude", "rate", "velocity_xy"):
            np.testing.assert_array_equal(self.m.block_base(b), self.m.flight_base)
        for b in ("velocity_z", "thrust", "translation"):
            np.testing.assert_array_equal(
                self.m.block_base(b), self.m.flight_base & ~self.m.ground_effect)
        with self.assertRaises(KeyError):
            self.m.block_base("not_a_block")

    def test_combine_block_ands_extra_terms(self):
        extra = np.zeros(self.m.n, dtype=bool)
        extra[: self.m.n // 2] = True
        out = self.m.combine_block("attitude", extra)
        self.assertTrue((out == (self.m.flight_base & extra)).all())


if __name__ == "__main__":
    unittest.main()
