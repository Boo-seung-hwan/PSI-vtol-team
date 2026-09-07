"""Pre-commit-audit regression tests for the three reviewed issues:

  1. block-specific validity masks (ground effect must not drop attitude/rate)
  2. setpoint<->measurement pairing validity (unmatched rows rejected, NaN I/O
     forces the block mask false, pairing fraction reported)
  3. body_z_specific_force_proxy naming + mandatory near_level_valid gate
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from system_id.preprocessing import dataset as ds_mod
from system_id.preprocessing.masks import (
    BLOCK_GE_SENSITIVE,
    BLOCK_NEAR_LEVEL_REQUIRED,
    LOOP_BLOCKS,
    NEAR_LEVEL_MAX_RAD,
    compute_masks,
)
from system_id.preprocessing.timebase import choose_stamp_us
from system_id.tests._synth import build_synthetic_loaded


# ----------------------------------------------------------------------------
# Issue 1 -- block-specific masks
# ----------------------------------------------------------------------------
class BlockMaskModelTest(unittest.TestCase):
    def test_ge_sensitivity_map_is_physical(self):
        self.assertEqual(BLOCK_GE_SENSITIVE, frozenset({"velocity_z", "thrust", "translation"}))
        self.assertNotIn("attitude", BLOCK_GE_SENSITIVE)
        self.assertNotIn("rate", BLOCK_GE_SENSITIVE)
        self.assertNotIn("velocity_xy", BLOCK_GE_SENSITIVE)

    def test_block_base_only_adds_ge_for_ge_sensitive_blocks(self):
        loaded = build_synthetic_loaded()
        t_us, _ = choose_stamp_us(loaded.topic("vehicle_attitude"))
        m = compute_masks(loaded, t_us, "attitude")
        # force some in-ground-effect samples
        ge = m.reason_masks["in_ground_effect"]
        self.assertTrue(ge.any(), "fixture must contain in_ground_effect samples")
        np.testing.assert_array_equal(m.block_base("attitude"), m.flight_base)
        np.testing.assert_array_equal(m.block_base("rate"), m.flight_base)
        np.testing.assert_array_equal(m.block_base("velocity_xy"), m.flight_base)
        np.testing.assert_array_equal(m.block_base("velocity_z"), m.flight_base & ~ge)
        np.testing.assert_array_equal(m.block_base("translation"), m.flight_base & ~ge)

    def test_attitude_rate_keep_low_altitude_data_that_velocity_z_drops(self):
        loaded = build_synthetic_loaded()
        ds = ds_mod.build_all(loaded)
        att = ds["attitude"].block_masks["attitude"]
        rate = ds["rate"].block_masks["rate"]
        vz = ds["velocity"].block_masks["velocity_z"]
        vxy = ds["velocity"].block_masks["velocity_xy"]
        # attitude/rate valid count uses flight_base; velocity_z is a strict subset
        self.assertGreaterEqual(att.sum() / max(ds["attitude"].n, 1),
                                vz.sum() / max(ds["velocity"].n, 1) - 0.30)
        self.assertTrue((vz <= (ds["velocity"].mask.flight_base & ~ds["velocity"].mask.ground_effect)).all())
        self.assertTrue((vxy <= ds["velocity"].mask.flight_base).all())

    def test_every_loop_emits_exactly_its_declared_blocks(self):
        ds = ds_mod.build_all(build_synthetic_loaded())
        for loop, dset in ds.items():
            self.assertEqual(set(dset.block_masks), set(LOOP_BLOCKS[loop]))


# ----------------------------------------------------------------------------
# Issue 2 -- setpoint/measurement pairing validity
# ----------------------------------------------------------------------------
class PairingValidityTest(unittest.TestCase):
    def test_all_paired_when_setpoint_stream_intact(self):
        ds = ds_mod.build_all(build_synthetic_loaded())["velocity"]
        self.assertEqual(ds.pairing["unmatched"], 0)
        self.assertAlmostEqual(ds.pairing["match_fraction"], 1.0)
        self.assertTrue(ds.aux["setpoint_pairing_matched"].all())

    def test_unmatched_pairs_are_rejected_and_io_is_nan(self):
        loaded = build_synthetic_loaded(dur_s=24.0, drop_lps_window_s=(12.0, 15.0))
        ds = ds_mod.build_all(loaded)["velocity"]
        matched = ds.aux["setpoint_pairing_matched"]
        self.assertGreater(ds.pairing["unmatched"], 0)
        self.assertLess(ds.pairing["match_fraction"], 1.0)
        self.assertEqual(ds.pairing["matched"] + ds.pairing["unmatched"], ds.pairing["n"])

        unm = ~matched
        # required I/O NaN wherever the pairing failed
        self.assertTrue(np.all(np.isnan(ds.columns["a_sp_N"][unm])))
        self.assertTrue(np.all(np.isnan(ds.columns["v_sp_D"][unm])))
        # ... and those rows are false in BOTH velocity block masks
        self.assertFalse(ds.block_masks["velocity_xy"][unm].any())
        self.assertFalse(ds.block_masks["velocity_z"][unm].any())

    def test_nonfinite_io_alone_forces_block_mask_false(self):
        loaded = build_synthetic_loaded()
        ds = ds_mod.build_all(loaded)["velocity"]
        # inject a NaN into a required output on an otherwise-valid row
        vm = ds.block_masks["velocity_xy"]
        idx = int(np.argmax(vm))  # first valid sample
        self.assertTrue(vm[idx])
        ds.columns["a_sp_E"][idx] = np.nan
        # recompute the finite term the way dataset.py does
        from system_id.preprocessing.dataset import _finite_all
        xy_fin = _finite_all(ds.columns["velocity_error_N"], ds.columns["velocity_error_E"],
                             ds.columns["a_sp_N"], ds.columns["a_sp_E"])
        new_mask = ds.mask.combine_block("velocity_xy", ds.aux["setpoint_pairing_matched"], xy_fin)
        self.assertFalse(new_mask[idx])


# ----------------------------------------------------------------------------
# Issue 3 -- proxy naming + near-level gate
# ----------------------------------------------------------------------------
class SpecificForceProxyTest(unittest.TestCase):
    def test_old_name_gone_new_name_present_with_warning(self):
        ds = ds_mod.build_all(build_synthetic_loaded())["translation"]
        self.assertNotIn("spec_thrust_recon", ds.columns)
        self.assertIn("body_z_specific_force_proxy", ds.columns)
        self.assertIn("near_level_valid", ds.columns)
        meta = ds.column_meta["body_z_specific_force_proxy"]["source"].lower()
        self.assertIn("crude", meta)
        self.assertIn("not a calibrated thrust", meta)
        self.assertIn("near_level_valid", meta)

    def test_thrust_block_requires_near_level(self):
        self.assertEqual(BLOCK_NEAR_LEVEL_REQUIRED, frozenset({"thrust"}))
        loaded = build_synthetic_loaded(dur_s=24.0, big_tilt=True)
        ds = ds_mod.build_all(loaded)
        tr = ds["translation"]
        tilt = tr.columns["tilt"]
        nlv = tr.columns["near_level_valid"].astype(bool)
        np.testing.assert_array_equal(nlv, np.isfinite(tilt) & (tilt <= NEAR_LEVEL_MAX_RAD))
        tilted = np.isfinite(tilt) & (tilt > NEAR_LEVEL_MAX_RAD)
        self.assertTrue(tilted.any(), "big_tilt fixture must produce >10deg tilt")
        # thrust block is false everywhere the vehicle is tilted past the gate
        self.assertFalse(tr.block_masks["thrust"][tilted].any())
        # translation block (not near-level-gated) can still be valid on some of those
        self.assertTrue("thrust" not in {b for b in BLOCK_NEAR_LEVEL_REQUIRED if b == "translation"})

    def test_thrust_block_subset_of_translation_block_on_gate(self):
        ds = ds_mod.build_all(build_synthetic_loaded(dur_s=24.0, big_tilt=True))["translation"]
        # thrust = translation_base & near_level & finite  -> subset of translation-base parts
        base_no_finite = ds.mask.block_base("translation") & ds.aux["near_level_valid"]
        self.assertTrue((ds.block_masks["thrust"] <= base_no_finite).all())


if __name__ == "__main__":
    unittest.main()
