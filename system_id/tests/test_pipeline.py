import json
import os
import tempfile
import unittest

import numpy as np

from system_id.preprocessing import dataset as ds_mod
from system_id.preprocessing.pipeline import (
    DEFAULT_SAMPLE_LOGS,
    preprocess_log,
    run_batch,
)
from system_id.preprocessing.schema import ALL_LOOPS, LOOP_COLUMNS
from system_id.tests._synth import build_synthetic_loaded

_SAMPLES_PRESENT = [p for p in DEFAULT_SAMPLE_LOGS if os.path.exists(p)]


class SyntheticPipelineTest(unittest.TestCase):
    """End-to-end on a synthetic in-memory log (no .ulg dependency)."""

    def setUp(self):
        self.loaded = build_synthetic_loaded(dur_s=24.0)
        self.datasets = ds_mod.build_all(self.loaded)

    def test_all_loops_built_with_declared_columns(self):
        self.assertEqual(set(self.datasets), set(ALL_LOOPS))
        for loop, dset in self.datasets.items():
            declared = {c.name for c in LOOP_COLUMNS[loop]}
            self.assertEqual(set(dset.columns), declared, f"column mismatch in {loop}")
            for name, arr in dset.columns.items():
                self.assertEqual(len(arr), dset.n, f"{loop}.{name} length")
            self.assertIn("t", dset.columns)
            self.assertTrue(np.all(np.diff(dset.columns["t"]) > 0))

    def test_frames_kept_separate(self):
        tr = self.datasets["translation"]
        # body specific force z ~ -9.8 in air; NED accel_D ~ 0 -> not the same column
        m = tr.mask.valid_strict
        if m.any():
            self.assertLess(np.nanmean(tr.columns["acc_b_z"][m]), -5.0)
            self.assertLess(abs(np.nanmean(tr.columns["acc_D"][m])), 3.0)

    def test_velocity_setpoint_column_is_local_position_setpoint(self):
        meta = self.datasets["velocity"].column_meta["v_sp_N"]
        self.assertIn("vehicle_local_position_setpoint", meta["source"])
        self.assertNotIn("trajectory_setpoint", meta["source"])

    def test_masks_have_all_reasons_and_block_masks(self):
        from system_id.preprocessing.masks import LOOP_BLOCKS, REJECTION_REASONS
        for loop, dset in self.datasets.items():
            self.assertEqual(set(dset.mask.reason_masks), set(REJECTION_REASONS))
            self.assertEqual(set(dset.block_masks), set(LOOP_BLOCKS[loop]))
            for b, m in dset.block_masks.items():
                self.assertEqual(len(m), dset.n)
                self.assertEqual(m.dtype, np.bool_)
        self.assertTrue(self.datasets["velocity"].mask.flight_base.any())

    def test_synthetic_roll_doublet_gives_a_usable_attitude_segment(self):
        from system_id.preprocessing.pipeline import _segments_for
        s = _segments_for("<synthetic>", self.datasets)
        self.assertIn("attitude", s)
        self.assertTrue(any(seg.duration_s >= 1.5 for seg in s["attitude"]))


class SyntheticWriteTest(unittest.TestCase):
    def test_manual_save_via_internal_api(self):
        from system_id.preprocessing.pipeline import _save_outputs, _segments_for
        from system_id.preprocessing.provenance import (
            DATASET_ROLE_VALIDATION, SOURCE_PROJECT_SAMPLE, build_provenance)
        from system_id.preprocessing.report import per_log_report

        loaded = build_synthetic_loaded()
        datasets = ds_mod.build_all(loaded)
        segs = _segments_for("<synthetic>", datasets)
        prov = build_provenance(loaded, SOURCE_PROJECT_SAMPLE, DATASET_ROLE_VALIDATION).to_dict()
        rep = per_log_report(loaded, datasets, segs, SOURCE_PROJECT_SAMPLE, DATASET_ROLE_VALIDATION)
        with tempfile.TemporaryDirectory() as tmp:
            _save_outputs(tmp, datasets, segs, prov, rep)
            for fn in ("columns.json", "segments.json", "provenance.json", "report.json", "report.txt"):
                self.assertTrue(os.path.exists(os.path.join(tmp, fn)))
            from system_id.preprocessing.masks import LOOP_BLOCKS
            for loop in ALL_LOOPS:
                npz = os.path.join(tmp, f"{loop}.npz")
                self.assertTrue(os.path.exists(npz))
                z = np.load(npz)
                self.assertIn("t", z.files)
                self.assertIn("_flight_base", z.files)
                self.assertIn("_ground_effect", z.files)
                for b in LOOP_BLOCKS[loop]:
                    self.assertIn(f"_block_{b}", z.files)
            with open(os.path.join(tmp, "segments.json")) as fh:
                seg = json.load(fh)
            self.assertIn("attitude", seg)      # keyed by BLOCK, not loop
            self.assertIn("velocity_z", seg)
            with open(os.path.join(tmp, "provenance.json")) as fh:
                p = json.load(fh)
            self.assertEqual(p["source_project"], "OTHER_PROJECT_SAMPLE")
            self.assertEqual(p["dataset_role"], "PIPELINE_VALIDATION_ONLY")


@unittest.skipUnless(_SAMPLES_PRESENT, "no sample ULogs present")
class SampleUlogIntegrationTest(unittest.TestCase):
    def test_all_present_samples_process_without_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = run_batch(_SAMPLES_PRESENT, out_root=tmp, write=True)
            self.assertEqual(len(results), len(_SAMPLES_PRESENT))
            for res in results:
                self.assertEqual(res.provenance["source_project"], "OTHER_PROJECT_SAMPLE")
                self.assertEqual(res.provenance["dataset_role"], "PIPELINE_VALIDATION_ONLY")
                self.assertEqual(set(res.datasets), set(ALL_LOOPS))
                # schema completeness
                for loop, dset in res.datasets.items():
                    self.assertEqual(
                        set(dset.columns), {c.name for c in LOOP_COLUMNS[loop]}
                    )
                # report is a gate, not a claim
                self.assertIn("does NOT declare", res.report["disclaimer"])
                # output files exist
                self.assertTrue(os.path.isdir(res.out_dir))
                self.assertTrue(os.path.exists(os.path.join(res.out_dir, "provenance.json")))

    def test_at_least_one_sample_has_usable_velocity_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = run_batch(_SAMPLES_PRESENT, out_root=tmp, write=False)
        usable = [
            r for r in results
            if r.report["blocks"]["velocity_xy"]["valid_s"] > 5.0
        ]
        self.assertTrue(usable, "expected >=1 sample log with >5 s usable velocity_xy data")

    def test_setpoint_pairing_fraction_reported_per_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = run_batch(_SAMPLES_PRESENT, out_root=tmp, write=False)
        for r in results:
            p = r.report["loops"]["velocity"]["setpoint_pairing"]
            self.assertEqual(p["matched"] + p["unmatched"], p["n"])
            self.assertGreaterEqual(p["match_fraction"], 0.0)
            self.assertLessEqual(p["match_fraction"], 1.0)

    def test_ground_effect_does_not_shrink_attitude_or_rate_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = run_batch(_SAMPLES_PRESENT, out_root=tmp, write=False)
        for r in results:
            b = r.report["blocks"]
            # attitude/rate use flight_base (no GE exclusion) -> valid >= GE-excluded blocks' fraction
            self.assertFalse(b["attitude"]["ge_sensitive"])
            self.assertFalse(b["rate"]["ge_sensitive"])
            self.assertTrue(b["velocity_z"]["ge_sensitive"])
            self.assertTrue(b["translation"]["ge_sensitive"])


if __name__ == "__main__":
    unittest.main()
