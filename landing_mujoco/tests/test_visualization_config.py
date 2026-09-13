import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from landing_mujoco.configs.visualization_config import (
    REPO_ROOT,
    VisualizationConfig,
    load_visualization_config,
)
from landing_mujoco.coordinates.transforms import frd_body_offset_to_mujoco_body, rotation_body_to_world_ned
from landing_mujoco.tests._helpers import load_reference_params
from landing_mujoco.visualization.x500_shell import load_native_assembly, resolve_alignment

X500_YAML = REPO_ROOT / "landing_mujoco" / "configs" / "x500_visualization.yaml"


class VisualizationConfigLoadTest(unittest.TestCase):
    def test_repository_yaml_loads(self):
        cfg = load_visualization_config(X500_YAML)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.model, "x500")
        self.assertEqual(cfg.uniform_scale, "auto")
        self.assertEqual(cfg.source_wheelbase_m, "auto")
        self.assertEqual(float(cfg.target_wheelbase_m), 0.737)
        self.assertEqual(cfg.translation_body_m, (0.0, 0.0, 0.0))
        self.assertEqual(cfg.rotation_rpy_rad, (0.0, 0.0, 0.0))
        self.assertTrue(cfg.assets_dir.is_absolute())

    def test_invalid_values_rejected(self):
        with self.assertRaises(ValueError):
            VisualizationConfig(enabled=True, uniform_scale="huge")
        with self.assertRaises(ValueError):
            VisualizationConfig(enabled=True, uniform_scale=-1.0)
        with self.assertRaises(ValueError):
            VisualizationConfig(enabled=True, target_wheelbase_m="auto")
        with self.assertRaises(ValueError):
            VisualizationConfig(enabled=True, model="iris")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.yaml"
            p.write_text("something_else: {}\n")
            with self.assertRaises(ValueError):
                load_visualization_config(p)


class AlignmentResolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = load_reference_params()
        cls.cfg = load_visualization_config(X500_YAML)
        cls.assembly = load_native_assembly(cls.cfg.assets_dir)

    def test_auto_scale_is_target_over_source(self):
        a = resolve_alignment(self.cfg, self.assembly, self.params)
        source = 2.0 * np.hypot(0.174, 0.174)
        self.assertAlmostEqual(a.source_wheelbase_m, source, places=12)
        self.assertEqual(a.target_wheelbase_m, 0.737)
        self.assertAlmostEqual(a.uniform_scale, 0.737 / source, places=12)
        self.assertAlmostEqual(a.uniform_scale, 1.4975, places=3)

    def test_explicit_scale(self):
        a = resolve_alignment(replace(self.cfg, uniform_scale=1.0), self.assembly, self.params)
        self.assertEqual(a.uniform_scale, 1.0)
        self.assertEqual(a.scale_mode, "explicit")

    def test_target_from_physical_params(self):
        a = resolve_alignment(replace(self.cfg, target_wheelbase_m="from_physical_params"),
                              self.assembly, self.params)
        self.assertEqual(a.target_wheelbase_m, self.params.geometry.wheelbase_m)
        self.assertIn("physical_params", a.target_wheelbase_origin)

    def test_target_change_is_config_only(self):
        a1 = resolve_alignment(self.cfg, self.assembly, self.params)
        a2 = resolve_alignment(replace(self.cfg, target_wheelbase_m=0.600), self.assembly, self.params)
        self.assertAlmostEqual(a2.uniform_scale / a1.uniform_scale, 0.600 / 0.737, places=12)

    def test_frd_offset_converted_through_coordinate_adapter(self):
        # 0.2 m UP in FRD is z = -0.2; MuJoCo body is Z-up.
        t, r = frd_body_offset_to_mujoco_body((0.1, 0.05, -0.2), (0.0, 0.0, np.pi / 2))
        np.testing.assert_allclose(t, [0.1, -0.05, 0.2], atol=1e-15)
        # +yaw in FRD (nose right) is a rotation about MuJoCo -Z.
        np.testing.assert_allclose(r, rotation_body_to_world_ned(0.0, 0.0, -np.pi / 2), atol=1e-15)


if __name__ == "__main__":
    unittest.main()
