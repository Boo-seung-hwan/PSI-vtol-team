"""Integrity, provenance and landing-gear partition of the derived x500 visual
bundle, and its MuJoCo reconstruction at NATIVE component coordinates (no SDF
pose, no display scale). Uses only the derived bundle -- never a vendor copy.
"""

import hashlib
import json
import unittest
from pathlib import Path

import mujoco
import numpy as np

from landing_mujoco.configs.visualization_config import REPO_ROOT

BUNDLE = REPO_ROOT / "landing_mujoco" / "assets" / "x500_visual"
UPSTREAM_REPOSITORY = "https://github.com/PX4/PX4-gazebo-models.git"
UPSTREAM_COMMIT = "d754381a1cecdd7f17050acd72bf5bf1327bced6"
PX4_AUTOPILOT_COMMIT = "85df8c2281c2466b30a121b22b0bf33dc69bcfe4"
UPSTREAM_LICENSE_SHA256 = "cee4ef94e73cd38fb2886f5d7e7a04d6488f54b1e70cd8b1b29d74f38bfdba5b"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_obj(path):
    v, vt, vn, f = [], [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                v.append(line[2:])
            elif line.startswith("vt "):
                vt.append(line[3:])
            elif line.startswith("vn "):
                vn.append(line[3:])
            elif line.startswith("f "):
                f.append(line[2:].replace("/", " ").split())
    v = np.array(" ".join(v).split(), dtype=np.float64).reshape(-1, 3)
    vt = np.array(" ".join(vt).split(), dtype=np.float64).reshape(-1, 2) if vt else None
    vn = np.array(" ".join(vn).split(), dtype=np.float64).reshape(-1, 3) if vn else None
    idx = np.array(f, dtype=np.int64).reshape(len(f), 3, -1) - 1
    out = {"pos": v[idx[:, :, 0]], "n_faces": len(f)}
    out["uv"] = vt[idx[:, :, 1]] if vt is not None else None
    out["normal"] = vn[idx[:, :, 2]] if vn is not None else None
    return out


class BundleIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((BUNDLE / "conversion_report.json").read_text())

    def test_required_files_present(self):
        for name in ["LICENSE", "THIRD_PARTY_NOTICES.md", "README.md", "conversion_report.json",
                     "x500_native_assembly.json"]:
            self.assertTrue((BUNDLE / name).is_file(), name)
        self.assertTrue((BUNDLE / "meshes").is_dir())
        self.assertTrue((BUNDLE / "textures" / "cf.png").is_file())

    def test_license_is_verbatim_upstream_license(self):
        self.assertEqual(sha256(BUNDLE / "LICENSE"), UPSTREAM_LICENSE_SHA256)
        self.assertEqual(self.report["license_copy"]["source"], "models/x500_base/LICENSE")
        self.assertTrue(self.report["license_copy"]["byte_identical_to_source"])
        self.assertIn("BSD 3-Clause", (BUNDLE / "LICENSE").read_text())

    def test_upstream_provenance(self):
        r = self.report
        self.assertEqual(r["upstream_source"]["repository"], UPSTREAM_REPOSITORY)
        self.assertEqual(r["upstream_source"]["commit"], UPSTREAM_COMMIT)
        self.assertEqual(r["upstream_source"]["referenced_by"]["commit"], PX4_AUTOPILOT_COMMIT)
        self.assertEqual(r["upstream_source"]["referenced_by"]["submodule_path"], "Tools/simulation/gz")
        self.assertTrue(r["pinned_source_verified"])
        self.assertTrue(r["source_verification"]["all_content_pins_match"])
        git = r["source_verification"]["git"]
        if git["available"]:
            self.assertEqual(git["head"], UPSTREAM_COMMIT)
            self.assertTrue(git["model_paths_clean"])
            self.assertTrue(git["model_paths_identical_to_pinned_commit"])
        manifest = json.loads((BUNDLE / "x500_native_assembly.json").read_text())
        self.assertEqual(manifest["upstream"], {"repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT})
        # every recorded source path is upstream-relative; nothing points at a local vendor copy
        sources = [c["source_file"] for c in r["components"].values()] + \
                  [c["source_file"] for c in r["stl_copies"].values()] + [r["textures"]["cf.png"]["source"]]
        self.assertTrue(all(s.startswith("models/x500_base/") for s in sources), sources)
        manifest_text = (BUNDLE / "x500_native_assembly.json").read_text()
        self.assertNotIn("vendor", manifest_text)
        self.assertNotIn("/home/", manifest_text)

    def test_report_gates_precision_and_tools(self):
        r = self.report
        self.assertTrue(r["gates_passed"])
        self.assertEqual(r["gate_failures"], [])
        self.assertEqual(r["formatting_precision"]["xyz_decimal_places"], 8)
        self.assertEqual(r["formatting_precision"]["uv_decimal_places"], 8)
        self.assertEqual(r["formatting_precision"]["normal_decimal_places"], 8)
        for pkg in ("python", "trimesh", "pycollada", "Pillow"):
            self.assertIn(pkg, r["conversion_environment"])

    def test_output_hashes_match_files(self):
        r = self.report
        for name, c in r["components"].items():
            self.assertEqual(sha256(BUNDLE / c["output_obj"]), c["output_obj_sha256"], name)
        for name, c in r["stl_copies"].items():
            self.assertEqual(sha256(BUNDLE / c["output"]), c["output_sha256"], name)
            self.assertEqual(c["source_sha256"], c["output_sha256"], name)
        tex = r["textures"]["cf.png"]
        self.assertEqual(sha256(BUNDLE / "textures" / "cf.png"), tex["output_sha256"])
        self.assertEqual(tex["source_sha256"], tex["output_sha256"])
        self.assertEqual(sha256(BUNDLE / "x500_native_assembly.json"), r["native_assembly_manifest"]["sha256"])
        self.assertEqual(sha256(BUNDLE / "LICENSE"), r["license_copy"]["sha256"])
        # every file in meshes/ and textures/ is accounted for in the report (no stale derived files)
        recorded = {c["output_obj"] for c in r["components"].values()} | \
                   {c["output"] for c in r["stl_copies"].values()} | {"textures/cf.png"}
        on_disk = {str(p.relative_to(BUNDLE)) for d in ("meshes", "textures") for p in (BUNDLE / d).iterdir()}
        self.assertEqual(on_disk, recorded)

    def test_landing_gear_partition(self):
        p = self.report["landing_gear_partition"]
        self.assertFalse(p["geometry_modified"])
        self.assertIn("physical contact geoms", p["decision"])
        for name, comp in p["components"].items():
            self.assertEqual(comp["exported_triangles"] + comp["excluded_landing_gear_triangles"],
                             comp["source_triangles"], name)
            self.assertEqual(sum(pc["triangles"] for pc in comp["pieces"]), comp["source_triangles"], name)
        self.assertEqual(sorted(self.report["excluded_components"]),
                         ["frame_landing_foam", "frame_landing_plastic", "frame_landing_rubber"])
        cf = p["components"]["frame_carbon_fiber"]
        gear = [pc for pc in cf["pieces"] if pc["class"] == "landing_gear"]
        self.assertEqual(len(gear), 4)                     # 2 leg tubes + 2 skid tubes
        self.assertEqual(sum(pc["triangles"] for pc in gear), 1376)
        m = p["z_rule_margins_m"]
        self.assertLess(m["highest_z_rule_gear_piece_min_z"], m["threshold"])
        self.assertGreater(m["lowest_upper_structure_piece_min_z"], m["threshold"])
        manifest = json.loads((BUNDLE / "x500_native_assembly.json").read_text())
        self.assertFalse(any(v["source"].get("geometry_id", "").startswith("Landing")
                             for v in manifest["meshes"].values()))
        self.assertIn("frame_carbon_fiber_upper", manifest["meshes"])

    def test_dropped_primitives_recorded(self):
        dropped = self.report["dropped_source_primitives"]
        self.assertEqual(len(dropped), 1)
        d = dropped[0]
        self.assertEqual(d["source_component"], "FMUK66-mesh")
        self.assertEqual(d["dropped_primitives"], "48 line segments")
        self.assertEqual(d["expected_visual_impact"], "none")
        self.assertEqual(d["physics_impact"], "none")

    def test_known_uv_tolerances_only_on_untextured_components(self):
        for tol in self.report["known_conversion_tolerances"]:
            comp = self.report["components"][tol["component"]]
            self.assertNotIn("texture", comp["source_material"]["diffuse"])
            self.assertFalse(tol["affects_exported_obj"])
        cf = self.report["components"]["frame_carbon_fiber_upper"]
        self.assertIn("texture", cf["source_material"]["diffuse"])
        self.assertEqual(cf["trimesh_cross_check"]["max_abs_uv_vs_trimesh"], 0.0)
        self.assertTrue(cf["trimesh_cross_check"]["texture_pixels_identical_to_bundle_file"])

    def test_manifest_has_no_physics(self):
        text = (BUNDLE / "x500_native_assembly.json").read_text().lower()
        for key in ('"mass"', '"inertia"', '"inertial"', '"collision"', '"motorconstant"', '"thrust"', '"plugin"'):
            self.assertNotIn(key, text)
        self.assertNotIn("oakd", text)


class NativeMuJoCoReconstructionTest(unittest.TestCase):
    """Every converted component, loaded into MuJoCo at its native coordinates,
    reproduces the exported OBJ per face corner (positions, UVs, normals) and
    the recorded source bounding boxes."""

    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((BUNDLE / "conversion_report.json").read_text())
        assets, geoms = [], []
        for name, c in cls.report["components"].items():
            assets.append(f'<mesh name="{name}" file="{BUNDLE / c["output_obj"]}"/>')
            geoms.append(f'<geom name="{name}" type="mesh" mesh="{name}" contype="0" conaffinity="0"/>')
        cls.model = mujoco.MjModel.from_xml_string(
            f"<mujoco><asset>{''.join(assets)}</asset><worldbody>{''.join(geoms)}</worldbody></mujoco>")
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def _corners(self, name):
        m, d = self.model, self.data
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        mid = m.geom_dataid[gid]
        R, p = d.geom_xmat[gid].reshape(3, 3), d.geom_xpos[gid]
        fadr, fnum = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        verts = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
        pos = (verts @ R.T + p)[m.mesh_face[fadr:fadr + fnum]]
        uv = normal = None
        if m.mesh_texcoordadr[mid] >= 0:
            tc = m.mesh_texcoord[m.mesh_texcoordadr[mid]:m.mesh_texcoordadr[mid] + m.mesh_texcoordnum[mid]]
            uv = tc.astype(np.float64)[m.mesh_facetexcoord[fadr:fadr + fnum]]
            uv[..., 1] = 1.0 - uv[..., 1]  # MuJoCo's documented-by-test OBJ V flip
        if m.mesh_normalnum[mid] > 0:
            nr = m.mesh_normal[m.mesh_normaladr[mid]:m.mesh_normaladr[mid] + m.mesh_normalnum[mid]]
            normal = (nr.astype(np.float64) @ R.T)[m.mesh_facenormal[fadr:fadr + fnum]]
        return fnum, pos, uv, normal

    def test_components_match_obj_and_source(self):
        dae_lo, dae_hi = {}, {}
        for name, c in self.report["components"].items():
            with self.subTest(component=name):
                fnum, pos, uv, normal = self._corners(name)
                obj = parse_obj(BUNDLE / c["output_obj"])
                self.assertEqual(fnum, c["counts"]["exported_triangles"])
                self.assertEqual(fnum, obj["n_faces"])
                self.assertLessEqual(float(np.max(np.abs(pos - obj["pos"]))), 1e-7)
                self.assertIsNotNone(uv)
                self.assertLessEqual(float(np.max(np.abs(uv - obj["uv"]))), 1e-6)
                self.assertTrue(c["normals_exported"])
                self.assertIsNotNone(normal)
                self.assertLessEqual(float(np.max(np.abs(normal - obj["normal"]))), 1e-6)
                lo, hi = pos.reshape(-1, 3).min(0), pos.reshape(-1, 3).max(0)
                np.testing.assert_allclose(lo, c["bounds_m"][0], rtol=0, atol=1e-7)
                np.testing.assert_allclose(hi, c["bounds_m"][1], rtol=0, atol=1e-7)
                src = Path(c["source_file"]).name
                dae_lo[src] = np.minimum(dae_lo.get(src, lo), lo)
                dae_hi[src] = np.maximum(dae_hi.get(src, hi), hi)
        for src, (lo_ref, hi_ref) in self.report["dae_exported_union_bounds_m"].items():
            np.testing.assert_allclose(dae_lo[src], lo_ref, rtol=0, atol=1e-7)
            np.testing.assert_allclose(dae_hi[src], hi_ref, rtol=0, atol=1e-7)
        # the approved NXP-HGD-CF acceptance reference applies to the full source (incl. landing gear)
        src_lo, src_hi = self.report["dae_source_union_bounds_m"]["NXP-HGD-CF.dae"]
        np.testing.assert_allclose(src_lo, [-0.198, -0.198, -0.253], atol=5e-4)
        np.testing.assert_allclose(src_hi, [0.198, 0.198, 0.026], atol=5e-4)
        # with the landing gear excluded, nothing exported reaches below the gear threshold
        self.assertGreater(dae_lo["NXP-HGD-CF.dae"][2], self.report["landing_gear_partition"]["z_rule_margins_m"]["threshold"])


if __name__ == "__main__":
    unittest.main()
