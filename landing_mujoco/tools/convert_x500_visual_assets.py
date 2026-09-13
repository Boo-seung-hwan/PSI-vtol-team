#!/usr/bin/env python3
"""Reproducible conversion of the PX4/Gazebo x500 VISUAL assets for MuJoCo.

CONVERSION-ONLY tool. Runs in a disposable environment with ``pycollada``,
``trimesh`` and ``Pillow`` installed, against an EXTERNAL checkout of the
pinned upstream models repository::

    git clone https://github.com/PX4/PX4-gazebo-models.git /tmp/PX4-gazebo-models
    git -C /tmp/PX4-gazebo-models checkout d754381a1cecdd7f17050acd72bf5bf1327bced6
    python3 -m venv /tmp/x500_mesh_convert
    source /tmp/x500_mesh_convert/bin/activate
    pip install trimesh pycollada Pillow
    python3 landing_mujoco/tools/convert_x500_visual_assets.py --source /tmp/PX4-gazebo-models

(``PX4-Autopilot@85df8c2281c2466b30a121b22b0bf33dc69bcfe4`` pins that same
commit as its ``Tools/simulation/gz`` submodule, so ``--source
<PX4-Autopilot>/Tools/simulation/gz`` works too.)

None of those packages are runtime/training dependencies: the output bundle
(``landing_mujoco/assets/x500_visual/``) is plain OBJ/STL/PNG/JSON that
MuJoCo loads directly. This script imports nothing from ``landing_mujoco``.

Source verification
-------------------
Every input file actually read must match its pinned sha256 (taken from the
git blobs at the pinned commit). If the source tree is a git checkout, its
HEAD, origin and cleanliness of the used model paths are also recorded. A
mismatch aborts unless ``--allow-unpinned-source`` is given, in which case the
report says so explicitly.

Scope
-----
x500 BODY visual shell: frame upper structure, motor bases, motor bells,
propellers. SDF collision, inertial, sensor, joint and plugin data are never
read into the bundle. OakD-Lite is not converted.

Landing gear (visual design decision, 2026-09-14)
------------------------------------------------
The x500 landing gear is NOT part of the bundle: in the MuJoCo viewer the
landing gear is represented by the PHYSICAL contact geoms. Each COLLADA
component is split into connected pieces (shared-vertex connectivity); a
whole piece is classified as landing gear if its COLLADA geometry is a
``Landing*`` component or it reaches below ``GEAR_MIN_Z_M`` in the native DAE
frame. Only non-gear pieces are exported; no triangle is modified. The
per-piece classification is recorded in ``conversion_report.json``.

Pipeline per COLLADA geometry component
---------------------------------------
source mesh coordinates
  -> accumulated COLLADA scene-node transform (baked here)
  -> component OBJ (independent v / vt / vn index streams)

The SDF component pose / mesh scale and the UGRP display alignment are NOT
applied here; they are recorded verbatim in ``x500_native_assembly.json`` and
composed at runtime (``landing_mujoco/visualization/x500_shell.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import collada
import numpy as np
import PIL
import trimesh
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "landing_mujoco" / "assets" / "x500_visual"

TOOL_VERSION = "1.1.0"
DECIMALS = 8  # text-format precision only; no decimation
FMT = f"{{:.{DECIMALS}f}}"

UPSTREAM = {
    "repository": "https://github.com/PX4/PX4-gazebo-models.git",
    "commit": "d754381a1cecdd7f17050acd72bf5bf1327bced6",
    "referenced_by": {
        "repository": "PX4-Autopilot",
        "commit": "85df8c2281c2466b30a121b22b0bf33dc69bcfe4",
        "submodule_path": "Tools/simulation/gz",
    },
    "model_paths_involved": ["models/x500_base", "models/x500", "models/x500_depth", "models/OakD-Lite"],
}

# sha256 of the git blobs at UPSTREAM["commit"] for every file this tool reads.
PINNED_INPUT_SHA256 = {
    "models/x500_base/model.sdf": "f0f04e4314eebde287eea2b0a962e31a9009f1d98470734a8137e888f78c5bf9",
    "models/x500_base/LICENSE": "cee4ef94e73cd38fb2886f5d7e7a04d6488f54b1e70cd8b1b29d74f38bfdba5b",
    "models/x500_base/meshes/NXP-HGD-CF.dae": "7321861b113866996dfa79b403def5666d4b796810840d2c057c7b51fc315b0a",
    "models/x500_base/meshes/5010Base.dae": "798396a1945c7ff55a0840b77869e27affdd35fed0d784fff96b68f50f160ea0",
    "models/x500_base/meshes/5010Bell.dae": "b90a1b2dd63a8ebbe1f3339ed18cef23673e0d8f03314fd9b9533a68b6b66668",
    "models/x500_base/meshes/1345_prop_ccw.stl": "ba197e0961117a3462b555d65edf3433ba8ff5691624100a185242cb5f456244",
    "models/x500_base/meshes/1345_prop_cw.stl": "7aa4b61c1ddc5304c9c1f5602eacf5e2ec5eafb95dd1f2a3368342f256122fc0",
    "models/x500_base/meshes/CF.png": "3a56ec48eeaaf7774e8d3e8e5e88f6973bfa5652c610974c3eb2dbec6ec85944",
}

# Hard acceptance gates.
GATE_POS_M = 1e-7
GATE_UV = 1e-7
GATE_NORMAL = 1e-7
GATE_TEXTURED_UV_VS_TRIMESH = 1e-7
KNOWN_TRIMESH_UV_TOLERANCE_UNTEXTURED = 1e-4

# Landing-gear partition rule (native DAE frame, meters).
GEAR_MIN_Z_M = -0.10
GEAR_COMPONENT_PREFIX = "Landing"
POSITION_MERGE_DECIMALS = 9

COMPONENT_OUTPUT_NAMES = {
    "NXP-HGD-CF.dae": {
        "LandingFoam-mesh": "frame_landing_foam",
        "LandingRubber-mesh": "frame_landing_rubber",
        "CarbonFiber-mesh": "frame_carbon_fiber",
        "Metal-mesh": "frame_metal",
        "RailsRubber-mesh": "frame_rails_rubber",
        "FMUK66-mesh": "frame_fmuk66",
        "RailsAntennaHolder-mesh": "frame_antenna_holder",
        "LandingPlastic-mesh": "frame_landing_plastic",
        "FMURubber-mesh": "frame_fmu_rubber",
    },
    "5010Base.dae": {"Body1_001-mesh": "motor_base_5010_stator"},
    "5010Bell.dae": {"Body1-mesh": "motor_bell_5010_side", "Body2-mesh": "motor_bell_5010_head"},
}
STL_OUTPUT_NAMES = {"1345_prop_ccw.stl": "prop_1345_ccw", "1345_prop_cw.stl": "prop_1345_cw"}

ASSUMED_SCRIPT_MATERIALS = {
    "Gazebo/DarkGrey": {
        "rgba": [0.175, 0.175, 0.175, 1.0],
        "provenance": "ASSUMED: value of Gazebo-classic 'Gazebo/DarkGrey' diffuse as defined in upstream "
                      "gazebo.material; that OGRE script is not part of PX4-gazebo-models and was not verified "
                      "against a local copy. Visual-only.",
    }
}

NOT_INTEGRATED_VISUALS = {
    "NXP_FMUK66_FRONT": "decorative textured <plane> decal (0.013 x 0.007 m, nxp.png PBR albedo); not a mesh",
    "NXP_FMUK66_TOP": "decorative textured <plane> decal (0.013 x 0.007 m, nxp.png PBR albedo); not a mesh",
    "RDDRONE_FMUK66_TOP": "decorative textured <plane> decal (0.032 x 0.0034 m, rd.png PBR albedo); not a mesh",
}


# ---------------------------------------------------------------------------
# source tree + provenance
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_models_dir(source: Path) -> Path:
    source = source.resolve()
    for cand in (source / "models", source):
        if (cand / "x500_base" / "model.sdf").is_file():
            return cand
    raise FileNotFoundError(f"{source}: expected a PX4-gazebo-models checkout (models/x500_base/model.sdf)")


class Upstream:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

    def rel(self, path: Path) -> str:
        return "models/" + str(Path(path).resolve().relative_to(self.models_dir))


def _git(models_dir: Path, *args) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(models_dir), *args], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify_source(up: Upstream) -> dict:
    content = {}
    for rel, pinned in PINNED_INPUT_SHA256.items():
        path = up.models_dir / rel[len("models/"):]
        actual = sha256(path) if path.is_file() else None
        content[rel] = {"sha256": actual, "pinned_sha256": pinned, "matches_pin": actual == pinned}

    git = {"available": False}
    toplevel = _git(up.models_dir, "rev-parse", "--show-toplevel")
    if toplevel and (Path(toplevel) / "models").resolve() != up.models_dir.resolve():
        # e.g. a plain copy nested inside some other repository: that repo's
        # HEAD/status say nothing about the models, so do not record them.
        git = {"available": False,
               "reason": f"source is not the models/ directory of a git checkout (enclosing repo: {toplevel})"}
        toplevel = None
    if toplevel:
        top = Path(toplevel)
        paths = [p for p in UPSTREAM["model_paths_involved"] if (top / p).exists()]
        if not paths:
            raise AssertionError("git checkout has none of the expected model paths")
        head = _git(top, "rev-parse", "HEAD")
        status = _git(top, "status", "--porcelain", "--", *paths)
        diff = _git(top, "diff", "--name-only", UPSTREAM["commit"], "--", *paths)
        git = {
            "available": True,
            "head": head,
            "head_is_pinned_commit": head == UPSTREAM["commit"],
            "origin": _git(top, "remote", "get-url", "origin"),
            "model_paths_status_porcelain": status,
            "model_paths_clean": status == "",
            "model_paths_diff_vs_pinned_commit": diff,
            "model_paths_identical_to_pinned_commit": diff == "",
        }
    return {
        "content_pins": content,
        "all_content_pins_match": all(c["matches_pin"] for c in content.values()),
        "git": git,
        "note": "content pins are authoritative; git fields are recorded when the source is a git checkout",
    }


def parse_pose(text: str | None) -> list[float]:
    if text is None:
        return [0.0] * 6
    vals = [float(v) for v in text.split()]
    if len(vals) != 6:
        raise ValueError(f"expected 6-value SDF pose, got {text!r}")
    return vals


# ---------------------------------------------------------------------------
# SDF: visual assembly only
# ---------------------------------------------------------------------------
def parse_sdf_visual_assembly(sdf_path: Path, models_dir: Path) -> dict:
    root = ET.parse(sdf_path).getroot()
    model = root.find("model")
    for el in model.iter("pose"):
        if "relative_to" in el.attrib:
            raise NotImplementedError(f"relative_to poses are not handled: {el.attrib}")

    links, visuals = [], []
    ignored = {"collision": 0, "inertial": 0, "sensor": 0,
               "joint": len(model.findall("joint")), "plugin": len(model.findall("plugin"))}
    for link in model.findall("link"):
        lname = link.attrib["name"]
        lpose = parse_pose(link.findtext("pose"))
        links.append({"name": lname, "pose_xyzrpy": lpose})
        ignored["collision"] += len(link.findall("collision"))
        ignored["inertial"] += len(link.findall("inertial"))
        ignored["sensor"] += len(link.findall("sensor"))
        for vis in link.findall("visual"):
            vname = vis.attrib["name"]
            mesh = vis.find("geometry/mesh")
            entry = {"name": vname, "link": lname, "link_pose_xyzrpy": lpose,
                     "visual_pose_xyzrpy": parse_pose(vis.findtext("pose"))}
            if mesh is None:
                entry.update({"integrated": False,
                              "reason": NOT_INTEGRATED_VISUALS.get(vname, "non-mesh visual geometry")})
                visuals.append(entry)
                continue
            uri = mesh.findtext("uri")
            if not uri.startswith("model://"):
                raise ValueError(f"unsupported mesh uri {uri}")
            entry.update({"integrated": True, "source_uri": uri, "source_file": models_dir / uri[len("model://"):],
                          "mesh_scale_xyz": [float(v) for v in (mesh.findtext("scale") or "1 1 1").split()],
                          "sdf_material_script": vis.findtext("material/script/name")})
            visuals.append(entry)
    return {"model_name": model.attrib["name"], "model_pose_xyzrpy": parse_pose(model.findtext("pose")),
            "links": links, "visuals": visuals, "ignored_non_visual_elements": ignored}


# ---------------------------------------------------------------------------
# COLLADA components
# ---------------------------------------------------------------------------
def _accumulate_geometry_nodes(node, parent_matrix, path, out):
    acc = parent_matrix @ getattr(node, "matrix", np.eye(4))
    for ch in getattr(node, "children", []):
        if isinstance(ch, collada.scene.GeometryNode):
            out.append((ch, acc, "/".join(path)))
        elif isinstance(ch, collada.scene.Node):
            _accumulate_geometry_nodes(ch, acc, path + [ch.id], out)


def collada_components(dae_path: Path) -> list[dict]:
    c = collada.Collada(str(dae_path))
    manual = []
    for n in c.scene.nodes:
        _accumulate_geometry_nodes(n, np.eye(4), [n.id], manual)
    bound = {}
    for bg in c.scene.objects("geometry"):
        bound.setdefault(bg.original.id, []).append(bg)

    comps = []
    for geom_node, acc, node_path in manual:
        gid = geom_node.geometry.id
        if len(bound[gid]) != 1:
            raise NotImplementedError(f"{gid}: multi-instance geometry not handled")
        bg = bound[gid][0]
        if not np.allclose(bg.matrix, acc, atol=1e-12):
            raise AssertionError(f"{gid}: manual node accumulation disagrees with pycollada bound matrix")
        m3, t = acc[:3, :3], acc[:3, 3]
        identity = bool(np.allclose(acc, np.eye(4), atol=1e-12))
        det = float(np.linalg.det(m3))
        normal_matrix = np.linalg.inv(m3).T

        tri_prims, dropped, materials = [], [], []
        for bprim in bg.primitives():
            if isinstance(bprim, collada.lineset.BoundLineSet):
                segs = bprim.vertex[bprim.vertex_index]
                lengths = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
                pts = segs.reshape(-1, 3)
                dropped.append({"type": "LineSet", "segments": int(len(segs)),
                                "segment_length_m_min_max": [float(lengths.min()), float(lengths.max())],
                                "bounds_m": [pts.min(0).tolist(), pts.max(0).tolist()],
                                "material": bprim.material.id if bprim.material else None})
                continue
            if not isinstance(bprim, collada.triangleset.BoundTriangleSet):
                raise NotImplementedError(f"unsupported primitive {type(bprim).__name__}")
            eff = bprim.material.effect
            diffuse = ({"texture": eff.diffuse.sampler.surface.image.path, "texcoord": eff.diffuse.texcoord}
                       if isinstance(eff.diffuse, collada.material.Map) else {"rgba": [float(x) for x in eff.diffuse]})
            materials.append({"material_id": bprim.material.id, "effect_id": eff.id, "diffuse": diffuse})
            tri_prims.append(bprim.original)
        if len({json.dumps(m["diffuse"], sort_keys=True) for m in materials}) != 1:
            raise NotImplementedError(f"{gid}: multiple materials inside one component")

        streams = {"v": [], "iv": [], "vt": [], "it": [], "vn": [], "in": []}
        normals_ok, normals_reason = True, None
        has_uv_all = True
        voff = toff = noff = 0
        for orig in tri_prims:
            v = orig.vertex @ m3.T + t
            vi = orig.vertex_index.reshape(-1, 3)
            has_uv = len(orig.texcoordset) > 0
            ti = orig.texcoord_indexset[0].reshape(-1, 3) if has_uv else None
            has_uv_all &= has_uv
            has_n = orig.normal is not None and orig.normal_index is not None and len(orig.normal) > 0
            n = ni = None
            if has_n:
                n = orig.normal.copy() if identity else orig.normal @ normal_matrix.T
                if not identity:
                    norms = np.linalg.norm(n, axis=1)
                    n = n / np.where(norms > 0, norms, 1.0)[:, None]
                ni = orig.normal_index.reshape(-1, 3)
                used = n[np.unique(ni)]
                if not np.all(np.isfinite(used)) or np.any(np.linalg.norm(used, axis=1) < 1e-6):
                    normals_ok, normals_reason = False, "non-finite or zero-length source normals"
            else:
                normals_ok, normals_reason = False, "source normals absent"
            if det < 0:
                vi = vi[:, ::-1]
                ti = ti[:, ::-1] if ti is not None else None
                ni = ni[:, ::-1] if ni is not None else None
            streams["v"].append(v); streams["iv"].append(vi + voff); voff += len(v)
            if ti is not None:
                streams["vt"].append(orig.texcoordset[0]); streams["it"].append(ti + toff); toff += len(orig.texcoordset[0])
            if ni is not None:
                streams["vn"].append(n); streams["in"].append(ni + noff); noff += len(n)

        s = {"v": np.concatenate(streams["v"]), "iv": np.concatenate(streams["iv"])}
        s["vt"] = np.concatenate(streams["vt"]) if has_uv_all else None
        s["it"] = np.concatenate(streams["it"]) if has_uv_all else None
        s["vn"] = np.concatenate(streams["vn"]) if normals_ok else None
        s["in"] = np.concatenate(streams["in"]) if normals_ok else None
        comps.append({
            "geometry_id": gid, "node_path": node_path, "node_transform": acc.tolist(),
            "node_transform_identity": identity, "node_transform_det": det,
            "material": materials[0], "dropped_primitives": dropped,
            "normals_exported": normals_ok, "normals_omitted_reason": None if normals_ok else normals_reason,
            "streams": s,
            "truth_corner_pos": s["v"][s["iv"]],
            "truth_corner_uv": s["vt"][s["it"]] if has_uv_all else None,
            "truth_corner_normal": s["vn"][s["in"]] if normals_ok else None,
        })
    return comps


def connected_pieces(comp: dict) -> tuple[np.ndarray, list[dict]]:
    """Face -> piece label via shared-vertex connectivity (positions merged
    by value). Pieces are ordered deterministically by their bounds."""
    s = comp["streams"]
    _, canon = np.unique(np.round(s["v"], POSITION_MERGE_DECIMALS), axis=0, return_inverse=True)
    canon = canon.reshape(-1)
    parent = np.arange(canon.max() + 1)

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    faces = canon[s["iv"]]
    for a, b, c in faces:
        ra = find(a)
        parent[find(b)] = ra
        parent[find(c)] = ra
    roots = np.array([find(a) for a in faces[:, 0]])
    pos = comp["truth_corner_pos"]
    pieces = []
    for r in np.unique(roots):
        mask = roots == r
        pts = pos[mask].reshape(-1, 3)
        pieces.append({"root": int(r), "triangles": int(mask.sum()), "min": pts.min(0), "max": pts.max(0)})
    pieces.sort(key=lambda p: (round(float(p["min"][2]), 9), round(float(p["min"][0]), 9),
                               round(float(p["min"][1]), 9), p["triangles"]))
    label = np.empty(len(roots), dtype=np.int64)
    for k, p in enumerate(pieces):
        label[roots == p["root"]] = k
    return label, pieces


def classify_pieces(comp: dict, pieces: list[dict]) -> list[dict]:
    is_landing_component = comp["geometry_id"].startswith(GEAR_COMPONENT_PREFIX)
    out = []
    for k, p in enumerate(pieces):
        reasons = []
        if is_landing_component:
            reasons.append(f"COLLADA component '{comp['geometry_id']}' is a {GEAR_COMPONENT_PREFIX}* component")
        if p["min"][2] < GEAR_MIN_Z_M:
            reasons.append(f"piece reaches z={p['min'][2]:+.4f} m < {GEAR_MIN_Z_M} m (native DAE frame)")
        out.append({"piece": k, "triangles": p["triangles"],
                    "bounds_m": [p["min"].tolist(), p["max"].tolist()],
                    "class": "landing_gear" if reasons else "upper_structure",
                    "reason": "; ".join(reasons) if reasons else "above gear threshold, non-Landing component"})
    return out


def _compact(values, indices):
    used, inverse = np.unique(indices.reshape(-1), return_inverse=True)
    return values[used], inverse.reshape(indices.shape)


def write_obj(path: Path, streams: dict, face_mask: np.ndarray, header: list[str]) -> dict:
    iv = streams["iv"][face_mask]
    v, iv = _compact(streams["v"], iv)
    vt = it = vn = inn = None
    if streams["vt"] is not None:
        vt, it = _compact(streams["vt"], streams["it"][face_mask])
    if streams["vn"] is not None:
        vn, inn = _compact(streams["vn"], streams["in"][face_mask])
    lines = [f"# {h}" for h in header]
    lines += ["v " + " ".join(FMT.format(x) for x in row) for row in v]
    if vt is not None:
        lines += ["vt " + " ".join(FMT.format(x) for x in row) for row in vt]
    if vn is not None:
        lines += ["vn " + " ".join(FMT.format(x) for x in row) for row in vn]
    for f in range(len(iv)):
        corners = []
        for k in range(3):
            a = iv[f, k] + 1
            if vt is not None and vn is not None:
                corners.append(f"{a}/{it[f, k] + 1}/{inn[f, k] + 1}")
            elif vt is not None:
                corners.append(f"{a}/{it[f, k] + 1}")
            elif vn is not None:
                corners.append(f"{a}//{inn[f, k] + 1}")
            else:
                corners.append(f"{a}")
        lines.append("f " + " ".join(corners))
    path.write_text("\n".join(lines) + "\n")
    return {"v": int(len(v)), "vt": int(len(vt)) if vt is not None else 0,
            "vn": int(len(vn)) if vn is not None else 0, "f": int(len(iv))}


def parse_obj_independent(path: Path) -> dict:
    v, vt, vn, faces = [], [], [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            vt.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("vn "):
            vn.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            corners = line.split()[1:]
            if len(corners) != 3:
                raise AssertionError("non-triangle face in exported OBJ")
            row = []
            for c in corners:
                parts = c.split("/") + [""] * 2
                row.append([int(p) if p else 0 for p in parts[:3]])
            faces.append(row)
    v, vt, vn, faces = np.array(v), np.array(vt), np.array(vn), np.array(faces)
    return {"corner_pos": v[faces[:, :, 0] - 1],
            "corner_uv": vt[faces[:, :, 1] - 1] if len(vt) else None,
            "corner_normal": vn[faces[:, :, 2] - 1] if len(vn) else None,
            "n_faces": int(len(faces))}


def trimesh_cross_check(dae_path: Path, comps: list[dict], texture_file: Path | None) -> dict:
    scene = trimesh.load(str(dae_path), force="scene")
    out = {}
    for comp in comps:
        gid = comp["geometry_id"]
        nodes = [n for n in scene.graph.nodes_geometry if scene.graph[n][1] == gid]
        if len(nodes) != 1:
            raise AssertionError(f"{gid}: trimesh found {len(nodes)} instances")
        transform, _ = scene.graph[nodes[0]]
        mesh = scene.geometry[gid].copy()
        mesh.apply_transform(transform)
        tm_pos = mesh.vertices[mesh.faces]
        entry = {"trimesh_transform_matches_collada": bool(np.allclose(transform, comp["node_transform"], atol=1e-12))}
        same_shape = tm_pos.shape == comp["truth_corner_pos"].shape
        entry["max_abs_pos_vs_trimesh_m"] = float(np.max(np.abs(tm_pos - comp["truth_corner_pos"]))) if same_shape else None
        uv = getattr(mesh.visual, "uv", None)
        if uv is not None and comp["truth_corner_uv"] is not None and same_shape:
            entry["max_abs_uv_vs_trimesh"] = float(np.max(np.abs(uv[mesh.faces] - comp["truth_corner_uv"])))
        if "texture" in comp["material"]["diffuse"]:
            img = getattr(mesh.visual.material, "baseColorTexture", None)
            entry["trimesh_resolved_texture_image"] = img is not None
            entry["texture_pixels_identical_to_bundle_file"] = bool(
                img is not None and texture_file is not None
                and np.array_equal(np.asarray(img.convert("RGBA")), np.asarray(Image.open(texture_file).convert("RGBA"))))
        out[gid] = entry
    return out


def stl_info(path: Path) -> dict:
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    if 84 + 50 * n != len(data):
        raise AssertionError(f"{path.name}: not a binary STL of consistent size")
    tri = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                        count=n, offset=84)
    pts = tri["v"].reshape(-1, 3).astype(np.float64)
    return {"format": "binary STL (80-byte header begins with 'solid')", "triangles": int(n),
            "bounds_native_units": [pts.min(0).tolist(), pts.max(0).tolist()]}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True,
                    help="PX4-gazebo-models checkout at the pinned commit (repo root or its models/ dir)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--allow-unpinned-source", action="store_true",
                    help="continue if input hashes differ from the pinned commit (recorded in the report)")
    args = ap.parse_args(argv)

    models_dir = resolve_models_dir(args.source)
    up = Upstream(models_dir)
    verification = verify_source(up)
    if not verification["all_content_pins_match"] and not args.allow_unpinned_source:
        bad = [k for k, v in verification["content_pins"].items() if not v["matches_pin"]]
        print("ABORT: source files differ from PX4-gazebo-models@" + UPSTREAM["commit"] + ":", *bad, sep="\n  ")
        return 2

    out = args.out_dir.resolve()
    mesh_dir, tex_dir = out / "meshes", out / "textures"
    # meshes/ and textures/ are fully tool-owned: clear them so no stale
    # derived file survives a regeneration. Hand-written README.md /
    # THIRD_PARTY_NOTICES.md in out-dir are never touched.
    for d in (mesh_dir, tex_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    sdf_path = models_dir / "x500_base" / "model.sdf"
    assembly = parse_sdf_visual_assembly(sdf_path, models_dir)

    report = {
        "schema_version": 2,
        "tool": {"script": "landing_mujoco/tools/convert_x500_visual_assets.py", "version": TOOL_VERSION},
        "conversion_environment": {
            "note": "disposable venv; these packages are NOT runtime/training dependencies",
            "python": sys.version.split()[0], "trimesh": trimesh.__version__,
            "pycollada": collada.__version__, "Pillow": PIL.__version__,
        },
        "upstream_source": UPSTREAM,
        "source_verification": verification,
        "pinned_source_verified": bool(verification["all_content_pins_match"]),
        "conversion_source_local_path": str(models_dir),
        "formatting_precision": {"xyz_decimal_places": DECIMALS, "uv_decimal_places": DECIMALS,
                                 "normal_decimal_places": DECIMALS,
                                 "note": "text-format precision reduction only; no decimation"},
        "acceptance_gates": {"max_position_deviation_m": GATE_POS_M, "max_uv_deviation": GATE_UV,
                             "max_normal_component_deviation": GATE_NORMAL,
                             "textured_component_uv_vs_trimesh": GATE_TEXTURED_UV_VS_TRIMESH},
        "scope": {
            "converted": "x500 body visual shell: frame upper structure (NXP-HGD-CF.dae), motor bases "
                         "(5010Base.dae), motor bells (5010Bell.dae), propellers (1345_prop_*.stl, copied)",
            "not_converted": ["OakD-Lite (camera; out of scope)", "x500 landing gear (see landing_gear_partition)"],
            "physics_data_ignored": "SDF collision, inertial, sensor, joint and plugin elements are never read into "
                                    "the bundle; the bundle carries no mass, inertia, CG, motor, thrust, contact or "
                                    "sensor parameters",
        },
        "sdf_visual_assembly": {
            "source": up.rel(sdf_path),
            "model_name": assembly["model_name"],
            "model_pose_xyzrpy": assembly["model_pose_xyzrpy"],
            "model_pose_handling": "NOT applied: x500_base's model <pose> is its standalone spawn pose; x500 "
                                   "includes x500_base with merge='true' and all visuals are link-relative",
            "ignored_non_visual_elements": assembly["ignored_non_visual_elements"],
        },
        "landing_gear_partition": {
            "decision": "2026-09-14 (user-approved): x500 landing gear hidden; landing gear represented in the "
                        "MuJoCo viewer by the physical contact geoms",
            "rule": f"whole connected piece is landing gear if its COLLADA geometry id starts with "
                    f"'{GEAR_COMPONENT_PREFIX}' OR its minimum z < {GEAR_MIN_Z_M} m in the native DAE frame",
            "connectivity": f"shared vertices, positions merged at {POSITION_MERGE_DECIMALS} decimals",
            "geometry_modified": False,
            "components": {},
        },
        "components": {},
        "excluded_components": {},
        "stl_copies": {},
        "textures": {},
        "known_conversion_tolerances": [],
        "dropped_source_primitives": [],
        "gates_passed": False,
    }

    failures = []
    manifest_meshes = {}
    exported_names_by_source = {}
    z_rule_upper_min, z_rule_gear_min = [], []

    cf_src = models_dir / "x500_base" / "meshes" / "CF.png"
    cf_out = tex_dir / "cf.png"
    shutil.copyfile(cf_src, cf_out)
    report["textures"]["cf.png"] = {
        "source": up.rel(cf_src), "source_sha256": sha256(cf_src), "output": "textures/cf.png",
        "output_sha256": sha256(cf_out), "byte_identical": sha256(cf_src) == sha256(cf_out),
        "note": "models/x500_base/materials/textures/CF.png is a byte-identical duplicate (not used by any mesh)",
    }
    texture_map = {"CF.png": "textures/cf.png"}

    dae_files = sorted({v["source_file"] for v in assembly["visuals"]
                        if v.get("integrated") and str(v["source_file"]).endswith(".dae")})
    for dae in dae_files:
        names = COMPONENT_OUTPUT_NAMES[dae.name]
        comps = collada_components(dae)
        if {c["geometry_id"] for c in comps} != set(names):
            failures.append(f"{dae.name}: unexpected component set")
            continue
        textured = any("texture" in c["material"]["diffuse"] for c in comps)
        cross = trimesh_cross_check(dae, comps, cf_out if textured else None)
        src_lo, src_hi = np.full(3, np.inf), np.full(3, -np.inf)
        exp_lo, exp_hi = np.full(3, np.inf), np.full(3, -np.inf)
        exported_names_by_source[dae.name] = []

        for comp in comps:
            gid = comp["geometry_id"]
            base = names[gid]
            label, pieces = connected_pieces(comp)
            classified = classify_pieces(comp, pieces)
            gear_idx = np.array([p["piece"] for p in classified if p["class"] == "landing_gear"], dtype=np.int64)
            upper_mask = ~np.isin(label, gear_idx)
            for p in classified:
                if comp["geometry_id"].startswith(GEAR_COMPONENT_PREFIX):
                    continue
                (z_rule_gear_min if p["class"] == "landing_gear" else z_rule_upper_min).append(p["bounds_m"][0][2])

            n_src = int(len(comp["truth_corner_pos"]))
            tlo = comp["truth_corner_pos"].reshape(-1, 3).min(0)
            thi = comp["truth_corner_pos"].reshape(-1, 3).max(0)
            src_lo, src_hi = np.minimum(src_lo, tlo), np.maximum(src_hi, thi)
            partition_entry = {
                "source_geometry_id": gid, "source_triangles": n_src,
                "exported_triangles": int(upper_mask.sum()), "excluded_landing_gear_triangles": int((~upper_mask).sum()),
                "pieces": classified,
            }
            report["landing_gear_partition"]["components"][base] = partition_entry
            if int(upper_mask.sum()) + int((~upper_mask).sum()) != n_src:
                failures.append(f"{base}: partition does not cover all triangles exactly once")

            xc = cross[gid]
            if not xc["trimesh_transform_matches_collada"]:
                failures.append(f"{base}: trimesh scene-graph transform disagrees with COLLADA accumulation")
            if xc.get("max_abs_pos_vs_trimesh_m") is None or xc["max_abs_pos_vs_trimesh_m"] > 1e-9:
                failures.append(f"{base}: independent trimesh geometry cross-check failed")
            diffuse = comp["material"]["diffuse"]
            is_textured = "texture" in diffuse
            if is_textured:
                if not (xc.get("trimesh_resolved_texture_image") and xc.get("texture_pixels_identical_to_bundle_file")):
                    failures.append(f"{base}: texture did not resolve to the bundled image")
                if xc.get("max_abs_uv_vs_trimesh", np.inf) > GATE_TEXTURED_UV_VS_TRIMESH:
                    failures.append(f"{base}: textured-component UV disagrees with trimesh")
            elif xc.get("max_abs_uv_vs_trimesh", 0.0) > KNOWN_TRIMESH_UV_TOLERANCE_UNTEXTURED:
                failures.append(f"{base}: untextured UV discrepancy exceeds known tolerance")

            for d in comp["dropped_primitives"]:
                report["dropped_source_primitives"].append({
                    "source_file": up.rel(dae), "source_component": gid, "output_component": base,
                    "dropped_primitives": f"{d['segments']} line segments", "detail": d,
                    "size": "approximately {:.2f}-{:.2f} mm".format(*(1000 * x for x in d["segment_length_m_min_max"])),
                    "reason": "unsupported/non-triangular internal CAD edges (COLLADA LineSet)",
                    "expected_visual_impact": "none", "physics_impact": "none",
                    "handling": "dropped; no replacement geometry synthesized",
                })

            if not upper_mask.any():
                report["excluded_components"][base] = {
                    "source_file": up.rel(dae), "source_geometry_id": gid, "source_triangles": n_src,
                    "reason": "entire component is x500 landing gear (represented by physical contact geoms)",
                    "source_material": comp["material"],
                }
                continue

            out_name = base if upper_mask.all() else f"{base}_upper"
            obj_path = mesh_dir / f"{out_name}.obj"
            counts = write_obj(obj_path, comp["streams"], upper_mask, [
                "x500 visual component (visual-only, no physics meaning)",
                f"source: {UPSTREAM['repository']}@{UPSTREAM['commit']} {up.rel(dae)} geometry={gid} node={comp['node_path']}",
                f"generated by landing_mujoco/tools/convert_x500_visual_assets.py v{TOOL_VERSION}",
                f"COLLADA node transform baked; {DECIMALS} decimal places; no decimation; "
                f"{'landing-gear pieces excluded' if not upper_mask.all() else 'all pieces'}",
            ])
            parsed = parse_obj_independent(obj_path)

            truth_pos = comp["truth_corner_pos"][upper_mask]
            errs = {"position_m": float(np.max(np.abs(parsed["corner_pos"] - truth_pos)))}
            if comp["truth_corner_uv"] is not None:
                errs["uv"] = float(np.max(np.abs(parsed["corner_uv"] - comp["truth_corner_uv"][upper_mask])))
            if comp["normals_exported"]:
                errs["normal_component"] = float(np.max(np.abs(parsed["corner_normal"] - comp["truth_corner_normal"][upper_mask])))
                errs["normal_unit_length_dev"] = float(np.max(np.abs(np.linalg.norm(parsed["corner_normal"], axis=2) - 1.0)))
            lo, hi = parsed["corner_pos"].reshape(-1, 3).min(0), parsed["corner_pos"].reshape(-1, 3).max(0)
            elo, ehi = truth_pos.reshape(-1, 3).min(0), truth_pos.reshape(-1, 3).max(0)
            bbox_dev = float(max(np.max(np.abs(lo - elo)), np.max(np.abs(hi - ehi))))
            exp_lo, exp_hi = np.minimum(exp_lo, lo), np.maximum(exp_hi, hi)

            if parsed["n_faces"] != int(upper_mask.sum()):
                failures.append(f"{out_name}: triangle count {parsed['n_faces']} != selected source {int(upper_mask.sum())}")
            if errs["position_m"] > GATE_POS_M or bbox_dev > GATE_POS_M:
                failures.append(f"{out_name}: position/bbox deviation above gate")
            if "uv" in errs and errs["uv"] > GATE_UV:
                failures.append(f"{out_name}: uv deviation above gate")
            if "normal_component" in errs and errs["normal_component"] > GATE_NORMAL:
                failures.append(f"{out_name}: normal deviation above gate")

            if not is_textured and xc.get("max_abs_uv_vs_trimesh", 0.0) > 0.0:
                report["known_conversion_tolerances"].append({
                    "component": out_name, "source_geometry": gid,
                    "kind": "trimesh cross-check UV discrepancy on an UNTEXTURED component",
                    "max_abs_uv": xc["max_abs_uv_vs_trimesh"],
                    "approx_pixels_on_1024px_map": xc["max_abs_uv_vs_trimesh"] * 1024,
                    "affects_exported_obj": False,
                    "note": "exported OBJ UVs come from pycollada index streams and meet the UV gate; no visual "
                            "effect (untextured)",
                })

            material_manifest = ({"rgba": [1.0, 1.0, 1.0, 1.0], "texture": texture_map[diffuse["texture"]]}
                                 if is_textured else {"rgba": diffuse["rgba"]})
            manifest_meshes[out_name] = {
                "file": f"meshes/{out_name}.obj", "material": material_manifest,
                "source": {"file": up.rel(dae), "geometry_id": gid, "node_path": comp["node_path"],
                           "pieces": "all" if upper_mask.all() else "upper_structure pieces only"},
            }
            exported_names_by_source[dae.name].append((gid, out_name))
            report["components"][out_name] = {
                "source_file": up.rel(dae), "source_file_sha256": sha256(dae),
                "source_geometry_id": gid, "source_node_path": comp["node_path"],
                "source_material": comp["material"],
                "node_transform_baked": comp["node_transform"],
                "node_transform_identity": comp["node_transform_identity"],
                "node_transform_det": comp["node_transform_det"],
                "normal_transform": ("as-is (identity node transform)" if comp["node_transform_identity"]
                                     else "inverse-transpose of node 3x3, renormalized"),
                "normals_exported": comp["normals_exported"],
                "normals_omitted_reason": comp["normals_omitted_reason"],
                "output_obj": f"meshes/{out_name}.obj", "output_obj_sha256": sha256(obj_path),
                "output_obj_bytes": obj_path.stat().st_size,
                "counts": {"source_triangles": n_src, "exported_triangles": int(upper_mask.sum()), **counts},
                "max_deviation_obj_vs_source": errs,
                "bbox_deviation_m": bbox_dev,
                "bounds_m": [lo.tolist(), hi.tolist()],
                "trimesh_cross_check": xc,
            }
        report.setdefault("dae_source_union_bounds_m", {})[dae.name] = [src_lo.tolist(), src_hi.tolist()]
        if np.all(np.isfinite(exp_lo)):
            report.setdefault("dae_exported_union_bounds_m", {})[dae.name] = [exp_lo.tolist(), exp_hi.tolist()]

    if z_rule_upper_min and z_rule_gear_min:
        report["landing_gear_partition"]["z_rule_margins_m"] = {
            "lowest_upper_structure_piece_min_z": float(min(z_rule_upper_min)),
            "highest_z_rule_gear_piece_min_z": float(max(z_rule_gear_min)),
            "threshold": GEAR_MIN_Z_M,
        }

    stl_files = sorted({v["source_file"] for v in assembly["visuals"]
                        if v.get("integrated") and str(v["source_file"]).endswith(".stl")})
    for stl in stl_files:
        out_name = STL_OUTPUT_NAMES[stl.name]
        dst = mesh_dir / f"{out_name}.stl"
        shutil.copyfile(stl, dst)
        scripts = {v["sdf_material_script"] for v in assembly["visuals"]
                   if v.get("integrated") and v["source_file"] == stl}
        if len(scripts) != 1 or next(iter(scripts)) not in ASSUMED_SCRIPT_MATERIALS:
            failures.append(f"{stl.name}: unexpected SDF material scripts {scripts}")
            continue
        script = next(iter(scripts))
        mat = ASSUMED_SCRIPT_MATERIALS[script]
        manifest_meshes[out_name] = {"file": f"meshes/{out_name}.stl", "material": {"rgba": mat["rgba"]},
                                     "source": {"file": up.rel(stl), "sdf_material_script": script}}
        report["stl_copies"][out_name] = {
            "source_file": up.rel(stl), "source_sha256": sha256(stl), "output": f"meshes/{out_name}.stl",
            "output_sha256": sha256(dst), "byte_identical": sha256(stl) == sha256(dst),
            "geometry_conversion": "none (byte copy)", **stl_info(dst),
            "sdf_material_script": script, "material_rgba": mat["rgba"], "material_provenance": mat["provenance"],
        }
        if sha256(stl) != sha256(dst):
            failures.append(f"{stl.name}: copy not byte-identical")

    instances, not_integrated = [], []
    for v in assembly["visuals"]:
        if not v.get("integrated"):
            not_integrated.append({"visual": v["name"], "link": v["link"], "reason": v["reason"]})
            continue
        src = v["source_file"]
        if src.suffix == ".dae":
            meshes = [name for _, name in exported_names_by_source[src.name]]
        else:
            meshes = [STL_OUTPUT_NAMES[src.name]]
        if not meshes:
            not_integrated.append({"visual": v["name"], "link": v["link"],
                                   "reason": "all pieces classified as x500 landing gear"})
            continue
        instances.append({"name": v["name"], "link": v["link"],
                          "link_pose_xyzrpy": v["link_pose_xyzrpy"], "visual_pose_xyzrpy": v["visual_pose_xyzrpy"],
                          "mesh_scale_xyz": v["mesh_scale_xyz"], "source_uri": v["source_uri"], "meshes": meshes})
    rotor_links = [{"name": l["name"], "pose_xyzrpy": l["pose_xyzrpy"]} for l in assembly["links"]
                   if l["name"].startswith("rotor_")]

    manifest = {
        "schema_version": 2,
        "description": "Native (unscaled) x500 visual assembly, transcribed programmatically from the upstream SDF. "
                       "Visual-only: contains no physical parameters.",
        "upstream": {"repository": UPSTREAM["repository"], "commit": UPSTREAM["commit"]},
        "frame": "x500_base SDF model frame: +x forward, +y left, +z up, meters",
        "pose_convention": "SDF [x y z roll pitch yaw]; R = Rz(yaw) Ry(pitch) Rx(roll)",
        "composition": "p_model = T_link( T_visual( mesh_scale * p_mesh ) ); p_mesh already includes the baked "
                       "COLLADA node transform",
        "landing_gear_representation": "physical_contact_geoms: x500 landing-gear pieces are excluded from this "
                                       "bundle (see conversion_report.json landing_gear_partition)",
        "source_sdf": up.rel(sdf_path),
        "source_sdf_sha256": sha256(sdf_path),
        "meshes": manifest_meshes,
        "visual_instances": instances,
        "rotor_links": rotor_links,
        "not_integrated_visuals": not_integrated,
    }
    manifest_path = out / "x500_native_assembly.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    lic_src, lic_dst = models_dir / "x500_base" / "LICENSE", out / "LICENSE"
    shutil.copyfile(lic_src, lic_dst)

    report["native_assembly_manifest"] = {"file": "x500_native_assembly.json", "sha256": sha256(manifest_path),
                                          "visual_instances": len(instances), "not_integrated_visuals": not_integrated}
    report["license_copy"] = {"source": up.rel(lic_src), "source_sha256": sha256(lic_src), "file": "LICENSE",
                              "sha256": sha256(lic_dst), "byte_identical_to_source": sha256(lic_dst) == sha256(lic_src)}
    report["gate_failures"] = failures
    report["gates_passed"] = not failures
    (out / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"source verified: {report['pinned_source_verified']}  git: "
          f"{verification['git'].get('head')} clean={verification['git'].get('model_paths_clean')}")
    print(f"exported components: {len(report['components'])}  excluded: {sorted(report['excluded_components'])}  "
          f"stl: {len(report['stl_copies'])}  instances: {len(instances)}  gates_passed: {report['gates_passed']}")
    for name, c in report["components"].items():
        e = c["max_deviation_obj_vs_source"]
        print(f"  {name:28s} tris {c['counts']['source_triangles']:6d}->{c['counts']['f']:6d} "
              f"pos {e['position_m']:.1e} uv {e.get('uv', float('nan')):.1e} n {e.get('normal_component', float('nan')):.1e}")
    for f in failures:
        print("GATE FAILURE:", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
