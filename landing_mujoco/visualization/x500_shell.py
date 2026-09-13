"""x500 visual shell: native reconstruction + UGRP display alignment.

VISUALIZATION ONLY. This module never touches mass, inertia, CG, joints,
actuators, physical motor positions, landing-gear contact geoms, or any
MuJoCo option. It only appends collision-inert (contype=0, conaffinity=0)
mesh geoms (group 2) plus their mesh/material/texture assets to an MJCF
whose ``vehicle`` body already carries an explicit ``<inertial>``.

Transform layers -- kept separate, both in the Python data and in the
emitted MJCF (nested compile-time ``<frame>`` elements, which add no bodies):

  1. converted mesh coordinates     (COLLADA node transforms already baked
                                     by tools/convert_x500_visual_assets.py)
  2. x500 SDF link pose             <frame name="x500vis_link_...">
     + x500 SDF visual pose         <geom pos/quat>
     + x500 SDF mesh scale          <mesh scale>
     = NATIVE x500 visual            (x500_base model frame, FLU, meters)
  3. UGRP display alignment         <frame name="x500vis_ugrp_alignment">
     p_body = t + R (s * p_native)   (uniform s factored into child positions
                                     and mesh scales, since frames are rigid)
"""

from __future__ import annotations

import itertools
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from landing_mujoco.configs.visualization_config import (
    AUTO,
    FROM_PHYSICAL_PARAMS,
    VisualizationConfig,
)
from landing_mujoco.coordinates.transforms import (
    SDF_MODEL_FLU_TO_MUJOCO_BODY,
    frd_body_offset_to_mujoco_body,
    rotation_matrix_to_mujoco_quaternion,
    sdf_pose_matrix,
)

PREFIX = "x500vis_"
VISUAL_GEOM_GROUP = 2

# Keys that must never appear in the visual manifest (physics separation).
FORBIDDEN_MANIFEST_KEYS = {"mass", "inertia", "inertial", "collision", "contact", "friction",
                           "motorconstant", "momentconstant", "thrust", "plugin", "sensor", "joint"}


def _fmt(values) -> str:
    return " ".join(f"{float(v):.17g}" for v in np.asarray(values, dtype=np.float64).reshape(-1))


# ---------------------------------------------------------------------------
# Layer 1+2: native assembly (from the bundle manifest)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MeshSpec:
    name: str
    file: Path
    rgba: tuple
    texture: Path | None


@dataclass(frozen=True)
class NativeInstance:
    name: str
    link: str
    link_pose_xyzrpy: tuple
    visual_pose_xyzrpy: tuple
    mesh_scale_xyz: tuple
    meshes: tuple

    @property
    def link_matrix(self) -> np.ndarray:
        return sdf_pose_matrix(self.link_pose_xyzrpy)

    @property
    def visual_matrix(self) -> np.ndarray:
        return sdf_pose_matrix(self.visual_pose_xyzrpy)

    def native_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(position, rotation) of the scaled-mesh frame in the native x500
        model frame, expressed with MuJoCo body axes (identical, FLU)."""
        t = self.link_matrix @ self.visual_matrix
        c = SDF_MODEL_FLU_TO_MUJOCO_BODY
        return c @ t[:3, 3], c @ t[:3, :3] @ c.T


@dataclass(frozen=True)
class NativeAssembly:
    assets_dir: Path
    manifest_sha256: str
    meshes: dict
    instances: tuple
    rotor_link_positions: dict

    def source_wheelbase_m(self) -> float:
        """Largest horizontal distance between rotor link origins (opposite
        motors), from the transcribed source SDF poses."""
        xy = [np.asarray(p[:2], dtype=np.float64) for p in self.rotor_link_positions.values()]
        if len(xy) < 2:
            raise ValueError("manifest has fewer than two rotor links; cannot derive source wheelbase")
        return float(max(np.linalg.norm(a - b) for a, b in itertools.combinations(xy, 2)))


def _check_no_physics_keys(obj, path="manifest"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in FORBIDDEN_MANIFEST_KEYS:
                raise ValueError(f"{path}.{k}: physical/sensor key not allowed in the visual manifest")
            _check_no_physics_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_no_physics_keys(v, f"{path}[{i}]")


def load_native_assembly(assets_dir: str | Path) -> NativeAssembly:
    import hashlib

    assets_dir = Path(assets_dir)
    manifest_path = assets_dir / "x500_native_assembly.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    _check_no_physics_keys(manifest)

    meshes = {}
    for name, m in manifest["meshes"].items():
        f = assets_dir / m["file"]
        if not f.is_file():
            raise FileNotFoundError(f"visual mesh missing: {f}")
        tex = m["material"].get("texture")
        tex_path = assets_dir / tex if tex else None
        if tex_path is not None and not tex_path.is_file():
            raise FileNotFoundError(f"visual texture missing: {tex_path}")
        meshes[name] = MeshSpec(name=name, file=f, rgba=tuple(m["material"]["rgba"]), texture=tex_path)

    instances = []
    for inst in manifest["visual_instances"]:
        for mesh in inst["meshes"]:
            if mesh not in meshes:
                raise KeyError(f"instance {inst['name']} references unknown mesh {mesh}")
        instances.append(NativeInstance(
            name=inst["name"], link=inst["link"],
            link_pose_xyzrpy=tuple(inst["link_pose_xyzrpy"]),
            visual_pose_xyzrpy=tuple(inst["visual_pose_xyzrpy"]),
            mesh_scale_xyz=tuple(inst["mesh_scale_xyz"]),
            meshes=tuple(inst["meshes"]),
        ))

    rotors = {r["name"]: tuple(r["pose_xyzrpy"][:3]) for r in manifest["rotor_links"]}
    return NativeAssembly(assets_dir=assets_dir, manifest_sha256=hashlib.sha256(raw).hexdigest(),
                          meshes=meshes, instances=tuple(instances), rotor_link_positions=rotors)


# ---------------------------------------------------------------------------
# Layer 3: UGRP display alignment
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VisualAlignment:
    uniform_scale: float
    scale_mode: str
    source_wheelbase_m: float
    source_wheelbase_origin: str
    target_wheelbase_m: float | None
    target_wheelbase_origin: str
    translation_body_frd_m: tuple
    rotation_rpy_frd_rad: tuple
    translation_mujoco_body_m: np.ndarray
    rotation_mujoco_body: np.ndarray

    def summary(self) -> dict:
        return {
            "uniform_scale": self.uniform_scale, "scale_mode": self.scale_mode,
            "source_wheelbase_m": self.source_wheelbase_m, "source_wheelbase_origin": self.source_wheelbase_origin,
            "target_wheelbase_m": self.target_wheelbase_m, "target_wheelbase_origin": self.target_wheelbase_origin,
            "translation_body_frd_m": list(self.translation_body_frd_m),
            "rotation_rpy_frd_rad": list(self.rotation_rpy_frd_rad),
            "translation_mujoco_body_m": self.translation_mujoco_body_m.tolist(),
            "rotation_mujoco_body": self.rotation_mujoco_body.tolist(),
        }


def _target_wheelbase_from_physical_params(physical_params) -> tuple[float, str]:
    geom = physical_params.geometry
    if geom.wheelbase_m is not None:
        return float(geom.wheelbase_m), "physical_params.geometry.wheelbase_m"
    motors = physical_params.motors.positions_body_m
    if motors and len(motors) >= 2:
        xy = [np.asarray(p[:2], dtype=np.float64) for p in motors.values()]
        return float(max(np.linalg.norm(a - b) for a, b in itertools.combinations(xy, 2))), \
            "physical_params.motors.positions_body_m (max pairwise horizontal distance)"
    raise ValueError("target_wheelbase_m=from_physical_params but physical params have neither "
                     "geometry.wheelbase_m nor motor positions")


def resolve_alignment(cfg: VisualizationConfig, assembly: NativeAssembly, physical_params) -> VisualAlignment:
    if cfg.source_wheelbase_m == AUTO:
        source, source_origin = assembly.source_wheelbase_m(), "auto: manifest rotor link poses"
    else:
        source, source_origin = float(cfg.source_wheelbase_m), "config"

    if cfg.target_wheelbase_m == FROM_PHYSICAL_PARAMS:
        target, target_origin = _target_wheelbase_from_physical_params(physical_params)
    else:
        target, target_origin = float(cfg.target_wheelbase_m), "config"

    if cfg.uniform_scale == AUTO:
        scale, mode = target / source, "auto: target_wheelbase_m / source_wheelbase_m"
    else:
        scale, mode = float(cfg.uniform_scale), "explicit"

    t_mj, r_mj = frd_body_offset_to_mujoco_body(cfg.translation_body_m, cfg.rotation_rpy_rad)
    return VisualAlignment(
        uniform_scale=scale, scale_mode=mode,
        source_wheelbase_m=source, source_wheelbase_origin=source_origin,
        target_wheelbase_m=target, target_wheelbase_origin=target_origin,
        translation_body_frd_m=tuple(cfg.translation_body_m), rotation_rpy_frd_rad=tuple(cfg.rotation_rpy_rad),
        translation_mujoco_body_m=t_mj, rotation_mujoco_body=r_mj,
    )


# ---------------------------------------------------------------------------
# Shell = assembly + alignment, and MJCF emission
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class X500VisualShell:
    assembly: NativeAssembly
    alignment: VisualAlignment

    def geom_pose_in_body(self, instance: NativeInstance) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expected final (position, rotation, mesh scale) of an instance's
        geoms in the MuJoCo vehicle body frame, composed directly in Python
        (independent of MuJoCo's <frame> handling; used for verification)."""
        p_native, r_native = instance.native_pose()
        a = self.alignment
        pos = a.translation_mujoco_body_m + a.uniform_scale * (a.rotation_mujoco_body @ p_native)
        rot = a.rotation_mujoco_body @ r_native
        mesh_scale = a.uniform_scale * np.asarray(instance.mesh_scale_xyz, dtype=np.float64)
        return pos, rot, mesh_scale

    def summary(self) -> dict:
        return {"model": "x500", "assets_dir": str(self.assembly.assets_dir),
                "manifest_sha256": self.assembly.manifest_sha256,
                "visual_instances": len(self.assembly.instances),
                "geoms": sum(len(i.meshes) for i in self.assembly.instances),
                "alignment": self.alignment.summary()}

    def append_to_mjcf(self, mujoco_el: ET.Element, vehicle_body_el: ET.Element) -> None:
        a = self.alignment
        s = a.uniform_scale
        asset = ET.SubElement(mujoco_el, "asset")

        texture_names = {}
        for mesh in self.assembly.meshes.values():
            if mesh.texture is not None and mesh.texture not in texture_names:
                name = f"{PREFIX}tex_{mesh.texture.stem}"
                ET.SubElement(asset, "texture", {"name": name, "type": "2d", "file": str(mesh.texture)})
                texture_names[mesh.texture] = name
        for mesh in self.assembly.meshes.values():
            attrs = {"name": f"{PREFIX}mat_{mesh.name}", "rgba": _fmt(mesh.rgba)}
            if mesh.texture is not None:
                attrs["texture"] = texture_names[mesh.texture]
            ET.SubElement(asset, "material", attrs)

        mesh_asset_names = {}
        for inst in self.assembly.instances:
            total_scale = s * np.asarray(inst.mesh_scale_xyz, dtype=np.float64)
            for mesh_name in inst.meshes:
                key = (mesh_name, tuple(np.round(total_scale, 15)))
                if key not in mesh_asset_names:
                    name = f"{PREFIX}mesh_{mesh_name}_{len(mesh_asset_names)}"
                    ET.SubElement(asset, "mesh", {"name": name, "file": str(self.assembly.meshes[mesh_name].file),
                                                  "scale": _fmt(total_scale)})
                    mesh_asset_names[key] = name

        vehicle_body_el.append(ET.Comment(
            " x500 visual shell (visual-only, contype=0 conaffinity=0, group 2): "
            "layer 3 UGRP alignment -> layer 2 SDF link -> SDF visual pose + mesh scale "))
        ugrp = ET.SubElement(vehicle_body_el, "frame", {
            "name": f"{PREFIX}ugrp_alignment",
            "pos": _fmt(a.translation_mujoco_body_m),
            "quat": _fmt(rotation_matrix_to_mujoco_quaternion(a.rotation_mujoco_body)),
        })
        c = SDF_MODEL_FLU_TO_MUJOCO_BODY
        link_frames = {}
        for inst in self.assembly.instances:
            if inst.link not in link_frames:
                lm = inst.link_matrix
                link_frames[inst.link] = ET.SubElement(ugrp, "frame", {
                    "name": f"{PREFIX}link_{inst.link}",
                    "pos": _fmt(s * (c @ lm[:3, 3])),
                    "quat": _fmt(rotation_matrix_to_mujoco_quaternion(c @ lm[:3, :3] @ c.T)),
                })
            vm = inst.visual_matrix
            total_scale = s * np.asarray(inst.mesh_scale_xyz, dtype=np.float64)
            for mesh_name in inst.meshes:
                ET.SubElement(link_frames[inst.link], "geom", {
                    "name": f"{PREFIX}{inst.name}__{mesh_name}",
                    "type": "mesh",
                    "mesh": mesh_asset_names[(mesh_name, tuple(np.round(total_scale, 15)))],
                    "material": f"{PREFIX}mat_{mesh_name}",
                    "pos": _fmt(s * (c @ vm[:3, 3])),
                    "quat": _fmt(rotation_matrix_to_mujoco_quaternion(c @ vm[:3, :3] @ c.T)),
                    "contype": "0",
                    "conaffinity": "0",
                    "group": str(VISUAL_GEOM_GROUP),
                })


def build_x500_visual_shell(cfg: VisualizationConfig | None, physical_params) -> X500VisualShell | None:
    """None when visualization is absent or disabled (MJCF is then exactly
    the physics-only model)."""
    if cfg is None or not cfg.enabled:
        return None
    assembly = load_native_assembly(cfg.assets_dir)
    return X500VisualShell(assembly=assembly, alignment=resolve_alignment(cfg, assembly, physical_params))
