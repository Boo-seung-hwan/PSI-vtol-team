"""Visualization-only configuration (x500 visual shell).

Deliberately separate from ``param_schema.UAVPhysicalParams``: nothing in
this file is a physical parameter, and nothing here is ever read by the
dynamics, controller, contact, observation, or reward code. Its only effect
is which purely cosmetic, collision-inert geoms get appended to the MJCF.

YAML schema (``landing_mujoco/configs/x500_visualization.yaml``)::

    visualization:
      enabled: true
      model: x500
      assets_dir: landing_mujoco/assets/x500_visual     # repo-root relative
      source_wheelbase_m: auto          # number, or auto (from manifest rotor links)
      target_wheelbase_m: 0.737         # number, or from_physical_params
      uniform_scale: auto               # number, or auto (= target / source)
      translation_body_m: [0.0, 0.0, 0.0]   # FRD policy body frame [m]
      rotation_rpy_rad: [0.0, 0.0, 0.0]     # FRD policy body frame, Rz.Ry.Rx [rad]

``translation_body_m`` / ``rotation_rpy_rad`` use the FRD policy body frame,
like every other ``*_body_m`` vector in ``landing_mujoco/configs``; they are
converted to the MuJoCo body frame only by
``landing_mujoco.coordinates.transforms``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

AUTO = "auto"
FROM_PHYSICAL_PARAMS = "from_physical_params"

Number = Union[float, int]


@dataclass(frozen=True)
class VisualizationConfig:
    enabled: bool = False
    model: str = "x500"
    assets_dir: Path = REPO_ROOT / "landing_mujoco" / "assets" / "x500_visual"
    source_wheelbase_m: Union[Number, str] = AUTO
    target_wheelbase_m: Union[Number, str] = FROM_PHYSICAL_PARAMS
    uniform_scale: Union[Number, str] = AUTO
    translation_body_m: tuple = (0.0, 0.0, 0.0)
    rotation_rpy_rad: tuple = (0.0, 0.0, 0.0)
    source_path: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.model != "x500":
            raise ValueError(f"unsupported visualization model {self.model!r} (only 'x500')")
        for name, allowed in (("source_wheelbase_m", {AUTO}),
                              ("target_wheelbase_m", {FROM_PHYSICAL_PARAMS}),
                              ("uniform_scale", {AUTO})):
            value = getattr(self, name)
            if isinstance(value, str):
                if value not in allowed:
                    raise ValueError(f"{name}: expected a number or one of {sorted(allowed)}, got {value!r}")
            elif not (isinstance(value, (int, float)) and float(value) > 0.0):
                raise ValueError(f"{name}: must be a positive number, got {value!r}")
        if len(self.translation_body_m) != 3 or len(self.rotation_rpy_rad) != 3:
            raise ValueError("translation_body_m and rotation_rpy_rad must be 3-vectors")
        if not np.all(np.isfinite(np.asarray(self.translation_body_m, dtype=float))) or \
                not np.all(np.isfinite(np.asarray(self.rotation_rpy_rad, dtype=float))):
            raise ValueError("translation_body_m / rotation_rpy_rad must be finite")


def load_visualization_config(path: str | Path) -> VisualizationConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "visualization" not in raw:
        raise ValueError(f"{path}: missing top-level 'visualization' section")
    v = raw["visualization"]
    assets_dir = Path(v.get("assets_dir", VisualizationConfig.assets_dir))
    if not assets_dir.is_absolute():
        assets_dir = REPO_ROOT / assets_dir
    return VisualizationConfig(
        enabled=bool(v.get("enabled", False)),
        model=v.get("model", "x500"),
        assets_dir=assets_dir,
        source_wheelbase_m=v.get("source_wheelbase_m", AUTO),
        target_wheelbase_m=v.get("target_wheelbase_m", FROM_PHYSICAL_PARAMS),
        uniform_scale=v.get("uniform_scale", AUTO),
        translation_body_m=tuple(float(x) for x in v.get("translation_body_m", (0.0, 0.0, 0.0))),
        rotation_rpy_rad=tuple(float(x) for x in v.get("rotation_rpy_rad", (0.0, 0.0, 0.0))),
        source_path=str(path),
    )
