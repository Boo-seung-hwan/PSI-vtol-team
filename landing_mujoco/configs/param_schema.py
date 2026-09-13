"""UAV physical-parameter schema, provenance, and fail-fast validation.

This module is the single place that knows how to load a
``landing_mujoco/configs/*.yaml`` physical-parameter file and turn it into a
validated ``UAVPhysicalParams`` object. It implements the
PROVISIONAL_REFERENCE / MEASURED_VEHICLE distinction requested for this
week's work:

* ``PROVISIONAL_REFERENCE`` (``tarot680b_reference.yaml``) — manufacturer /
  derived / synthetic values, runnable today, used ONLY to exercise the
  MuJoCo software pipeline (coordinate transforms, actuator plumbing,
  contact, observation contract). Every non-informational field in this file
  is tagged with ``source`` / ``confidence`` / ``replace_before_real_training``
  metadata in the YAML.

* ``MEASURED_VEHICLE`` (``ugrp_vehicle_measured.yaml``) — the future research
  configuration. Required fields are ``null`` until the real UAV is measured.
  Loading this file fails fast (``MissingMeasurementError``) and lists
  exactly which measurements are still missing, matching the format asked
  for in the task spec.

No numeric value is ever invented by this module. The only computed value is
the provisional inertia tensor, and only when ``inertia_estimation_mode:
auto`` is explicitly set (see ``landing_mujoco/dynamics/inertia_estimation.py``)
-- this is a clearly-tagged geometric approximation, never presented as
measured.

Swapping PROVISIONAL_REFERENCE for MEASURED_VEHICLE (once real numbers exist)
requires editing YAML only -- nothing here or in ``landing_mujoco/dynamics/``
needs to change.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


class ParameterSet(str, enum.Enum):
    PROVISIONAL_REFERENCE = "PROVISIONAL_REFERENCE"
    MEASURED_VEHICLE = "MEASURED_VEHICLE"


class MissingMeasurementError(RuntimeError):
    """Raised when a config declares fields required to run the MuJoCo plant,
    but one or more of them is still ``null``.

    ``missing`` is the ordered list of missing field labels, in the exact
    format used for logging/CLI output.
    """

    def __init__(self, parameter_set: "ParameterSet", missing: list[str], source_path: str):
        self.parameter_set = parameter_set
        self.missing = missing
        self.source_path = source_path
        label = {
            ParameterSet.MEASURED_VEHICLE: "measured",
            ParameterSet.PROVISIONAL_REFERENCE: "provisional reference",
        }[parameter_set]
        lines = [
            f"ERROR: {label} vehicle model incomplete",
            f"Config: {source_path}",
            "",
            "Missing:",
        ] + [f"- {name}" for name in missing]
        super().__init__("\n".join(lines))


@dataclass
class MassProperties:
    mass_kg: Optional[float] = None
    cg_body_m: Optional[np.ndarray] = None
    ixx: Optional[float] = None
    iyy: Optional[float] = None
    izz: Optional[float] = None
    inertia_estimation_mode: str = "manual"  # "auto" | "manual"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Geometry:
    wheelbase_m: Optional[float] = None
    arm_length_m: Optional[float] = None
    frame_footprint_m: Optional[tuple[float, float]] = None
    frame_height_m: Optional[float] = None
    ground_clearance_m: Optional[float] = None
    landing_gear_points_body_m: Optional[np.ndarray] = None  # (N,3)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotorSpec:
    positions_body_m: Optional[dict[str, np.ndarray]] = None  # {"M1": (3,), ...}
    spin_directions: Optional[dict[str, str]] = None  # {"M1": "CW"|"CCW", ...}
    mass_kg_each: Optional[float] = None
    model: Optional[str] = None
    kv_rpm_per_v: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def ordered_names(self) -> list[str]:
        if not self.positions_body_m:
            return []
        return sorted(self.positions_body_m.keys())


@dataclass
class PropellerSpec:
    model: Optional[str] = None
    diameter_m: Optional[float] = None
    pitch_m: Optional[float] = None
    blade_count: Optional[int] = None


@dataclass
class BatterySpec:
    model: Optional[str] = None
    nominal_voltage_v: Optional[float] = None
    mass_kg: Optional[float] = None
    position_body_m: Optional[np.ndarray] = None


@dataclass
class ESCSpec:
    model: Optional[str] = None
    protocol: Optional[str] = None
    max_current_a: Optional[float] = None


@dataclass
class ThrustSpec:
    max_collective_thrust_n: Optional[float] = None
    thrust_to_weight_max: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class IdentifiedResponse:
    tau_roll_s: Optional[float] = None
    tau_pitch_s: Optional[float] = None
    tau_thrust_s: Optional[float] = None
    tau_yaw_s: Optional[float] = None
    delay_s: Optional[float] = None
    k_roll: Optional[float] = None
    k_pitch: Optional[float] = None
    k_thrust: Optional[float] = None
    k_yaw: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def gain(self, axis: str) -> float:
        """Return the identified gain for ``axis`` in {roll,pitch,thrust,yaw},
        defaulting to unity when not specified (never fabricated -- unity gain
        is the neutral/no-op default, not an invented number)."""
        value = getattr(self, f"k_{axis}")
        return 1.0 if value is None else float(value)


@dataclass
class UAVPhysicalParams:
    parameter_set: ParameterSet
    vehicle_name: str
    mass_properties: MassProperties
    geometry: Geometry
    motors: MotorSpec
    propellers: PropellerSpec
    battery: BatterySpec
    esc: ESCSpec
    thrust: ThrustSpec
    identified_response: IdentifiedResponse
    source_path: str

    def hover_thrust_n(self, gravity_mps2: float) -> float:
        return float(self.mass_properties.mass_kg) * float(gravity_mps2)

    def thrust_margin(self, gravity_mps2: float) -> float:
        """T_max / (m*g). Raises if either is missing -- caller must have
        already validated completeness."""
        return float(self.thrust.max_collective_thrust_n) / self.hover_thrust_n(gravity_mps2)

    def summary(self) -> dict[str, Any]:
        """Compact, log/info-friendly summary. Always includes which
        parameter set is active -- required by the task spec so every
        MuJoCo evaluation visibly reports PROVISIONAL vs MEASURED."""
        return {
            "parameter_set": self.parameter_set.value,
            "vehicle_name": self.vehicle_name,
            "source_path": self.source_path,
            "mass_kg": self.mass_properties.mass_kg,
            "max_collective_thrust_n": self.thrust.max_collective_thrust_n,
        }


def landing_reference_point_body_m(geometry: Geometry) -> np.ndarray:
    """Nominal landing-gear touchdown reference point, in the FRD body
    frame, relative to CG: ``r_landing^B``.

    Deliberately does NOT assume this equals ``geometry.ground_clearance_m``
    (a separate, informational scalar) -- it is derived explicitly from the
    actual per-leg contact geometry (``landing_gear_points_body_m``, the
    same points ``mjcf_builder`` uses for MuJoCo collision geoms) whenever
    that is available, by averaging the per-leg points. This is the single
    point a symmetric (or near-symmetric) gear layout rests on.

    Falls back to ``[0, 0, ground_clearance_m]`` (x/y assumed centered)
    ONLY when per-leg points are not given -- a distinct, clearly separate
    code path, not a silent assumption that the two are the same value.
    """
    if geometry.landing_gear_points_body_m is not None and len(geometry.landing_gear_points_body_m) > 0:
        return np.mean(geometry.landing_gear_points_body_m, axis=0).astype(np.float64)
    if geometry.ground_clearance_m is not None:
        return np.array([0.0, 0.0, float(geometry.ground_clearance_m)], dtype=np.float64)
    raise ValueError(
        "cannot determine a landing reference point: geometry has neither "
        "landing_gear_points_body_m nor ground_clearance_m"
    )


# ---------------------------------------------------------------------------
# Required-for-runtime fields.
#
# This list intentionally matches the exact set named in the task's own
# "ERROR: measured vehicle model incomplete" example: total_mass_kg,
# cg_body_m, Ixx, Iyy, Izz, motor_positions_body_m, max_collective_thrust_n,
# tau_roll_s, tau_pitch_s, tau_thrust_s, actuator_delay_s. Motor spin
# directions, K_*, and propeller/battery/ESC specs are informational /
# default-to-neutral (unity gain) and are NOT required to run the v0
# IdentifiedWrenchActuation model -- they matter for a future rotor-level
# backend, not this one.
# ---------------------------------------------------------------------------
_REQUIRED_FIELDS: list[tuple[str, Any]] = [
    ("total_mass_kg", lambda p: p.mass_properties.mass_kg),
    ("cg_body_m", lambda p: p.mass_properties.cg_body_m),
    ("Ixx", lambda p: p.mass_properties.ixx),
    ("Iyy", lambda p: p.mass_properties.iyy),
    ("Izz", lambda p: p.mass_properties.izz),
    ("motor_positions_body_m", lambda p: p.motors.positions_body_m),
    ("max_collective_thrust_n", lambda p: p.thrust.max_collective_thrust_n),
    ("tau_roll_s", lambda p: p.identified_response.tau_roll_s),
    ("tau_pitch_s", lambda p: p.identified_response.tau_pitch_s),
    ("tau_thrust_s", lambda p: p.identified_response.tau_thrust_s),
    ("actuator_delay_s", lambda p: p.identified_response.delay_s),
]


def missing_required_fields(params: UAVPhysicalParams) -> list[str]:
    """Return the ordered list of required-field labels that are still None."""
    missing = []
    for label, getter in _REQUIRED_FIELDS:
        try:
            value = getter(params)
        except AttributeError:
            value = None
        if value is None:
            missing.append(label)
    return missing


def _vec3(raw: Optional[list]) -> Optional[np.ndarray]:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"expected a 3-vector, got shape {arr.shape}: {raw}")
    return arr


def _parse_mass_properties(raw: dict) -> MassProperties:
    inertia_raw = raw.get("inertia_kgm2") or {}
    return MassProperties(
        mass_kg=raw.get("mass_kg"),
        cg_body_m=_vec3(raw.get("cg_body_m")),
        ixx=inertia_raw.get("ixx"),
        iyy=inertia_raw.get("iyy"),
        izz=inertia_raw.get("izz"),
        inertia_estimation_mode=raw.get("inertia_estimation_mode", "manual"),
        meta=raw.get("meta", {}),
    )


def _parse_geometry(raw: dict) -> Geometry:
    footprint = raw.get("frame_footprint_m")
    landing_gear = raw.get("landing_gear_points_body_m")
    return Geometry(
        wheelbase_m=raw.get("wheelbase_m"),
        arm_length_m=raw.get("arm_length_m"),
        frame_footprint_m=tuple(footprint) if footprint else None,
        frame_height_m=raw.get("frame_height_m"),
        ground_clearance_m=raw.get("ground_clearance_m"),
        landing_gear_points_body_m=(
            np.asarray(landing_gear, dtype=np.float64) if landing_gear else None
        ),
        meta=raw.get("meta", {}),
    )


def _parse_motors(raw: dict) -> MotorSpec:
    positions_raw = raw.get("positions_body_m")
    positions = None
    if positions_raw:
        positions = {name: _vec3(pos) for name, pos in positions_raw.items()}
    return MotorSpec(
        positions_body_m=positions,
        spin_directions=raw.get("spin_directions"),
        mass_kg_each=raw.get("mass_kg_each"),
        model=raw.get("model"),
        kv_rpm_per_v=raw.get("kv_rpm_per_v"),
        meta=raw.get("meta", {}),
    )


def _parse_propellers(raw: dict) -> PropellerSpec:
    return PropellerSpec(
        model=raw.get("model"),
        diameter_m=raw.get("diameter_m"),
        pitch_m=raw.get("pitch_m"),
        blade_count=raw.get("blade_count"),
    )


def _parse_battery(raw: dict) -> BatterySpec:
    pos = raw.get("position_body_m")
    return BatterySpec(
        model=raw.get("model"),
        nominal_voltage_v=raw.get("nominal_voltage_v"),
        mass_kg=raw.get("mass_kg"),
        position_body_m=_vec3(pos) if pos else None,
    )


def _parse_esc(raw: dict) -> ESCSpec:
    return ESCSpec(
        model=raw.get("model"),
        protocol=raw.get("protocol"),
        max_current_a=raw.get("max_current_a"),
    )


def _parse_thrust(raw: dict) -> ThrustSpec:
    return ThrustSpec(
        max_collective_thrust_n=raw.get("max_collective_thrust_n"),
        thrust_to_weight_max=raw.get("thrust_to_weight_max"),
        meta=raw.get("meta", {}),
    )


def _parse_identified_response(raw: dict) -> IdentifiedResponse:
    return IdentifiedResponse(
        tau_roll_s=raw.get("tau_roll_s"),
        tau_pitch_s=raw.get("tau_pitch_s"),
        tau_thrust_s=raw.get("tau_thrust_s"),
        tau_yaw_s=raw.get("tau_yaw_s"),
        delay_s=raw.get("delay_s"),
        k_roll=raw.get("K_roll"),
        k_pitch=raw.get("K_pitch"),
        k_thrust=raw.get("K_thrust"),
        k_yaw=raw.get("K_yaw"),
        meta=raw.get("meta", {}),
    )


def load_uav_params(path: str | Path, *, validate: bool = True) -> UAVPhysicalParams:
    """Load and (by default) validate a UAV physical-parameter YAML file.

    If ``mass_properties.inertia_estimation_mode == "auto"`` and any of
    Ixx/Iyy/Izz is still null, this computes a provisional inertia tensor via
    ``landing_mujoco.dynamics.inertia_estimation`` (clearly tagged, never
    called "measured") before validation runs.

    If ``validate`` is True (the default) and any required field (see
    ``_REQUIRED_FIELDS``) is still ``None`` after that, raises
    ``MissingMeasurementError`` listing exactly which fields are missing --
    for BOTH parameter sets (a mis-populated reference config is a config bug
    just as much as an incomplete measured one).
    """
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    parameter_set = ParameterSet(raw["parameter_set"])

    params = UAVPhysicalParams(
        parameter_set=parameter_set,
        vehicle_name=raw.get("vehicle_name", "unknown"),
        mass_properties=_parse_mass_properties(raw.get("mass_properties", {})),
        geometry=_parse_geometry(raw.get("geometry", {})),
        motors=_parse_motors(raw.get("motors", {})),
        propellers=_parse_propellers(raw.get("propellers", {})),
        battery=_parse_battery(raw.get("battery", {})),
        esc=_parse_esc(raw.get("esc", {})),
        thrust=_parse_thrust(raw.get("thrust", {})),
        identified_response=_parse_identified_response(raw.get("identified_response", {})),
        source_path=str(path),
    )

    mp = params.mass_properties
    if mp.inertia_estimation_mode == "auto" and (
        mp.ixx is None or mp.iyy is None or mp.izz is None
    ):
        # Deferred import to avoid a hard dependency for callers that only
        # need the schema (e.g. the CLI missing-field check).
        from landing_mujoco.dynamics.inertia_estimation import build_provisional_inertia

        ixx, iyy, izz = build_provisional_inertia(params)
        mp.ixx = ixx if mp.ixx is None else mp.ixx
        mp.iyy = iyy if mp.iyy is None else mp.iyy
        mp.izz = izz if mp.izz is None else mp.izz

    if validate:
        missing = missing_required_fields(params)
        if missing:
            raise MissingMeasurementError(parameter_set, missing, str(path))

    return params
