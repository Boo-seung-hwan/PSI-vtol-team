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

* ``MEASURED_VEHICLE`` (``ugrp_vehicle_measured.yaml``) — the real research
  configuration. Populated with mass, full motor geometry, CG, battery
  position, spin layout and component specs (2026-09-16) and the measured
  diagonal inertia tensor Ixx/Iyy/Izz (2026-09-19); max collective thrust and
  the identified closed-loop response (tau_*, actuator delay) are still
  ``null``. Loading this file fails fast (``MissingMeasurementError``) and
  lists exactly which measurements are still missing. The inertia's
  provenance (bifilar reduction, per-axis std/variance/n, outlier policy,
  ASSUMED_ZERO_FOR_V0 products of inertia) lives in the free-form
  ``mass_properties.meta`` dict: it is recorded, and consumed by nothing.

  That file also carries a ``raw_measurements`` block (see
  ``RawMeasurements``) which this module parses but NOTHING in
  ``landing_mujoco/dynamics/`` consumes. It preserves the as-measured
  numbers and the definition of the datum they were taken in, so that every
  converted body-frame coordinate stays independently recomputable from its
  source. It is optional, so configs that omit it
  (``tarot680b_reference.yaml``) parse unchanged.

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
import math
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


# How ``Geometry.landing_gear_points_body_m`` is to be read.
#
#   geom_center       legacy / reference-config reading: each point IS the
#                     centre of the simulated contact sphere. The physical
#                     contact surface is then one sphere radius BELOW the
#                     point, so a point at z = h rests the CG at h + radius.
#   physical_contact  each point is where the real gear touches the ground in
#                     the landed pose (PHYSICAL). The sphere centre is a
#                     SIMULATION quantity derived by the MJCF builder
#                     (``landing_gear_sphere_centers_body_m``) and never
#                     stored: centre_z = point_z - sphere_radius (FRD, +z down).
LANDING_GEAR_POINTS_GEOM_CENTER = "geom_center"
LANDING_GEAR_POINTS_PHYSICAL_CONTACT = "physical_contact"
LANDING_GEAR_POINT_SEMANTICS = (
    LANDING_GEAR_POINTS_GEOM_CENTER,
    LANDING_GEAR_POINTS_PHYSICAL_CONTACT,
)


@dataclass
class Geometry:
    wheelbase_m: Optional[float] = None
    arm_length_m: Optional[float] = None
    # Arm angle away from +x (forward), degrees. 45.0 == symmetric X-frame.
    # Stays None unless a specific vehicle's frame symmetry is confirmed --
    # it is the documented basis for any derived motor XY.
    arm_angle_deg: Optional[float] = None
    frame_footprint_m: Optional[tuple[float, float]] = None
    frame_height_m: Optional[float] = None
    # PHYSICAL: vertical distance from the body origin (the CG) DOWN to the
    # ground / landing-contact plane in the normal landed pose, i.e. the body
    # frame (FRD, +z down) z of the ground plane. Never a simulation-geom
    # offset: the contact-sphere centre is ``ground_clearance_m - radius`` and
    # is derived by the MJCF builder, not stored here.
    ground_clearance_m: Optional[float] = None
    landing_gear_points_body_m: Optional[np.ndarray] = None  # (N,3)
    # How to read ``landing_gear_points_body_m`` (see the constants above).
    # The default is the legacy reading so configs that predate this field
    # (tarot680b_reference.yaml) keep their exact behavior.
    landing_gear_points_semantics: str = LANDING_GEAR_POINTS_GEOM_CENTER
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.landing_gear_points_semantics not in LANDING_GEAR_POINT_SEMANTICS:
            raise ValueError(
                "landing_gear_points_semantics must be one of "
                f"{LANDING_GEAR_POINT_SEMANTICS}, got "
                f"{self.landing_gear_points_semantics!r}"
            )


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
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatterySpec:
    model: Optional[str] = None
    nominal_voltage_v: Optional[float] = None
    mass_kg: Optional[float] = None
    position_body_m: Optional[np.ndarray] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ESCSpec:
    model: Optional[str] = None
    protocol: Optional[str] = None
    max_current_a: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)


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
class RawMeasurements:
    """As-measured numbers in a physical measurement datum that has NOT yet
    been reconciled with the FRD body frame.

    This block is deliberately separate from every ``*_body_m`` field: it is
    read by nothing in ``landing_mujoco/dynamics/`` and never reaches MuJoCo.
    It preserves the measurements verbatim so that every converted body-frame
    coordinate stays independently recomputable from its source.

    ``datum_status`` gates conversion. While it is ``PENDING_DEFINITION``,
    only ABSOLUTE separations are meaningful. Once ``RESOLVED``, the datum
    origin and up-axis are recorded and ``body_z_from_raw_height`` defines
    the conversion.

    The ``abs_*`` accessors are kept AFTER resolution on purpose: they are
    computed straight from the raw pair, so they remain an independent check
    that a stored body-frame magnitude still matches its measurement. A
    signed accessor derived from the same conversion would prove nothing.
    """

    datum_status: str = "PENDING_DEFINITION"
    datum_note: Optional[str] = None
    datum_origin: Optional[str] = None
    datum_up_axis: Optional[str] = None
    datum_resolved_date: Optional[str] = None
    cg_raw_m: Optional[np.ndarray] = None
    motor_plane_raw_z_m: Optional[float] = None
    battery_center_raw_m: Optional[np.ndarray] = None
    motor_radial_distance_m: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def datum_resolved(self) -> bool:
        """True only once the datum origin and +Z direction are recorded.
        Guard any datum->FRD conversion on this."""
        return self.datum_status == "RESOLVED"

    def body_z_from_raw_height(self, raw_height_m: float) -> float:
        """Convert a raw-datum height into the FRD body-frame z.

        Valid only for a ``+z_up`` datum measured against a surface, with the
        body origin at the CG: the conversion is an origin shift to the CG
        plus a Z sign flip, i.e. ``z_body = cg_raw_z - raw_z``. A positive
        result therefore means BELOW the CG (FRD +Z is down).

        Raises if the datum is unresolved, if its up-axis is not ``+z_up``,
        or if the CG reference is missing -- it never guesses a sign.
        """
        if not self.datum_resolved:
            raise ValueError(
                f"cannot convert: datum_status is {self.datum_status!r}, not "
                "'RESOLVED' -- the datum origin and +Z direction must be "
                "recorded before any raw height becomes a body-frame value"
            )
        if self.datum_up_axis != "+z_up":
            raise ValueError(
                f"unsupported datum_up_axis {self.datum_up_axis!r}; this "
                "conversion is defined only for '+z_up'"
            )
        if self.cg_raw_m is None:
            raise ValueError("cannot convert: raw_measurements.cg_raw_m is null")
        return float(self.cg_raw_m[2]) - float(raw_height_m)

    def abs_motor_plane_to_cg_m(self) -> Optional[float]:
        """|motor_plane_z - cg_z|, computed from the raw pair."""
        if self.motor_plane_raw_z_m is None or self.cg_raw_m is None:
            return None
        return abs(float(self.motor_plane_raw_z_m) - float(self.cg_raw_m[2]))

    def abs_cg_to_battery_m(self) -> Optional[float]:
        """|cg_z - battery_center_z|, computed from the raw pair."""
        if self.battery_center_raw_m is None or self.cg_raw_m is None:
            return None
        return abs(float(self.cg_raw_m[2]) - float(self.battery_center_raw_m[2]))


def motor_xy_from_radial_distance(
    radial_distance_m: float, arm_angle_deg: float
) -> dict[str, np.ndarray]:
    """Motor XY in the FRD body frame (+x forward, +y right, so left is -y)
    for a symmetric X-layout quad, given the center-to-motor radial distance.

    ``radial_distance_m`` is the CENTER-to-motor-center distance (what
    ``geometry.arm_length_m`` holds), NOT the opposite-motor diagonal and NOT
    the x/y component. Each returned point satisfies
    ``hypot(x, y) == radial_distance_m`` exactly.

    ``arm_angle_deg`` is the angle of each arm away from the +x (forward)
    axis, and is deliberately REQUIRED with no default: 45 degrees is an
    assumption about a specific airframe, and a default would let callers
    inherit it silently. This is a geometric helper, not a claim that any
    particular vehicle has that arm angle.
    """
    r = float(radial_distance_m)
    theta = math.radians(float(arm_angle_deg))
    x, y = r * math.cos(theta), r * math.sin(theta)
    return {
        "FL": np.array([x, -y], dtype=np.float64),
        "FR": np.array([x, y], dtype=np.float64),
        "RL": np.array([-x, -y], dtype=np.float64),
        "RR": np.array([-x, y], dtype=np.float64),
    }


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
    # Optional provenance block. Defaulted so existing configs that omit it
    # (tarot680b_reference.yaml) parse unchanged, and so the single
    # construction site below stays the only one that must know about it.
    raw_measurements: RawMeasurements = field(default_factory=RawMeasurements)

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
    (a separate scalar) -- it is derived explicitly from the actual per-leg
    contact geometry (``landing_gear_points_body_m``) whenever that is
    available, by averaging the per-leg points. This is the single point a
    symmetric (or near-symmetric) gear layout rests on.

    What that point IS depends on ``geometry.landing_gear_points_semantics``:
    with ``physical_contact`` it is the physical touchdown point (where the
    real gear meets the ground, so ``altitude_agl`` reaches 0 at true
    contact); with the legacy ``geom_center`` it is the mean of the simulated
    sphere CENTRES, one sphere radius above the physical contact surface.

    Falls back to ``[0, 0, ground_clearance_m]`` ONLY when per-leg points are
    not given -- a distinct, clearly separate code path, not a silent
    assumption that the two are the same value. ``ground_clearance_m`` is the
    PHYSICAL CG-to-ground distance, so this fallback is a physical touchdown
    point. Its x/y are ASSUMED centred under the CG (per-leg x/y unknown); that
    assumption only affects the lateral offset of the reference point under
    tilt. It creates NO contact geometry: ``mjcf_builder`` builds collision
    geoms from ``landing_gear_points_body_m`` only.
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
        arm_angle_deg=raw.get("arm_angle_deg"),
        frame_footprint_m=tuple(footprint) if footprint else None,
        frame_height_m=raw.get("frame_height_m"),
        ground_clearance_m=raw.get("ground_clearance_m"),
        landing_gear_points_body_m=(
            np.asarray(landing_gear, dtype=np.float64) if landing_gear else None
        ),
        landing_gear_points_semantics=raw.get(
            "landing_gear_points_semantics", LANDING_GEAR_POINTS_GEOM_CENTER
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
        meta=raw.get("meta", {}),
    )


def _parse_battery(raw: dict) -> BatterySpec:
    pos = raw.get("position_body_m")
    return BatterySpec(
        model=raw.get("model"),
        nominal_voltage_v=raw.get("nominal_voltage_v"),
        mass_kg=raw.get("mass_kg"),
        position_body_m=_vec3(pos) if pos else None,
        meta=raw.get("meta", {}),
    )


def _parse_esc(raw: dict) -> ESCSpec:
    return ESCSpec(
        model=raw.get("model"),
        protocol=raw.get("protocol"),
        max_current_a=raw.get("max_current_a"),
        meta=raw.get("meta", {}),
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


def _parse_raw_measurements(raw: dict) -> RawMeasurements:
    return RawMeasurements(
        datum_status=raw.get("datum_status", "PENDING_DEFINITION"),
        datum_note=raw.get("datum_note"),
        datum_origin=raw.get("datum_origin"),
        datum_up_axis=raw.get("datum_up_axis"),
        datum_resolved_date=str(raw["datum_resolved_date"]) if raw.get("datum_resolved_date") else None,
        cg_raw_m=_vec3(raw.get("cg_raw_m")),
        motor_plane_raw_z_m=raw.get("motor_plane_raw_z_m"),
        battery_center_raw_m=_vec3(raw.get("battery_center_raw_m")),
        motor_radial_distance_m=raw.get("motor_radial_distance_m"),
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
        raw_measurements=_parse_raw_measurements(raw.get("raw_measurements", {})),
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
