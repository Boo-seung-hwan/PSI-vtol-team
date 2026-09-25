"""Parameterized MJCF generation for the quadrotor vehicle body.

Builds the MJCF XML programmatically from a ``UAVPhysicalParams`` instance
rather than hand-authoring a static ``models/uav_quadrotor.xml`` -- this is
what lets next week's measured config produce a different, correctly-scaled
model with zero code changes (task requirement: "changing m/CG/inertia/
motor positions must require editing YAML only").

Body-frame / CG relationship (documented per task requirement)
----------------------------------------------------------------
The MuJoCo ``vehicle`` body's local frame origin is defined to coincide with
the vehicle CG. Mass and inertia are given via an explicit ``<inertial pos="0
0 0" .../>`` element at that origin, so:

* ``mass_properties.cg_body_m`` positions the vehicle body's local origin
  relative to whatever "vehicle reference frame" the geometry was measured
  in -- for v0 all other body-frame vectors (motor positions, landing-gear
  points, battery position) are stored ALREADY relative to CG (matching how
  the physical measurement ``r_Mi - r_CG`` is naturally reported), so they
  are used directly as MJCF child-geom offsets with no further translation.
* Because an explicit ``<inertial>`` is present, geom mass/density on the
  body's collision/visual geoms is NOT used to compute body mass or inertia
  (MuJoCo only infers mass/inertia from geoms when no ``<inertial>`` is
  given) -- this is what prevents the measured mass/inertia from being
  silently duplicated by visual geometry, per the task's explicit warning.

Ground contact
--------------
Only the landing-gear points get contact-enabled geoms (small spheres of
radius ``LEG_CONTACT_RADIUS_M``). Physical contact geometry and its simulation
representation are kept apart: with ``landing_gear_points_semantics:
physical_contact`` the config stores where the real gear touches the ground
and the builder derives each sphere centre one radius above it
(``landing_gear_sphere_centers_body_m``), so the CG rests at the physical
``ground_clearance_m``. The legacy ``geom_center`` reading (reference config)
is unchanged. With no per-leg points there are no contact geoms.

The center-body box and the arm/motor markers are visual/collision-disabled
(``contype="0" conaffinity="0"``) -- they do not need to visually reproduce
every component, but the landing-gear contact points must be physically
meaningful (task requirement), and keeping the main body out of collision
avoids spurious contacts overlapping the legs at spawn.

Friction note (documented discrepancy, not silently transferred): the legacy
``LandingConfig.ground_friction_xy`` was a velocity-damping factor applied
manually in Python (``vel[:2] *= (1 - ground_friction_xy)``). MuJoCo now
owns tangential contact resolution natively via geom friction coefficients,
which are NOT numerically equivalent to that damping factor. We reuse
``ground_friction_xy`` as the geom sliding-friction coefficient because it
is the closest available knob, but this is a genuine physical-behavior
change from the legacy model, not a hidden retune -- see ``MUJOCO_MODEL.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from xml.etree import ElementTree as ET
from xml.dom import minidom

import numpy as np

from landing_mujoco.configs.param_schema import LANDING_GEAR_POINTS_PHYSICAL_CONTACT
from landing_mujoco.coordinates.transforms import frd_vector_to_mujoco_body

if TYPE_CHECKING:
    from landing_mujoco.configs.param_schema import Geometry, UAVPhysicalParams

# SIMULATION representation of a landing-gear foot: a sphere of this radius.
# This is NOT a physical parameter of the vehicle (nothing measured it) and is
# never stored in a physical-parameter YAML.
LEG_CONTACT_RADIUS_M = 0.02
ARM_MARKER_RADIUS_M = 0.012
MOTOR_MARKER_RADIUS_M = 0.03
# Geom group for the placeholder markers when the x500 visual shell is on
# (MuJoCo's viewer shows groups 0-2 by default; 3 is hidden but toggleable).
PLACEHOLDER_MARKER_GROUP_WITH_SHELL = 3


def _coord(value: float) -> str:
    """MJCF text for a coordinate: the legacy 4-dp form when it round-trips
    exactly (so existing MJCFs stay byte-identical), full-precision ``repr``
    otherwise (so no measured value is silently rounded to 0.1 mm)."""
    short = f"{value:.4f}"
    return short if float(short) == value else repr(value)


def landing_gear_sphere_centers_body_m(
    geometry: "Geometry", radius: float = LEG_CONTACT_RADIUS_M
) -> Optional[np.ndarray]:
    """Centres of the simulated landing-gear contact spheres, FRD body frame.

    This is the seam between the PHYSICAL contact geometry and its SIMULATION
    representation. ``geometry.landing_gear_points_body_m`` holds either
    (``landing_gear_points_semantics``):

    * ``physical_contact`` -- the points where the real gear touches the ground
      in the landed pose (z = ``ground_clearance_m`` for a level landing). The
      sphere must TOUCH the ground there, so its centre sits one ``radius``
      above the contact point, toward the CG: in FRD (+z down)
      ``centre_z = point_z - radius`` (e.g. 0.241 - 0.020 = 0.221). The offset
      is along body z, i.e. it assumes the gear stands vertically in the
      landed pose. x/y are unchanged.
    * ``geom_center`` (legacy default) -- the points are already sphere
      centres and are returned unchanged, so existing configs keep their exact
      MJCF and resting height.

    Returns ``None`` when no per-leg points exist (per-leg x/y unknown): no
    contact geoms are invented from ``ground_clearance_m`` alone.
    """
    points = geometry.landing_gear_points_body_m
    if points is None:
        return None
    points = np.asarray(points, dtype=np.float64)
    if geometry.landing_gear_points_semantics == LANDING_GEAR_POINTS_PHYSICAL_CONTACT:
        centers = points.copy()
        centers[:, 2] -= float(radius)
        return centers
    return points


def build_mjcf(
    params: "UAVPhysicalParams",
    *,
    physics_dt: float,
    ground_friction_xy: float = 0.9,
    initial_altitude_m: float = 2.0,
    visual_shell=None,
) -> str:
    """Return an MJCF XML string for the vehicle described by ``params``.

    ``initial_altitude_m`` only sets the MJCF's nominal default pose (MuJoCo
    requires *some* starting qpos); the environment overwrites qpos/qvel at
    every ``reset()`` from ``InitialStateSampler`` regardless.

    ``visual_shell`` (an ``landing_mujoco.visualization.x500_shell
    .X500VisualShell`` or None): when given, collision-inert cosmetic geoms
    are APPENDED after every physics element has been emitted. When None
    (default) the output is exactly the physics-only MJCF.
    """
    mp = params.mass_properties
    geom = params.geometry
    motors = params.motors

    if mp.mass_kg is None or mp.ixx is None or mp.iyy is None or mp.izz is None:
        raise ValueError(
            "build_mjcf requires mass_kg and Ixx/Iyy/Izz to be resolved "
            "(load_uav_params should have raised MissingMeasurementError "
            "already if they were not)"
        )

    mujoco_el = ET.Element("mujoco", {"model": "uav_quadrotor"})
    ET.SubElement(mujoco_el, "compiler", {"angle": "radian"})
    ET.SubElement(
        mujoco_el,
        "option",
        {
            "timestep": f"{physics_dt:.6f}",
            "gravity": "0 0 -9.80665",
            "integrator": "RK4",
        },
    )

    default = ET.SubElement(mujoco_el, "default")
    ET.SubElement(
        default,
        "geom",
        {"contype": "0", "conaffinity": "0", "friction": "1.0 0.005 0.0001"},
    )

    worldbody = ET.SubElement(mujoco_el, "worldbody")
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "ground",
            "type": "plane",
            "size": "50 50 0.1",
            "pos": "0 0 0",
            "contype": "1",
            "conaffinity": "1",
            "friction": f"{ground_friction_xy:.4f} 0.005 0.0001",
            "rgba": "0.5 0.5 0.5 1",
        },
    )

    vehicle = ET.SubElement(
        worldbody,
        "body",
        {"name": "vehicle", "pos": f"0 0 {initial_altitude_m:.3f}"},
    )
    ET.SubElement(vehicle, "freejoint", {"name": "vehicle_free"})
    # Frame note for the inertia written below: the config tensor is in the
    # FRD policy body frame, MuJoCo's local body frame is FLU, and the two are
    # related by R = diag(1, -1, -1). For a DIAGONAL tensor R I R^T == I, so
    # Ixx/Iyy/Izz pass through unconverted. This is true ONLY because the
    # products of inertia are (assumed) zero: a non-zero Ixy/Ixz would flip
    # sign (Ixy -> -Ixy, Ixz -> -Ixz, Iyz -> +Iyz) and would need converting.
    ET.SubElement(
        vehicle,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": f"{mp.mass_kg:.6f}",
            "diaginertia": f"{mp.ixx:.8f} {mp.iyy:.8f} {mp.izz:.8f}",
        },
    )

    # With the x500 visual shell enabled, the simple placeholder markers below
    # (collision-inert and massless under the explicit <inertial>) move to a
    # geom group hidden by default, so the default view shows the shell plus
    # the PHYSICAL landing-gear contact geoms (which are never modified).
    # Without a shell, nothing here changes.
    marker_group = {"group": str(PLACEHOLDER_MARKER_GROUP_WITH_SHELL)} if visual_shell is not None else {}

    # Center body -- visual/collision-disabled box sized from frame
    # footprint/height. Purely cosmetic; mass comes from <inertial> above.
    if geom.frame_footprint_m is not None and geom.frame_height_m is not None:
        lx, ly = geom.frame_footprint_m
        lz = geom.frame_height_m
        ET.SubElement(
            vehicle,
            "geom",
            {
                "name": "body_box",
                "type": "box",
                "size": f"{lx / 2:.4f} {ly / 2:.4f} {lz / 2:.4f}",
                "pos": "0 0 0",
                "rgba": "0.2 0.2 0.8 0.6",
                **marker_group,
            },
        )

    # Arm markers (visual only) + motor markers (visual only) at each
    # measured-or-provisional motor position (already relative to CG).
    # Config vectors are in the FRD policy body frame; MJCF geom pos/fromto
    # are in MuJoCo's local (FLU) body frame, so every vector is passed
    # through the one coordinate adapter before being written out.
    if motors.positions_body_m:
        for name in sorted(motors.positions_body_m.keys()):
            x, y, z = (
                float(v)
                for v in frd_vector_to_mujoco_body(motors.positions_body_m[name])
            )
            # Motor coordinates must reach the compiled model without the
            # 0.1 mm rounding of the legacy ``.4f`` text (measured a =
            # 0.25455844... would become 0.2546, i.e. a 0.36006 m radius).
            # ``_coord`` keeps the legacy text whenever it is lossless, so
            # existing configs stay byte-identical. These markers are
            # massless and collision-disabled: no dynamics are affected.
            ET.SubElement(
                vehicle,
                "geom",
                {
                    "name": f"arm_{name}",
                    "type": "capsule",
                    "fromto": f"0 0 0 {_coord(x)} {_coord(y)} {_coord(z)}",
                    "size": f"{ARM_MARKER_RADIUS_M:.4f}",
                    "rgba": "0.1 0.1 0.1 1",
                    **marker_group,
                },
            )
            ET.SubElement(
                vehicle,
                "geom",
                {
                    "name": f"motor_{name}",
                    "type": "sphere",
                    "pos": f"{_coord(x)} {_coord(y)} {_coord(z)}",
                    "size": f"{MOTOR_MARKER_RADIUS_M:.4f}",
                    "rgba": "0.8 0.1 0.1 1",
                    **marker_group,
                },
            )

    # Landing-gear contact spheres -- the only geoms with collision enabled.
    # ``sphere_centers`` are SIMULATION positions, derived from the physical
    # contact points by ``landing_gear_sphere_centers_body_m``.
    sphere_centers = landing_gear_sphere_centers_body_m(geom)
    if sphere_centers is not None:
        for i, point in enumerate(sphere_centers):
            x, y, z = (float(v) for v in frd_vector_to_mujoco_body(point))
            ET.SubElement(
                vehicle,
                "geom",
                {
                    "name": f"leg_{i}",
                    "type": "sphere",
                    # Same lossless-or-legacy rule as the motor markers:
                    # landing_reference_point_body_m() reads the full-precision
                    # Python array, so the collision geoms must not be rounded
                    # to 0.1 mm behind its back.
                    "pos": f"{_coord(x)} {_coord(y)} {_coord(z)}",
                    "size": f"{LEG_CONTACT_RADIUS_M:.4f}",
                    "contype": "1",
                    "conaffinity": "1",
                    "friction": f"{ground_friction_xy:.4f} 0.005 0.0001",
                    "rgba": "0.9 0.6 0.1 1",
                },
            )

    if visual_shell is not None:
        visual_shell.append_to_mjcf(mujoco_el, vehicle)

    xml_bytes = ET.tostring(mujoco_el, encoding="unicode")
    return minidom.parseString(xml_bytes).toprettyxml(indent="  ")
