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
Only the landing-gear points get contact-enabled geoms (small spheres).
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

from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET
from xml.dom import minidom

from landing_mujoco.coordinates.transforms import frd_vector_to_mujoco_body

if TYPE_CHECKING:
    from landing_mujoco.configs.param_schema import UAVPhysicalParams

LEG_CONTACT_RADIUS_M = 0.02
ARM_MARKER_RADIUS_M = 0.012
MOTOR_MARKER_RADIUS_M = 0.03
# Geom group for the placeholder markers when the x500 visual shell is on
# (MuJoCo's viewer shows groups 0-2 by default; 3 is hidden but toggleable).
PLACEHOLDER_MARKER_GROUP_WITH_SHELL = 3


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
            ET.SubElement(
                vehicle,
                "geom",
                {
                    "name": f"arm_{name}",
                    "type": "capsule",
                    "fromto": f"0 0 0 {x:.4f} {y:.4f} {z:.4f}",
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
                    "pos": f"{x:.4f} {y:.4f} {z:.4f}",
                    "size": f"{MOTOR_MARKER_RADIUS_M:.4f}",
                    "rgba": "0.8 0.1 0.1 1",
                    **marker_group,
                },
            )

    # Landing-gear contact points -- the only geoms with collision enabled.
    if geom.landing_gear_points_body_m is not None:
        for i, point in enumerate(geom.landing_gear_points_body_m):
            x, y, z = (float(v) for v in frd_vector_to_mujoco_body(point))
            ET.SubElement(
                vehicle,
                "geom",
                {
                    "name": f"leg_{i}",
                    "type": "sphere",
                    "pos": f"{x:.4f} {y:.4f} {z:.4f}",
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
