"""Provisional (NOT measured) inertia-tensor estimation.

This is a clearly-documented geometric approximation used ONLY when a
physical-parameter config sets ``mass_properties.inertia_estimation_mode:
auto`` (currently only ``tarot680b_reference.yaml``). It is never used for
``ugrp_vehicle_measured.yaml`` -- that file uses ``inertia_estimation_mode:
manual`` and requires literal ``Ixx/Iyy/Izz`` once measured.

Model (per the task spec): a two-body point-mass decomposition about the CG,

    J = J_center + sum_i m_i [ (r_i^T r_i) I - r_i r_i^T ]

* ``J_center`` -- the "everything that isn't a motor" mass (frame, avionics,
  battery, wiring) approximated as a uniform-density solid cuboid using
  ``geometry.frame_footprint_m`` (Lx, Ly) and ``geometry.frame_height_m``
  (Lz), centered at the CG. This is a crude shape approximation, not a CAD
  model.
* Four (or however many are configured) motor/arm point masses at their
  measured-or-provisional ``motors.positions_body_m`` (already expressed
  relative to CG), each of mass ``motors.mass_kg_each``.

Only the diagonal terms are returned (Ixx, Iyy, Izz) -- consistent with
CLAUDE.md's instruction not to introduce arbitrary off-diagonal products of
inertia. For a perfectly symmetric X-quad motor layout the products of
inertia from the point-mass sum cancel exactly by construction; this
function does not attempt to compute or report them.

The result MUST always be tagged ``source: temporary_geometric_approximation``,
``confidence: low`` wherever it is stored/logged, and MUST NOT be described
as measured inertia.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from landing_mujoco.configs.param_schema import UAVPhysicalParams


def build_provisional_inertia(params: "UAVPhysicalParams") -> tuple[float, float, float]:
    """Return a provisional (Ixx, Iyy, Izz) in kg*m^2. Raises ValueError if
    the geometry/motor fields this estimate needs are missing -- it does not
    silently fall back to invented numbers."""
    mp = params.mass_properties
    motors = params.motors
    geom = params.geometry

    if mp.mass_kg is None:
        raise ValueError("cannot estimate inertia: mass_properties.mass_kg is null")
    if not motors.positions_body_m:
        raise ValueError("cannot estimate inertia: motors.positions_body_m is empty/null")
    if motors.mass_kg_each is None:
        raise ValueError("cannot estimate inertia: motors.mass_kg_each is null")
    if geom.frame_footprint_m is None or geom.frame_height_m is None:
        raise ValueError(
            "cannot estimate inertia: geometry.frame_footprint_m / "
            "frame_height_m is null"
        )

    n_motors = len(motors.positions_body_m)
    motor_mass_total = n_motors * float(motors.mass_kg_each)
    center_mass = float(mp.mass_kg) - motor_mass_total
    if center_mass <= 0.0:
        raise ValueError(
            f"cannot estimate inertia: motor_mass_total ({motor_mass_total:.3f} kg) "
            f">= total mass ({mp.mass_kg:.3f} kg)"
        )

    lx, ly = geom.frame_footprint_m
    lz = geom.frame_height_m

    # Uniform-density solid cuboid about its own center (the "center body").
    ixx = (1.0 / 12.0) * center_mass * (ly**2 + lz**2)
    iyy = (1.0 / 12.0) * center_mass * (lx**2 + lz**2)
    izz = (1.0 / 12.0) * center_mass * (lx**2 + ly**2)

    # Parallel-axis point-mass contribution of each motor, diagonal terms only.
    for pos in motors.positions_body_m.values():
        x, y, z = (float(v) for v in pos)
        m_i = float(motors.mass_kg_each)
        ixx += m_i * (y**2 + z**2)
        iyy += m_i * (x**2 + z**2)
        izz += m_i * (x**2 + y**2)

    return ixx, iyy, izz
