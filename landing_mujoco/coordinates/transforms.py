"""Centralized NED/FRD <-> MuJoCo coordinate adapter.

This is the ONLY place in ``landing_mujoco`` allowed to contain a sign flip,
axis swap, or frame-specific rotation. Every other module works either
entirely in NED/FRD (the existing ``landing_rl`` policy convention) or
entirely in MuJoCo's native frame, and calls into this module at the
boundary.

World frame
-----------
Policy/control convention (unchanged, matches ``landing_rl``): **NED**,
+X = North, +Y = East, +Z = Down.

MuJoCo world convention chosen here: **NWU** (+X = North, +Y = West,
+Z = Up). This is deliberately NOT ENU -- NWU relates to NED by a single
fixed 180-degree rotation about the shared North (X) axis, which is exactly
the diagonal reflection ``diag(1, -1, -1)``. Two consequences that motivate
this choice over ENU (which would need an axis swap + sign flip, not just a
sign flip):

1. ``diag(1, -1, -1)`` is its own inverse (R_x(pi) . R_x(pi) = I), so the
   SAME matrix converts NED->MuJoCo and MuJoCo->NED, and the same matrix
   converts FRD->MuJoCo-body and MuJoCo-body->FRD.
2. Gravity "just works": MuJoCo's ``option gravity="0 0 -9.81"`` (down is
   -Z in a Z-up world) is exactly the NED gravity vector ``[0, 0, +g]``
   passed through this same matrix -- no separate gravity bookkeeping is
   needed anywhere in the MuJoCo dynamics code.

Body frame
----------
Policy/control convention (unchanged): **FRD**, +X = Forward, +Y = Right,
+Z = Down. MuJoCo body frame chosen here: **FLU** (+X = Forward,
+Y = Left, +Z = Up) -- related to FRD by the identical ``diag(1, -1, -1)``
matrix, for the same reason as the world frame above.

Because MuJoCo's free-joint angular velocity (``qvel[3:6]``) is already
expressed in the LOCAL body frame (a documented MuJoCo convention), body
rates convert with the same ``REFLECT`` matrix as any other body-frame
vector -- no extra bookkeeping.

Attitude
--------
``rotation_body_to_world_ned`` builds the exact same aerospace
Rz(yaw).Ry(pitch).Rx(roll) matrix used throughout ``landing_rl``
(``legacy_dynamics.py::_rotation_body_to_ned``) -- kept here, verbatim in
structure, as the one canonical definition ``landing_mujoco`` uses. Because
both the world and body frames are related to NED/FRD by the SAME fixed
rotation ``REFLECT`` (itself a proper rotation: it equals R_x(pi)), the
MuJoCo-frame body-to-world rotation is a similarity transform:

    R_mujoco = REFLECT @ R_ned @ REFLECT

(``REFLECT`` is self-inverse, so this conjugation is also self-inverse.)
``policy_euler_to_mujoco_quaternion`` / ``mujoco_quaternion_to_policy_euler``
implement this and are round-trip tested in
``landing_mujoco/tests/test_coordinate_transforms.py``.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

# R_x(pi): the single fixed rotation relating NED<->MuJoCo(NWU) and
# FRD<->MuJoCo-body(FLU). Self-inverse (REFLECT @ REFLECT == I).
REFLECT = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def ned_position_to_mujoco(pos_ned: np.ndarray) -> np.ndarray:
    return REFLECT @ np.asarray(pos_ned, dtype=np.float64)


def mujoco_position_to_ned(pos_mj: np.ndarray) -> np.ndarray:
    return REFLECT @ np.asarray(pos_mj, dtype=np.float64)


def ned_velocity_to_mujoco(vel_ned: np.ndarray) -> np.ndarray:
    return REFLECT @ np.asarray(vel_ned, dtype=np.float64)


def mujoco_velocity_to_ned(vel_mj: np.ndarray) -> np.ndarray:
    return REFLECT @ np.asarray(vel_mj, dtype=np.float64)


# Acceleration and any other linear (non-attitude) quantity transform
# identically to velocity -- REFLECT is a constant matrix, so it commutes
# with time differentiation.
ned_accel_to_mujoco = ned_velocity_to_mujoco
mujoco_accel_to_ned = mujoco_velocity_to_ned


def frd_vector_to_mujoco_body(vec_frd: np.ndarray) -> np.ndarray:
    return REFLECT @ np.asarray(vec_frd, dtype=np.float64)


def mujoco_body_vector_to_frd(vec_mj_body: np.ndarray) -> np.ndarray:
    return REFLECT @ np.asarray(vec_mj_body, dtype=np.float64)


def rotation_body_to_world_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) . Ry(pitch) . Rx(roll). Identical in structure to
    ``landing_rl.dynamics.legacy_dynamics.LegacyVehicleDynamics._rotation_body_to_ned``
    -- the one canonical definition ``landing_mujoco`` uses."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def euler_from_rotation_ned(r: np.ndarray) -> tuple[float, float, float]:
    """Inverse of ``rotation_body_to_world_ned`` for the same Rz.Ry.Rx form."""
    roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
    pitch = math.atan2(-float(r[2, 0]), math.hypot(float(r[2, 1]), float(r[2, 2])))
    yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    return roll, pitch, yaw


def policy_euler_to_mujoco_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """(roll, pitch, yaw) in the NED/FRD policy convention -> MuJoCo
    ``qpos[3:7]`` quaternion, shape (4,), order (w, x, y, z)."""
    r_ned = rotation_body_to_world_ned(roll, pitch, yaw)
    r_mj = REFLECT @ r_ned @ REFLECT
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, r_mj.flatten())
    return quat


def mujoco_quaternion_to_policy_euler(quat: np.ndarray) -> tuple[float, float, float]:
    """MuJoCo ``qpos[3:7]`` quaternion (w, x, y, z) -> (roll, pitch, yaw) in
    the NED/FRD policy convention."""
    quat = np.asarray(quat, dtype=np.float64)
    r_mj = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(r_mj, quat)
    r_mj = r_mj.reshape(3, 3)
    r_ned = REFLECT @ r_mj @ REFLECT
    return euler_from_rotation_ned(r_ned)


# ---------------------------------------------------------------------------
# Visualization-only conventions (x500 visual shell). No physics code uses
# these; they live here so that every frame convention stays in one file.
# ---------------------------------------------------------------------------

# The Gazebo/SDF x500 model frame is FLU (+x forward, +y left, +z up), which
# is exactly the MuJoCo vehicle body frame chosen above. The native x500
# visual assembly is therefore placed WITHOUT any axis remapping.
SDF_MODEL_FLU_TO_MUJOCO_BODY = np.eye(3, dtype=np.float64)


def sdf_pose_matrix(pose_xyzrpy) -> np.ndarray:
    """4x4 homogeneous transform for an SDF ``<pose>x y z roll pitch yaw</pose>``.

    SDF uses the same fixed-axis Rz(yaw).Ry(pitch).Rx(roll) composition as
    ``rotation_body_to_world_ned``; that function is reused purely as the
    generic rotation builder (no NED meaning here)."""
    x, y, z, roll, pitch, yaw = (float(v) for v in pose_xyzrpy)
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rotation_body_to_world_ned(roll, pitch, yaw)
    t[:3, 3] = (x, y, z)
    return t


def frd_body_offset_to_mujoco_body(translation_frd, rotation_rpy_frd) -> tuple[np.ndarray, np.ndarray]:
    """A rigid body-fixed offset expressed in the FRD policy body frame
    (translation [m] + roll/pitch/yaw [rad], Rz.Ry.Rx) -> the same offset in
    the MuJoCo (FLU) body frame: (translation, 3x3 rotation)."""
    t_mj = frd_vector_to_mujoco_body(translation_frd)
    r_frd = rotation_body_to_world_ned(*(float(a) for a in rotation_rpy_frd))
    return t_mj, REFLECT @ r_frd @ REFLECT


def rotation_matrix_to_mujoco_quaternion(rot: np.ndarray) -> np.ndarray:
    """3x3 rotation -> MuJoCo (w, x, y, z) quaternion."""
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=np.float64).flatten())
    return quat
