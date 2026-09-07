"""Frame / signal transforms for SI preprocessing.

Conventions (fixed once, applied identically to setpoint and measurement):

  * Quaternions are PX4 order ``[w, x, y, z]`` describing the rotation that
    takes a vector from body FRD to NED (``R = R(q)``, ``v_ned = R v_body``).
  * Euler angles are the intrinsic ``x-y-z`` (roll, pitch, yaw) decomposition
    of that same rotation, in radians. Derived from the quaternion ONLY --
    never read from a PX4 ``roll_body`` style field (absent in newer logs).
  * Tilt is the angle between the body "down" axis and world down ``(0,0,1)``
    in NED, i.e. ``acos(R[2, 2])``.

Nothing here converts normalized thrust to m/s^2 or otherwise fabricates
physical scaling.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

FRAME_NED = "NED"
FRAME_FRD = "FRD"


def _as_wxyz(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion array must have last dim 4, got {q.shape}")
    return q


def _to_scipy_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """scipy expects [x, y, z, w]."""
    q = _as_wxyz(q_wxyz)
    return np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1)


def normalize_quat(q_wxyz) -> np.ndarray:
    q = _as_wxyz(q_wxyz)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return q / n


def quat_to_euler_xyz(q_wxyz) -> np.ndarray:
    """Return ``[..., 3]`` array of (roll, pitch, yaw) in radians.

    A zero/degenerate quaternion maps to zeros rather than NaN so a single bad
    sample does not poison a whole segment; callers rely on validity masks to
    drop such samples for other reasons.
    """
    q = _as_wxyz(q_wxyz)
    scalar = q.ndim == 1
    q2 = q.reshape(-1, 4)
    norm = np.linalg.norm(q2, axis=1)
    good = norm > 1e-9
    out = np.zeros((q2.shape[0], 3), dtype=np.float64)
    if np.any(good):
        rot = Rotation.from_quat(_to_scipy_xyzw(q2[good]))
        out[good] = rot.as_euler("xyz", degrees=False)
    return out[0] if scalar else out.reshape(q.shape[:-1] + (3,))


def euler_xyz_columns(q_wxyz):
    """Convenience: returns (roll, pitch, yaw) as three 1-D arrays."""
    e = quat_to_euler_xyz(q_wxyz)
    return e[..., 0], e[..., 1], e[..., 2]


def tilt_from_quat(q_wxyz) -> np.ndarray:
    """Angle (rad) between the body-down axis and world-down, i.e. acos(R[2,2]).

    Computed directly from the quaternion: ``R[2, 2] = 1 - 2 (x^2 + y^2)`` for a
    unit ``[w, x, y, z]``.
    """
    q = normalize_quat(q_wxyz)
    x = q[..., 1]
    y = q[..., 2]
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return np.arccos(np.clip(r22, -1.0, 1.0))


def thrust_vector_magnitude(tx, ty, tz) -> np.ndarray:
    """Euclidean norm of a (normalized) thrust vector; frame-agnostic."""
    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)
    tz = np.asarray(tz, dtype=np.float64)
    return np.sqrt(tx * tx + ty * ty + tz * tz)


def wrap_pi(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return np.arctan2(np.sin(a), np.cos(a))
