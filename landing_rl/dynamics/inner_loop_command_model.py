"""Inner-loop command generation: velocity command -> attitude/thrust setpoint.

Control/physics boundary extraction. This module lifts ONLY the
controller-side portion of ``LegacyVehicleDynamics.advance_free_flight`` --
what used to be the private method
``LegacyVehicleDynamics._velocity_command_to_inner_loop_setpoints`` (itself a
verbatim copy of ``LandingEnv._velocity_command_to_inner_loop_setpoints`` in
``mujoco_rl/envs/env_prototype.py``) -- into its own component, with no
behavior change.

This approximates the PX4 cascade:
    velocity command -> acceleration demand -> attitude/thrust setpoint.

What is NOT here (stays in ``LegacyVehicleDynamics.advance_free_flight``)
--------------------------------------------------------------------------
* The motor-cutoff override of ``thrust_accel_setpoint`` -- it depends on
  ``ContactModel``'s persistent ``motor_cutoff`` state, a physics-side
  concept, so it is applied by the caller immediately after ``compute()``
  returns, exactly where it already sits in the pre-extraction code.
* Everything downstream of the setpoint: the attitude/rate controller proxy,
  body-rate response, thrust response, rotation, gravity/drag/wind/process
  noise, and velocity/position integration. All of that remains physics-side
  in ``LegacyVehicleDynamics``.

Ownership
---------
``InnerLoopCommandModel`` owns ONLY ``cfg``. It does NOT own an RNG, a
``VehicleState``, a ``ContactModel``, a ``LandingEnv`` reference, or any
target/wind/response-alpha state. ``compute()`` consumes ZERO RNG -- this
extraction moves no RNG draw, adds none, and reorders none.

No behavior change. The OLD-vs-NEW exact parity gate
(``test_legacy_regression_contract.py``), the expanded structural freeze
(``test_structural_freeze.py``), and the PPO / VecNormalize checkpoint gate
(``test_checkpoint_compatibility.py``) remain authoritative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from landing_rl.dynamics.vehicle_state import VehicleState


@dataclass
class InnerLoopCommand:
    """Output of one ``InnerLoopCommandModel.compute()`` call.

    Plain data -- no methods, no RNG, no cfg. Field names/shapes/units match
    ``VehicleState.accel_cmd`` / ``attitude_setpoint`` / ``thrust_accel_setpoint``
    exactly (this is what gets written into those fields, verbatim, by
    ``LegacyVehicleDynamics.advance_free_flight`` -- after the motor-cutoff
    override is applied to ``thrust_accel_setpoint``)."""

    accel_cmd: np.ndarray
    attitude_setpoint: np.ndarray
    thrust_accel_setpoint: float


class InnerLoopCommandModel:
    """Controller-side velocity-command-to-setpoint conversion.

    Stateless apart from the shared config. ``compute()`` is a pure function
    of its arguments and ``self.cfg``.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def compute(
        self,
        state: VehicleState,
        v_cmd: np.ndarray,
        dt: float,
        target_yaw: float,
    ) -> InnerLoopCommand:
        """Convert velocity command to attitude/thrust setpoints.

        Verbatim copy of the former
        ``LegacyVehicleDynamics._velocity_command_to_inner_loop_setpoints``
        with ``state`` already being the explicit input (no ``self.vel`` /
        ``self.target_yaw`` compatibility mirrors to translate).

        NED convention is used. Positive z acceleration means downward
        acceleration, so it is produced by reducing collective thrust below
        hover thrust.
        """
        dt_safe = max(float(dt), 1e-6)
        g = float(self.cfg.gravity_mps2)

        # Desired NED acceleration from velocity error. This replaces the old
        # direct first-order velocity response with a physically interpretable
        # acceleration request.
        accel_cmd = (np.asarray(v_cmd, dtype=np.float64) - state.vel) / max(
            self.cfg.vel_cmd_tau_s,
            dt_safe,
        )

        axy_norm = float(np.linalg.norm(accel_cmd[:2]))
        if axy_norm > self.cfg.max_cmd_accel_xy_mps2:
            accel_cmd[:2] *= self.cfg.max_cmd_accel_xy_mps2 / (axy_norm + 1e-9)
        accel_cmd[2] = float(np.clip(
            accel_cmd[2],
            -self.cfg.max_cmd_accel_z_mps2,
            self.cfg.max_cmd_accel_z_mps2,
        ))

        # Near-hover multicopter tilt approximation in NED.
        # +roll produces +East acceleration. +pitch produces -North acceleration.
        denom = max(g - float(accel_cmd[2]), 1e-3)
        roll_sp = math.atan2(float(accel_cmd[1]), denom)
        pitch_sp = math.atan2(-float(accel_cmd[0]), denom)

        roll_sp = float(np.clip(
            roll_sp,
            -self.cfg.max_tilt_target_rad,
            self.cfg.max_tilt_target_rad,
        ))
        pitch_sp = float(np.clip(
            pitch_sp,
            -self.cfg.max_tilt_target_rad,
            self.cfg.max_tilt_target_rad,
        ))

        yaw_sp = float(target_yaw)
        thrust_sp = float(np.clip(
            g - float(accel_cmd[2]),
            self.cfg.min_thrust_accel_mps2,
            self.cfg.max_thrust_accel_mps2,
        ))

        attitude_sp = np.array([roll_sp, pitch_sp, yaw_sp], dtype=np.float64)
        return InnerLoopCommand(
            accel_cmd=accel_cmd,
            attitude_setpoint=attitude_sp,
            thrust_accel_setpoint=thrust_sp,
        )
