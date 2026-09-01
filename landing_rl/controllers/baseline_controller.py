"""Baseline deterministic velocity controller.

Phase 2 extraction. This module holds ONLY the deterministic velocity-control
arithmetic that previously lived on ``LandingEnv`` as:

    LandingEnv._pid_velocity              -> BaselineController.pid_velocity
    LandingEnv._scale_action             -> BaselineController.scale_action
    LandingEnv._limit_velocity_command   -> BaselineController.combine_and_limit
    (inline descent gate #2 in step())   -> BaselineController.apply_descent_gate

The arithmetic is copied verbatim from ``mujoco_rl/envs/env_prototype.py``
(the immutable reference). Nothing is vectorized, algebraically simplified, or
reordered. Even mathematically-equivalent rewrites are avoided so last-bit
floating-point results are unchanged.

Ownership
---------
The controller owns ONLY ``cfg`` (the existing ``LandingConfig`` instance).
It does NOT own or touch: position, velocity, target, previous action, delay
queues, RNG, contact state, or any episode state. It consumes ZERO RNG.
``LandingEnv`` remains responsible for orchestration, including the
target-valid residual gate (``valid or cfg.apply_residual_when_target_invalid``).
"""

from __future__ import annotations

import numpy as np


class BaselineController:
    """Nominal outer-loop velocity controller + RL residual combiner.

    Stateless apart from the shared config. All methods are pure functions of
    their arguments and ``self.cfg``.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Nominal PID velocity (was LandingEnv._pid_velocity)
    #
    # ``altitude_agl_before`` is the value LandingEnv computed via
    # ``_altitude_agl()`` for the current step, before dynamics. In the legacy
    # code ``_pid_velocity`` recomputed ``self._altitude_agl()`` at this point;
    # ``self.pos`` is unchanged between the two call sites, so the passed-in
    # value is the identical float. ``_altitude_agl`` itself is NOT extracted.
    # ------------------------------------------------------------------
    def pid_velocity(self, target, valid, pos, vel, altitude_agl_before):
        if (not valid) and self.cfg.freeze_control_when_target_invalid:
            return np.zeros(3, dtype=np.float64)

        error = target - pos

        vx = self.cfg.kp_xy * error[0] - self.cfg.kd_xy * vel[0]
        vy = self.cfg.kp_xy * error[1] - self.cfg.kd_xy * vel[1]
        vz = self.cfg.kp_z * error[2] - self.cfg.kd_z * vel[2]

        vxy = np.array([vx, vy], dtype=np.float64)
        norm_xy = float(np.linalg.norm(vxy))
        if norm_xy > self.cfg.max_pid_xy_mps:
            vxy *= self.cfg.max_pid_xy_mps / (norm_xy + 1e-9)

        altitude_agl = altitude_agl_before
        xy_error = float(np.linalg.norm(error[:2]))

        # Descent gate #1: logically part of the PID calculation, before final
        # velocity-command saturation. Kept separate from descent gate #2.
        if (
            xy_error < self.cfg.descent_xy_gate_m
            and altitude_agl < self.cfg.descent_start_height_m
            and altitude_agl > self.cfg.success_altitude_m
        ):
            vz = max(vz, self.cfg.landing_descent_bias_mps)

        vz = float(np.clip(vz, -self.cfg.max_pid_z_mps, self.cfg.max_pid_z_mps))

        return np.array([vxy[0], vxy[1], vz], dtype=np.float64)

    # ------------------------------------------------------------------
    # Residual velocity scaling (was LandingEnv._scale_action)
    # ------------------------------------------------------------------
    def scale_action(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        return np.array(
            [
                action[0] * self.cfg.residual_xy_mps,
                action[1] * self.cfg.residual_xy_mps,
                action[2] * self.cfg.residual_z_mps,
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # v_cmd = limit(v_pid + v_residual)   (was LandingEnv._limit_velocity_command
    # applied to the sum computed in step())
    # ------------------------------------------------------------------
    def combine_and_limit(self, v_pid, v_residual):
        v_cmd = np.asarray(v_pid + v_residual, dtype=np.float64).copy()

        vxy_norm = float(np.linalg.norm(v_cmd[:2]))
        if vxy_norm > self.cfg.max_cmd_xy_mps:
            v_cmd[:2] *= self.cfg.max_cmd_xy_mps / (vxy_norm + 1e-9)

        v_cmd[2] = float(
            np.clip(v_cmd[2], -self.cfg.max_cmd_z_mps, self.cfg.max_cmd_z_mps)
        )
        return v_cmd

    # ------------------------------------------------------------------
    # Descent gate #2 (was inline in LandingEnv.step, AFTER combine_and_limit).
    # Mutates and returns v_cmd in place, matching the legacy behavior.
    # ------------------------------------------------------------------
    def apply_descent_gate(
        self,
        v_cmd,
        control_target_valid,
        control_xy_error,
        altitude_agl_before,
        ground_contact,
    ):
        if (
            control_target_valid
            and control_xy_error < self.cfg.descent_xy_gate_m
            and altitude_agl_before < self.cfg.descent_start_height_m
            and not ground_contact
        ):
            if altitude_agl_before < 0.10:
                v_cmd[2] = max(v_cmd[2], 0.06)
            else:
                v_cmd[2] = max(v_cmd[2], self.cfg.landing_descent_bias_mps)
        return v_cmd
