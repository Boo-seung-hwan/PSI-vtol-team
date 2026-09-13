"""MuJoCo-backed rigid-body plant.

This is the module that actually satisfies "MuJoCo must integrate the
rigid-body state; do NOT manually Euler-integrate position or orientation in
Python": ``MuJoCoDynamics`` never assigns ``state.pos``/``state.vel``/
``state.attitude`` from a Python-computed integration step. It only ever
reads them back from ``mujoco.MjData`` (via the coordinate adapter) after
calling ``mujoco.mj_step``.

One control step (``dt``, ~0.05 s) is subdivided into ``physics_dt``-sized
MuJoCo substeps (``SimulationConfig``). Per substep:

    1. read current attitude + body rates back from MuJoCo (the ONLY
       source of truth for these -- no parallel Python integration)
    2. ``IdentifiedWrenchActuation`` turns (attitude error, body rates,
       thrust setpoint) into a body torque + collective thrust, using the
       vehicle's measured-or-provisional J/tau/K/T_max
    3. thrust + drag + wind + translational process noise are summed into
       one world-frame force; torque is rotated body->world using MuJoCo's
       own current orientation (``data.xmat``)
    4. ``xfrc_applied`` is set and ``mujoco.mj_step`` advances one
       ``physics_dt``

The attitude/thrust SETPOINT (from ``InnerLoopCommandModel``, reused
unchanged from ``landing_rl``) is computed ONCE per control step, matching
the existing 20 Hz control cadence -- only the identified inner-loop
response is integrated at the finer physics rate.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from landing_mujoco.configs.param_schema import UAVPhysicalParams, landing_reference_point_body_m
from landing_mujoco.configs.simulation_config import SimulationConfig
from landing_mujoco.coordinates.transforms import (
    frd_vector_to_mujoco_body,
    mujoco_body_vector_to_frd,
    mujoco_position_to_ned,
    mujoco_quaternion_to_policy_euler,
    mujoco_velocity_to_ned,
    ned_position_to_mujoco,
    ned_velocity_to_mujoco,
    rotation_body_to_world_ned,
    policy_euler_to_mujoco_quaternion,
)
from landing_mujoco.dynamics.actuation_model import IdentifiedWrenchActuation
from landing_mujoco.dynamics.contact import begin_step, classify_touchdown
from landing_mujoco.dynamics.mjcf_builder import build_mjcf
from landing_mujoco.visualization.x500_shell import build_x500_visual_shell
from landing_rl.contact.contact_model import ContactResult, ContactState
from landing_rl.dynamics.inner_loop_command_model import InnerLoopCommandModel
from landing_rl.dynamics.process_noise import ProcessNoiseSampler


class MuJoCoDynamics:
    """Owns the MuJoCo model/data, the identified actuation model, and the
    ground-contact classifier. Not a ``VehicleDynamicsBackend`` (that
    Protocol's signature is specific to the legacy alpha-blend response
    scheme -- see the Phase 0 report); this class has its own natural
    ``step`` signature, called directly by ``MujocoLandingEnv``.
    """

    def __init__(self, params: UAVPhysicalParams, control_cfg, sim_cfg: SimulationConfig, visualization=None):
        self.params = params
        self.cfg = control_cfg
        self.sim_cfg = sim_cfg
        self.mass_kg = float(params.mass_properties.mass_kg)
        # Visualization-only (see landing_mujoco/visualization/x500_shell.py).
        # None / disabled -> the MJCF is exactly the physics-only model.
        self.visual_shell = build_x500_visual_shell(visualization, params)
        # Ground effect is a physical-clearance phenomenon: it must trigger
        # based on the landing gear's actual height above the ground, not
        # the CG's raw NED z (see MUJOCO_MODEL.md's landing-gear-standoff
        # section) -- computed from the same physical geometry the env uses
        # for its task-level altitude_agl, but independently (this module
        # owns no task/target state).
        self.r_landing_body = landing_reference_point_body_m(params.geometry)

        xml = build_mjcf(
            params,
            physics_dt=sim_cfg.physics_dt,
            ground_friction_xy=control_cfg.ground_friction_xy,
            visual_shell=self.visual_shell,
        )
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "vehicle")
        ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self._ground_geom_id = ground_id
        self._leg_geom_ids = {
            i
            for i in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("leg_")
        }

        self.inner_loop = InnerLoopCommandModel(control_cfg)
        self.actuation = IdentifiedWrenchActuation(params, sim_cfg.physics_dt)
        self.process_noise = ProcessNoiseSampler(control_cfg)

        self.contact_state = ContactState()
        self.contact_result = ContactResult()

        self._prev_vel_ned = np.zeros(3, dtype=np.float64)
        self.last_applied_force_world = np.zeros(3, dtype=np.float64)
        self.last_applied_torque_world = np.zeros(3, dtype=np.float64)
        self.last_desired_body_rates = np.zeros(3, dtype=np.float64)
        self.last_delayed_body_rate_command = np.zeros(3, dtype=np.float64)
        self.last_desired_collective_thrust_n = 0.0
        self.last_collective_thrust_n = 0.0

    # ------------------------------------------------------------------
    def reset(self, pos_ned: np.ndarray, vel_ned: np.ndarray, attitude_rpy: np.ndarray) -> None:
        quat = policy_euler_to_mujoco_quaternion(*[float(a) for a in attitude_rpy])
        pos_mj = ned_position_to_mujoco(pos_ned)
        vel_mj = ned_velocity_to_mujoco(vel_ned)

        self.data.qpos[:] = 0.0
        self.data.qpos[0:3] = pos_mj
        self.data.qpos[3:7] = quat
        self.data.qvel[:] = 0.0
        self.data.qvel[0:3] = vel_mj
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.actuation.reset(hover_accel=float(self.cfg.gravity_mps2))
        self.contact_state = ContactState()
        self.contact_result = ContactResult()
        self._prev_vel_ned = vel_ned.copy()

    # ------------------------------------------------------------------
    def _read_state_ned(self):
        pos_ned = mujoco_position_to_ned(self.data.qpos[0:3])
        vel_ned = mujoco_velocity_to_ned(self.data.qvel[0:3])
        roll, pitch, yaw = mujoco_quaternion_to_policy_euler(self.data.qpos[3:7])
        attitude = np.array([roll, pitch, yaw], dtype=np.float64)
        body_rates_frd = mujoco_body_vector_to_frd(self.data.qvel[3:6])
        return pos_ned, vel_ned, attitude, body_rates_frd

    def _legs_touching_ground(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            geoms = {int(c.geom1), int(c.geom2)}
            if self._ground_geom_id in geoms and geoms & self._leg_geom_ids:
                return True
        return False

    def _gear_altitude_agl(self, pos_ned: np.ndarray, attitude: np.ndarray) -> float:
        """Physical clearance of the landing-gear reference point above the
        ground plane, using the vehicle's CURRENT (not desired/level)
        attitude -- ground effect depends on real clearance, not a task
        target."""
        r_current = rotation_body_to_world_ned(*attitude) @ self.r_landing_body
        gear_z = float(pos_ned[2] + r_current[2])
        return max(0.0, float(self.cfg.ground_z_m - gear_z))

    def _ground_effect_factor(self, altitude_agl: float, ground_contact_prev: bool) -> float:
        height = max(float(self.cfg.ground_effect_height_m), 1e-6)
        if altitude_agl >= height or ground_contact_prev:
            return 1.0
        closeness = 1.0 - altitude_agl / height
        factor = 1.0 + float(self.cfg.ground_effect_gain) * closeness * closeness
        return float(np.clip(factor, 1.0, self.cfg.ground_effect_max_factor))

    # ------------------------------------------------------------------
    def step(
        self,
        state,
        v_cmd: np.ndarray,
        dt: float,
        rng,
        wind_accel_ned: np.ndarray,
        target_yaw: float,
        motor_cutoff: bool,
        ground_contact_prev: bool,
    ) -> ContactResult:
        """Advance one control step. ``state`` (a ``VehicleState``) is
        updated FROM MuJoCo's readback at the start and end of this call --
        never Euler-integrated here."""
        dt_safe = max(float(dt), 1e-6)
        vel_before_ned = self._prev_vel_ned.copy()

        pos_ned, vel_ned, attitude, body_rates_frd = self._read_state_ned()
        state.pos, state.vel = pos_ned, vel_ned
        state.attitude, state.body_rates = attitude, body_rates_frd

        inner = self.inner_loop.compute(state, np.asarray(v_cmd, dtype=np.float64), dt_safe, target_yaw)
        thrust_sp = (
            float(self.cfg.motor_cutoff_thrust_accel_mps2) if motor_cutoff else inner.thrust_accel_setpoint
        )
        state.attitude_setpoint = inner.attitude_setpoint.copy()
        state.thrust_accel_setpoint = float(thrust_sp)
        state.accel_cmd = inner.accel_cmd.copy()

        altitude_agl = self._gear_altitude_agl(pos_ned, attitude)
        ge_factor = self._ground_effect_factor(altitude_agl, ground_contact_prev)
        state.ground_effect_factor = ge_factor

        n_sub = self.sim_cfg.physics_substeps(dt_safe)
        physics_dt = dt_safe / n_sub

        noise_scale = math.sqrt(dt_safe / max(self.cfg.dt, 1e-6))
        accel_noise_ned = self.process_noise.sample_translational_accel(rng, noise_scale, dt_safe)
        body_rate_noise = self.process_noise.sample_body_rate(rng, noise_scale)
        thrust_noise = self.process_noise.sample_thrust(rng, noise_scale)

        # Stochastic actuator/aerodynamic disturbances are applied once per
        # control step (matching the legacy per-control-step draw cadence),
        # not re-injected every physics substep.
        self.data.qvel[3:6] += frd_vector_to_mujoco_body(body_rate_noise)
        self.actuation.thrust_accel = float(
            np.clip(
                self.actuation.thrust_accel + thrust_noise,
                0.0,
                self.actuation.max_thrust_accel_physical,
            )
        )

        for _ in range(n_sub):
            _, vel_ned_sub, attitude_sub, body_rates_sub = self._read_state_ned()

            actuation_out = self.actuation.compute_substep(
                attitude=attitude_sub,
                attitude_setpoint=state.attitude_setpoint,
                omega_frd=body_rates_sub,
                thrust_accel_setpoint=state.thrust_accel_setpoint,
                physics_dt=physics_dt,
                control_cfg=self.cfg,
                motor_cutoff=motor_cutoff,
            )

            torque_mj_body = frd_vector_to_mujoco_body(actuation_out.torque_body_frd)
            r_body_to_world_mj = np.asarray(self.data.xmat[self.body_id]).reshape(3, 3)
            torque_mj_world = r_body_to_world_mj @ torque_mj_body

            r_frd_to_ned = rotation_body_to_world_ned(*attitude_sub)
            body_z_in_ned = r_frd_to_ned[:, 2]
            effective_thrust_accel = actuation_out.thrust_accel * ge_factor
            thrust_force_ned = -effective_thrust_accel * self.mass_kg * body_z_in_ned

            drag_force_ned = -self.mass_kg * np.array(
                [
                    self.cfg.linear_drag_xy * vel_ned_sub[0],
                    self.cfg.linear_drag_xy * vel_ned_sub[1],
                    self.cfg.linear_drag_z * vel_ned_sub[2],
                ],
                dtype=np.float64,
            )
            wind_force_ned = self.mass_kg * np.asarray(wind_accel_ned, dtype=np.float64)
            noise_force_ned = self.mass_kg * accel_noise_ned

            total_force_ned = thrust_force_ned + drag_force_ned + wind_force_ned + noise_force_ned
            total_force_mj = ned_velocity_to_mujoco(total_force_ned)

            self.data.xfrc_applied[self.body_id, 0:3] = total_force_mj
            self.data.xfrc_applied[self.body_id, 3:6] = torque_mj_world

            mujoco.mj_step(self.model, self.data)

            self.last_applied_force_world = total_force_mj.copy()
            self.last_applied_torque_world = torque_mj_world.copy()
            self.last_desired_body_rates = actuation_out.desired_body_rates.copy()
            self.last_delayed_body_rate_command = actuation_out.delayed_body_rate_command.copy()
            self.last_desired_collective_thrust_n = actuation_out.desired_collective_thrust_n
            self.last_collective_thrust_n = actuation_out.collective_thrust_n

        self.data.xfrc_applied[self.body_id, :] = 0.0

        pos_ned, vel_ned, attitude, body_rates_frd = self._read_state_ned()

        begin_step(self.contact_result)
        ground_now = self._legs_touching_ground()
        classify_touchdown(
            self.cfg, ground_now, vel_before_ned, vel_ned, attitude,
            self.contact_state, self.contact_result,
        )

        state.prev_accel = state.accel.copy()
        state.accel = (vel_ned - vel_before_ned) / dt_safe
        state.pos, state.vel = pos_ned, vel_ned
        state.attitude, state.body_rates = attitude, body_rates_frd
        state.thrust_accel = self.actuation.thrust_accel
        state.yaw_rate = float(body_rates_frd[2])

        self._prev_vel_ned = vel_ned.copy()

        return self.contact_result
