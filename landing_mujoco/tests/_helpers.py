from pathlib import Path

import numpy as np

from landing_mujoco.configs.param_schema import load_uav_params
from landing_mujoco.configs.simulation_config import SimulationConfig
from landing_mujoco.dynamics.mujoco_dynamics import MuJoCoDynamics
from landing_rl.dynamics.vehicle_state import VehicleState
from landing_rl.envs.landing_env import LandingConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_YAML = REPO_ROOT / "landing_mujoco" / "configs" / "tarot680b_reference.yaml"


def load_reference_params():
    return load_uav_params(REFERENCE_YAML)


def deterministic_control_cfg(**overrides) -> LandingConfig:
    from dataclasses import replace

    from landing_mujoco.envs.mujoco_landing_env import deterministic_overrides

    cfg = deterministic_overrides(LandingConfig())
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def make_dynamics(physics_dt: float = 0.002, **cfg_overrides) -> MuJoCoDynamics:
    params = load_reference_params()
    cfg = deterministic_control_cfg(**cfg_overrides)
    sim_cfg = SimulationConfig(control_dt=cfg.dt, physics_dt=physics_dt)
    return MuJoCoDynamics(params, cfg, sim_cfg)


def fresh_state(pos=None, vel=None, attitude=None, thrust_accel=9.80665) -> VehicleState:
    return VehicleState(
        pos=np.zeros(3) if pos is None else np.asarray(pos, dtype=np.float64),
        vel=np.zeros(3) if vel is None else np.asarray(vel, dtype=np.float64),
        accel=np.zeros(3),
        prev_accel=np.zeros(3),
        attitude=np.zeros(3) if attitude is None else np.asarray(attitude, dtype=np.float64),
        yaw_rate=0.0,
        body_rates=np.zeros(3),
        thrust_accel=thrust_accel,
        attitude_setpoint=np.zeros(3),
        thrust_accel_setpoint=thrust_accel,
        accel_cmd=np.zeros(3),
        ground_effect_factor=1.0,
    )
