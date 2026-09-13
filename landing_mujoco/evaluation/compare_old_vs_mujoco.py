#!/usr/bin/env python3
"""Old (legacy analytic) LandingEnv vs new MujocoLandingEnv, same scenario.

Runs all 8 deterministic scenarios through BOTH the existing
``landing_rl.envs.landing_env.LandingEnv`` and the new
``landing_mujoco.envs.mujoco_landing_env.MujocoLandingEnv``, with zero
policy action (open-loop PID only) and a forced identical initial
condition, and reports what physically changed:

    1. hover
    2. north_velocity   (position error along North -> commands +North velocity)
    3. east_velocity    (position error along East  -> commands +East velocity)
    4. climb
    5. descent
    6. roll_transient    (nonzero initial roll, self-corrects toward level)
    7. pitch_transient   (nonzero initial pitch, self-corrects toward level)
    8. landing_approach  (offset + altitude -> full descent to touchdown)

For each scenario, records position/velocity/acceleration/roll/pitch/yaw/
body-rates/thrust_accel/v_cmd and reports peak tilt, peak speed, peak
accel, steady-state velocity, a generic rise-time/settling-time/overshoot
transient analysis on the scenario's dominant position channel, and (for
``landing_approach``) touchdown speed. Two plots per scenario are saved:
``<name>_kinematics.png`` (pos/vel/accel, N/E/D) and
``<name>_attitude_thrust_command.png`` (roll/pitch/yaw, body rates,
thrust_accel, v_cmd).

Per the task spec: the trajectories do NOT need to be identical, and MuJoCo
is NOT tuned to reproduce the legacy trajectories -- this tool exists to
make the DIFFERENCE legible, not to erase it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from landing_mujoco.configs.param_schema import load_uav_params
from landing_mujoco.envs.mujoco_landing_env import MujocoLandingEnv, deterministic_overrides
from landing_rl.envs.landing_env import LandingEnv as OldLandingEnv
from landing_rl.envs.landing_env import LandingConfig

OUTPUT_DIR = REPO_ROOT / "landing_mujoco" / "evaluation" / "compare_old_vs_mujoco_output"

TRACE_KEYS = ["pos", "vel", "accel", "roll", "pitch", "yaw", "body_rates", "thrust_accel", "v_cmd"]


@dataclass
class Scenario:
    name: str
    pos: np.ndarray
    vel: np.ndarray
    attitude: np.ndarray
    n_steps: int = 80


SCENARIOS = [
    Scenario("hover", pos=np.array([0.0, 0.0, -3.0]), vel=np.zeros(3), attitude=np.zeros(3)),
    Scenario("north_velocity", pos=np.array([-3.0, 0.0, -3.0]), vel=np.zeros(3), attitude=np.zeros(3)),
    Scenario("east_velocity", pos=np.array([0.0, -3.0, -3.0]), vel=np.zeros(3), attitude=np.zeros(3)),
    Scenario("climb", pos=np.array([0.0, 0.0, -1.0]), vel=np.zeros(3), attitude=np.zeros(3)),
    Scenario("descent", pos=np.array([0.0, 0.0, -5.0]), vel=np.zeros(3), attitude=np.zeros(3)),
    Scenario("roll_transient", pos=np.array([0.0, 0.0, -3.0]), vel=np.zeros(3), attitude=np.array([0.3, 0.0, 0.0])),
    Scenario("pitch_transient", pos=np.array([0.0, 0.0, -3.0]), vel=np.zeros(3), attitude=np.array([0.0, 0.3, 0.0])),
    Scenario("landing_approach", pos=np.array([-2.0, 1.0, -4.0]), vel=np.zeros(3), attitude=np.zeros(3), n_steps=400),
]


def _force_initial_state(env, scenario: Scenario, is_mujoco: bool) -> None:
    st = env._vehicle_state
    st.pos[:] = scenario.pos
    st.vel[:] = scenario.vel
    st.attitude[:] = scenario.attitude
    st.attitude_setpoint[:] = scenario.attitude
    if is_mujoco:
        env.mujoco_dynamics.reset(scenario.pos.copy(), scenario.vel.copy(), scenario.attitude.copy())
    env.prev_potential = env._potential()
    env.target_true = np.zeros(3, dtype=np.float64)
    env.obs_target = np.zeros(3, dtype=np.float64)


def run_scenario(env, scenario: Scenario, is_mujoco: bool) -> dict:
    env.reset(seed=0)
    _force_initial_state(env, scenario, is_mujoco)

    trace = {k: [] for k in TRACE_KEYS}
    trace["ground_contact"] = []
    zero_action = np.zeros(3, dtype=np.float32)
    for _ in range(scenario.n_steps):
        obs, reward, term, trunc, info = env.step(zero_action)
        trace["pos"].append(info["pos"].copy())
        trace["vel"].append(info["vel"].copy())
        trace["accel"].append(info["accel"].copy())
        trace["roll"].append(info["roll"])
        trace["pitch"].append(info["pitch"])
        trace["yaw"].append(info["yaw"])
        trace["body_rates"].append(info["body_rates"].copy())
        trace["thrust_accel"].append(info["thrust_accel"])
        trace["v_cmd"].append(info["v_cmd"].copy())
        trace["ground_contact"].append(bool(info["ground_contact"]))
        if term or trunc:
            break

    for key in TRACE_KEYS:
        trace[key] = np.array(trace[key])
    return trace


def analyze_transient(signal: np.ndarray, dt: float) -> dict:
    """Generic rise time (10-90%), settling time (within 5% of final
    value), and overshoot for a 1-D signal, measured from its own start/end
    values (works for any monotonic-ish step-like response; scenarios with
    negligible motion, e.g. hover, just report near-zero/degenerate
    values)."""
    start, final = float(signal[0]), float(signal[-1])
    delta = final - start
    t = np.arange(len(signal)) * dt

    if abs(delta) < 1e-6:
        return {"rise_time_s": 0.0, "settling_time_s": 0.0, "overshoot_pct": 0.0}

    frac = (signal - start) / delta
    idx10 = np.argmax(frac >= 0.10) if np.any(frac >= 0.10) else len(signal) - 1
    idx90 = np.argmax(frac >= 0.90) if np.any(frac >= 0.90) else len(signal) - 1
    rise_time = float(t[idx90] - t[idx10]) if idx90 >= idx10 else 0.0

    band = 0.05 * abs(delta)
    within_band = np.abs(signal - final) <= band
    settling_idx = len(signal) - 1
    for i in range(len(signal) - 1, -1, -1):
        if not within_band[i]:
            settling_idx = min(i + 1, len(signal) - 1)
            break
        settling_idx = 0
    settling_time = float(t[settling_idx])

    if delta > 0:
        overshoot = max(0.0, float(np.max(signal)) - final)
    else:
        overshoot = max(0.0, final - float(np.min(signal)))
    overshoot_pct = 100.0 * overshoot / abs(delta) if abs(delta) > 1e-9 else 0.0

    return {"rise_time_s": rise_time, "settling_time_s": settling_time, "overshoot_pct": overshoot_pct}


def summarize(trace: dict, dt: float) -> dict:
    pos, vel = trace["pos"], trace["vel"]
    tilt = np.hypot(trace["roll"], trace["pitch"])

    # Dominant position channel: the axis with the largest net displacement.
    net_change = np.abs(pos[-1] - pos[0])
    dominant_axis = int(np.argmax(net_change))
    transient = analyze_transient(pos[:, dominant_axis], dt)

    touchdown_speed = float("nan")
    if np.any(trace["ground_contact"]):
        first_contact = int(np.argmax(trace["ground_contact"]))
        touchdown_speed = float(np.linalg.norm(vel[first_contact]))

    summary = {
        "final_pos": pos[-1],
        "final_vel": vel[-1],
        "peak_tilt_deg": float(np.degrees(np.max(tilt))),
        "peak_speed": float(np.max(np.linalg.norm(vel, axis=1))),
        "peak_accel": float(np.max(np.linalg.norm(trace["accel"], axis=1))),
        "steady_state_speed": float(np.linalg.norm(vel[-1])),
        f"rise_time_s(axis={dominant_axis})": transient["rise_time_s"],
        "settling_time_s": transient["settling_time_s"],
        "overshoot_pct": transient["overshoot_pct"],
    }
    if not np.isnan(touchdown_speed):
        summary["touchdown_speed"] = touchdown_speed
    return summary


def plot_kinematics(name: str, old_trace: dict, new_trace: dict) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    labels = ["N", "E", "D"]
    t_old = np.arange(len(old_trace["pos"]))
    t_new = np.arange(len(new_trace["pos"]))
    cols = [("pos", "pos [m]"), ("vel", "vel [m/s]"), ("accel", "accel [m/s^2]")]

    for i in range(3):
        for j, (key, ylabel) in enumerate(cols):
            axes[i, j].plot(t_old, old_trace[key][:, i], label="old")
            axes[i, j].plot(t_new, new_trace[key][:, i], label="mujoco", linestyle="--")
            axes[i, j].set_ylabel(f"{labels[i]} {ylabel}")

    axes[0, 0].legend()
    for j, (_, title) in enumerate(cols):
        axes[0, j].set_title(f"{name}: {title}")
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{name}_kinematics.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_attitude_thrust_command(name: str, old_trace: dict, new_trace: dict) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    t_old = np.arange(len(old_trace["pos"]))
    t_new = np.arange(len(new_trace["pos"]))

    for i, label in enumerate(["roll", "pitch", "yaw"]):
        axes[0].plot(t_old, old_trace[label], label=f"old {label}", linestyle="-")
        axes[0].plot(t_new, new_trace[label], label=f"mujoco {label}", linestyle="--")
    axes[0].set_ylabel("attitude [rad]")
    axes[0].legend(fontsize=7, ncol=3)
    axes[0].set_title(f"{name}: roll/pitch/yaw")

    for i, label in enumerate(["p", "q", "r"]):
        axes[1].plot(t_old, old_trace["body_rates"][:, i], label=f"old {label}")
        axes[1].plot(t_new, new_trace["body_rates"][:, i], label=f"mujoco {label}", linestyle="--")
    axes[1].set_ylabel("body rates [rad/s]")
    axes[1].legend(fontsize=7, ncol=3)

    axes[2].plot(t_old, old_trace["thrust_accel"], label="old")
    axes[2].plot(t_new, new_trace["thrust_accel"], label="mujoco", linestyle="--")
    axes[2].set_ylabel("thrust_accel [m/s^2]")
    axes[2].legend(fontsize=7)

    for i, label in enumerate(["N", "E", "D"]):
        axes[3].plot(t_old, old_trace["v_cmd"][:, i], label=f"old {label}")
        axes[3].plot(t_new, new_trace["v_cmd"][:, i], label=f"mujoco {label}", linestyle="--")
    axes[3].set_ylabel("v_cmd [m/s]")
    axes[3].set_xlabel("step")
    axes[3].legend(fontsize=7, ncol=3)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{name}_attitude_thrust_command.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> int:
    physical_params = load_uav_params(
        REPO_ROOT / "landing_mujoco" / "configs" / "tarot680b_reference.yaml"
    )
    base_cfg = deterministic_overrides(LandingConfig())

    print(f"{'scenario':<18} {'metric':<26} {'old':>14} {'mujoco':>14}")
    print("-" * 76)

    for scenario in SCENARIOS:
        old_env = OldLandingEnv(config=base_cfg)
        new_env = MujocoLandingEnv(physical_params, config=base_cfg, deterministic_physics=False)

        old_trace = run_scenario(old_env, scenario, is_mujoco=False)
        new_trace = run_scenario(new_env, scenario, is_mujoco=True)

        old_summary = summarize(old_trace, base_cfg.dt)
        new_summary = summarize(new_trace, base_cfg.dt)

        for key in old_summary:
            old_v = old_summary[key]
            new_v = new_summary.get(key, float("nan"))
            if isinstance(old_v, np.ndarray):
                old_s = np.array2string(old_v, precision=3)
                new_s = np.array2string(new_v, precision=3) if isinstance(new_v, np.ndarray) else f"{new_v:.4f}"
            else:
                old_s = f"{old_v:.4f}"
                new_s = f"{new_v:.4f}" if not isinstance(new_v, np.ndarray) else np.array2string(new_v, precision=3)
            print(f"{scenario.name:<18} {key:<26} {old_s:>14} {new_s:>14}")

        kin_path = plot_kinematics(scenario.name, old_trace, new_trace)
        att_path = plot_attitude_thrust_command(scenario.name, old_trace, new_trace)
        print(f"  -> plots saved: {kin_path.relative_to(REPO_ROOT)}, {att_path.relative_to(REPO_ROOT)}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
