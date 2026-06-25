import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from envs.landing_env import LandingEnv


OUT_DIR = "./runs/trajectory_plots"


def run_episode(name, policy_fn, seed=3000):
    env = LandingEnv()
    obs, _ = env.reset(seed=seed)

    log = {
        "t": [],
        "pos": [],
        "vel": [],
        "action": [],
        "xy_error": [],
        "z_error": [],
        "speed": [],
        "reward": [],
        "success": [],
    }

    done = False
    step = 0
    last_info = {}

    while not done:
        action = policy_fn(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        pos = info["pos"]
        vel = info["vel"]
        xy_error = info["xy_error"]
        z_error = info["z_error"]
        speed = float(np.linalg.norm(vel))

        log["t"].append(step * env.cfg.dt)
        log["pos"].append(pos.copy())
        log["vel"].append(vel.copy())
        log["action"].append(np.asarray(action, dtype=float).copy())
        log["xy_error"].append(xy_error)
        log["z_error"].append(z_error)
        log["speed"].append(speed)
        log["reward"].append(reward)
        log["success"].append(bool(info.get("success", False)))

        last_info = info
        step += 1

    for key in ["t", "pos", "vel", "action", "xy_error", "z_error", "speed", "reward"]:
        log[key] = np.asarray(log[key])

    print(
        f"{name}: "
        f"steps={len(log['t'])}, "
        f"success={last_info.get('success', False)}, "
        f"xy={last_info.get('xy_error', np.nan):.4f}, "
        f"z={last_info.get('z_error', np.nan):.4f}"
    )

    return log


def save_plot_xy(pid_log, ppo_log):
    plt.figure()
    plt.plot(pid_log["pos"][:, 0], pid_log["pos"][:, 1], label="PID only")
    plt.plot(ppo_log["pos"][:, 0], ppo_log["pos"][:, 1], label="PPO residual")
    plt.scatter([0.0], [0.0], marker="x", label="Target")
    plt.xlabel("x NED [m]")
    plt.ylabel("y NED [m]")
    plt.title("XY trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "xy_trajectory.png"), dpi=160)
    plt.close()


def save_plot_altitude(pid_log, ppo_log):
    # NED에서 z는 아래 방향 양수입니다.
    # 보기 쉽게 altitude = -z 로 변환합니다.
    pid_alt = -pid_log["pos"][:, 2]
    ppo_alt = -ppo_log["pos"][:, 2]

    plt.figure()
    plt.plot(pid_log["t"], pid_alt, label="PID only")
    plt.plot(ppo_log["t"], ppo_alt, label="PPO residual")
    plt.xlabel("time [s]")
    plt.ylabel("altitude above target [m]")
    plt.title("Altitude profile")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "altitude_time.png"), dpi=160)
    plt.close()


def save_plot_xy_error(pid_log, ppo_log):
    plt.figure()
    plt.plot(pid_log["t"], pid_log["xy_error"], label="PID only")
    plt.plot(ppo_log["t"], ppo_log["xy_error"], label="PPO residual")
    plt.xlabel("time [s]")
    plt.ylabel("XY error [m]")
    plt.title("Horizontal error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "xy_error_time.png"), dpi=160)
    plt.close()


def save_plot_z_error(pid_log, ppo_log):
    plt.figure()
    plt.plot(pid_log["t"], pid_log["z_error"], label="PID only")
    plt.plot(ppo_log["t"], ppo_log["z_error"], label="PPO residual")
    plt.xlabel("time [s]")
    plt.ylabel("Z error [m]")
    plt.title("Vertical error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "z_error_time.png"), dpi=160)
    plt.close()


def save_plot_speed(pid_log, ppo_log):
    plt.figure()
    plt.plot(pid_log["t"], pid_log["speed"], label="PID only")
    plt.plot(ppo_log["t"], ppo_log["speed"], label="PPO residual")
    plt.xlabel("time [s]")
    plt.ylabel("speed [m/s]")
    plt.title("Vehicle speed")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "speed_time.png"), dpi=160)
    plt.close()


def save_plot_vertical_velocity(pid_log, ppo_log):
    plt.figure()
    plt.plot(pid_log["t"], pid_log["vel"][:, 2], label="PID only")
    plt.plot(ppo_log["t"], ppo_log["vel"][:, 2], label="PPO residual")
    plt.xlabel("time [s]")
    plt.ylabel("vz NED [m/s]")
    plt.title("Vertical velocity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "vertical_velocity_time.png"), dpi=160)
    plt.close()


def save_plot_action_norm(pid_log, ppo_log):
    pid_action_norm = np.linalg.norm(pid_log["action"], axis=1)
    ppo_action_norm = np.linalg.norm(ppo_log["action"], axis=1)

    plt.figure()
    plt.plot(pid_log["t"], pid_action_norm, label="PID only")
    plt.plot(ppo_log["t"], ppo_action_norm, label="PPO residual")
    plt.xlabel("time [s]")
    plt.ylabel("normalized action norm")
    plt.title("Action magnitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "action_norm_time.png"), dpi=160)
    plt.close()


def save_plot_ppo_action_components(ppo_log):
    plt.figure()
    plt.plot(ppo_log["t"], ppo_log["action"][:, 0], label="action x")
    plt.plot(ppo_log["t"], ppo_log["action"][:, 1], label="action y")
    plt.plot(ppo_log["t"], ppo_log["action"][:, 2], label="action z")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.axhline(-1.0, linestyle="--", linewidth=1)
    plt.xlabel("time [s]")
    plt.ylabel("normalized action")
    plt.title("PPO residual action components")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ppo_action_components.png"), dpi=160)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = PPO.load("./runs/ppo_landing_residual", device="cpu")

    def pid_policy(obs):
        return np.zeros(3, dtype=np.float32)

    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    # 같은 seed를 써야 같은 초기조건에서 비교됩니다.
    seed = 3000

    pid_log = run_episode("PID only", pid_policy, seed=seed)
    ppo_log = run_episode("PPO residual", ppo_policy, seed=seed)

    save_plot_xy(pid_log, ppo_log)
    save_plot_altitude(pid_log, ppo_log)
    save_plot_xy_error(pid_log, ppo_log)
    save_plot_z_error(pid_log, ppo_log)
    save_plot_speed(pid_log, ppo_log)
    save_plot_vertical_velocity(pid_log, ppo_log)
    save_plot_action_norm(pid_log, ppo_log)
    save_plot_ppo_action_components(ppo_log)

    print()
    print(f"Saved plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
