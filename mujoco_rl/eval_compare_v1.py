import numpy as np
from stable_baselines3 import PPO

from envs.landing_env import LandingEnv


def run_eval(name, policy_fn, n_eval=100):
    env = LandingEnv()

    successes = 0
    final_xy_errors = []
    final_z_errors = []
    steps_list = []
    rewards = []

    for ep in range(n_eval):
        obs, _ = env.reset(seed=2000 + ep)
        done = False
        total_reward = 0.0
        last_info = {}

        steps = 0
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            last_info = info
            steps += 1

        success = bool(last_info.get("success", False))
        successes += int(success)
        final_xy_errors.append(last_info.get("xy_error", np.nan))
        final_z_errors.append(last_info.get("z_error", np.nan))
        steps_list.append(steps)
        rewards.append(total_reward)

    print()
    print(f"===== {name} =====")
    print(f"Success rate: {successes}/{n_eval} = {successes / n_eval * 100:.1f}%")
    print(f"Mean final XY error: {np.nanmean(final_xy_errors):.4f} m")
    print(f"Mean final Z error:  {np.nanmean(final_z_errors):.4f} m")
    print(f"Mean steps:          {np.mean(steps_list):.1f}")
    print(f"Mean reward:         {np.mean(rewards):.1f}")


def main():
    model = PPO.load("./runs/ppo_landing_residual", device="cpu")

    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    def pid_only_policy(obs):
        # action=0 means no RL residual, PID only
        return np.zeros(3, dtype=np.float32)

    rng = np.random.default_rng(0)

    def random_residual_policy(obs):
        return rng.uniform(-1.0, 1.0, size=3).astype(np.float32)

    run_eval("PID only", pid_only_policy, n_eval=100)
    run_eval("Random residual", random_residual_policy, n_eval=100)
    run_eval("PPO residual", ppo_policy, n_eval=100)


if __name__ == "__main__":
    main()
