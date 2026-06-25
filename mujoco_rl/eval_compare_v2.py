import numpy as np
from stable_baselines3 import PPO

from envs.landing_env_real2sim import LandingEnv
from collections import Counter

def run_eval(name, policy_fn, n_eval=200):
    env = LandingEnv()

    successes = 0
    final_xy_errors = []
    final_z_errors = []
    steps_list = []
    rewards = []
    failures = 0
    truncations = 0
    failure_reasons = Counter()

    for ep in range(n_eval):
        obs, _ = env.reset(seed=5000 + ep)
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
        failed = bool(last_info.get("failed", False))

        successes += int(success)
        failures += int(failed)
        truncations += int((not success) and (not failed))

        final_xy_errors.append(last_info.get("xy_error", np.nan))
        final_z_errors.append(last_info.get("z_error", np.nan))
        steps_list.append(steps)
        rewards.append(total_reward)
        reason = last_info.get("failure_reason", "none")
        if failed:
            failure_reasons[reason] += 1

    print()
    print(f"===== {name} =====")
    print(f"Success rate:       {successes}/{n_eval} = {successes / n_eval * 100:.1f}%")
    print(f"Failure count:      {failures}/{n_eval}")
    print(f"Timeout count:      {truncations}/{n_eval}")
    print(f"Mean final XY error:{np.nanmean(final_xy_errors):.4f} m")
    print(f"Mean final Z error: {np.nanmean(final_z_errors):.4f} m")
    print(f"Mean steps:         {np.mean(steps_list):.1f}")
    print(f"Mean reward:        {np.mean(rewards):.1f}")
    print(f"Failure reasons:    {dict(failure_reasons)}")

def main():
    model = PPO.load("./runs/ppo_landing_residual_new_final", device="cpu")

    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    def pid_only_policy(obs):
        return np.zeros(3, dtype=np.float32)

    rng = np.random.default_rng(0)

    def random_residual_policy(obs):
        return rng.uniform(-1.0, 1.0, size=3).astype(np.float32)

    run_eval("PID only v2 env", pid_only_policy, n_eval=200)
    run_eval("Random residual new env", random_residual_policy, n_eval=200)
    run_eval("PPO residual new env", ppo_policy, n_eval=200)


if __name__ == "__main__":
    main()
