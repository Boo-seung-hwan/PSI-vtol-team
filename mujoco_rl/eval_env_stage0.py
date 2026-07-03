import numpy as np
from collections import Counter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.env_prototype import LandingEnv, LandingConfig


def make_config():
    return LandingConfig(
        init_xy_range_m=4.0,
        init_altitude_min_m=2.0,
        init_altitude_max_m=5.0,
        init_vel_range_mps=0.20,

        dt_jitter_std=0.004,

        action_delay_steps_min=0,
        action_delay_steps_max=2,
        obs_delay_steps_min=0,
        obs_delay_steps_max=2,

        target_dropout_prob=0.01,
        target_outlier_prob=0.0,
        target_stale_prob=0.02,
        target_noise_xy_std_m=0.02,
        target_noise_z_std_m=0.015,

        wind_accel_xy_max_mps2=0.0,
        wind_accel_z_max_mps2=0.0,

        contact_enabled=False,
        success_requires_contact=False,

        init_yaw_range_rad=0.0,
        w_yaw=0.0,
    )

def make_stage0_env():
    return Monitor(LandingEnv(make_config()))


def make_vecnormalize_env(seed: int = 0):
    env = DummyVecEnv([make_stage0_env])

    # 학습 때 저장한 VecNormalize statistics 복원
    env = VecNormalize.load("./runs/vecnormalize_v3_stage0.pkl", env)

    # 평가 모드
    env.training = False
    env.norm_reward = False

    env.seed(seed)
    return env


def run_eval(name, policy_fn, n_eval=200):
    env = make_vecnormalize_env(seed=5000)

    successes = 0
    failures = 0
    truncations = 0

    final_xy_errors = []
    final_z_errors = []
    steps_list = []
    rewards = []
    failure_reasons = Counter()

    for ep in range(n_eval):
        env.seed(5000 + ep)
        obs = env.reset()

        done = False
        total_reward = 0.0
        last_info = {}
        steps = 0

        while not done:
            action = policy_fn(obs)

            # VecEnv API:
            # obs: shape (n_envs, obs_dim)
            # action: shape (n_envs, action_dim)
            obs, reward, done_array, infos = env.step(action)

            done = bool(done_array[0])
            total_reward += float(reward[0])
            last_info = infos[0]
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

    env.close()

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
    # PPO용 env를 만들어서 model에 연결
    ppo_env = make_vecnormalize_env(seed=0)

    model = PPO.load(
        "./runs/ppo_landing_residual_v3_stage1_final.zip",
        env=ppo_env,
        device="cpu",
    )

    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    def pid_only_policy(obs):
        return np.zeros((1, 3), dtype=np.float32)

    rng = np.random.default_rng(0)

    def random_residual_policy(obs):
        return rng.uniform(-1.0, 1.0, size=(1, 3)).astype(np.float32)

    run_eval("PID only stage0 env", pid_only_policy, n_eval=200)
    run_eval("Random residual stage0 env", random_residual_policy, n_eval=200)
    run_eval("PPO residual stage0 env", ppo_policy, n_eval=200)

    ppo_env.close()


if __name__ == "__main__":
    main()