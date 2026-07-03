from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.env_prototype import LandingEnv, LandingConfig


N_ENVS = 4


def make_config():
    return LandingConfig(
        max_steps=420,
        residual_z_mps=0.05,

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


def make_env():
    def _init():
        return Monitor(LandingEnv(make_config()))
    return _init


def main():
    env = DummyVecEnv([make_env() for _ in range(N_ENVS)])

    # Stage 0에서 저장한 normalization 통계 불러오기
    env = VecNormalize.load("./runs/vecnormalize_v3_stage0.pkl", env)

    # Stage 1에서는 새 환경 통계에 적응하도록 계속 업데이트
    env.training = True
    env.norm_reward = True

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // N_ENVS, 1),
        save_path="./runs/checkpoints_v3_stage1_rewardfix1",
        name_prefix="ppo_landing_v3_stage1_rewardfix1",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    model = PPO.load(
        "./runs/ppo_landing_residual_v3_stage0_final.zip",
        env=env,
        device="cpu",
        tensorboard_log="./runs/tensorboard_v3_stage1_rewardfix1",
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
        tb_log_name="PPO_stage1_rewardfix1",
    )

    model.save("./runs/ppo_landing_residual_v3_stage1_rewardfix1_final")
    env.save("./runs/vecnormalize_v3_stage1_rewardfix1.pkl")

    print("Stage 1 rewardfix1 training finished.")
    print("Saved model: ./runs/ppo_landing_residual_v3_stage1_rewardfix1_final.zip")
    print("Saved VecNormalize: ./runs/vecnormalize_v3_stage1_rewardfix1.pkl")


if __name__ == "__main__":
    main()