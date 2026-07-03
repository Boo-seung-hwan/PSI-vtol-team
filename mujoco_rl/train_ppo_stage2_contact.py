from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.env_prototype import LandingEnv, LandingConfig


N_ENVS = 4


def make_config():
    return LandingConfig(
        # Stage 1 config
        max_steps=500,

        residual_xy_mps = 0.25,
        residual_z_mps=0.05,

        init_xy_range_m=6.0,
        init_altitude_min_m=2.0,
        init_altitude_max_m=7.0,
        init_vel_range_mps=0.60,

        dt_jitter_std=0.010,

        action_delay_steps_min=1,
        action_delay_steps_max=5,
        obs_delay_steps_min=1,
        obs_delay_steps_max=5,

        target_dropout_prob=0.05,
        target_outlier_prob=0.02,
        target_stale_prob=0.08,
        target_noise_xy_std_m=0.06,
        target_noise_z_std_m=0.04,

        wind_accel_xy_max_mps2=0.10,
        wind_accel_z_max_mps2=0.02,
        ground_effect_gain= 0.005,
        landing_descent_bias_mps=0.14,


        contact_enabled=True,
        success_requires_contact=True,

        init_yaw_range_rad=0.5,
        w_yaw=0.05,
    )


def make_env():
    def _init():
        return Monitor(LandingEnv(make_config()))
    return _init


def main():
    env = DummyVecEnv([make_env() for _ in range(N_ENVS)])

    env = VecNormalize.load("./runs/vecnormalize_v3_stage2_contact.pkl", env)

    # Stage 1에서는 새 환경 통계에 적응하도록 계속 업데이트
    env.training = True
    env.norm_reward = True

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // N_ENVS, 1),
        save_path="./runs/checkpoints_v4_stage2_rewardfix1",
        name_prefix="ppo_landing_v4_stage2_rewardfix1",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    model = PPO.load(
        "./runs/ppo_landing_residual_v3_stage2_contact_final.zip",
        env=env,
        device="cpu",
        tensorboard_log="./runs/tensorboard_v4_stage2_rewardfix1",
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
        tb_log_name="PPO_v4_stage2_rewardfix1",
    )

    model.save("./runs/ppo_landing_residual_v4_stage2_contact_final")
    env.save("./runs/vecnormalize_v4_stage2_contact.pkl")

    print("Stage 1 rewardfix1 training finished.")
    print("Saved model: ./runs/ppo_landing_residual_v4_stage2_contact_final.zip")
    print("Saved VecNormalize: ./runs/vecnormalize_v4_stage2_contact.pkl")


if __name__ == "__main__":
    main()
