from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.landing_env_real2sim import LandingEnv


def main():
    env = LandingEnv()
    check_env(env, warn=True)
    env = Monitor(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./runs/checkpoints_new",
        name_prefix="ppo_landing_new",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./runs/tensorboard_v3",
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        ),
        device="cpu",
    )

    model.learn(
        total_timesteps=300_000,
        callback=checkpoint_callback,
    )

    model.save("./runs/ppo_landing_residual_new_final")

    print("Training finished.")
    print("Saved model: ./runs/ppo_landing_residual_new_final.zip")


if __name__ == "__main__":
    main()
