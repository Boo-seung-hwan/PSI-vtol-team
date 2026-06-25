from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from envs.landing_env import LandingEnv


def main():
    env = LandingEnv()
    check_env(env, warn=True)

    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./runs/tensorboard",
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        ),
        device="cpu",
    )

    model.learn(total_timesteps=100_000)
    model.save("./runs/ppo_landing_residual")

    print("Training finished.")
    print("Saved model: ./runs/ppo_landing_residual.zip")


if __name__ == "__main__":
    main()
