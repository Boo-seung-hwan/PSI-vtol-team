"""Fast environment smoke check for the canonical landing_rl baseline.

    python -m landing_rl.evaluation.smoke

Imports the canonical ``LandingEnv``, instantiates it, resets with a fixed
seed, verifies the 16-D observation / 3-D action contract, steps the PID-only
(zero residual) policy, checks every output is finite, checks that a seeded
reset is reproducible, and closes the environment. It needs no PPO model, no
VecNormalize file and no MuJoCo, and finishes in well under a second.

It changes no configuration: ``LandingConfig()`` defaults are used.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from landing_rl.envs.landing_env import LandingConfig, LandingEnv

OBS_DIM = 16
ACTION_DIM = 3


def run_smoke(seed: int = 0, n_steps: int = 30) -> dict:
    """Run the smoke check; raises ``AssertionError`` on any contract breach."""
    env = LandingEnv(LandingConfig())
    try:
        assert env.observation_space.shape == (OBS_DIM,), env.observation_space.shape
        assert env.action_space.shape == (ACTION_DIM,), env.action_space.shape
        assert np.array_equal(env.action_space.low, -np.ones(ACTION_DIM))
        assert np.array_equal(env.action_space.high, np.ones(ACTION_DIM))

        obs, _ = env.reset(seed=seed)
        assert obs.shape == (OBS_DIM,), obs.shape
        assert obs.dtype == np.float32, obs.dtype
        assert np.isfinite(obs).all(), obs
        assert obs[-1] in (0.0, 1.0), f"target_valid flag must be 0/1, got {obs[-1]}"
        first_obs = obs.copy()

        zero_action = np.zeros(ACTION_DIM, dtype=np.float32)  # PID only
        steps = 0
        for _ in range(n_steps):
            obs, reward, terminated, truncated, info = env.step(zero_action)
            steps += 1
            assert obs.shape == (OBS_DIM,), obs.shape
            assert np.isfinite(obs).all(), obs
            assert math.isfinite(reward), reward
            assert np.isfinite(info["v_cmd"]).all(), info["v_cmd"]
            assert np.array_equal(info["v_residual"], np.zeros(ACTION_DIM)), info["v_residual"]
            if terminated or truncated:
                break

        env.reset(seed=seed)
        obs_again, _ = env.reset(seed=seed)
        assert np.array_equal(obs_again, first_obs), "seeded reset is not reproducible"
    finally:
        env.close()

    return {"obs_dim": OBS_DIM, "action_dim": ACTION_DIM, "steps": steps, "seed": seed}


def main() -> int:
    result = run_smoke()
    print(
        "landing_rl smoke test passed: "
        f"obs=({result['obs_dim']},) action=({result['action_dim']},) "
        f"PID-only steps={result['steps']} seed={result['seed']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
