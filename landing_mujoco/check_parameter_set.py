#!/usr/bin/env python3
"""CLI: validate a physical-parameter config, or run a short smoke episode.

    python3 -m landing_mujoco.check_parameter_set --parameter-set tarot680b_reference
    python3 -m landing_mujoco.check_parameter_set --parameter-set ugrp_vehicle_measured

The reference set is runnable today. The measured set fails fast and lists
exactly which measurements are still missing (task spec section 11).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from landing_mujoco.configs.param_schema import MissingMeasurementError, load_uav_params

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
KNOWN_SETS = {
    "tarot680b_reference": CONFIG_DIR / "tarot680b_reference.yaml",
    "ugrp_vehicle_measured": CONFIG_DIR / "ugrp_vehicle_measured.yaml",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parameter-set",
        required=True,
        choices=sorted(KNOWN_SETS.keys()),
        help="Which physical-parameter config to validate.",
    )
    parser.add_argument(
        "--run-smoke-episode",
        action="store_true",
        help="If the config is complete, also construct MujocoLandingEnv and "
        "run a short deterministic smoke episode.",
    )
    args = parser.parse_args(argv)

    path = KNOWN_SETS[args.parameter_set]
    try:
        params = load_uav_params(path)
    except MissingMeasurementError as e:
        print(str(e), file=sys.stderr)
        return 1

    print(f"OK: {args.parameter_set} is complete and runnable.")
    for key, value in params.summary().items():
        print(f"  {key}: {value}")

    if args.run_smoke_episode:
        import numpy as np

        from landing_mujoco.envs.mujoco_landing_env import MujocoLandingEnv
        from landing_rl.envs.landing_env import LandingConfig

        env = MujocoLandingEnv(params, config=LandingConfig(), deterministic_physics=True)
        obs, info = env.reset(seed=0)
        for _ in range(20):
            obs, reward, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
            if term or trunc:
                break
        print(f"Smoke episode OK ({args.run_smoke_episode=}): "
              f"parameter_set={info['parameter_set']}, final pos={info['pos']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
