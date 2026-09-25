import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import mujoco
import gymnasium
import stable_baselines3
import numpy as np

print("MuJoCo version:", mujoco.__version__)
print("Gymnasium version:", gymnasium.__version__)
print("Stable-Baselines3 version:", stable_baselines3.__version__)
print("NumPy version:", np.__version__)

# Canonical-environment smoke check (16-D obs, 3-D action, PID-only steps, finite outputs).
from landing_rl.evaluation.smoke import main as landing_env_smoke

status = landing_env_smoke()
if status == 0:
    print("Smoke test passed.")
sys.exit(status)
