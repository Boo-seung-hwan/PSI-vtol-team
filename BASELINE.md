# RL Baseline v0.1

What a new team member must know about the UGRP precision-landing RL baseline as it exists **today**. Everything here
was checked against source code, tests, or saved artifacts; where something could not be verified it says so.

| | **CURRENT RL BASELINE** | EXPERIMENTAL / NEXT PHYSICS BACKEND |
|---|---|---|
| Package | `landing_rl/` | `landing_mujoco/` |
| Environment | `landing_rl.envs.landing_env.LandingEnv` | `landing_mujoco.envs.mujoco_landing_env.MujocoLandingEnv` |
| Physics | analytical, compact 6DoF-inspired, mass-normalized | MuJoCo rigid body |
| Vehicle | none: no mass, inertia or geometry anywhere in the dynamics | measured UGRP parameters exist (6.408 kg, measured Ixx/Iyy/Izz, 0.360 m X-frame, skid contacts) |
| Trained PPO policy | **yes, this is what the PPO baseline was trained and evaluated on** | **no** |
| Can be built from the measured vehicle? | not applicable | **no**: 5 propulsion / response fields are still unmeasured |

## MuJoCo status (read this)

`landing_mujoco` is **experimental / provisional**. The measured UGRP vehicle parameters
(`landing_mujoco/configs/ugrp_vehicle_measured.yaml`) reach the compiled MuJoCo rigid-body model, but
**the currently trained PPO baseline was NOT trained in the measured MuJoCo model**, and no policy has been trained or
evaluated in it. `MujocoLandingEnv` cannot even be constructed from the measured set yet:

```bash
python -m landing_mujoco.check_parameter_set --parameter-set ugrp_vehicle_measured   # exits 1 by design
# Missing: max_collective_thrust_n, tau_roll_s, tau_pitch_s, tau_thrust_s, actuator_delay_s
```

Only the reference set (`tarot680b_reference`, a different, unmeasured airframe) can run the MuJoCo environment.
As implemented, the MuJoCo path applies thrust as `mass x acceleration` and torque as `J x desired angular
acceleration + w x Jw` (`landing_mujoco/dynamics/actuation_model.py`, `identified_inner_loop.py`), so the measured mass
and inertia largely cancel out of the closed-loop flight response and matter mainly for contact and for the `T_max / m`
limit (a reading of the code, not a simulation experiment). Treat any statement of the form "the policy was trained on the measured 6.408 kg vehicle" as false.

## Canonical environment

`landing_rl/envs/landing_env.py` (`LandingEnv`, `LandingConfig`). It is the environment used by every training and
evaluation script in `mujoco_rl/` and by the trained PPO baseline. It is an analytical, compact 6DoF-inspired model
(velocity command -> acceleration demand -> attitude / thrust setpoint -> first-order body-rate and thrust response ->
NED translation, with drag, wind, process noise, ground effect and a simple contact / bounce model). It does not model
motors, rotors or PX4 itself. `mujoco_rl/envs/env_prototype.py` is the frozen legacy reference kept for the regression
tests; do not use it for new work.

Configurations are Python literals in the scripts (`make_config()`); there are no config files. The bare
`LandingConfig()` defaults (no wind, no target noise, no ground effect) match **no** trained distribution.

## Policy architecture

```
v_cmd = saturate( v_PID + delta_v_PPO )         then a landing-descent gate
        |             |
        |             +-- bounded residual: 3-D action in [-1, 1] scaled to m/s
        +-- baseline PID velocity controller on the (delayed, noisy, stale) target measurement
v_cmd -> attitude / thrust setpoint -> plant model -> next state
```

PPO does not replace attitude, rate or actuator control. Implementation: `landing_rl/controllers/baseline_controller.py`,
orchestrated in `LandingEnv.step()` in `landing_rl/envs/landing_env.py`. Checkpoint network: actor and critic MLPs with 2 hidden
layers x 128 units (read from the checkpoint).

## Observation contract (16-D, `float32`)

Order as built by `LandingEnv._get_obs()` and documented in the `LandingEnv` docstring:

| index | name | definition |
|---|---|---|
| 0-2 | `dx, dy, dz` | `obs_target - pos` [m], NED; `obs_target` is the delayed / noisy / stale / possibly dropped target measurement |
| 3-5 | `vx, vy, vz` | `vel + N(0, 0.015)` [m/s], NED |
| 6-8 | `ax, ay, az` | `accel + N(0, 0.05)` [m/s^2], clipped to +-4.0; `accel` is the finite difference `(vel - vel_before) / dt` |
| 9-11 | `roll, pitch, yaw_error` | `[roll, pitch, wrap_pi(target_yaw - yaw)] + N(0, 1 deg)` [rad] |
| 12-14 | `prev_action_x, _y, _z` | previous raw (pre-delay) normalized action, clipped to [-1, 1] |
| 15 | `target_valid` | 1.0 if the delayed measurement is valid, 0.0 on dropout |

Space bounds: +-[30, 30, 30, 6, 6, 6, 4, 4, 4, pi, pi, pi, 1, 1, 1, 1]. The frozen VecNormalize statistics normalize
this vector; the policy never sees raw observations.

## Action contract (3-D)

`Box(-1, 1, shape=(3,), float32)`, a normalized residual velocity command in NED axes, clipped to [-1, 1]:

```
residual_x = action[0] * 0.25 m/s     (North)
residual_y = action[1] * 0.25 m/s     (East)
residual_z = action[2] * 0.05 m/s     (Down; positive = faster descent)
```

(`LandingConfig.residual_xy_mps = 0.25`, `residual_z_mps = 0.05`; the final command is limited to
`|v_xy| <= 0.90`, `|v_z| <= 0.38` m/s.) The residual is zeroed while the target is invalid.

## Coordinate convention

Policy and analytical environment: **NED** world (+x North, +y East, **+z Down**), **FRD** body, attitude
`R = Rz(yaw) Ry(pitch) Rx(roll)`. The pad target is the origin, `z = 0` is the ground, altitude is `-z`.
**A positive z velocity is downward, i.e. descent.** Positive roll produces East acceleration and positive pitch produces
North-negative acceleration (`landing_rl/dynamics/inner_loop_command_model.py`). Verified by code and by
`landing_rl/tests/test_initial_state.py::test_ned_altitude_convention`; the same commands are checked end to end through
the MuJoCo stack in `landing_mujoco/tests/test_sign_conventions.py`. The MuJoCo backend converts NED/FRD to its own
NWU/FLU frames in one place (`landing_mujoco/coordinates/transforms.py`); the policy interface is unchanged.

## Control rate

20 Hz (`LandingConfig.dt = 0.05 s`), with per-step timing jitter in the evaluated configs (`dt_jitter_std = 0.010`).
Episodes are capped at `max_steps = 500` in the baseline evaluation config.

## Model / normalization pair of record

| | file | SHA256 |
|---|---|---|
| PPO model | `ppo_landing_residual_v4_stage2_contact_final.zip` | `9649fcd5e669c099193f9fd05687c22190625642fa373aedb530a49ef8b9949b` |
| VecNormalize | `vecnormalize_v4_stage2_contact.pkl` | `de32e096d7d778a50f06d996f1e8bcd7c340f97fd015348fce108237d23bbfe3` |

**These files are not stored in git** (git-ignored, no shared location yet). Full hashes, expected locations, how to
supply them (`--model`, `--vecnorm`, `UGRP_RL_ARTIFACT_DIR`) and the historical stages are in
[`MODEL_ARTIFACTS.md`](MODEL_ARTIFACTS.md). The two files must always be used together.

## Canonical evaluation

| script | purpose | notes |
|---|---|---|
| `mujoco_rl/eval_compare_v2.py` | matched-seed PID vs random residual vs PPO residual, 200 episodes | printed labels say "stage1 env" for historical reasons; the configuration is the **stage2_eval** config of record, which is identical to the `D_wind` case below |
| `mujoco_rl/eval_robustness_paper.py` | 5-case robustness matrix (A_nominal, B_delay, C_target, D_wind, E_mixed), PID vs PPO, Wilson intervals, paired McNemar | E_mixed is identical to the stage-2 training configuration (`train_ppo_stage2_contact.py`) |

```bash
python -m mujoco_rl.eval_compare_v2 --policy pid                     # no model needed
python -m mujoco_rl.eval_compare_v2 --policy ppo --model M.zip --vecnorm V.pkl
python -m mujoco_rl.eval_robustness_paper --policy pid --output-dir OUT
python -m mujoco_rl.eval_robustness_paper --model M.zip --vecnorm V.pkl --output-dir OUT
```

`--policy pid` (and `random`) never load a PPO model or VecNormalize file. `--policy ppo` and the default `all` require
both and stop with an explicit message if they are missing. **Do not use** `mujoco_rl/eval_env_stage0.py` or
`mujoco_rl/scripts/plot_trajectory.py`: both are deprecated (see below).

## Evaluation seeds

| script | seeds | policies |
|---|---|---|
| `eval_compare_v2.py` | `5000 + episode`, episodes 0..199, i.e. **5000-5199** | PID, PPO, and a random residual drawn from one `default_rng(0)` stream |
| `eval_robustness_paper.py` | `7000 + episode`, i.e. **7000-7199** (`--seed-start`, `--n-eval`) | PID and PPO on identical seeds (paired); random residual only in E_mixed, `default_rng(100000 + seed)` |

PPO is evaluated deterministically. Training was **not seeded**.

## Expected reference result

Verified, not assumed. The archived outputs of 2026-08-31 (outside git, hashes in `MODEL_ARTIFACTS.md`) were reproduced
on 2026-09-25 with the current scripts, current environment code, Python 3.10.12, numpy 2.2.6, gymnasium 1.3.0,
stable-baselines3 2.9.0, torch 2.12.1, mujoco 3.4.0.

`eval_compare_v2` (seeds 5000-5199, 200 episodes). The full printed output, including the timeout episode details, is
identical to the archived file; the PID line was also reproduced with no artifacts present, from a different working
directory.

| policy | success | failures | timeouts | mean final XY error | mean reward |
|---|---|---|---|---|---|
| PID only | **190 / 200 (95.0%)** | 9 (all `excessive_bounce`) | 1 | 0.1005 m | -171.7 |
| random residual | 183 / 200 (91.5%) | 16 (all `excessive_bounce`) | 1 | 0.1120 m | -254.3 |
| PPO residual | **191 / 200 (95.5%)** | 8 (all `excessive_bounce`) | 1 | 0.0990 m | -159.1 |

`eval_robustness_paper` (seeds 7000-7199, 200 episodes per cell, success counts; Wilson 95% intervals and every episode
row are in `robustness_summary.csv` / `robustness_episodes.csv`). The three result CSVs written by the current script are
**byte-identical** to the archived ones (`robustness_summary.csv`, `robustness_episodes.csv`,
`robustness_paired_pid_vs_ppo.csv`).

| case | PID | PPO | random (E_mixed only) | PID fail -> PPO success / PID success -> PPO fail | exact McNemar p |
|---|---|---|---|---|---|
| A_nominal | 187 (93.5%) | 188 (94.0%) | | 1 / 0 | 1.000 |
| B_delay | 189 (94.5%) | 189 (94.5%) | | 1 / 1 | 1.000 |
| C_target | 179 (89.5%) | 182 (91.0%) | | 3 / 0 | 0.250 |
| D_wind | 184 (92.0%) | 187 (93.5%) | | 4 / 1 | 0.375 |
| E_mixed | 175 (87.5%) | 179 (89.5%) | 177 (88.5%) | 4 / 0 | 0.125 |

**What this does and does not show.** It documents that the baseline is reproducible. It is not evidence that PPO
beats PID: in every archived case the paired exact McNemar p-value is 0.125 or larger and the Wilson intervals overlap.
The dominant failure in this environment is `excessive_bounce`, for PID as well as PPO.

## Clean-room reproduction

Performed on a separate fresh Ubuntu machine, fresh clone of `refactor/landing-rl-architecture` at commit
`553fdfd44da9762ab878294b7dabb891c4172ad9`, Python 3.10.12, new venv, `pip install -r requirements.txt` (before
the matplotlib fix below).

* `landing_rl` tests: 233 OK, skipped=1 (artifact-dependent checkpoint test) without artifacts; 238 OK, no skip with
  the pair of record present.
* `landing_mujoco` tests: 221 OK.
* `system_id` tests: 67 run, OK, skipped=6 (a `ResourceWarning` in `test_loader.py` was observed but did not fail
  the suite).
* `python -m landing_rl.evaluation.smoke`: PASS, `obs=(16,)`, `action=(3,)`.
* `eval_compare_v2 --policy pid`: 190/200 = 95.0%, matching the table above.
* `eval_compare_v2 --policy ppo`: 191/200 = 95.5%, matching the table above, after copying in the pair of record
  and verifying its SHA256 against `MODEL_ARTIFACTS.md`.

Two setup issues surfaced by this run were fixed after `553fdfd` (see `requirements.txt` and the README quick
start): `matplotlib` is required by `landing_rl/tests/test_entry_point_migration.py` (it loads
`mujoco_rl/scripts/plot_trajectory.py`, which imports `matplotlib.pyplot` at module scope) but was listed as
commented-out/optional in `requirements.txt`; and a fresh Debian/Ubuntu install may need the OS `python3-venv`
package before `python3 -m venv` works. The clean-room run above installed matplotlib by hand after hitting the
first issue; it predates both fixes.

Independently re-run 2026-09-28 against the tree that adds those two fixes (commit `9cb8027`, a docs-only change
on top of `553fdfd`), on WSL2, system Python 3.10.12 (not a clean venv; matplotlib and the rest of
`requirements.txt` were already present, so this run does not by itself validate the requirements.txt fix), with
the pair of record supplied via `UGRP_RL_ARTIFACT_DIR`. Every result above was reproduced identically: 233/238,
221, 67 (skipped=6, same `ResourceWarning`), smoke PASS, PID 190/200, PPO 191/200 = 95.5%.

This confirms the baseline is reproducible across two machines and across the docs-only fix commit. It is not
additional evidence that PPO statistically outperforms PID (see the McNemar/Wilson caveat above), and the PPO
policy was still trained on the analytical `landing_rl` environment, not the measured MuJoCo model.

## Training lineage (what is and is not reproducible)

Training from scratch is **not** reproducible from this repository. Facts read from the artifacts (`num_timesteps` of
each checkpoint, sample counts and the `target_valid` mean of each VecNormalize file):

| artifact | parent | target dropout while training (derived from its VecNormalize) |
|---|---|---|
| stage 0 | none | 0% |
| stage 1 (`train_ppo_v3_long.py`) | stage 0 (script) | 1.0% |
| stage 1 rewardfix1 (`train_ppo_stage1_1m.py`) | stage 0 (script) | 1.0% |
| stage 2 v3 | **stage 0** (inferred, no script) | 5.0% |
| **v4 stage 2 = baseline** (`train_ppo_stage2_contact.py`) | stage 2 v3 (script) | 5.0% |

The stage-1, rewardfix1 and stage-2 v3 checkpoints all sit at exactly 1,515,520 timesteps (stage 0 = 507,904 plus one
1M-step run each), so they are **sibling branches of stage 0, not a chain**. There is no script for stage 0 or for stage 2
v3. The `make_config()` in `eval_env_stage0.py` is the stage-1 configuration, not stage 0's (stage 0's VecNormalize
saw no invalid targets). Training scripts still read and write `./runs/...` relative to the working directory (run them
from `mujoco_rl/`).

## Deprecated scripts

* `mujoco_rl/eval_env_stage0.py`: legacy. It evaluated the stage-1 model with stage-0 normalization; the VecNormalize
  file is now the stage-1 one saved with that model. Marked DEPRECATED, prints a warning, not a baseline tool.
* `mujoco_rl/scripts/plot_trajectory.py`: legacy. Loads a pre-v3 model that is not part of the lineage and applies no
  VecNormalize. Marked DEPRECATED, prints a warning; no correct pair can be established, so it was not repaired.

## Known gaps

* **Analytical PID != the real PX4 / ROS2 landing controller.** Simulation: `kp_xy 0.45, kd_xy 0.18, kp_z 0.35,
  kd_z 0.12`, no integral term (`landing_rl/envs/landing_env.py`). Vehicle-side node: `kp 0.35 / 0.35 / 0.22`,
  `ki 0.008 / 0.008 / 0.003`, `kd 0.16 / 0.16 / 0.10`, output limits 0.8 / 0.8 / 0.35, plus deadband, derivative
  filter, slew-rate limit and target smoothing
  (`ros2/ws/src/drone_control/drone_control/precision_landing_controller.py`). The residual was trained on top of the
  simulated PID, not the deployed one.
* **Measured MuJoCo environment is not the training baseline** (see above), and its propulsion / closed-loop parameters
  are unmeasured.
* **Full system identification is not complete.** `system_id/` has preprocessing only; identification, validation and
  results are empty; no UGRP closed-loop parameter has been identified; PX4 gain-tuning status is not recorded.
* **Deployment / inference is not integrated.** No PPO inference code exists under `ros2/`, `jetson/` or `px4/`, and
  how each of the 16 observation elements is produced on the vehicle (acceleration, attitude, previous action) is not
  specified. The camera-to-NED conversion uses yaw only (no roll / pitch compensation), which the simulation does not model.
* No shared hosting for the model artifacts; no seeded or scripted from-scratch training path (above).
