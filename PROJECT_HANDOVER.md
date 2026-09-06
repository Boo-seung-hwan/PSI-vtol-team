# UGRP Precision-Landing Project Handover

Last updated: 2026-09-07 (consistency-audit pass; see §9, 2026-09-07 entry)
Repository: `/home/qntmdghkss/drone_stack_rl_refactor` (a Git worktree of the UGRP drone_stack repository; per CLAUDE.md §5 this path is not guaranteed stable across sessions/machines — always re-verify with `git rev-parse --show-toplevel`)
Branch: `refactor/landing-rl-architecture`
HEAD: `a2260a3988da71f2c266776e2b267e390dd23538` ("Add expanded structural freeze gate")

This document is created for the first time in this session. No prior
`PROJECT_HANDOVER.md` existed in this repository before this entry (verified:
`ls PROJECT_HANDOVER.md` → no such file, this session). Everything below is
sourced from actual source code read this session, `git log`/`git show`
output, and one real test-suite execution. Nothing is reconstructed from
memory or inferred beyond what the cited evidence supports.

## Status labels

`[CURRENT]` `[FACT]` `[CONFIRMED]` `[DECISION]` `[INTERPRETATION]` `[PROPOSAL]`
`[TODO]` `[OPEN]` `[CAUTION]` `[SUPERSEDED]` `[HISTORICAL]` `[CORRECTION]`

---

# 0. CURRENT CANONICAL STATUS

## 0.1 Research objective

[FACT] Develop and validate a lightweight RL precision-landing method for an
onboard UAV (Jetson + PX4), where PPO produces a **bounded residual velocity
correction** on top of a nominal controller, not a replacement for PX4
attitude/rate/actuator control. Canonical relation: `v_cmd = v_nominal +
Δv_RL`. Source: `CLAUDE.md` §1 (current file on disk this session).

## 0.2 Current control architecture

[CONFIRMED — code-verified] The architecture CLAUDE.md §1 describes is
implemented exactly as `v_cmd = v_pid + v_residual`, then saturated and
gated, in `landing_rl/envs/landing_env.py:790-813` (byte-identical logic in
`mujoco_rl/envs/env_prototype.py:1013-1034`). PX4 itself is not present in
this Python simulator — the "PX4 layer" below the velocity command is a proxy
model (see §0.4).

## 0.3 Current live implementation

[CONFIRMED] `mujoco_rl/envs/env_prototype.py :: LandingEnv` is the
environment actually imported by every training and evaluation script in the
repository — `mujoco_rl/train_ppo_stage1_1m.py`, `train_ppo_stage2_contact.py`,
`train_ppo_v3_long.py`, `eval_compare_v2.py`, `eval_env_stage0.py`,
`eval_robustness_paper.py`, `scripts/plot_trajectory.py` all contain
`from envs.env_prototype import LandingEnv, LandingConfig`.

[CONFIRMED] `landing_rl/envs/landing_env.py :: LandingEnv` is a modular
decomposition of the same class (component graph in §1) that is **not**
wired into any training or evaluation entry point: `landing_rl/training/`
and `landing_rl/evaluation/` are empty directories (0 files, verified this
session via `find`/`ls`).

## 0.4 Current simulator abstraction

[CONFIRMED] Reduced-order, closed-loop-response **proxy** model — not full
motor/rotor physics. The code's own docstring states this directly: *"This is
a compact 6DoF-inspired plant model. It does not simulate motor mixing or
rotor angular momentum"* (`mujoco_rl/envs/env_prototype.py:728-730`, verbatim
in `landing_rl/dynamics/legacy_dynamics.py:224-226`). This matches the
modeling direction declared in `CLAUDE.md` §2 ("REDUCED-ORDER PX4 CLOSED-LOOP
MODEL... do NOT assume... a full motor-level Newton-Euler simulator").

**Full Newton-Euler rigid-body dynamics: NOT IMPLEMENTED.** There is no
inertia tensor and no `J·ω̇ = τ − ω×Jω` torque/angular-momentum equation
anywhere in `landing_rl/` or `mujoco_rl/`. Body attitude instead evolves via a
first-order low-pass filter toward a commanded rate
(`legacy_dynamics.py:241-294`).

**Mass / inertia in dynamics: NOT USED.** Repository-wide grep for
`mass|inertia|kg\b` in `landing_rl/`+`mujoco_rl/` (excluding tests) returns
zero hits for any dynamics parameter — the model works entirely in
specific-force (m/s²) units (`thrust_accel`, a "collective thrust per unit
mass" scalar state), so mass never needs to appear. This matches `CLAUDE.md`
§4's instruction not to invent inertia/mass values, since none is currently
consumed by the simulator at all.

## 0.5 Structural freeze / known-good baseline

[CONFIRMED — verified this session by actually running the suite]

```
cd /home/qntmdghkss/drone_stack_rl_refactor
python3 -m unittest discover -s landing_rl/tests -p "test_*.py"
```

Result: **192 tests, 0 failures, 0 errors, exit code 0, "OK"** (full run,
109.25s). This includes the 5 `test_checkpoint_compatibility.py` tests, which
ran (were not skipped) — meaning the external primary checkpoint artifacts
at `/home/qntmdghkss/drone_stack/mujoco_rl/runs/ppo_landing_residual_v4_stage2_contact_final.zip`
and `vecnormalize_v4_stage2_contact.pkl` were found and loaded successfully
against both the OLD (`mujoco_rl`) and NEW (`landing_rl`) environments.

The structural-freeze matrix (`test_structural_freeze.py`) reported: **140
paired OLD-vs-NEW episodes + 6 NEW-only-determinism episodes = 46,958 total
steps**, all under exact (`np.array_equal`, no tolerance) parity, across 4
configs (`default`, `stage0_eval`, `stage2_eval`, `stage2_train`) and forced
contact/bounce/ground-effect/motor-cutoff regimes.

- **Pre-refactor frozen baseline**: tag `pre-landing-rl-refactor-2026-09-02`
  → commit `1bd64aa` (`git tag -l` confirms only this one tag exists; full
  tag message and detail kept in §3, not repeated here).
- **Last known-good structural-equivalence commit**: HEAD, `a2260a3` —
  regression suite passes at this exact commit, verified this session.

## 0.6 Current system-identification strategy

These are two separate claims — an active *research direction* and the
*implementation state* — kept deliberately distinct:

[DECISION — see §2 items 4–5] The active SI direction is to identify the
**closed-loop** command→response mapping (`v_cmd→velocity/accel`,
`attitude cmd→attitude`, `rate cmd→body-rate`, `thrust cmd→vertical accel`,
command timing→delay, perception→observation delay/noise/dropout), not
individual motor/propeller physics, and to treat final closed-loop SI as
gated on PX4 gains being sufficiently stabilized. This is a currently-active
canonical research direction (recorded in `CLAUDE.md` §2/§3/§19/§29), not a
statement about what has been built.

[CONFIRMED] **`system_id/` is entirely unimplemented.** `system_id/preprocessing/`,
`identification/`, `validation/`, `results/` all exist as directories but
contain **0 files** (verified via `find`/`ls -la` this session). NOT
IMPLEMENTED. The active-direction claim above does not imply any of this
exists yet — it does not.

## 0.7 Current PX4 / gain-tuning status

NOT RECORDED / NOT VERIFIED. No PX4 gain-tuning log, ULog analysis output, or
gain-stabilization status document was found anywhere in this repository
(`px4/`, `system_id/`, `docs/` contain no such record). `README.md` §2
pins a specific PX4 SITL version (`release/1.15`, commit `85df8c2281`) for
the Docker/Gazebo/ROS2 stack, but this is a **software-version pin for
simulation**, not evidence about real-vehicle gain-tuning state. Whether PX4
gains are "sufficiently stabilized" per `CLAUDE.md` §3's precondition for
final closed-loop SI is UNKNOWN from repository evidence alone.

## 0.8 Current PPO / evaluation status

[CONFIRMED] The primary checkpoint of record is
`ppo_landing_residual_v4_stage2_contact_final.zip` +
`vecnormalize_v4_stage2_contact.pkl`, held externally at
`/home/qntmdghkss/drone_stack/mujoco_rl/runs/` (outside this worktree,
read-only, not modified this session). "Primary" here comes from
`landing_rl/tests/test_checkpoint_compatibility.py:9-10,77-79` explicitly
naming this pair as the target artifact — not from file recency — and the
checkpoint-compatibility tests passed against it this session (§0.5).

Earlier-stage checkpoints, periodic SB3 checkpoints, TensorBoard logs, and
two eval-output files also exist in that same directory; see §4 and §8 for
the full inventory and an explicit note on which claims about their
ordering are code-verified versus mtime-only. File contents of the two
`.txt` eval outputs were **not read** this session — their existence and
mtimes are FACT; their quantitative content is NOT VERIFIED here.

[CONFIRMED] Evaluation config of record = `mujoco_rl/eval_compare_v2.py ::
make_config()` (the "stage2_eval" config). Verified to match
`landing_rl/tests/test_legacy_regression_contract.py :: CONFIG_SPECS`'s
`"stage2_eval"` entry by the passing test
`test_config_of_record_matches_eval_compare_v2` (part of the 192-test run
above).

## 0.9 Major known gaps

- `landing_rl/` is structurally verified equivalent but **not wired** into
  any training/evaluation entry point (§0.3).
- `system_id/` is **completely unimplemented** — 0 files in all 4 subdirectories (§0.6).
- `landing_rl/configs/{environment,training,vehicle}/` are empty directories
  — no config files exist there; all effective training/eval configs
  currently live as Python literals inside individual `mujoco_rl/*.py` scripts.
- PX4 gain-tuning status is NOT RECORDED anywhere in-repo (§0.7).
- No `PROJECT_HANDOVER.md` existed before this session.
- `README.md:5-9` states *"Current working branch: orange-fix"*, but the
  actually-checked-out branch this session is `refactor/landing-rl-architecture`
  — a documentation/Git mismatch (README not updated since the branch changed).
- Several disturbance/noise/reward config fields default to `0.00` in the
  shared `LandingConfig` dataclass with the previously-used nonzero value
  left only as a trailing comment (`wind_accel_*`, `target_noise_*`,
  `ground_effect_gain`, `w_yaw`) — see §7 items 2, and the `"default"` /
  `LandingConfig()` config is therefore not representative of any trained
  policy's actual training distribution.

## 0.10 Immediate next step

[OPEN] — no user-approved "next step" decision is recorded anywhere in this
repository (this is the first handover; nothing to inherit). Do not treat any
of the following as decided: extracting reward/termination into its own
module, wiring `landing_rl` into training/evaluation, populating
`landing_rl/configs/*`, or starting `system_id/preprocessing`. These are only
structurally implied by `CLAUDE.md` §14.2/§17/§19 as areas the project intends
to eventually cover — they are `[PROPOSAL]`, not `[DECISION]`, until the user
explicitly picks one.

---

# 1. Project architecture

[CONFIRMED — verified by passing regression/structural-freeze tests, §0.5]

```
Training entry point (mujoco_rl/train_ppo_*.py; landing_rl/training/ is empty)
    ↓  from envs.env_prototype import LandingEnv, LandingConfig
Environment construction
    mujoco_rl/envs/env_prototype.py :: LandingEnv          [LIVE reference]
    landing_rl/envs/landing_env.py  :: LandingEnv          [modular, unwired]
        ↓
Controller           landing_rl/controllers/baseline_controller.py :: BaselineController
Dynamics / plant      landing_rl/dynamics/plant_model.py :: PlantModel
                        → landing_rl/dynamics/legacy_dynamics.py :: LegacyVehicleDynamics
                        → landing_rl/contact/contact_model.py :: ContactModel
Perception / obs      landing_rl/perception/{target_measurement,obs_latency,observation_noise}.py
Disturbance / delay   landing_rl/disturbances/disturbance_model.py,
                        landing_rl/envs/{action_latency,loop_timing}.py
Contact               landing_rl/contact/contact_model.py
Reward / termination  inline in landing_rl/envs/landing_env.py :: LandingEnv.step (lines 820-935)
                        — NOT extracted into its own module in either implementation
```

`mujoco_rl/envs/env_prototype.py` implements the identical logic inline
(monolithic, 1221 lines) rather than through the component split above. Both
implementations are proven byte-identical in behavior by the OLD-vs-NEW
parity suite (§0.5).

Current observation/action contract (both implementations, identical):

- Observation `(16,) float32`: `[dx,dy,dz, vx,vy,vz, ax,ay,az, roll,pitch,
  yaw_error, prev_action_x,y,z, target_valid]` — position error uses the
  delayed/noisy/dropout-aware target measurement; velocity/accel/attitude
  channels are instantaneous + Gaussian-noised (not delayed).
- Action `(3,) ∈[-1,1]³`: normalized residual velocity command, scaled to
  `Δv = a·[residual_xy_mps, residual_xy_mps, residual_z_mps] = a·[0.25,0.25,0.05]`.

Nominal controller structure: PD velocity law (`kp_xy=0.45, kd_xy=0.18,
kp_z=0.35, kd_z=0.12`) with two descent-gate heuristics near the pad, then
`v_cmd = clip(v_pid + Δv)` — `landing_rl/controllers/baseline_controller.py`.

---

# 2. Major research decisions

The following are recorded as `[DECISION]` because they are already present
as an approved, standing project direction in `CLAUDE.md` §29 ("CURRENT
HIGH-LEVEL DECISIONS TO RESPECT... Unless PROJECT_HANDOVER.md records a later
superseding decision, respect the following"). No code-inference was used to
produce this list — it is transcribed verbatim-in-substance from the
instruction file as the existing canonical decision record.

**Provenance**: `CLAUDE.md` is user-maintained local tooling, not a file
Claude authored or may edit on its own initiative — `CLAUDE.md` §23 states it
"may be edited only when the user explicitly requests an instruction
update," and it is deliberately kept out of Git (§34). Its presence and
content in the working tree is therefore evidence of user authorship/
approval, not Claude inference. §29's list is treated here as the
bootstrap set of user-approved canonical decisions this project already
had before this handover existed; this handover did not create, infer, or
select any of the 8 items below — it only cites where each already lives.

1. `[DECISION]` Use bounded residual PPO at the velocity-setpoint level.
   (`CLAUDE.md` §29.1)
2. `[DECISION]` Preserve PX4 low-level stabilization (attitude/body-rate/
   actuator control is not replaced by RL). (`CLAUDE.md` §29.2)
3. `[DECISION]` Use a reduced-order closed-loop vehicle model in the near
   term. (`CLAUDE.md` §29.3)
4. `[DECISION]` Prioritize closed-loop command-response system identification
   over component-level (motor/propeller) identification. (`CLAUDE.md` §29.4)
5. `[DECISION]` Perform final closed-loop SI only after relevant PX4 gain
   tuning is sufficiently stabilized. (`CLAUDE.md` §29.5)
6. `[DECISION]` Do not immediately rebuild the simulator as a full
   motor-level Newton-Euler model without evidence that the added fidelity is
   necessary. (`CLAUDE.md` §29.6)
7. `[DECISION]` Treat simulator fidelity as a validation problem against real
   UAV data, not physics-engine sophistication. (`CLAUDE.md` §29.7)

`CLAUDE.md` §29 itself notes: *"These are architectural/research principles,
not permission to assume specific numerical parameter values."*

8. `[DECISION]` A structural refactor must not secretly change observation/
   action/controller-gain/reward/termination/delay/noise/contact/dynamics/PPO
   behavior; structural refactor and physics/behavior change must stay
   separated in Git and in this document. (`CLAUDE.md` §15) — this decision
   is the one actively enforced by the passing structural-freeze test suite
   in §0.5.

No other research decision (e.g., a specific next architecture phase, a
specific SI method, a specific hardware target date) was found recorded as
user-approved anywhere in the repository. Any such item is `[OPEN]` /
`[PROPOSAL]` until recorded here with explicit approval.

---

# 3. Simulator development history

[HISTORICAL — reconstructed from `git log`/`git show`, this session;
message text quoted verbatim, no invented detail]

Full commit list obtained via `git log --oneline --all --decorate`, filtered
to the landing-RL-relevant subsequence (oldest → newest):

| commit | message |
|---|---|
| `41a7239` | Add MuJoCo RL landing prototype |
| `2249a86` (tag: `rl-landing-prototype`) | Add MuJoCo RL landing prototype (same content point, tagged) |
| `d0b41a3` | Add initial residual RL landing environment and baseline comparison |
| `7dddd1c` | Add MuJoCo landing RL stage environments |
| `4068894` | Update landing RL evaluation and remove obsolete training scripts |
| `ad34334` | Remove legacy landing environments and training files |
| `1bd64aa` (tag: `pre-landing-rl-refactor-2026-09-02`) | Update Jetson configuration and scripts — **frozen pre-refactor baseline** |

Tag detail (`git show pre-landing-rl-refactor-2026-09-02`, this session):
annotated tag, message *"UGRP landing stack before modular RL environment
refactor"*, tagger Boo-seung-hwan, dated 2026-09-02. `git tag -l` confirms
this is the only tag in the repository.

Then, on branch `refactor/landing-rl-architecture` (all 16 commits present
and inspected this session):

| commit | message | what it extracted (verified by reading the resulting file this session) |
|---|---|---|
| `697bd62` | Add legacy landing environment regression contract | `landing_rl/tests/test_legacy_regression_contract.py` — the exact OLD-vs-NEW parity harness |
| `4b1d9c8` | Add parity-verified landing environment copy | `landing_rl/envs/landing_env.py` created as byte-for-byte copy of `mujoco_rl/envs/env_prototype.py` |
| `e94754b` | Extract deterministic landing baseline controller | `landing_rl/controllers/baseline_controller.py` (`pid_velocity`, `scale_action`, `combine_and_limit`, `apply_descent_gate`) |
| `369a8e4` | Extract per-episode wind disturbance model | `landing_rl/disturbances/disturbance_model.py` |
| `29cc099` | Extract landing control-loop timing sampler | `landing_rl/envs/loop_timing.py` |
| `ed728f1` | Add PPO and VecNormalize compatibility regression | `landing_rl/tests/test_checkpoint_compatibility.py` |
| `ee6cd37` | Extract landing action-path latency | `landing_rl/envs/action_latency.py` |
| `192e803` | Extract raw target measurement model | `landing_rl/perception/target_measurement.py` |
| `828e13b` | Extract target observation latency model | `landing_rl/perception/obs_latency.py` |
| `4c7ec6a` | Extract per-episode response alpha sampling | `landing_rl/dynamics/response_alphas.py` |
| `e21a0d4` | Extract landing initial-state sampling | `landing_rl/envs/initial_state.py` |
| `f0c0469` | Extract vehicle observation noise sampling | `landing_rl/perception/observation_noise.py` |
| `3aea4e7` | Extract dynamics process noise sampling | `landing_rl/dynamics/process_noise.py` |
| `59ee330` | Extract landing ground contact model | `landing_rl/contact/contact_model.py` |
| `7667e07` | Extract legacy vehicle dynamics backend | `landing_rl/dynamics/legacy_dynamics.py`, `landing_rl/dynamics/vehicle_state.py` |
| `6725096` | Extract plant orchestration model | `landing_rl/dynamics/plant_model.py` |
| `a2260a3` (HEAD) | Add expanded structural freeze gate | `landing_rl/tests/test_structural_freeze.py` — expanded seed/config/regime matrix + structural-invariant checks |

Each extraction commit's stated intent ("no behavior change, verbatim copy")
is what the passing regression suite in §0.5 actually verifies at HEAD — this
is not asserted from the commit messages alone.

---

# 4. PPO training and evaluation history

[CONFIRMED — existence, filenames, and mtimes] from an external artifact
directory listing this session, `/home/qntmdghkss/drone_stack/mujoco_rl/runs/`
(read-only, not modified). The checkpoint/VecNormalize files below are listed
in filename-implied stage order:

| stage (by filename) | checkpoint | VecNormalize | mtime |
|---|---|---|---|
| stage0 | `ppo_landing_residual_v3_stage0_final.zip` | `vecnormalize_v3_stage0.pkl` | 2026-06-27 14:02 |
| stage1 | `ppo_landing_residual_v3_stage1_final.zip` | `vecnormalize_v3_stage1.pkl` | 2026-06-27 16:28 |
| stage1_rewardfix1 | `ppo_landing_residual_v3_stage1_rewardfix1_final.zip` | `vecnormalize_v3_stage1_rewardfix1.pkl` | 2026-06-28 17:16 |
| stage2_contact (v3) | `ppo_landing_residual_v3_stage2_contact_final.zip` | `vecnormalize_v3_stage2_contact.pkl` | 2026-06-30 14:14 |
| stage2_rewardfix1 (v4, **primary**, §0.8) | `ppo_landing_residual_v4_stage2_contact_final.zip` | `vecnormalize_v4_stage2_contact.pkl` | 2026-06-30 14:59 |

`[CAUTION]` The row order above matches both the filename convention and the
file mtimes, and the two happen to agree — but mtime is filesystem metadata
only (it can reflect a copy/touch time, not necessarily a training-completion
time) and by itself does not prove training lineage. Do not read this table
as a verified curriculum sequence; read the split below instead:

- `[CONFIRMED — code-verified]` Two of the four stage-to-stage transitions
  are directly evidenced by training-script code read this session:
  `mujoco_rl/train_ppo_stage1_1m.py` and `mujoco_rl/train_ppo_v3_long.py`
  each `PPO.load("./runs/ppo_landing_residual_v3_stage0_final.zip", env=env,
  ...)` + `VecNormalize.load("./runs/vecnormalize_v3_stage0.pkl", ...)` and
  save to the `v3_stage1_final`/`vecnormalize_v3_stage1.pkl` pair — i.e.
  stage0→stage1 is code-confirmed. `mujoco_rl/train_ppo_stage2_contact.py`
  loads `ppo_landing_residual_v3_stage2_contact_final.zip` +
  `vecnormalize_v3_stage2_contact.pkl` and saves to the v4
  `stage2_contact_final`/`vecnormalize_v4_stage2_contact.pkl` pair — i.e.
  stage2_contact(v3)→stage2_rewardfix1(v4) is code-confirmed. Both scripts
  use `reset_num_timesteps=False` and continue the loaded `VecNormalize`
  running statistics — for these two transitions specifically, the
  curriculum is confirmed to carry forward both policy weights and
  observation/reward normalization state, not weights alone.
- `[CAUTION / NOT VERIFIED]` The other two transitions implied by the table
  order — stage1→stage1_rewardfix1, and stage1(or stage1_rewardfix1)→
  stage2_contact(v3) — have **no corresponding training script anywhere in
  this repository**. No script that loads a `v3_stage1*` checkpoint and
  saves a `v3_stage1_rewardfix1_final.zip` or `v3_stage2_contact_final.zip`
  was found. Their position in the sequence above is an
  `[INTERPRETATION]` from filename convention + mtime ordering only, not a
  code-verified fact. Likewise, no `stage0` training script exists in this
  repository at all — see the note below the table.

Eval outputs present but **not read** this session (content NOT VERIFIED,
existence/mtime is FACT): `eval_compare_v2_paper_200ep.txt` (2026-08-31
04:23), `eval_robustness_paper.txt` (2026-08-31 11:23), `paper_eval/`
directory (2026-08-31 11:48).

`eval_compare_v2.py` runs matched-seed (seed 5000+ep) comparisons of
PID-only / random-residual / PPO-residual policies against the v4 checkpoint
under the "stage2_eval" config (§0.8). `eval_robustness_paper.py` defines a
5-case robustness matrix (`A_nominal`, `B_delay`, `C_target`, `D_wind`,
`E_mixed`) varying delay/target-uncertainty/wind, with paired PID-vs-PPO
McNemar-style output — its results were not read this session.

**No `stage0` training script was found in this repository** — only its
output artifacts exist externally; the stage0 config is only recoverable via
`mujoco_rl/eval_env_stage0.py :: make_config()`.

---

# 5. System-identification track

[CONFIRMED] `system_id/preprocessing/`, `system_id/identification/`,
`system_id/validation/`, `system_id/results/` all exist as directories with
**zero files** in each (verified via `find . -type f` and `ls -la` on all
four this session). **NOT IMPLEMENTED.**

No ULog ingestion code, no delay/velocity/attitude/rate/thrust identification
code, no validation/plotting code, and no exported parameter sets exist
anywhere in the repository at HEAD. This directly matches `CLAUDE.md` §19's
description of `system_id/`'s *intended* responsibilities — none of them are
built yet.

---

# 6. PX4 / hardware / real-flight track

This session did not perform a deep audit of `px4/`, `ros2/`, or `jetson/`
source code — only `README.md` (top-level setup doc) and directory listings
were inspected. The following is limited to what that gives:

[FACT, from `README.md` §2] The ROS2 mission code targets PX4 1.15-style
unversioned topics (`/fmu/out/vehicle_status`, `/fmu/out/vehicle_local_position`,
`/fmu/in/trajectory_setpoint`, `/fmu/in/offboard_control_mode`). Recommended
PX4 pin: branch `release/1.15`, commit `85df8c2281`. This is a SITL/Gazebo
software configuration, not a statement about real-hardware gain tuning.

[FACT, from `README.md` §9] Vision pipeline uses a YOLO model
`ros2/ws/src/drone_vision/models/aruco_best.pt` (class 0 = `yolo-marker`,
untracked binary, not committed).

NOT VERIFIED / NOT RECORDED: real-flight test history, PX4 gain-tuning
values or status, any flight-log/ULog dataset, any hardware measurement
(mass, inertia, thrust curve, etc.). None of these were found anywhere in the
repository. Per `CLAUDE.md` §4, such values must remain `UNKNOWN` /
`NOT MEASURED` until direct evidence appears.

---

# 7. Known bugs and technical debt

All items below were found by direct source reading this session (and, where
noted, corroborated by a passing test that explicitly freezes the behavior
as intentional-but-dead-code rather than a live bug):

1. **Dead per-episode RNG parameters.** `vel_response_alpha` (3,) and
   `attitude_response_alpha` (2,) are sampled every `reset()` and reported in
   `info`, but are never read by any dynamics computation
   (`landing_rl/dynamics/response_alphas.py:6-9`; confirmed dead by the
   module's own docstring, and by the passing test
   `test_H_vel_and_attitude_response_alpha_consume_rng_but_stay_dynamics_inactive`
   in `test_structural_freeze.py`). `cfg.attitude_process_noise_std_rad` is
   likewise declared but never consumed
   (`landing_rl/dynamics/process_noise.py:33-35`; frozen by the passing test
   `test_G_attitude_process_noise_std_rad_is_behaviorally_unused`).
2. **Disturbance/noise/reward channels default OFF** in the shared
   `LandingConfig` dataclass, with the previously-used nonzero values left
   only as trailing comments: `wind_accel_xy/z_max_mps2 = 0.00  #0.08/#0.02`,
   `target_noise_xy/z_std_m = 0.00  #0.04/#0.03`,
   `ground_effect_gain = 0.00  #0.18`, `w_yaw = 0.00  #0.05`
   (`landing_rl/envs/landing_env.py:108-109,116-117,192,251`, mirrored in
   `mujoco_rl/envs/env_prototype.py`). Actual nonzero values only appear
   hand-copied (with differing numbers) inside individual training/eval
   scripts. **The bare `LandingConfig()` default is not representative of
   any policy's actual training distribution.**
3. **`success_yaw_error_rad = π` is a structural no-op** in
   `kinematic_success` (`landing_env.py:236,887`), since `yaw_error_abs ∈
   [0,π]` always — the yaw term never actually gates success, despite
   `w_yaw` weighting it in the reward.
4. **`train_ppo_v3_long.py` and `train_ppo_stage1_1m.py` are near-duplicate
   files** — identical `make_config()`, identical checkpoint load/save paths.
5. **`train_ppo_stage2_contact.py` prints stale stage labels** — `"Stage 1
   rewardfix1 training finished."` (lines 92-95) despite being the
   stage2/v4 script; leftover from copy-paste.
6. **`eval_env_stage0.py` mismatches checkpoint and VecNormalize stage** —
   loads `vecnormalize_v3_stage0.pkl` (stage0 statistics) but
   `ppo_landing_residual_v3_stage1_final.zip` (a stage1 checkpoint) in
   `main()`.
7. **Misleading parameter name, acknowledged in-code**:
   `process_noise_vel_std_mps` is consumed as an *acceleration*-noise
   standard deviation, not a velocity-noise one; the module's own docstring
   states the name is kept only for RNG-stream compatibility
   (`landing_rl/dynamics/process_noise.py:29-32`).
8. **`README.md` branch pointer is stale** — states *"Current working
   branch: orange-fix"* (`README.md:5-9`); the actual checked-out branch this
   session is `refactor/landing-rl-architecture`.
9. **`landing_rl/configs/{environment,training,vehicle}/` are empty** — no
   externalized config files exist; every effective config is a Python
   literal embedded in a `mujoco_rl/*.py` script.

---

# 8. Experiment / artifact registry

[CONFIRMED, external directory `/home/qntmdghkss/drone_stack/mujoco_rl/runs/`,
read-only inspection this session, not part of this worktree]

| artifact | type | mtime | notes |
|---|---|---|---|
| `ppo_landing_residual_v3_stage0_final.zip` | PPO checkpoint | 2026-06-27 14:02 | stage0, contact disabled |
| `vecnormalize_v3_stage0.pkl` | VecNormalize | 2026-06-27 14:02 | paired with above |
| `ppo_landing_residual_v3_stage1_final.zip` | PPO checkpoint | 2026-06-27 16:28 | stage1 |
| `vecnormalize_v3_stage1.pkl` | VecNormalize | 2026-06-27 16:28 | paired with above |
| `ppo_landing_residual_v3_stage1_rewardfix1_final.zip` | PPO checkpoint | 2026-06-28 17:16 | reward-fixed stage1 |
| `vecnormalize_v3_stage1_rewardfix1.pkl` | VecNormalize | 2026-06-28 17:16 | paired with above |
| `ppo_landing_residual_v3_stage2_contact_final.zip` | PPO checkpoint | 2026-06-30 14:14 | contact enabled |
| `vecnormalize_v3_stage2_contact.pkl` | VecNormalize | 2026-06-30 14:14 | paired with above |
| `ppo_landing_residual_v4_stage2_contact_final.zip` | PPO checkpoint | 2026-06-30 14:59 | **primary, per test code §0.5/§0.8** (not "primary" merely because it is newest) |
| `vecnormalize_v4_stage2_contact.pkl` | VecNormalize | 2026-06-30 14:59 | paired with above |
| `checkpoints_v3*`, `checkpoints_v4*` (5 dirs) | SB3 periodic checkpoints | Jun 26 – Jun 30 | intermediate `CheckpointCallback` output, not individually inspected |
| `tensorboard_v3*`, `tensorboard_v4*` (5 dirs) | TensorBoard logs | Jun 27 – Jul 12 | not read this session |
| `eval_compare_v2_paper_200ep.txt` | eval output | 2026-08-31 04:23 | not read this session |
| `eval_robustness_paper.txt` | eval output | 2026-08-31 11:23 | not read this session |
| `paper_eval/` | eval output dir | 2026-08-31 11:48 | not enumerated this session |

`[CAUTION]` The `mtime` column above is raw filesystem metadata, listed for
reference only. It is not, by itself, proof of training order or lineage —
see §4 for which stage-to-stage transitions are additionally confirmed by
in-repo training-script code versus inferred from filename/mtime alone.

Regression/test artifacts (this worktree, `landing_rl/tests/`, tracked in
Git): 16 test files, 6,721 lines total, all passing at HEAD (§0.5).

---

# 9. Session / milestone history

Only sessions/events with direct evidence (Git commits or an actually-run
command this session) are recorded. No prior chat/session log was available
to reconstruct earlier sessions beyond what Git already records (§3, §4).

### 2026-09-06 — First `PROJECT_HANDOVER.md` created (this session)

`[FACT]` No `PROJECT_HANDOVER.md` existed before this session. This document
was created as the first canonical handover, per user request, as a
read-then-write documentation audit (no application source code modified).

IMPLEMENTATION: created `PROJECT_HANDOVER.md` at the worktree root. No other
file modified.

VALIDATION: ran the full `landing_rl/tests/` regression/parity/structural-freeze
suite (`python3 -m unittest discover -s landing_rl/tests -p "test_*.py"`):
**192/192 passed**, including the 5 real-checkpoint compatibility tests
against the external `mujoco_rl/runs/` artifacts (§0.5, §0.8).

`[CAUTION]` This session did not deeply audit `px4/`, `ros2/`, or `jetson/`
source; §6 above is limited to what `README.md` and directory listings show.
It also did not read the content of the two external `.txt` eval output
files or `paper_eval/` (§4, §8) — only their existence/mtime.

GIT STATE: branch `refactor/landing-rl-architecture`, HEAD `a2260a3`, working
tree clean except one untracked, unrelated file
(`CLAUDE_revised.md:Zone.Identifier` — a Windows download marker, not part of
this task, not modified/staged).

`[TODO]` User to review this handover and decide the actual next canonical
step (§0.10); nothing here should be treated as decided until then.

### 2026-09-07 — Pre-commit consistency audit (no facts changed)

`[FACT]` Before the first commit of this document, a targeted consistency
pass was performed: (1) trimmed §0 (esp. §0.5, §0.8) to remove
historical-provenance detail that belongs in §3/§4/§8 instead, relocating it
there rather than deleting it; (2) added an explicit provenance statement in
§2 clarifying that all 8 `[DECISION]` items are transcribed from
user-maintained `CLAUDE.md` content, not Claude inference; (3) in §0.6,
split the SI *research direction* (`[DECISION]`, cross-referenced to §2) from
the `system_id/` *implementation state* (`[CONFIRMED]` empty), which had
previously been merged under one `[FACT]` label; (4) in §4 and §8, downgraded
checkpoint-lineage claims that rested only on filesystem mtime — only 2 of
the 4 apparent stage-to-stage transitions are actually confirmed by
training-script code found in this repository; the other 2 have no
corresponding script and are now marked `[CAUTION]`/`[INTERPRETATION]`
instead of implied `[CONFIRMED]` fact. No new claims were added and no
existing `[CONFIRMED]`/`[FACT]` item was found to be substantively wrong —
this was a label-strength and section-placement correction, not a factual
correction. No application source code was touched.
