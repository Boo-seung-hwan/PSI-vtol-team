# UGRP Precision-Landing Project Handover

Last updated: 2026-09-25 (RL Baseline v0.1 packaging pass -- repository engineering only, NO research-behaviour change; see §0.12 and the 2026-09-25 §9 entry. `[SUPERSEDED]` the "UNCOMMITTED" / "Not pushed" statements below about the 2026-09-19 measured-vehicle work: it is now COMMITTED as `6c4fd81` (parameters, MJCF injection, tests) and `a666ab8` (`MUJOCO_MODEL.md`) on top of `6306699`; the handover file itself is committed last. Still NOT pushed.)
Previous revision, 2026-09-19 (was "latest" until 2026-09-25; fourth pass: the skid-type landing gear is
recorded as four PHYSICAL contact points DERIVED from the confirmed skid length
0.300 m / spacing 0.310 m under a CENTERED_SYMMETRY_ASSUMPTION — `landing_gear_
points_body_m` in `ugrp_vehicle_measured.yaml`; four contact spheres now exist
in the measured MJCF; landed and ±5° tilted contact verified; see §0.11.5 and
the fourth 2026-09-19 §9 entry. Third pass: landed-pose geometry fixed —
`ground_clearance_m = 0.241` stored as a PHYSICAL parameter, the 0.020 m gear
sphere radius kept as a SIMULATION constant in the MJCF builder, and the 0.221 m
sphere-centre offset DERIVED, never stored; dynamically verified, see §0.11.5
and the third 2026-09-19 §9 entry. Second pass: `MUJOCO_MODEL.md` rewritten,
documentation only. First pass: bifilar-measured Ixx/Iyy/Izz recorded in
`landing_mujoco/configs/ugrp_vehicle_measured.yaml` and shown to reach the
COMPILED MuJoCo model — mass 6.408 kg, diagonal inertia, CG-origin inertial
frame and motor markers verified in `mjModel`; the analytical `landing_rl`
path is untouched (218/218 identical). Propulsion / closed-loop response are
still unmeasured, so `MuJoCoDynamics` / `MujocoLandingEnv` still cannot be
built from the measured config — see §0.11.5 and the 2026-09-19 §9 entry.
UNCOMMITTED on top of `6306699`.)
Before that (2026-09-16): measurement datum and 45° X-frame symmetry
confirmed by the user, so CG / motor / battery coordinates were promoted to
confirmed FRD body-frame values in
`landing_mujoco/configs/ugrp_vehicle_measured.yaml` — storage/provenance
only, runtime physics UNCHANGED (`landing_rl` 218/218 identical); see §0.11
and both 2026-09-16 §9 entries. **Inertia (Ixx/Iyy/Izz) is now the only
remaining geometry/mass blocker.** This revision is UNCOMMITTED on top of
`6306699`.
Before that: interactive MuJoCo viewer SIGSEGV diagnosis
on the user's WSLg desktop — local graphics-driver fault, NO code/model change;
that revision of the file was then-uncommitted on top of `6306699` and is
carried forward here, see its §9 entry. Before that: x500 visual shell,
visualization only, split-shell
landing gear, pinned PX4-gazebo-models provenance, vendor copy removed;
committed as "Add x500 visual shell to MuJoCo landing environment" on top of
`7b231ef` — see the 2026-09-14 §9 entries and the SESSION CLOSEOUT entry. Earlier: 2026-09-13 entry-point migration +
VehicleDynamicsBackend seam + provisional `landing_mujoco/` MuJoCo pipeline)
Repository: `/home/qntmdghkss/drone_stack_rl_refactor` (a Git worktree of the UGRP drone_stack repository; per CLAUDE.md §5 this path is not guaranteed stable across sessions/machines — always re-verify with `git rev-parse --show-toplevel`)
Branch: `refactor/landing-rl-architecture`
HEAD (verified 2026-09-14, viewer-diagnosis session): `6306699871122cd985a3e35ebd1b93ce575aeaa4` — "Add x500 visual shell to MuJoCo landing environment", parent `7b231ef082c5124b0140639b79e5f518599f6f06` ("Add provisional MuJoCo landing simulation pipeline"). The viewer-diagnosis revision of this file was written on top of `6306699` and left UNCOMMITTED; if it has since been committed, resolve that commit with `git log -1 --format=%H -- PROJECT_HANDOVER.md`. The 2026-09-13 work is committed as `53e1ce6` (entry-point migration), `e3b408f` (VehicleDynamicsBackend seam), `7b231ef` (provisional MuJoCo pipeline). The 2026-09-14 x500 visual shell is committed in the visualization commit (all under `landing_mujoco/` plus `MUJOCO_MODEL.md` and this file; nothing under `landing_rl/` or `mujoco_rl/`). Not pushed.
[HISTORICAL — superseded by the line above] Before the 2026-09-13 commits, HEAD was `9a154df` and all 2026-09-13 work was uncommitted.

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

`[SUPERSEDED 2026-09-13 — see below]` Until 2026-09-13, `mujoco_rl/envs/
env_prototype.py :: LandingEnv` was the environment actually imported by
every training and evaluation script in the repository, and `landing_rl/
envs/landing_env.py :: LandingEnv` was a structurally-verified but unwired
modular copy. This was accurate as of HEAD `bc40963`/`a2260a3` and is kept
here for history; it is no longer the current state.

`[CONFIRMED — 2026-09-13, source-verified, still UNCOMMITTED]` The canonical
live environment is now **`landing_rl/envs/landing_env.py :: LandingEnv`**.
All 7 scripts that previously imported `env_prototype` were switched to
`from landing_rl.envs.landing_env import LandingEnv, LandingConfig` (an
import-line-only change, plus a small repo-root `sys.path` bootstrap in each
file — `make_config()` bodies, checkpoint/VecNormalize path literals, and
all other script content are byte-unchanged):

  - `mujoco_rl/train_ppo_stage1_1m.py`
  - `mujoco_rl/train_ppo_v3_long.py`
  - `mujoco_rl/train_ppo_stage2_contact.py`
  - `mujoco_rl/eval_compare_v2.py`
  - `mujoco_rl/eval_env_stage0.py`
  - `mujoco_rl/eval_robustness_paper.py`
  - `mujoco_rl/scripts/plot_trajectory.py`

`mujoco_rl/envs/env_prototype.py :: LandingEnv` is **kept, unmodified, as the
frozen legacy reference** — `landing_rl/tests/test_legacy_regression_contract.py`
continues to load it by file path and compare it against the NEW
implementation. It is not imported by any live script any more, but it is
NOT deleted or wrapped, per explicit instruction (preserve as reference,
don't force-merge behavior).

`landing_rl/training/` and `landing_rl/evaluation/` remain empty directories
(0 files) — the migration changed *which module* the existing
`mujoco_rl/*.py` scripts import, not their location; no new training/eval
entry point was created under `landing_rl/`.

Evidence: `landing_rl/tests/test_entry_point_migration.py` (new, 7 tests) —
identity-checks every migrated script's `LandingEnv`/`LandingConfig` against
`landing_rl.envs.landing_env`, re-verifies each script's unedited
`make_config()` still resolves to the exact same field values as before
(via `dataclasses.asdict` comparison against the frozen `CONFIG_SPECS`
table, not by re-typing values), confirms checkpoint/VecNormalize path
literals are untouched, and re-runs exact OLD-vs-NEW parity (seeds 0–9 ×
5 action patterns × the 4 named configs + the 5 `eval_robustness_paper.py`
cases = 210 episode-pairs, 0 mismatches) through the real, migrated script
files. A manual smoke test additionally attached the real external
checkpoint (`ppo_landing_residual_v4_stage2_contact_final.zip` +
`vecnormalize_v4_stage2_contact.pkl`) to `eval_compare_v2.py`'s own
(migrated) `make_config()` + `landing_rl.LandingEnv` and ran `PPO.load` →
`VecNormalize.load` → `reset` → `predict` → `step` successfully.

`[CAUTION]` This migration is **uncommitted** (§ header). A session that
starts from a fresh `git clone`/checkout of the last commit (`9a154df`)
will NOT see this change until it is committed — do not assume the
migration is permanent until a commit exists and is reported here.

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

`[CONFIRMED — 2026-09-13, UNCOMMITTED]` **MuJoCo-integration interface seam
added, MuJoCo itself still NOT implemented.** `landing_rl/dynamics/
vehicle_dynamics_backend.py` defines a `VehicleDynamicsBackend`
(`typing.Protocol`, `@runtime_checkable`) matching
`LegacyVehicleDynamics.advance_free_flight`'s exact current signature —
`PlantModel.__init__`'s `dynamics` parameter is now type-hinted against this
Protocol instead of the concrete `LegacyVehicleDynamics` class. This is a
type-hint/documentation-only change: `LegacyVehicleDynamics` gained no base
class and no code change, `PlantModel`'s public constructor
(`PlantModel(cfg, dynamics, contact)`) is unchanged, and `LandingEnv`'s
default construction still wires exactly one runtime backend —
`LegacyVehicleDynamics` — with no swap mechanism, factory, or config option
added. No `import mujoco`, no MJCF, no qpos/qvel handling, no frame/quaternion
adapter, and no new physics parameter exist anywhere in the repository as a
result of this change (verified by grep this session — `mujoco` is still
only imported in `mujoco_rl/scripts/smoke_test.py`, a version-check script).
Evidence: `landing_rl/tests/test_vehicle_dynamics_backend.py` (new, 12
tests, A–E per the task's own naming: Legacy satisfies the contract /
PlantModel accepts the contract / default backend stays Legacy / a mock
backend can be injected below `LandingEnv` untouched / exact OLD-vs-NEW
parity unchanged by the seam).

`[CONFIRMED — 2026-09-13, UNCOMMITTED]` **Controller-side command generation
split from physics inside `LegacyVehicleDynamics`, `VehicleDynamicsBackend`
contract unchanged.** The former private method
`LegacyVehicleDynamics._velocity_command_to_inner_loop_setpoints`
(`v_cmd → accel_cmd → attitude/thrust setpoint`, a PX4 velocity/attitude
controller approximation — exactly the stage the prior session's
`[INTERPRETATION]` note flagged as a KEEP-OUTSIDE-MUJOCO candidate) now
lives on its own component, `landing_rl/dynamics/inner_loop_command_model.py
:: InnerLoopCommandModel.compute(state, v_cmd, dt, target_yaw) ->
InnerLoopCommand(accel_cmd, attitude_setpoint, thrust_accel_setpoint)`.
`LegacyVehicleDynamics` constructs one internally in `__init__` (own only
`cfg`, stateless, zero RNG) and calls it at the exact call site the old
method occupied. The motor-cutoff override on `thrust_accel_setpoint` was
deliberately NOT moved — it reads `ContactModel`-owned persistent state (a
physics-side concept) and stays in `advance_free_flight`, applied to the
returned command immediately afterward, exactly where it sat before.
`advance_free_flight`'s signature (the `VehicleDynamicsBackend` contract)
is byte-identical to before this split — the boundary is realized entirely
as composition inside `LegacyVehicleDynamics`, not by moving the call
across the `PlantModel` boundary. Everything downstream of the setpoint
(attitude/rate controller proxy, body-rate/thrust response, rotation,
gravity/drag/wind/process-noise, integration) is untouched.
Evidence: `landing_rl/tests/test_inner_loop_command_model.py` (new, 6 tests,
A–E per the task's own naming).

## 0.5 Structural freeze / known-good baseline

[CONFIRMED — verified this session by actually running the suite]

```
cd /home/qntmdghkss/drone_stack_rl_refactor
python3 -m unittest discover -s landing_rl/tests -p "test_*.py"
```

Result (re-verified 2026-09-13, post-InnerLoopCommandModel split):
**218 tests, 0 failures, 0 errors, exit code 0, "OK"** (full run, 263.7s;
was 192 tests / 109.25s at the point this section was first written — the
delta is the entry-point-migration, VehicleDynamicsBackend, and
InnerLoopCommandModel test files added across the three 2026-09-13 §9
entries, not a change to what's being tested). This includes the 5
`test_checkpoint_compatibility.py` tests, which ran (were not skipped) —
meaning the external primary checkpoint artifacts
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

[CONFIRMED — 2026-09-07] **`system_id/preprocessing/` is now implemented**
(raw `.ulg` → validated topic extraction → per-loop timebase → frame-consistent
derived signals → validity/saturation/contact masks → contiguous SI segments →
per-loop datasets → provenance + data-quality report). It fits **nothing** and
writes no parameters. `system_id/identification/`, `system_id/validation/`,
`system_id/results/` remain **empty — NOT IMPLEMENTED**. See §5 for detail.
The active SI *direction* above is unchanged; no UGRP numeric parameter has
been identified.

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

- `[SUPERSEDED 2026-09-13]` ~~`landing_rl/` is structurally verified
  equivalent but **not wired** into any training/evaluation entry point.~~
  Resolved this session: all 7 `mujoco_rl/*.py` scripts now import
  `landing_rl.envs.landing_env` (§0.3) — **but the change is still
  uncommitted**, so this gap is only closed in the current working tree,
  not yet in Git history.
- `[SUPERSEDED 2026-09-13, second same-day entry — see §9 "MuJoCo v0
  pipeline"]` ~~`landing_rl/dynamics/vehicle_dynamics_backend.py`'s
  `VehicleDynamicsBackend` Protocol (§0.4) is an interface seam only — no
  second backend (`MuJoCoDynamics` or otherwise) exists yet.~~ A MuJoCo
  backend now exists, but NOT as an implementation of that Protocol and NOT
  wired into `landing_rl.envs.landing_env.LandingEnv` / `PlantModel` — it is
  a separate, additive package, `landing_mujoco/`, with its own
  `MujocoLandingEnv(gym.Env)` (own controller/perception/reward wiring,
  reusing `landing_rl` components directly rather than going through
  `PlantModel`). `landing_rl/envs/landing_env.py` is UNCHANGED by this work
  (verified: no diff). See §9 for full detail. Still true and NOT resolved:
  no numeric UGRP physical parameter exists anywhere (mass, CG, inertia,
  motor geometry, `T_max`, `tau_roll/pitch/thrust`, `T_delay` are all
  `null` in `landing_mujoco/configs/ugrp_vehicle_measured.yaml`, pending
  next week's measurement per explicit user statement this session).
  `[CORRECTION — 2026-09-19]` That last sentence has been false since
  2026-09-16 and is now much further from true: mass, CG, motor geometry,
  battery position (2026-09-16) and the diagonal inertia Ixx/Iyy/Izz
  (2026-09-19) are populated in that YAML. Still `null`: `T_max`,
  `tau_roll/pitch/thrust`, `T_delay` and the landing-gear contact geometry.
  See §0.11.
- `system_id/` is **partially implemented**: `preprocessing/` exists (§5);
  `identification/`, `validation/`, `results/` are still empty. No UGRP
  parameter identified. The only ULogs available are OTHER-PROJECT sample
  logs (`OTHER_PROJECT_SAMPLE` / `PIPELINE_VALIDATION_ONLY`), usable for
  pipeline validation only — no number derived from them may enter a UGRP
  config (§5, §9).
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
- `[SUPERSEDED 2026-09-13, second same-day landing_mujoco entry — see §9]`
  ~~Landing-gear standoff vs. `success_altitude_m` incompatibility, NOT
  fixed.~~ **RESOLVED**, without changing `success_altitude_m` (still
  `0.05m`, byte-unchanged): `MujocoLandingEnv` now distinguishes
  `pad_target_ned` (the pad) from `vehicle_touchdown_target_ned` (the
  desired CG position at a correct level touchdown, offset from the pad by
  the landing-gear standoff via `p_landing^N = p_vehicle^N + R_B^N
  r_landing^B`); `xy_error`/`z_error`/dx,dy,dz are computed against the
  latter. `altitude_agl` is kept as a separate physical-clearance quantity
  (gear-point height above ground at current attitude), independent of
  `z_error` by construction. See §9 and `MUJOCO_MODEL.md`.
- `[OPEN — 2026-09-14]` The x500 visual shell has **not yet been visually
  inspected by the user** in an interactive viewer (overall appearance, shell
  orientation, scale 1.4975221199266826, motor / propeller / landing-gear
  alignment, ground alignment, CG/body-reference relationship). The only
  visual evidence so far is OSMesa offscreen renders. On the user's WSL
  desktop the interactive viewer segfaults on the default Intel Iris Xe D3D12
  adapter — a local graphics-driver fault, NOT a repository/model bug; the
  candidate workaround is `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` (fallback
  `LIBGL_ALWAYS_SOFTWARE=1`), not yet visually confirmed. See the §9 entry
  "interactive MuJoCo viewer SIGSEGV on WSLg". This does not change §0.10.

## 0.10 Immediate next step

`[CURRENT — 2026-09-25]` Before the baseline is shared: the user reviews and approves pushing the local baseline commits (§0.12); nothing has been pushed. The physical-model next step below (propulsion calibration) is unchanged and independent of this.

`[CURRENT — 2026-09-19]` **Physical-inertia injection is complete;
propulsion calibration is pending.** The measured mass, CG, diagonal inertia
and motor geometry now reach the compiled MuJoCo rigid body (§0.11.5). The
minimum measurements still needed before the measured config can drive
`MuJoCoDynamics` / `MujocoLandingEnv` are: `max_collective_thrust_n`,
`tau_roll_s`, `tau_pitch_s`, `tau_thrust_s`, `actuator_delay_s`.
`[CORRECTION — fourth 2026-09-19 pass]` the landing-gear part of this list is
gone: `ground_clearance_m = 0.241 m` and the four skid contact points are
recorded (§0.11.5); only the propulsion fields remain. The exact per-leg offset
of the skid footprint from the CG is still an assumption (centred). No tuning was
started this session.

`[SUPERSEDED — 2026-09-19, by the item above]` `[CURRENT — 2026-09-16,
second pass]` **Inertia measurement (Ixx / Iyy /
Izz) is now the single remaining blocker** for a complete vehicle model.
Both coordinate blockers are resolved: the measurement datum is defined
(floor, +Z up → `z_body = cg_raw − raw`) and the exact 45° X-frame symmetry
is confirmed, so CG, full 3-D motor positions and battery position are now
recorded in the FRD body frame. Mass, motor geometry, spin layout and
component specs were recorded earlier the same day. All of it remains
storage/provenance only — no physics, no MuJoCo implementation, no behavior
change. `max_collective_thrust_n` and the identified closed-loop response
(`tau_*`, `delay_s`) also remain unmeasured, but neither blocks inertia
work. See §0.11.

`[HISTORICAL — partially satisfied by the 2026-09-16 session]` The
2026-09-14 closeout specified: the next session starts with **real UGRP
vehicle parameter measurement / injection** into
`landing_mujoco/configs/ugrp_vehicle_measured.yaml` — NOT further
visualization work, and NOT PPO training. Recommended order: mass → CG →
inertia → motor positions → landing-gear geometry → max collective thrust →
roll/pitch/thrust response identification → actuator delay → real-flight vs
MuJoCo validation. See the final 2026-09-14 §9 closeout entry. Mass and
motor radial distance are now done; CG is measured but datum-blocked;
inertia onward remain open.

`[HISTORICAL — superseded by the item above]` The text below predates the
2026-09-14 closeout.

[OPEN] — no user-approved "next step" decision is recorded. The 2026-09-07
phase implemented `system_id/preprocessing/`; the 2026-09-13 session
migrated the canonical entry point to `landing_rl`, added the
`VehicleDynamicsBackend` interface seam, built a full provisional-parameter
MuJoCo pipeline in `landing_mujoco/`, and (in a fourth same-day pass)
resolved the landing-gear-standoff/`success_altitude_m` interaction and
completed all 8 `compare_old_vs_mujoco.py` scenarios (see §9). All of this
is user-directed scope, and the first two items remain uncommitted — the
natural follow-ons are still `[PROPOSAL]`, not `[DECISION]`, until the user
picks one:
  * **commit the 2026-09-13 working-tree changes** — CLAUDE.md §24 asks for
    separate commits per concern; this session's own analysis (§9) proposes
    exactly three: (A) entry-point migration, (B) `VehicleDynamicsBackend`
    seam, (C) the entire `landing_mujoco/` package (including the
    landing-gear-standoff fix and finished comparison harness — all
    developed and tested together as one coherent unit of work);
  * obtain **dedicated UGRP SI flights** on the pinned PX4 1.15 in offboard
    **velocity** mode (the sample logs are position-mode, other-project) and
    with a raised `SDLOG_PROFILE` rate for `vehicle_local_position_setpoint`;
  * then start `system_id/identification/` (attitude-setpoint→attitude and
    rate-setpoint→body-rate first; the PX4 velocity→accel and accel→attitude
    mappings are source-known and only need validation, not fitting — see the
    2026-09-07 "command-path" audit clarification);
  * **measure the real UGRP vehicle** (mass, CG, inertia, motor geometry,
    `T_max`, `tau_roll/pitch/thrust`, `T_delay`) and fill in
    `landing_mujoco/configs/ugrp_vehicle_measured.yaml` — explicitly
    expected "next week" per this session's user statement; this is now the
    ONLY remaining blocker before real-parameter physical validation, since
    the landing-gear-standoff task-semantics issue is resolved;
  * unrelated legacy-RL items (reward/termination extraction, populating
    `landing_rl/configs/*`) remain open.

## 0.11 Vehicle physical parameters

`[CURRENT — 2026-09-16]` Canonical source:
**`landing_mujoco/configs/ugrp_vehicle_measured.yaml`**
(schema + loader: `landing_mujoco/configs/param_schema.py`; validator CLI:
`python3 -m landing_mujoco.check_parameter_set --parameter-set
ugrp_vehicle_measured`). This is the ONLY place real UGRP vehicle numbers
are recorded. `tarot680b_reference.yaml` describes a **different,
unmeasured airframe** and must never be used to back-fill it.

`[SUPERSEDED — 2026-09-19]` ~~`[CAUTION]` Recording ≠ applying. None of the
values below are wired into any runtime physics.~~

`[CORRECTION — 2026-09-19]` That statement is no longer true for MuJoCo.
`landing_mujoco/dynamics/mjcf_builder.py` consumes `mass_properties.mass_kg`,
`ixx/iyy/izz` and `motors.positions_body_m` from this YAML, and the
**compiled `mjModel`** built from it carries the measured values (§0.11.5).
What is still true: (a) the analytical `landing_rl` legacy dynamics is fully
mass-normalized — it carries no mass, inertia or geometry at all (§0.11.4) —
so its behavior is UNCHANGED; (b) `load_uav_params()` on this config still
raises `MissingMeasurementError` (5 propulsion / response fields), so
`MuJoCoDynamics` / `MujocoLandingEnv` / `check_parameter_set` still cannot be
built from it — only the rigid-body MJCF can, via `validate=False`.
Everything not listed as consumed in §0.11.5 (raw datum, component specs,
spin directions, battery position, inertia provenance) remains
record-only.

### 0.11.1 Parameter table

Status vocabulary: **MEASURED** (physically measured on the assembled
vehicle) · **USER_PROVIDED_SPEC** (component model designation / nameplate
figure supplied by the user; not read from a datasheet held in this
repository, not independently verified) · **USER_REPORTED_OBSERVATION** (a
reported observation of how the vehicle is built — weaker than MEASURED: no
instrument reading, no independent cross-check) · **DERIVED** (computed from
another recorded value under a stated assumption) · **UNKNOWN** (no evidence
yet). Tags are asserted by tests, so promoting one is a visible change.

| Parameter | Value | Unit | Status | Source / Note |
|---|---|---|---|---|
| total mass | 6.408 | kg | MEASURED | 6408 g, 2026-09-16. `[CORRECTION 2026-09-19]` the weighing configuration IS now stated (user): the full actual flight configuration — battery, ~1.8 kg ballast brick, motors, ESCs, frame, electronics, landing gear, wiring. Applied once, to the single MuJoCo root body; nothing is added on top |
| motor radial distance | 0.360 | m | MEASURED | center → motor center. **Not** the opposite-motor diagonal. Stored as `geometry.arm_length_m` |
| motor spin — FL | CW | — | USER_REPORTED_OBSERVATION | X-frame; diagonals share direction. Weaker than MEASURED: a reported build observation, not an instrument reading; **not** cross-checked against the PX4 actuator/motor assignment |
| motor spin — FR | CCW | — | USER_REPORTED_OBSERVATION | |
| motor spin — RL | CCW | — | USER_REPORTED_OBSERVATION | |
| motor spin — RR | CW | — | USER_REPORTED_OBSERVATION | |
| arm angle | 45.0 | ° | USER_CONFIRMED | exact symmetric X-frame, confirmed 2026-09-16; the basis for the derived motor XY |
| measurement datum | floor, +Z up | — | USER_CONFIRMED | resolved 2026-09-16 → `z_body_FRD = cg_raw_z − raw_z`. See §0.11.2 |
| CG (raw datum) | [0, 0, 0.241] | m | MEASURED | height above floor; preserved in `raw_measurements` for provenance |
| motor plane z (raw datum) | 0.334 | m | MEASURED | same datum |
| battery center (raw datum) | [0, 0, 0.220] | m | MEASURED | same datum |
| **physical ground clearance** `ground_clearance_m` (CG → ground plane, landed pose) | **0.241** (body z of the ground = **+0.241**) | m | MEASURED (raw CG height) · landed-pose datum USER_CONFIRMED 2026-09-19 | **stored** in the YAML as `geometry.ground_clearance_m` (third 2026-09-19 pass); see §0.11.2, §0.11.5 |
| gear sphere radius | 0.020 | m | SIMULATION representation — not measured, **not stored** in the YAML | `LEG_CONTACT_RADIUS_M` in `mjcf_builder.py` |
| sphere-centre offset (body z) | 0.221 (= 0.241 − 0.020) | m | DERIVED — **never stored**; must not be entered as clearance or contact point | `landing_gear_sphere_centers_body_m()` |
| **CG (body, FRD)** | [0, 0, 0] | m | DERIVED | body origin **is** the CG; now a reconciled statement, not a default |
| **motor positions (body, FRD)** | FL [+a,−a,−0.093] · FR [+a,+a,−0.093] · RL [−a,−a,−0.093] · RR [−a,+a,−0.093], a = 0.2545584412271571 | m | DERIVED | XY from `arm_length_m`+`arm_angle_deg`; Z from the datum conversion. Motors sit **above** the CG ⇒ negative in FRD |
| **battery (body, FRD)** | [0, 0, +0.021] | m | DERIVED | positive ⇒ battery sits **below** the CG |
| opposite-motor span | 0.720 | m | DERIVED | max pairwise horizontal distance; `wheelbase_m` left `null` on purpose — see §0.11.3a |
| motor model | T-MOTOR MN501-S IP45 | — | USER_PROVIDED_SPEC | |
| motor KV | 360 | rpm/V | USER_PROVIDED_SPEC | from the "KV360" designation only |
| ESC model | HOBBYWING Skywalker V2 60A | — | USER_PROVIDED_SPEC | 3–6S, 7 A BEC built in |
| ESC max current | 60 | A | USER_PROVIDED_SPEC | continuous, per designation |
| propeller model | T-MOTOR MS1704 | — | USER_PROVIDED_SPEC | model string only; **not** decoded into diameter/pitch |
| battery model | Poly-Tronics 8th Gen 10000 mAh 22.2 V 6S1P 75C+ XT90-S | — | USER_PROVIDED_SPEC | |
| battery nominal voltage | 22.2 | V | USER_PROVIDED_SPEC | 6S1P nominal |
| hover thrust, total | 62.8401 | N | DERIVED | `m·g`, g = 9.8065 (`LandingConfig.gravity_mps2`). With g = 9.80665 (landing_mujoco): 62.8410 N |
| hover thrust, per motor | 15.7100 | N | DERIVED | = 1.602 kgf. Assumes **equal sharing** across 4 motors — the real CG offset means sharing is not exactly equal |
| **Ixx** (roll) | **0.153184** | kg·m² | **MEASURED** | bifilar, manual stopwatch, n = 10, sample std 0.004362. Recorded 2026-09-19. See §0.11.5 |
| **Iyy** (pitch) | **0.126285** | kg·m² | **MEASURED** | bifilar, n = 10, sample std 0.003041 |
| **Izz** (yaw) | **0.149050** | kg·m² | **MEASURED** | bifilar, n = 9 (8.48 s outlier excluded), sample std 0.001698 |
| Ixy, Ixz, Iyz | 0, 0, 0 | kg·m² | **ASSUMED_ZERO_FOR_V0** | **NOT measured.** Zero is a v0 modelling assumption, never "measured zero". MuJoCo receives `diaginertia` only; recorded in `mass_properties.meta.inertia_products_kgm2` |
| inertia R² | N/A | — | N/A | manual stopwatch periods; no damped-sinusoid fit exists. Not fabricated |
| max collective thrust | — | N | UNKNOWN | MN501-S + MS1704 combination not bench-tested |
| thrust / torque coefficients | — | — | UNKNOWN | propulsion bench test pending |
| motor time constant, ESC delay | — | s | UNKNOWN | not measured |
| motor mass (each) | — | kg | UNKNOWN | |
| battery mass | — | kg | UNKNOWN | |
| frame footprint / height | — | m | UNKNOWN | not measured. `frame_height_m` is **not** the 0.334 m motor height |
| skid effective length | **0.300** | m | USER_CONFIRMED | skid long axis = body X; recorded in `meta` of `landing_gear_points_body_m` |
| skid centre-line spacing | **0.310** | m | USER_CONFIRMED | same |
| skid half length / half spacing | 0.150 / 0.155 | m | DERIVED (length / 2, spacing / 2) | x / y of the contact points |
| **landing-gear contact points** (FRD) | LF [+0.150, −0.155, +0.241] · LR [−0.150, −0.155, +0.241] · RF [+0.150, +0.155, +0.241] · RR [−0.150, +0.155, +0.241] | m | **DERIVED_FROM_MEASURED_DIMENSIONS + CENTERED_SYMMETRY_ASSUMPTION** — *not* four measured coordinates | `geometry.landing_gear_points_body_m` (fourth pass), `landing_gear_points_semantics: physical_contact` |
| exact per-leg offset of the footprint from the CG | — | m | **MODELLING ASSUMPTION** (centred, symmetric) — not measured | `[CORRECTION 2026-09-19, fourth pass]` supersedes the previous "per-leg x/y UNKNOWN" row |
| τ_roll, τ_pitch, τ_thrust, τ_yaw, delay | — | s | UNKNOWN | closed-loop SI not yet performed (§0.6; gated on PX4 gain stabilization per CLAUDE.md §3) |

### 0.11.2 `[RESOLVED — 2026-09-16]` Measurement datum → body frame

`[SUPERSEDED]` This subsection previously recorded the datum ambiguity as
the project's blocking issue. The user resolved it on 2026-09-16; the
original `[OPEN]` text is preserved in the 2026-09-16 §9 entries.

`[FACT — USER_CONFIRMED]` The raw z-coordinates (CG 0.241, motor plane
0.334, battery center 0.220) are heights measured from the **floor**, with
**+Z pointing up**. The simulation body frame is **FRD** (+X forward,
+Y right, +Z **down**) with its **origin at the CG**. The two therefore
differ by an origin shift to the CG plus a Z sign flip:

```
z_body_FRD = cg_raw_z − raw_z
```

| Quantity | Raw (floor, +up) | Body (FRD, +down) | Physical reading |
|---|---|---|---|
| CG | 0.241 m | **0.000 m** | the body origin, by definition |
| motor plane | 0.334 m | **−0.093 m** | motors sit 93 mm **above** the CG |
| battery center | 0.220 m | **+0.021 m** | battery sits 21 mm **below** the CG |

The raw numbers remain in `raw_measurements:` (`datum_status: RESOLVED`,
`datum_origin: floor`, `datum_up_axis: +z_up`) so the provenance chain stays
auditable. `RawMeasurements.body_z_from_raw_height()` is the one definition
of the conversion; it refuses to run on an unresolved datum or a non-`+z_up`
datum rather than guessing a sign. Tests recompute every stored body-frame z
from the raw pair, and one test asserts the sign convention is not
accidentally inverted.

`[CAUTION]` Resolving the datum did **not** determine `frame_height_m`
(0.334 is a motor height, not a bounding dimension), `ground_clearance_m`,
or `landing_gear_points_body_m`. All three stay `null`, enforced by test.
Ground clearance in particular would require confirming the vehicle rested
on its own landing gear on the datum floor — nobody stated that, and
`landing_reference_point_body_m()` falls back to `[0,0,ground_clearance_m]`
when gear points are absent, so a guess would silently become the touchdown
reference. `[TODO]` Measure the landing-gear contact geometry.

`[CORRECTION — 2026-09-19]` "nobody stated that" is superseded. `[FACT —
USER_CONFIRMED]` The raw heights were measured with the vehicle in its
**normal landed pose, landing gear on the ground**, ground plane = 0, height
measured upward: CG 0.241 m, motor plane 0.334 m, battery centre 0.220 m above
the ground. In the FRD CG-origin frame the CG stays `[0, 0, 0]` and the
nominal **ground plane is at z = +0.241 m** (CG → ground = 0.241 m); writing
"CG body z = +0.241" would be wrong. What this does and does not settle
(detail: `MUJOCO_MODEL.md`, "Measured vehicle: ground plane…"):
  * `[FACT]` The **vertical** CG→ground distance is known, and matches the
    repo's definition of `ground_clearance_m` (CG down to the lowest contact
    point in the landed pose).
  * `[SUPERSEDED — see the `[DECISION]` below]` `[CAUTION]` It is still **not safe to enter 0.241 as a gear-point z**:
    `mjcf_builder.py` models each gear point as the *centre* of a 0.02 m-radius
    contact sphere, so a 0.241 m gear point rests the CG ≈ 0.261 m up
    (read-only simulation, synthetic x/y), while ≈ 0.221 m reproduces the
    measured 0.241 m. Which convention to adopt is `[OPEN]`, not decided.
  * `[SUPERSEDED — fourth 2026-09-19 pass: x/y now derived from measured skid dimensions under a centred-symmetry assumption]` `[OPEN — x/y part still open]` Per-leg contact **x/y remain UNKNOWN**; `ground_clearance_m` alone
    would stop `landing_reference_point_body_m()` raising but would create
    **no contact geoms** (leg geoms are built only from
    `landing_gear_points_body_m`), and the propulsion blockers remain.
  * `[SUPERSEDED — YAML updated in the third pass]` `[CAUTION]`
    `ugrp_vehicle_measured.yaml` still carries the old "not stated" wording
    (header comment; `geometry.meta.ground_clearance_m`) and still has both gear
    fields `null`. Left as is on purpose — that pass was documentation-only.

`[DECISION — user, 2026-09-19, third pass]` `[SUPERSEDES]` the convention
question above. Physical contact geometry and its MuJoCo representation are
**separate quantities with separate owners**:

```
PHYSICAL (YAML)                      SIMULATION REPRESENTATION (mjcf_builder.py)
  ground_clearance_m = 0.241           gear sphere radius        = 0.020
  landing contact point z = +0.241     gear sphere centre z (FRD) = 0.221 (derived)
```

  * `ground_clearance_m = 0.241` is stored (source MEASURED, dated
    2026-09-19). **0.221 must never be stored as clearance or contact point.**
  * `Geometry.landing_gear_points_semantics` (`geom_center` legacy default |
    `physical_contact`) says how `landing_gear_points_body_m` is read. The
    measured YAML declares `physical_contact`; the reference YAML has no field ⇒
    `geom_center` ⇒ **its MJCF is byte-identical** (hash pin unchanged).
  * `mjcf_builder.landing_gear_sphere_centers_body_m()` derives
    `centre_z = point_z − radius` (FRD); `build_mjcf` uses it. With no per-leg
    points there are **no contact geoms** — none are invented.
  * Dynamic verification (ground plane z = 0, gravity, released 0.5 m up, settle):
    CG world z = 0.24092 m (4 legs) / 0.24089 m (3 legs), sphere bottom world z
    = −0.079 / −0.105 mm (soft-contact sinking, linear in load ≈ 5 µm/N,
    tolerance ±2 mm ≈ 6× the worst single-contact case). Negative control
    (0.241 read as the sphere centre) rests at 0.2609 m, +19.9 mm.
  * `[SUPERSEDED — fourth 2026-09-19 pass]` `[OPEN]` Per-leg x/y are still UNKNOWN and
    not estimated; the tests use **synthetic** x/y (three layouts; the landed
    height does not depend on them). The tests now use the canonical skid
    footprint (see §0.11.5).

### 0.11.3 `[RESOLVED — 2026-09-16]` Arm angle confirmed; motor geometry promoted

`[SUPERSEDED]` This subsection previously recorded the ±45° arm angle as an
unconfirmed assumption, which kept the motor XY out of
`motors.positions_body_m`.

`[FACT — USER_CONFIRMED]` The frame is an **exact symmetric 45° X-frame**,
with center-to-motor radius R = 0.360 m, so `a = R/√2 = 0.2545584412271571`
is now a derivation from confirmed geometry rather than a borrowed
assumption. Recorded as `geometry.arm_angle_deg: 45.0`.

With both the arm angle and the datum resolved, the full 3-D motor
coordinates are now written into **`motors.positions_body_m`** — the field
`mjcf_builder.py` actually consumes (FRD, +y right so left is −y):

```
FL [+a, −a, −0.093]    FR [+a, +a, −0.093]
RL [−a, −a, −0.093]    RR [−a, +a, −0.093]
```

The interim `provisional_geometry:` block was **removed**, not renamed:
keeping motor XY in two places would have created exactly the drift its own
test was written to prevent. The drift guard survives in stronger form —
tests recompute XY from `arm_length_m` + `arm_angle_deg` and compare against
the stored positions, so there is one source of truth.

`[CAUTION]` `tarot680b_reference.yaml` is untouched. Its ±45° assumption
remains an explicitly-tagged, low-confidence statement about a *different*
airframe and must not be cited as evidence about this vehicle.

### 0.11.3a `[CAUTION]` "arm length" vs "wheelbase" vs "a" — naming trap

Three different lengths are in play and the repository's names for them are
already muddled. For this vehicle:

| Quantity | Value | Meaning |
|---|---|---|
| `geometry.arm_length_m` | 0.360 m | **center → motor center** (what was measured) |
| `geometry.wheelbase_m` | null | opposite-motor span = 0.720 m, now derivable from the confirmed symmetry — but deliberately **left null**, see below |
| `a` (the x/y component) | 0.2546 m | = `arm_length_m`/√2; **confirmed** since 2026-09-16, no longer an assumption |

The 0.360 m figure is consistent with how `tarot680b_reference.yaml` uses
`arm_length_m` (it sets `arm_length_m = wheelbase/2 = 0.3685`), so the
semantics match. But **MUJOCO_MODEL.md §D calls `a = wheelbase/(2√2)` the
"radial distance"**, which is wrong — `a` is the x/y *component*; the radial
distance is `wheelbase/2`. Anyone applying §D's formula with 0.360 as a
"wheelbase" would get `a = 0.127 m`, off by a factor of 2.

`[DECISION — 2026-09-16]` `wheelbase_m` stays `null` even though it is now
derivable. `x500_shell._target_wheelbase_from_physical_params` already
derives the 0.720 m span from `positions_body_m` (max pairwise horizontal
distance) whenever `wheelbase_m` is absent, so storing a literal would only
create a second source that can disagree with the positions. A test
(`test_opposite_motor_span_derives_to_720mm`) pins that derivation, and
`test_arm_length_is_radial_not_diagonal` pins the radius interpretation.

`[TODO]` Fix MUJOCO_MODEL.md §D's wording (root-level file, out of this
session's write scope — see the §9 entries).
`[RESOLVED — 2026-09-19]` Done in a documentation-only pass the user
explicitly authorized: `MUJOCO_MODEL.md` §D now defines `R` (radial distance,
0.360 m) and `a = R/√2` (x/y component, 0.2545584412271571 m) separately and
carries a correction note. The full rewrite is listed in the 2026-09-19 §9
entry's follow-up.

### 0.11.4 `[FACT]` Current parameter ownership (audit, 2026-09-16)

| Item | Owner | Note |
|---|---|---|
| mass, CG, inertia, geometry, motor position/direction, thrust limits, motor/ESC/prop/battery metadata | `landing_mujoco/configs/*.yaml` via `param_schema.py` | the only structured owner |
| (no vehicle physical parameters at all) | `system_id/`, `mujoco_rl/` | verified by grep: **zero** references to `param_schema`/`landing_mujoco`. `system_id/` holds ULog signal schemas only; `mujoco_rl/env_prototype.py` duplicates the mass-normalized `LandingConfig` fields |
| (no vehicle model) | `models/`, `px4/`, `px4_assets/` SDF/URDF | other projects' airframes (X8 VTOL, mono_cam); **no** SDF/URDF/MJCF describes the UGRP vehicle |
| mass-normalized thrust limits | `landing_rl/envs/landing_env.py` `LandingConfig` | `max_thrust_accel_mps2 = 1.80·9.8065`, `min_thrust_accel_mps2 = 0.25·9.8065` — accelerations, **not** forces |
| gravity (RL/analytical) | `LandingConfig.gravity_mps2 = 9.8065` | |
| gravity (MuJoCo) | hardcoded `9.80665` in `actuation_model.py`, `mjcf_builder.py` MJCF, several tests | |

`[FACT]` **`landing_rl` legacy dynamics contains no mass, inertia, CG or
geometry at all** — it is mass-normalized throughout (thrust is carried as
`thrust_accel` in m/s²). This is why recording real mass/geometry cannot
change legacy behavior, and why connecting them later is a deliberate,
separate physics change.

`[OPEN]` **Duplicate/inconsistent gravity constant**: `landing_rl` uses
9.8065, `landing_mujoco` hardcodes 9.80665. Pre-existing, recorded here,
deliberately **NOT** reconciled — changing either is a behavior change
requiring its own regression pass. At 6.408 kg the two differ by 0.0009 N in
hover thrust (immaterial), but they should be unified under one owner
eventually.

`[OPEN]` The canonical vehicle-parameter source currently lives **inside
`landing_mujoco/`**, while the target architecture has analytical/SI and
MuJoCo consuming one shared source. `landing_rl/configs/vehicle/` exists but
is an **empty, untracked placeholder** (no files in `git ls-files`).
`[PROPOSAL]` (not approved) promote the parameter source to a
backend-neutral location once a second consumer actually exists — premature
today.

### 0.11.5 `[CURRENT — 2026-09-19]` Measured inertia → compiled MuJoCo model

**Data flow (verified by reading the code and by inspecting a compiled
`mjModel`):**

```
ugrp_vehicle_measured.yaml
  → param_schema.load_uav_params(validate=False)       [parse only; no logic changed]
  → mjcf_builder.build_mjcf(params, physics_dt=…)      [one explicit <inertial> at the CG]
  → mujoco.MjModel.from_xml_string(xml)
  → body_mass / body_inertia / body_ipos / body_iquat / geom_pos (motor markers)
```

| YAML field | Reaches compiled `mjModel`? | As |
|---|---|---|
| `mass_properties.mass_kg` | **yes** | `body_mass[vehicle]` = 6.408 |
| `mass_properties.inertia_kgm2.{ixx,iyy,izz}` | **yes** | `body_inertia[vehicle]` = [0.153184, 0.126285, 0.14905] |
| `mass_properties.cg_body_m` | implicit | inertial `pos="0 0 0"` — the body origin **is** the CG (`body_ipos` = 0, `body_iquat` = identity) |
| `motors.positions_body_m` | **yes**, markers only | `arm_*` / `motor_*` geoms, FRD→FLU via `frd_vector_to_mujoco_body`; massless, collision-disabled |
| `geometry.landing_gear_points_body_m` | **yes** (fourth pass) | four PHYSICAL contact points DERIVED from skid length 0.300 / spacing 0.310 under a CENTERED_SYMMETRY_ASSUMPTION; read as `physical_contact`, turned into `leg_0..leg_3` sphere centres (`z − 0.020`, i.e. 0.221) by the builder; FRD→FLU via the repository helper |
| `geometry.ground_clearance_m` (0.241) | not into the MJCF | PHYSICAL; only the fallback `r_landing^B = [0, 0, 0.241]` of `landing_reference_point_body_m()` reads it; creates no geom |
| gear sphere radius 0.020 / centre offset 0.221 | builder only | SIMULATION / DERIVED — `LEG_CONTACT_RADIUS_M`, `landing_gear_sphere_centers_body_m()`; never in a physical YAML |
| `mass_properties.meta.*` (inertia provenance, products, included hardware) | no | record-only |
| `battery.position_body_m`, `motors.spin_directions`, `raw_measurements`, component specs | no | record-only |
| `thrust.*`, `identified_response.*` | not needed for the MJCF; **required** by `IdentifiedWrenchActuation` | still `null` |

**Mass / inertia ownership — no double counting.** `mass_kg` and `Ixx/Iyy/Izz`
were measured on the full flight configuration, so they already contain
battery, ballast brick, motors, ESCs, frame, electronics, landing gear and
wiring. MuJoCo therefore carries **exactly one** inertial body:
`vehicle`, mass 6.408, principal inertia = the measured diagonal, an explicit
`<inertial>` element (which makes MuJoCo ignore geom-derived mass). Every
child geom (markers, x500 visual shell) is massless. Component masses
(`battery.mass_kg`, `motors.mass_kg_each`) stay `null` and are never added.
Verified on the compiled model: `Σ body_mass` = `mj_getTotalmass` = 6.408,
exactly one non-zero-mass body, and enabling the x500 visual shell (31 geoms
vs 9) leaves `body_mass/inertia/ipos/iquat/subtreemass` bit-identical.

**Compiled `mjModel` readout** (`physics_dt` 0.002, gravity `[0,0,−9.80665]`):
`body_mass[vehicle]` = 6.408 · `body_inertia` = [0.153184, 0.126285, 0.14905] ·
`body_ipos` = [0,0,0] · `body_iquat` = [1,0,0,0] · motor markers (FLU) at
(±0.2545584412271571, ±0.2545584412271571, +0.093), radius 0.360 m to machine
precision, FRD round-trip recovering the YAML literals. MuJoCo did not
modify the tensor: the `<compiler>` carries only `angle="radian"` (no
`balanceinertia` / `boundmass` / `boundinertia` / `inertiafromgeom`), and the
triangle inequalities hold with margin (0.153184 ≤ 0.275335;
0.126285 ≤ 0.302234; 0.149050 ≤ 0.279469).

**Inertia measurement (bifilar, manual stopwatch, recorded 2026-09-19).**
Full raw periods, D/L per axis and statistics are in
`mass_properties.meta.inertia_kgm2` and are re-reduced by tests.

| Axis | D [m] | L [m] | n | T_mean [s] | mean [kg·m²] | sample std | sample var |
|---|---|---|---|---|---|---|---|
| Ixx (roll) | 0.090 | 0.427 | 10 | 4.50433 | 0.153184 | 0.004362 | 1.903e-5 |
| Iyy (pitch) | 0.185 | 0.424 | 10 | 1.98267 | 0.126285 | 0.003041 | 9.246e-6 |
| Izz (yaw) | 0.163 | 0.505 | 9 | 2.66815 | 0.149050 | 0.001698 | 2.884e-6 |

Yaw outlier: the 8.48 s trial was judged an outlier candidate by the user and
excluded (n = 9); it is preserved in the YAML, and including it would give
0.150874 / std 0.005984 (n = 10). Canonical Izz is the excluded value.
Uncertainty is **recorded only** — not wired into any randomization.

`[FACT]` Reproduction check: `I = m·g·D²·T²/(16π²L)` with T = total/3, then
the **mean of per-trial I** (not I at the mean period), reproduces all
reported means, stds, variances and the with-outlier Izz **exactly at
g = 9.8065**. The user did not state g; 9.8065 is *inferred* (it equals
`LandingConfig.gravity_mps2`). The MuJoCo constant 9.80665 would move Ixx by
+3e-6. The canonical values are the user's literals, not recomputed.

`[CAUTION]` What this does NOT establish:
  * The off-diagonal products are an assumption, not a measurement. Frame
    trap for any future use: FRD→FLU (`diag(1,−1,−1)`) leaves diagonal terms
    unchanged but flips `Ixy`, `Ixz`; the builder passes `diaginertia`
    through unconverted, which is only valid for a diagonal tensor.
  * `Ixx/Iyy ≈ 1.21` on a nominally symmetric X-frame. `[INTERPRETATION]`
    the ballast brick (0.09 × 0.21 × 0.06 m, long in y *if* those are body
    axes) points the same way, but no evidence shows it explains the gap.
    Not adjusted toward symmetry.
  * R² is N/A (manual stopwatch); confidence is tagged `medium`, not `high`.
  * The measured MJCF has **no ground-contact geoms** (gear geometry
    unknown), so free-fall through the ground plane is expected and contact
    behavior with this vehicle is untested.
  * Force-level checks (free fall = −g, hover-equivalent force m·g ⇒ zero
    acceleration, torque τ ⇒ α = τ/I per axis, inertia bound to body axes)
    prove the wiring, **not** that the real vehicle can produce that
    thrust: `max_collective_thrust_n` is UNKNOWN. Hover thrust m·g =
    62.8401 N (g = 9.8065) / 62.8410 N (g = 9.80665), 15.71 N ≈ 1.602 kgf per
    motor under an equal-sharing assumption — DERIVED, not a maximum.

`[OPEN]` `MuJoCoDynamics(params)` cannot be constructed from the measured
config. `[CORRECTION — 2026-09-19, third pass]` This used to be two
independent reasons; the first is gone: `landing_reference_point_body_m()` no
longer raises (`ground_clearance_m = 0.241` ⇒ `[0, 0, 0.241]`, x/y assumed
centred). What remains is `IdentifiedWrenchActuation.__init__`, which requires
`thrust.max_collective_thrust_n` and `identified_response.tau_*` / `delay_s`
(with `validate=False` it stops with a `TypeError` on `float(None)`; with
validation `load_uav_params` raises `MissingMeasurementError` first).
`_REQUIRED_FIELDS` was deliberately **not** relaxed.

### 0.11.6 `[CURRENT — 2026-09-19, fourth pass]` Skid-type landing gear footprint

`[FACT]` The real gear is two continuous parallel skid bars along body X.
`[USER_CONFIRMED]` skid effective length **0.300 m**, centre-line spacing
**0.310 m**; `[MEASURED]` CG → ground **0.241 m**. `[DERIVED]` half length
0.150 m, half spacing 0.155 m. `[MODELLING ASSUMPTION]` the footprint is centred
and symmetric about the CG in x and y (`CENTERED_SYMMETRY_ASSUMPTION`) — the
actual per-leg offset from the CG was **not** measured.

Recorded in `landing_gear_points_body_m` as PHYSICAL contact points (FRD, +Y
right; order left-front, left-rear, right-front, right-rear; z = 0.241):
`[+0.150, −0.155]`, `[−0.150, −0.155]`, `[+0.150, +0.155]`, `[−0.150, +0.155]`.
`[CAUTION]` These four coordinates are **not** "fully measured point
coordinates": provenance is `DERIVED_FROM_MEASURED_DIMENSIONS` +
`CENTERED_SYMMETRY_ASSUMPTION`.

`[FACT]` Compiled MuJoCo geometry (existing conversion path, no builder change):
four collision spheres of radius 0.020 m, centres in FLU
`[+0.150, +0.155, −0.221]`, `[−0.150, +0.155, −0.221]`,
`[+0.150, −0.155, −0.221]`, `[−0.150, −0.155, −0.221]`; sphere bottoms at
z = −0.241. The 0.221 is derived by `landing_gear_sphere_centers_body_m()`, never
stored. **Limitation:** two continuous bars are approximated by four endpoint
spheres — a deliberate v0 simplification, not a line-contact model; a
capsule/cylinder skid is a possible future change, not done.

### 0.12 `[CURRENT — 2026-09-25]` RL Baseline v0.1 packaging

`[DECISION — as stated in the user's 2026-09-25 session brief]` Scope of the pass: make the repository a first
reproducible team baseline WITHOUT changing research behaviour (no reward / PID / observation / action / randomization /
physics change, no retraining, no checkpoint regeneration, no landing_mujoco redesign, no merge, no push).
Baseline of record as stated there: canonical environment `landing_rl/envs/landing_env.py`; contract 16-D obs / 3-D
normalized residual action (XY ±0.25 m/s, Z ±0.05 m/s) at 20 Hz; model generation of record
`ppo_landing_residual_v4_stage2_contact_final.zip` + `vecnormalize_v4_stage2_contact.pkl`; main evaluation scripts
`eval_compare_v2.py` and `eval_robustness_paper.py`.

`[FACT]` Commits created (local, on `refactor/landing-rl-architecture`, on top of `6306699`):

| commit | content |
|---|---|
| `6c4fd81` | measured UGRP parameters (`param_schema.py`, `ugrp_vehicle_measured.yaml`), MJCF injection (`mjcf_builder.py`), `test_ugrp_measured_params.py`, `test_measured_mjcf_injection.py`, `test_param_schema.py` update |
| `a666ab8` | `MUJOCO_MODEL.md` rewrite (documentation) |
| `7d65074` | `landing_rl/evaluation/artifacts.py`; CLI + CWD-independent paths for `eval_compare_v2.py` / `eval_robustness_paper.py`; test gate without machine paths; `eval_env_stage0.py` pairing fix + DEPRECATED; `plot_trajectory.py` DEPRECATED |
| `b633024` | `BASELINE.md`, `MODEL_ARTIFACTS.md` |
| `9dd5563` | `requirements.txt`, README quickstart, `landing_rl/evaluation/smoke.py`, `smoke_test.py` |
| (this file) | `PROJECT_HANDOVER.md` |

`MUJOCO_LOG.TXT` (GLFW "could not create window" log from the viewer crash) is a debug artifact and was deliberately
NOT committed; it is still untracked in the working tree.

`[FACT]` Evaluation interface now: `--policy {all,pid,random,ppo}` (`eval_compare_v2`), `--policy {all,pid}`
(`eval_robustness_paper`), `--model`, `--vecnorm` (`--vecnormalize` kept as an alias). Defaults resolve independent of
the working directory: `$UGRP_RL_ARTIFACT_DIR`, else `<repo>/mujoco_rl/runs`; results default to
`mujoco_rl/runs/paper_eval`, never the artifact directory. PID / random need no artifacts. Checkpoint gate
(`test_checkpoint_compatibility.py`) reads the same location; without artifacts it skips with a message naming the
files, and `UGRP_RL_REQUIRE_ARTIFACTS=1` makes that a failure.

`[FACT]` NON-REGRESSION. (a) Nothing under `landing_rl/{envs,controllers,dynamics,perception,contact,disturbances}`,
`landing_mujoco/{dynamics,envs,coordinates}`, `mujoco_rl/envs` or the three `train_*.py` scripts changed; the
`make_config()`, `COMMON_CONFIG`, `CASE_OVERRIDES` and statistics helpers in the eval scripts are AST-identical to
`HEAD`. (b) With the pair of record on this machine (Python 3.10.12, numpy 2.2.6, gymnasium 1.3.0, SB3 2.9.0, torch
2.12.1, mujoco 3.4.0): the `eval_compare_v2` output (PID 190/200, random 183/200, PPO 191/200, seeds 5000-5199) is
identical to the archived `eval_compare_v2_paper_200ep.txt`, and `robustness_summary.csv`,
`robustness_episodes.csv`, `robustness_paired_pid_vs_ppo.csv` (seeds 7000-7199) are BYTE-IDENTICAL to the archived
2026-08-31 files (archive hashes in `MODEL_ARTIFACTS.md`). PID-only was also reproduced (190/200) with no artifacts
present from a different working directory.

`[FACT]` TESTS (unittest, the README commands; artifacts supplied via `UGRP_RL_ARTIFACT_DIR`, strict mode on):
`landing_rl` 238 tests OK in 190 s; `landing_mujoco` 221 OK in 55.7 s; `system_id` 67 OK in 8.8 s -- 526 tests,
0 failures, 0 errors, 0 skipped. Without artifacts (fresh-clone behaviour) `landing_rl` runs 233 tests and reports 1
class-level skip (the 5 checkpoint-gate tests). The 238 = 218 previous + 17 (`test_evaluation_artifacts.py`) + 2
(`test_smoke.py`) + 1 (`test_C_canonical_eval_scripts_use_pair_of_record`). `system_id` integration tests that need
the git-ignored sample ULogs would skip on a fresh clone (count NOT VERIFIED). `requirements.txt` resolves in a
fresh venv (`pip install --dry-run`); the config-loading suites also pass on PyYAML 6.0.3 (159 tests); an actual
from-scratch install was NOT performed.

`[FACT — new, from artifact metadata]` TRAINING LINEAGE. `num_timesteps` of `v3_stage1`, `v3_stage1_rewardfix1` and
`v3_stage2_contact` are all exactly 1,515,520 (stage 0 = 507,904 + one 1M-step run each) and all three VecNormalize
files have `obs_rms.count` 1,515,528; `v4` is 2,523,136 / 2,523,148. The `target_valid` mean of each VecNormalize
gives the invalid-target fraction seen while training: stage 0 0.0%, stage 1 1.0%, stage 1 rewardfix1 1.0%, stage 2 v3
5.0%, v4 5.0% (from stage 2 v3, script-confirmed). `[INTERPRETATION]` stage 2 v3 was branched from stage 0, not from
stage 1: the three v3 checkpoints are sibling branches, not a chain. There is no script for stage 0 or for stage 2 v3.

`[CORRECTION — 2026-09-25]` §4 above states that the stage-0 configuration "is only recoverable via
`mujoco_rl/eval_env_stage0.py :: make_config()`" and orders the checkpoints as a chain. Corrected: that `make_config()`
has 1% target dropout, equal to the STAGE-1 training configuration (`train_ppo_v3_long.py`); stage 0's VecNormalize
saw no invalid target at all, so the stage-0 training configuration is UNKNOWN. The order stage 1 -> rewardfix1 -> stage
2 in the §4 table is not supported by the artifacts (see the lineage above). The original §4 text is kept.

`[RESOLVED — 2026-09-25]` §7 items 6 (`eval_env_stage0.py` model/VecNormalize mismatch) and 8 (README branch pointer):
the stage-1 model is now paired with the stage-1 VecNormalize saved with it by `train_ppo_v3_long.py` (script-confirmed;
the script is also marked DEPRECATED and warns at run time), and the README branch note was replaced.
`plot_trajectory.py` (loads `./runs/ppo_landing_residual`, no VecNormalize, default `LandingConfig()`) was NOT repaired
because no correct historical pair can be established; it is marked DEPRECATED. Neither is a baseline tool.

`[CAUTION]` What this pass does NOT establish: the reference result is a reproducibility statement, not a performance
claim -- in the archived evaluation every paired PID-vs-PPO McNemar p-value is >= 0.125 and the dominant failure for
both is `excessive_bounce`. The simulated PID (`kp_xy 0.45, kd_xy 0.18, kp_z 0.35, kd_z 0.12`, no integral) is not the
deployed ROS2 controller (`precision_landing_controller.py`: kp 0.35/0.35/0.22, ki 0.008/0.008/0.003, kd
0.16/0.16/0.10, output limits 0.8/0.8/0.35, deadband, derivative filter, slew limit, target EMA filter). The trained
policy has never run in the measured MuJoCo model (which cannot be constructed from the measured set), and no PPO
inference / deployment code exists under `ros2/`, `jetson/` or `px4/`.

`[OPEN]` (1) The model artifacts have no shared hosting; the files exist only in the maintainer's local
`drone_stack/mujoco_rl/runs/`. (2) Stage 0 and stage 2 v3 training scripts are missing; training is unseeded and the
train scripts still use `./runs` relative to the working directory. (3) Whether `main` or `refactor/landing-rl-architecture`
is the team baseline branch is undecided; README states the RL baseline is NOT on `main`. (4) The 6 local commits above
plus this handover commit are unpushed.

`[TODO]` User approval, then push (never force). Decide artifact hosting. Decide whether to reconstruct or record as lost
the stage-0 / stage-2-v3 scripts. Document how each of the 16 observations is produced on the vehicle.

---

# 1. Project architecture

[CONFIRMED — verified by passing regression/structural-freeze tests, §0.5;
updated 2026-09-13 for the entry-point migration + backend seam, §0.3/§0.4 —
UNCOMMITTED]

```
Training entry point (mujoco_rl/train_ppo_*.py, eval_*.py, scripts/plot_trajectory.py)
    ↓  from landing_rl.envs.landing_env import LandingEnv, LandingConfig   [2026-09-13]
Environment construction
    landing_rl/envs/landing_env.py  :: LandingEnv          [LIVE, canonical]
    mujoco_rl/envs/env_prototype.py :: LandingEnv          [frozen legacy reference only —
                                                             no longer imported by any script]
        ↓
Controller           landing_rl/controllers/baseline_controller.py :: BaselineController
Dynamics / plant      landing_rl/dynamics/plant_model.py :: PlantModel
                        → VehicleDynamicsBackend (Protocol, landing_rl/dynamics/    [2026-09-13]
                          vehicle_dynamics_backend.py) — interface seam only
                          → landing_rl/dynamics/legacy_dynamics.py :: LegacyVehicleDynamics
                            [the only implementation; still the only runtime backend]
                          → (future) MuJoCoDynamics — NOT implemented
                        → landing_rl/contact/contact_model.py :: ContactModel
Perception / obs      landing_rl/perception/{target_measurement,obs_latency,observation_noise}.py
Disturbance / delay   landing_rl/disturbances/disturbance_model.py,
                        landing_rl/envs/{action_latency,loop_timing}.py
Contact               landing_rl/contact/contact_model.py [unchanged this session; ContactModel
                        architecture itself was explicitly out of scope for the backend refactor]
Reward / termination  inline in landing_rl/envs/landing_env.py :: LandingEnv.step (lines 820-935)
                        — NOT extracted into its own module in either implementation
```

`mujoco_rl/envs/env_prototype.py` implements the identical logic inline
(monolithic, 1221 lines) rather than through the component split above. Both
implementations are proven byte-identical in behavior by the OLD-vs-NEW
parity suite (§0.5), which continues to load `env_prototype.py` by file path
as the frozen reference regardless of which one is the live import target.

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

## 5.1 `system_id/preprocessing/` — IMPLEMENTED 2026-09-07 (working tree, not committed)

[FACT] First SI implementation phase. Scope was strictly preprocessing —
**no parameter identification, no dynamics backend**. Modules
(`system_id/preprocessing/`):

| module | responsibility |
|---|---|
| `ulog_loader.py` | `pyulog` wrapper: SHA256, PX4 version/hw/airframe/params, one `TopicData` per topic; `ULogLoadError` for missing/corrupt file or missing **required** topic; missing **optional** topics (e.g. `esc_status`) recorded, never fatal |
| `schema.py` | version-tolerant signal schema; per-loop `ColumnSpec` (name/unit/frame/source); `rates_setpoint_fields` resolves scalar `roll/pitch/yaw` vs `xyz[0..2]`; Euler ALWAYS derived from quaternions (no `roll_body` dependence); `NEAR_LEVEL_MAX_RAD` (10°); `SCHEMA_VERSION="0.2.0"` |
| `frames.py` | `quat_to_euler_xyz` (intrinsic x-y-z), `tilt_from_quat`, thrust-vector magnitude, `wrap_pi`; degenerate quaternion → zeros not NaN |
| `timebase.py` | per-loop canonical grids (velocity ~10 Hz, attitude ~20 Hz, rate ~50 Hz, translation ~20 Hz) — **never one global fastest rate**; `timestamp_sample` for measurement topics else `timestamp`; `vehicle_local_position_setpoint` nearest-matched (`nearest_match`/`sample_aligned`, ±`CYCLE_PAIR_TOL_US`=15 ms) to the `vehicle_local_position` sample of its own MPC cycle; ZOH for slow→fast setpoints only |
| `masks.py` | ONE `flight_base` mask (8 base reasons: `not_armed / not_offboard / not_mc_pos / landed / ground_contact / maybe_landed / motor_saturated / accel_sentinel`) + `ground_effect` term + `MaskResult.block_base(block)` / `combine_block(block, *extra)`. `in_ground_effect` rejects ONLY `velocity_z / thrust / translation` (`BLOCK_GE_SENSITIVE`); `attitude / rate / velocity_xy` keep low-altitude data. `thrust` additionally requires `near_level` (`BLOCK_NEAR_LEVEL_REQUIRED`). Named constants `MOTOR_SAT_HIGH=0.98`, `MOTOR_SAT_LOW=0.02`, `MOTOR_SAT_DILATION_S=0.03`, `ACCEL_SENTINEL_ABS_MPS2=30.0`, `NEAR_LEVEL_MAX_RAD`. `valid_core`/`valid_strict` kept as read-only back-compat aliases. |
| `segments.py` | contiguous valid runs only (never concatenates disconnected intervals); `MIN_SEGMENT_S=1.5`; splits on internal time gaps; stores per-segment source/start/end/duration + preceding-gap rejection summary. Segmentation is **per SI block** (each block uses its own block mask). |
| `dataset.py` | four per-loop datasets; NED acceleration and body specific force kept in separate named columns, never summed; velocity setpoint column is `vehicle_local_position_setpoint.vx/vy/vz` (NOT `trajectory_setpoint.velocity`). Each `LoopDataset` also carries `block_masks` (final bool[n] per block = `block_base & near_level? & required-I/O-finite & setpoint-pairing-matched?`), `aux` masks, and `pairing` stats. A row whose required input/output is NaN, or whose `vehicle_local_position_setpoint` pairing failed, is `False` for that block. `body_z_specific_force_proxy` (renamed from `spec_thrust_recon`) + a mandatory `near_level_valid` column carry an explicit "CRUDE proxy, not a calibrated thrust" warning in their metadata. |
| `provenance.py` | SHA256, PX4 version, hardware, airframe + MPC/MC parameter subsets (of the LOGGED vehicle), tool versions, git commit/dirty, UTC timestamp, `schema_version`; explicit `source_project` + `dataset_role` |
| `report.py` | data-quality / excitation gate — **per SI block** (attitude / rate / velocity_xy / velocity_z / thrust / translation): valid duration, segments, per-channel GOOD/MARGINAL/POOR (std + range + I/O correlation caps); plus per-loop timebase diagnostics and the setpoint-pairing match fraction. Disclaimer: *"does NOT declare any physical model identified"* |
| `pipeline.py` | orchestrator `preprocess_log()` + `run_batch()` + `python -m system_id.preprocessing.pipeline` CLI; writes `<loop>.npz` (columns + `_flight_base` / `_ground_effect` / `_reject_<reason>` / `_block_<name>` / `_aux_<name>`) + `columns.json` / `segments.json` (keyed by block) / `provenance.json` / `report.{json,txt}` |

**Derived-data location:** `system_id/derived/sample_other_project/<logstem>/`
(added to `.gitignore`; regenerable, never a source of truth). Raw ULogs are
read-only and were not moved or deleted.

**Tests:** `system_id/tests/` (plain `unittest`, mirrors `landing_rl/tests/`
style). **67 tests** — synthetic-fixture unit tests for quaternion conversion,
timebase/stamp selection, ZOH, nearest-match cycle alignment, saturation mask
+ dilation, sentinel rejection, segment splitting, `rates_setpoint` version
tolerance, missing-optional-topic handling, provenance labelling, **block-mask
model + GE-sensitivity map, setpoint-pairing / unmatched-row / NaN-I/O
rejection, `body_z_specific_force_proxy` naming + near-level gate**; plus a
synthetic end-to-end write/round-trip and an integration test over the four
sample ULogs (skips if absent). `python3 -m unittest discover -s
system_id/tests` → **67 passed** (2026-09-07, pre-commit audit).

## 5.2 Sample ULogs — OTHER-PROJECT, validation only

[FACT] `landing_rl/flight_log_ulg/{03_49_26,05_34_35,06_56_24,08_53_49}.ulg`
(gitignored via `*.ulg`). Provenance recorded as
`source_project = OTHER_PROJECT_SAMPLE`, `dataset_role =
PIPELINE_VALIDATION_ONLY`. They are a **`PX4_FMU_V6C` quadcopter on PX4
`main` / `v1.17.0-alpha`** — NOT the UGRP vehicle and NOT the pinned UGRP PX4
1.15. They are offboard **position**-mode flights (not velocity mode). All
four parse; schema is identical across them; `esc_status` absent (handled).

[CAUTION] **No numeric value derived from these logs is a UGRP parameter.**
The MPC/MC parameters stored in each `provenance.json` are the *logged
other-project vehicle's* controller settings, kept only as provenance. Nothing
was written to `system_id/results/`, `landing_rl/configs/`, or any identified
dynamics/training config.

## 5.3 What is still NOT implemented

`system_id/identification/`, `system_id/validation/`, `system_id/results/` are
**empty**. No delay / velocity / attitude / rate / thrust identification, no
validation/plotting, no exported parameter set. `IdentifiedClosedLoopDynamics`
does not exist; `LegacyVehicleDynamics` and `PlantModel` are untouched.

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
   `main()`. `[RESOLVED — 2026-09-25, commit 7d65074]` now paired with `vecnormalize_v3_stage1.pkl`;
   script marked DEPRECATED (§0.12).
7. **Misleading parameter name, acknowledged in-code**:
   `process_noise_vel_std_mps` is consumed as an *acceleration*-noise
   standard deviation, not a velocity-noise one; the module's own docstring
   states the name is kept only for RNG-stream compatibility
   (`landing_rl/dynamics/process_noise.py:29-32`).
8. `[RESOLVED — 2026-09-25, commit 9dd5563]` **`README.md` branch pointer is stale** — states *"Current working
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

### 2026-09-07 — `system_id/preprocessing/` implemented (first SI implementation phase)

DATE / PHASE: 2026-09-07 — SI preprocessing pipeline (preprocessing only;
explicitly NO parameter identification, NO dynamics backend).

`[FACT]` Implemented `system_id/preprocessing/` (10 modules + package init) +
`system_id/tests/` (7 `test_*.py` modules + `_synth.py` fixture builder + init;
53 tests) + `system_id/__init__.py`. Added `system_id/derived/` to `.gitignore`.
Full module/responsibility table and constants are in §5.1.

IMPLEMENTATION: new files only. No file under `landing_rl/` or `mujoco_rl/`
modified. No ROS2 / PX4 / Docker file modified. `.gitignore` gained one entry
(`system_id/derived/`). `PROJECT_HANDOVER.md` updated (§0.6, §0.9, §0.10, §5,
this entry).

ARCHITECTURE CLARIFICATION RECORDED (from the user, this phase): the future
grey-box model must separate **source-verified PX4 controller dynamics**
(velocity controller; acceleration→attitude/thrust map — both defined by PX4
source + MPC params) from **identified vehicle response** (attitude-setpoint→
attitude, thrust-setpoint→effective thrust, optionally rate-setpoint→body-rate,
translational residuals). The end-to-end `v_cmd → v` stays a *validation*
mapping, not one opaque fitted plant block. Preprocessing was written to
encode these semantics: the velocity dataset labels its setpoint column as
`vehicle_local_position_setpoint.vx/vy/vz` (PX4 `_vel_sp`), NOT
`trajectory_setpoint.velocity`, and records the logged MPC gains as provenance.

VALIDATION:
  * `python3 -m unittest discover -s system_id/tests -p "test_*.py"` →
    **53 passed, 0 fail** (2026-09-07).
  * `python3 -m unittest discover -s landing_rl/tests -p "test_*.py"` →
    **192 passed, 0 fail**, 109.8 s — byte-identical to the §0.5 structural-
    freeze baseline (46,958 paired steps, 5 real-checkpoint compatibility
    tests ran and passed). **Paper/legacy behaviour unchanged.**
  * `python3 -m system_id.preprocessing.pipeline` → all 4 sample ULogs
    processed, outputs written to `system_id/derived/sample_other_project/`
    (confirmed gitignored via `git check-ignore`).

RESULT (data-quality gate over the 4 OTHER-PROJECT sample logs — these are
NOT UGRP numbers; recorded only to show the pipeline runs and to characterise
the fixtures):

| log | velocity valid (strict) | attitude valid | rate valid | longest seg |
|---|---|---|---|---|
| 03_49_26 | ~6.6 s (1 seg) | ~9.4 s | ~10.7 s | ~2.1 s |
| 05_34_35 | ~41 s (6 seg) | ~45.6 s | ~49.6 s | ~4.0 s |
| 06_56_24 | ~31.7 s (3 seg) | ~35.9 s | ~39.4 s | ~2.4 s |
| 08_53_49 | ~46.9 s (6 seg) | ~51.7 s | ~56.5 s | ~6.1 s |

Excitation grades (sample logs): velocity-error→accel N/E GOOD, D
MARGINAL/POOR; roll/pitch-setpoint→attitude MARGINAL–GOOD; roll/pitch-rate
POOR (yaw-rate GOOD); thrust MARGINAL (near-hover, thrust-marginal airframe).
Matches the 2026-09-06 read-only audit within grid-rate differences.

`[INTERPRETATION]` The pipeline is ready to ingest real UGRP ULogs. The four
sample logs are sufficient to exercise every code path (schema drift,
saturation, sentinel, multi-rate, segmentation, provenance).

`[CAUTION]` NOTHING has been identified. The sample logs cannot demonstrate
the sim's Block 1 (`v_cmd → accel demand`) as defined, because they are
offboard-position-mode, other-project flights. Roll/pitch-rate and thrust
identification are not feasible from them regardless.

`[OPEN]` UGRP identification needs dedicated flights: pinned PX4 1.15,
offboard **velocity** mode, raised `SDLOG_PROFILE` rate for
`vehicle_local_position_setpoint`, bounded single-axis excitation (CLAUDE.md
§31), and the CLAUDE.md §3 gain-stabilisation precondition satisfied.

`[TODO]` Next phase (needs user approval): `system_id/identification/` starting
with attitude-setpoint→attitude and rate-setpoint→body-rate; validate (not
fit) the PX4 velocity→accel and accel→attitude/thrust maps.

GIT STATE: branch `refactor/landing-rl-architecture`, HEAD `bc40963`, NOT
committed. Working tree: `M .gitignore`, untracked `system_id/` (new package +
tests) and `system_id/derived/` (gitignored outputs), plus the pre-existing
untracked `CLAUDE_revised.md:Zone.Identifier`.

UGRP NUMERIC PARAMETERS IDENTIFIED: NO.
SAMPLE-LOG NUMBERS TRANSFERRED TO UGRP: NO.
IDENTIFIED DYNAMICS BACKEND: NOT IMPLEMENTED.

### 2026-09-07 — `system_id/preprocessing/` pre-commit consistency audit (behaviour changed)

`[FACT]` Same-day follow-up to the entry above. Three reviewed issues, all
confirmed and corrected (preprocessing only; still no identification, no
dynamics backend; `LegacyVehicleDynamics` / `PlantModel` untouched;
`landing_rl/` unchanged).

1. **Block-specific validity masks.** The single `valid_strict` mask (which
   excluded `in_ground_effect` for every loop) was discarding low-altitude
   attitude/rate data that ground effect cannot contaminate. Replaced with one
   `flight_base` mask + per-block masks: `attitude` / `rate` / `velocity_xy`
   use `flight_base`; `velocity_z` / `thrust` / `translation` additionally
   exclude `in_ground_effect`; `thrust` additionally requires `near_level`.
   Segmentation and the report are now **per SI block**. `valid_core` /
   `valid_strict` remain as read-only aliases.

2. **Setpoint/measurement pairing validity.** `vehicle_local_position_setpoint`
   ↔ `vehicle_local_position` nearest-match (±15 ms) verified on all four
   sample logs: **100 % matched, 0 unmatched** in every log. Added an explicit
   `setpoint_pairing_matched` aux mask; unmatched rows now force both velocity
   block masks `False` and their `a_sp_* / v_sp_*` columns are NaN there;
   required-I/O-finite terms fold into every block mask (a NaN input or output
   makes that block `False`). Per-log pairing fraction is reported
   (`report.loops.<loop>.setpoint_pairing`). New synthetic tests cover the
   unmatched-pair and NaN-I/O rejection paths.

3. **Specific-thrust proxy naming / safety.** `spec_thrust_recon` renamed to
   **`body_z_specific_force_proxy`** (= `-vehicle_acceleration.xyz[2]`), with
   BOTH a mandatory `near_level_valid` column/gate (tilt ≤ `NEAR_LEVEL_MAX_RAD`
   = 10°) and an explicit metadata warning ("CRUDE proxy … NOT a calibrated
   thrust measurement"). The `thrust` block mask is `False` wherever the
   vehicle is tilted past the gate, so identification code cannot pick it up as
   a generally valid thrust signal.

IMPLEMENTATION: `system_id/preprocessing/{schema,masks,dataset,report,pipeline}.py`
edited; `system_id/tests/{_synth,test_masks,test_pipeline}.py` updated;
`system_id/tests/test_blocks_and_pairing.py` added. `SCHEMA_VERSION` /
`PREPROCESSING_VERSION` bumped `0.1.0 → 0.2.0`. No non-`system_id/` file
changed (`.gitignore` and this document aside).

VALIDATION:
  * `python3 -m unittest discover -s system_id/tests` → **67 passed, 0 fail**.
  * `python3 -m unittest discover -s landing_rl/tests` → **192 passed, 0 fail**
    (162.9 s) — structural-freeze baseline unchanged.
  * `python3 -m system_id.preprocessing.pipeline` → all 4 sample logs
    reprocessed; per-block report + `_block_*` / `_aux_*` npz arrays +
    block-keyed `segments.json` written under the gitignored derived path.

RESULT (sample-log data-quality gate — NOT UGRP numbers): per-block valid
seconds now higher for attitude/rate (they keep GE samples), e.g. 08_53_49
attitude 52.1 s / rate 57.0 s (were 51.7 / 56.5 s under the old GE-excluding
`valid_strict`). Setpoint pairing 100 % on all four logs. Near-level gate trims
~1–5 % of the thrust block (gentle flights).

GIT STATE: branch `refactor/landing-rl-architecture`, HEAD `bc40963`, NOT
committed. `M .gitignore`, `M PROJECT_HANDOVER.md`, untracked `system_id/`.

UGRP NUMERIC PARAMETERS IDENTIFIED: NO.
SAMPLE-LOG NUMBERS TRANSFERRED TO UGRP: NO.
LEGACY DYNAMICS MODIFIED: NO.
PLANTMODEL MODIFIED: NO.

### 2026-09-13 — Canonical-environment entry-point migration (mujoco_rl → landing_rl)

`[FACT]` Switched the actually-imported `LandingEnv`/`LandingConfig` in
every real train/eval script from `mujoco_rl/envs/env_prototype.py` to
`landing_rl/envs/landing_env.py`. Purely a structural/import-path change —
no observation/action/reward/dynamics/controller/randomization/RNG-ordering
change, per CLAUDE.md §15's structural-freeze rule. See §0.3 for full detail
(superseding the pre-2026-09-13 §0.3 text, kept above for history).

IMPLEMENTATION: 7 files modified, each by exactly the same pattern (repo-root
`sys.path` bootstrap + import-line swap; `make_config()` bodies and
checkpoint/VecNormalize path literals byte-unchanged):
`mujoco_rl/{train_ppo_stage1_1m,train_ppo_v3_long,train_ppo_stage2_contact,
eval_compare_v2,eval_env_stage0,eval_robustness_paper}.py`,
`mujoco_rl/scripts/plot_trajectory.py`. 1 new file:
`landing_rl/tests/test_entry_point_migration.py` (7 tests). No file under
`mujoco_rl/envs/`, `landing_rl/dynamics|contact|controllers|perception|
disturbances/`, or any config default was touched.

VALIDATION:
  * Baseline (pre-migration) `python3 -m unittest discover -s landing_rl/tests
    -p "test_*.py"` → **192 passed, 0 fail, 147.5s** (includes the 5
    checkpoint-compatibility tests, 0 skipped).
  * New `test_entry_point_migration.py` in isolation → **7 passed, 42.3s**:
    identity-checks (migrated `LandingEnv`/`LandingConfig` IS
    `landing_rl.envs.landing_env`'s, IS NOT `env_prototype`'s), config-value
    equality against the frozen `CONFIG_SPECS` table (by resolved
    `dataclasses.asdict`, not re-typed literals), checkpoint/VecNormalize
    path-literal presence, and exact OLD-vs-NEW parity — seeds 0–9 × 5
    action patterns (zero / pseudo_random / saturation / fixed_plus /
    fixed_minus) × the 4 named script configs (`default`, `stage0_eval`,
    `stage2_eval`, `stage2_train`) + the 5 `eval_robustness_paper.py` cases
    (`A_nominal`…`E_mixed`) = **210 episode-pairs, 0 mismatches**.
  * Post-migration full suite → **199 passed, 0 fail, 0 skipped, 260.4s**
    (192 baseline + 7 new; structural-freeze's own 140-paired/46,958-step
    matrix re-ran unchanged and still passed).
  * Manual checkpoint-compatibility smoke test: `eval_compare_v2.py`'s own
    (migrated) `make_config()` + `landing_rl.LandingEnv` attached to the
    real external checkpoint
    (`/home/qntmdghkss/drone_stack/mujoco_rl/runs/
    ppo_landing_residual_v4_stage2_contact_final.zip` +
    `vecnormalize_v4_stage2_contact.pkl`, read-only, not modified) —
    `PPO.load` → `VecNormalize.load` → `reset` → `predict(deterministic=True)`
    → `step` all succeeded; `observation_space.shape==(16,)`,
    `action_space.shape==(3,)`.
  * `python3 -m unittest discover -s system_id/tests` → **67 passed**
    (unaffected, confirming isolation).
  * Import smoke test: all 7 migrated files load standalone via
    `importlib.util.spec_from_file_location` and resolve
    `LandingEnv.__module__ == "landing_rl.envs.landing_env"`.

`[DECISION — user-directed, this session]` `mujoco_rl/envs/env_prototype.py`
is kept, unmodified, as the frozen legacy reference (not deleted, not
wrapped) — `test_legacy_regression_contract.py` continues to load it by
file path.

`[CAUTION]` **Uncommitted.** `git status --short` shows 7 modified +
1 untracked file from this entry alone. `landing_rl/training/` and
`landing_rl/evaluation/` are still empty — this migration changed the
import target inside the existing `mujoco_rl/*.py` scripts, it did not
create new entry points under `landing_rl/`.

GIT STATE: branch `refactor/landing-rl-architecture`, HEAD `9a154df`,
working tree dirty (this entry's files, uncommitted).

### 2026-09-13 — VehicleDynamicsBackend interface seam (MuJoCo pre-work, no MuJoCo)

`[FACT]` Added a minimal interface seam so `PlantModel` depends on a
`VehicleDynamicsBackend` contract rather than the concrete
`LegacyVehicleDynamics` class, as prework for an eventual `MuJoCoDynamics`
backend. Explicitly NOT a MuJoCo implementation: no `import mujoco`, no
MJCF, no qpos/qvel handling, no actuator mapping, no frame/quaternion
adapter, no new physics parameter, no `ContactModel` change.

IMPLEMENTATION:
  * New `landing_rl/dynamics/vehicle_dynamics_backend.py` —
    `VehicleDynamicsBackend`, a `typing.Protocol` (`@runtime_checkable`)
    with one method, `advance_free_flight(...)`, copied verbatim from
    `LegacyVehicleDynamics.advance_free_flight`'s existing signature (no new
    argument invented).
  * `landing_rl/dynamics/plant_model.py` — `PlantModel.__init__`'s
    `dynamics` parameter now type-hinted `VehicleDynamicsBackend`; docstring
    updated. Public constructor signature unchanged
    (`PlantModel(cfg, dynamics, contact)`); zero new instance attributes.
  * `landing_rl/dynamics/legacy_dynamics.py` — docstring-only addition
    noting `LegacyVehicleDynamics` satisfies the Protocol structurally, with
    no inheritance/code change.
  * `landing_rl/dynamics/__init__.py` — exports `VehicleDynamicsBackend`.
  * New `landing_rl/tests/test_vehicle_dynamics_backend.py` (12 tests, A–E
    per the task's own labels: A `LegacyVehicleDynamics` satisfies the
    contract; B `PlantModel` accepts the contract (including a from-scratch
    duck-typed object, not a `LegacyVehicleDynamics` subclass); C
    `LandingEnv()`'s default backend stays `LegacyVehicleDynamics`; D a mock
    backend runs through `PlantModel.step()` with zero `LandingEnv` source
    change required (`landing_env.py`'s source text contains neither
    `"VehicleDynamicsBackend"` nor `"MuJoCo"`, asserted directly); E exact
    OLD-vs-NEW parity unchanged by the seam, seeds 0–4 × 3 patterns × the 4
    named configs).

DESIGN CHOICE: `typing.Protocol` over `abc.ABC` or bare duck typing — no
file under `landing_rl/` uses `abc`/`abstractmethod` anywhere (grep-verified
this session); every existing component is a plain class with no base-class
hierarchy. `ABC` would force a new base class onto `LegacyVehicleDynamics`,
a class whose shape `test_structural_freeze.py` already freezes
(`set(vars(env.legacy_dynamics)) == {"cfg", "process_noise"}`). `Protocol`
gives an `isinstance()`-checkable, discoverable contract with **zero** code
change to `LegacyVehicleDynamics`.

VALIDATION:
  * `test_vehicle_dynamics_backend.py` in isolation → **12 passed, 11.2s**.
  * Full `landing_rl/tests` suite → **211 passed, 0 fail, 0 skipped,
    281.5s** (199 prior + 12 new; `test_structural_freeze.py`'s existing
    `Freeze00_ComponentGraphAndOwnershipTest` — which already asserts
    `set(vars(env.plant)) == {"cfg", "dynamics", "contact"}` and
    `set(vars(env.legacy_dynamics)) == {"cfg", "process_noise"}` — was
    re-run UNMODIFIED and still passed, since this refactor adds no
    instance attribute to either class). `test_structural_freeze.py` itself
    was NOT edited this session — the new backend-contract-specific freeze
    assertions the task asked for were added additively in
    `test_vehicle_dynamics_backend.py` instead, to avoid any risk of
    weakening the existing frozen file.
  * Manual re-run of the checkpoint-compatibility smoke test (same
    external v4 checkpoint as above) against the post-refactor code →
    passed; `isinstance(env.plant.dynamics, LegacyVehicleDynamics)` and
    `isinstance(env.plant.dynamics, VehicleDynamicsBackend)` both `True`
    under the actual migrated `eval_compare_v2.py` script.

`[INTERPRETATION]` Chain-boundary analysis for a future `MuJoCoDynamics`
(not acted on this session): the current `v_cmd → accel_cmd → attitude/
thrust setpoint` stage inside `_velocity_command_to_inner_loop_setpoints`
is a PX4 velocity/attitude *controller* approximation, not vehicle physics,
and is a KEEP-OUTSIDE-MUJOCO candidate; the `attitude/thrust setpoint →
body-rate/thrust response → rigid-body acceleration → velocity/position`
stage is the physics proxy a MuJoCo backend would actually replace; the
response-alpha time constants, ground-effect functional form, and
ground-friction coefficient are UNCERTAIN / REQUIRE SYSTEM IDENTIFICATION
regardless of which backend is used. No code was changed to reflect this
split — it is filed here as a `[PROPOSAL]`-level note for the next MuJoCo
phase.

`[CAUTION]` **Uncommitted**, layered on top of the same-day entry-point
migration above (also uncommitted). No second backend implementation
exists; `LandingEnv`/`PlantModel` still have no runtime way to select a
non-default backend (no factory, no config flag) — this was a deliberate
scope limit ("do not over-abstract"), not a gap to fill reflexively.

GIT STATE: branch `refactor/landing-rl-architecture`, HEAD `9a154df`,
working tree dirty (this entry's files + the entry-point migration above,
both uncommitted, no unrelated changes present).

### 2026-09-13 — InnerLoopCommandModel: controller/physics split inside LegacyVehicleDynamics (no MuJoCo)

`[FACT]` Acted on the `[INTERPRETATION]` note filed in the entry above:
extracted the CONTROL-side portion of `LegacyVehicleDynamics.advance_free_flight`
(the former private method `_velocity_command_to_inner_loop_setpoints`:
`v_cmd → accel_cmd → attitude/thrust setpoint`, a PX4 velocity/attitude
controller proxy) into its own component, `InnerLoopCommandModel`, leaving
every PHYSICS-side stage (attitude/rate controller proxy + body-rate
response, thrust response, body-to-NED rotation, gravity/drag/wind/process
noise, velocity/position integration) untouched inside
`LegacyVehicleDynamics`. Explicitly NOT a MuJoCo implementation and not a
behavior change — CLAUDE.md §15's structural-freeze rule applies; no
observation/action/reward/gain/noise/delay/contact value changed.

CALL-GRAPH / CLASSIFICATION (verified from source before editing, per
CLAUDE.md §13/§30):
  * A `v_cmd → accel_cmd` and B `accel_cmd → attitude/thrust setpoint`
    (legacy_dynamics.py:138-203, pre-refactor line numbers) = CONTROL-SIDE →
    extracted.
  * The motor-cutoff override on `thrust_accel_setpoint`
    (legacy_dynamics.py:240-241) reads `ContactModel`'s persistent
    `motor_cutoff` state (physics-side concept) — deliberately NOT moved;
    stays in `advance_free_flight`, applied to the `InnerLoopCommand`
    immediately after `compute()` returns, exactly where it sat before.
  * C attitude/rate controller proxy + body-rate response, D thrust
    response, E rotation/gravity/ground-effect/drag/wind, F
    velocity/position integration = PHYSICS-SIDE → untouched, stay in
    `LegacyVehicleDynamics.advance_free_flight`.

IMPLEMENTATION:
  * New `landing_rl/dynamics/inner_loop_command_model.py` —
    `InnerLoopCommand` (plain `@dataclass`: `accel_cmd`,
    `attitude_setpoint`, `thrust_accel_setpoint`, matching the existing
    `VehicleState`/`ContactState`/`ContactResult` plain-dataclass
    convention) and `InnerLoopCommandModel(cfg)` (`compute(state, v_cmd,
    dt, target_yaw) -> InnerLoopCommand`, stateless, owns only `cfg`, zero
    RNG parameters). Body is a verbatim lift of the removed private method.
  * `landing_rl/dynamics/legacy_dynamics.py` — `__init__` now also
    constructs `self.inner_loop_command_model = InnerLoopCommandModel(cfg)`
    internally (own only `cfg`; **no** change to how `LandingEnv`
    constructs `LegacyVehicleDynamics(self.cfg, self.process_noise)` —
    zero edits to `landing_env.py`). `advance_free_flight` calls
    `self.inner_loop_command_model.compute(...)` at the old call site, then
    applies the motor-cutoff override to the returned
    `thrust_accel_setpoint` exactly as before. The old private method is
    deleted (no dead code / no thin wrapper kept). Signature of
    `advance_free_flight` — the `VehicleDynamicsBackend` contract —
    byte-unchanged.
  * `landing_rl/dynamics/__init__.py` — exports `InnerLoopCommand`,
    `InnerLoopCommandModel`.
  * `landing_rl/tests/test_legacy_dynamics.py` — the ownership-set
    assertion `set(vars(lvd)) == {"cfg", "process_noise"}` widened to
    `{"cfg", "process_noise", "inner_loop_command_model"}` (renamed test),
    plus a new `test_inner_loop_command_model_owns_only_cfg`.
  * `landing_rl/tests/test_structural_freeze.py` —
    `Freeze00_ComponentGraphAndOwnershipTest`'s ownership-set assertion for
    `env.legacy_dynamics` widened the same way, a new
    `assertIsInstance(env.legacy_dynamics.inner_loop_command_model,
    InnerLoopCommandModel)` + per-component no-RNG/no-env check added to
    the component loop, and a new `inspect.signature(...)` check pinning
    `advance_free_flight`'s parameter list (proof the backend contract
    itself didn't move).
  * New `landing_rl/tests/test_inner_loop_command_model.py` (6 tests, A–E
    per the task's own labels): A/B/C/D `InnerLoopCommandModel.compute()`
    matches an INDEPENDENT reference (`test_legacy_dynamics.py`'s own
    `_LegacyFreeFlightReplica._velocity_command_to_inner_loop_setpoints`,
    never `LegacyVehicleDynamics` itself) exactly on `accel_cmd` /
    `attitude_setpoint` / `thrust_accel_setpoint`, across 5 init states ×
    4 `v_cmd` (incl. a saturating one) × 3 `dt` × 3 `target_yaw`; plus a
    dedicated check that the motor-cutoff override is NOT part of
    `compute()`'s output and stays a `LegacyVehicleDynamics` responsibility.
    E: full env-trajectory exact OLD-vs-NEW parity, reusing
    `test_legacy_regression_contract.py`'s comparator — seeds 0–9 ×
    (`zero`/`pseudo_random`/`saturation`) × 4 configs, plus seeds 0–9 ×
    (+axis/−axis fixed commands) × 4 configs (reusing
    `test_entry_point_migration.py`'s `_run_paired_actions` helper for the
    axis patterns, per the task's request to cover `+axis`/`-axis`
    explicitly).

DESIGN CHOICE: `InnerLoopCommandModel` is constructed and owned INSIDE
`LegacyVehicleDynamics`, not injected from `LandingEnv`. A future
`MuJoCoDynamics` backend would have its own, possibly different,
command-generation strategy (or might reuse this one if the PX4
attitude+thrust boundary is chosen — see the analysis below); injecting a
Legacy-specific collaborator from `LandingEnv` would leak backend-internal
detail across the `VehicleDynamicsBackend` seam that `PlantModel`/
`LandingEnv` are not supposed to know about. This is also why the backend
contract (`advance_free_flight`'s signature) was left unchanged rather than
adopting the CLAUDE.md-suggested `advance_free_flight(state, command, ...)`
form: the source shows `v_cmd` is genuinely sufficient at that boundary
today, and changing it would have meant `PlantModel` or `LandingEnv` calling
`InnerLoopCommandModel` directly — a call-graph change beyond what this task
authorized ("컨트롤러-side command generation만 분리한다").

VALIDATION:
  * `test_inner_loop_command_model.py` in isolation → **6 passed, 28.6s**.
  * Full `landing_rl/tests` suite → **218 passed, 0 fail, 0 skipped,
    263.7s** (211 prior + 1 net new in `test_legacy_dynamics.py` [rename +
    1 new] + 6 new in `test_inner_loop_command_model.py`).
    `test_structural_freeze.py`'s full expanded matrix (140 paired
    OLD-vs-NEW episodes, 46,958 steps) re-ran and matched byte-for-byte
    against the pre-refactor run recorded in §0.5 above.
  * Manual smoke test: `LandingEnv().reset(seed=0)` + 5 `step()` calls,
    confirming `env.legacy_dynamics.inner_loop_command_model` exists and
    the env runs end-to-end.

`[INTERPRETATION]` MuJoCo future-backend input-boundary analysis (not acted
on this session — no MuJoCo code written):

| Boundary | What MuJoCo would receive | Pro | Con / what stays UNIDENTIFIED |
|---|---|---|---|
| A. attitude_setpoint + thrust_setpoint | Same as today's setpoint, fed to a MuJoCo attitude/thrust controller (e.g. a PD actuator model or MuJoCo's own PD actuators) | Reuses `InnerLoopCommandModel` unchanged; closest to "swap only the physics" | Requires MuJoCo-side attitude+thrust controller gains — NOT IDENTIFIED; still not modeling real PX4 rate-loop dynamics |
| B. body-rate_setpoint + thrust_setpoint | `rate_cmd` (currently computed inside `advance_free_flight`'s attitude/rate proxy) + thrust | Closer to PX4's actual rate-controller boundary (PX4's innermost loop takes body-rate setpoints); would let a future rate-controller model be identified from ULog independently of attitude dynamics | Requires moving the attitude→rate proxy out of `LegacyVehicleDynamics` too — bigger boundary change than this task authorized; rate-controller gains for the real vehicle NOT IDENTIFIED |
| C. body torque + collective thrust | Direct actuator-level command | Matches true rigid-body dynamics (`J·ω̇ = τ − ω×Jω`) most directly; enables real 6DoF MuJoCo simulation | Requires inertia tensor, thrust coefficient, motor/ESC dynamics, propeller aerodynamics — ALL explicitly `NOT MEASURED` per CLAUDE.md §4; would violate the "do not invent physical parameters" rule today |

Recommendation (not a decision — proposal only, pending user approval):
boundary **A** is the most defensible near-term target, because it requires
inventing nothing new (`InnerLoopCommandModel`'s existing output is already
exactly this) and keeps every unidentified physical quantity (rate-loop
gains, inertia, thrust coefficient, motor/ESC dynamics, propeller
aerodynamics) explicitly `NOT IDENTIFIED` rather than silently assumed.
Boundary B is the more PX4-realistic target for eventual sim-to-real
system identification (§19), since PX4's actual innermost onboard loop is
body-rate, but adopting it now would require a second control/physics
extraction (attitude→rate) beyond this session's scope.

`[CAUTION]` **Uncommitted**, layered on top of both same-day entries above
(also uncommitted). `MuJoCoDynamics` remains NOT IMPLEMENTED; no boundary
in the table above has been chosen or built.

GIT STATE: branch `refactor/landing-rl-architecture`, HEAD `9a154df`,
working tree dirty (this entry's files + both entries above, all
uncommitted, no unrelated changes present).

---

### 2026-09-13 — MuJoCo v0 pipeline (`landing_mujoco/`), PROVISIONAL_REFERENCE only

`[FACT]` Implemented, per explicit user directive, a complete but
provisional-parameter MuJoCo precision-landing pipeline in a new, additive
package `landing_mujoco/` (does not modify `landing_rl/` or `mujoco_rl/` —
verified: `git status --short` shows zero changes under either directory
from this work). The user stated real UGRP mass/CG/inertia/motor-geometry/
`T_max`/`tau_*`/`T_delay` measurements are expected "next week" and asked
for the software pipeline to be built now against a clearly-separated
PROVISIONAL_REFERENCE parameter set (Tarot TL680B manufacturer/derived/
synthetic values), with a MEASURED_VEHICLE config that fails fast and lists
missing fields until real numbers are supplied.

`[DECISION]` (user-directed this session, not this session's own proposal):
build the MuJoCo plant now at the "boundary C" level flagged in the prior
same-day §9 entry's input-boundary analysis (body torque + collective
thrust, identified per-axis K/tau/T_delay response, measured/provisional J)
— superseding that entry's own "boundary A is more defensible near-term"
recommendation. The user's reasoning: this week's goal is pipeline
validation, not calibrated fidelity, so provisional numbers are acceptable
as long as they are never presented as measured and the architecture
requires zero code changes to swap them for real ones.

IMPLEMENTATION (all new files, `landing_mujoco/`):
  * `configs/param_schema.py` — `UAVPhysicalParams` dataclass tree,
    `ParameterSet.{PROVISIONAL_REFERENCE,MEASURED_VEHICLE}`,
    `load_uav_params()` (auto-computes provisional inertia when
    `inertia_estimation_mode: auto`, then validates required fields),
    `MissingMeasurementError` (exact "ERROR: measured vehicle model
    incomplete / Missing: - ..." format the user specified).
  * `configs/tarot680b_reference.yaml` — runnable now; every
    non-informational field tagged `source`/`confidence`/
    `replace_before_real_training: true`. Values: `mass_kg=3.57`,
    `cg_body_m=[0,0,0]`, `wheelbase_m=0.737`, motor positions derived
    `a=wheelbase/(2*sqrt2)=0.2606m` square-X, `T_max=70.02N`
    (`thrust_to_weight_max=2.0`, synthetic), `tau_roll_s=tau_pitch_s=0.15`,
    `tau_thrust_s=0.20`, `tau_yaw_s=0.30`, `delay_s=0.02`, all `K_*=1.0`
    (all synthetic). Explicitly NOT the Tarot 680 Pro/TL68P00 (hexacopter)
    and NOT the X8 VTOL model already in `models/drone_default/`.
  * `configs/ugrp_vehicle_measured.yaml` — the future config; every
    research-critical field `null`, `inertia_estimation_mode: manual` (no
    auto-fallback).
  * `configs/simulation_config.py` — `SimulationConfig(control_dt,
    physics_dt, deterministic_physics)`, validates `physics_dt <=
    control_dt`, computes substep count without hardcoding it.
  * `coordinates/transforms.py` — single coordinate adapter. World:
    NED<->MuJoCo-NWU; body: FRD<->MuJoCo-FLU; both related by the SAME
    matrix `REFLECT=diag(1,-1,-1)` (`=R_x(pi)`, a proper rotation, self-
    inverse) — chosen specifically because it makes NED gravity `[0,0,+g]`
    equal MuJoCo-world gravity `[0,0,-g]` with no separate bookkeeping, and
    because one matrix serves both directions and both position/velocity
    and body vectors. Attitude conversion via conjugation
    `R_mj = REFLECT @ R_ned @ REFLECT` + `mujoco.mju_mat2Quat`/
    `mju_quat2Mat`.
  * `dynamics/inertia_estimation.py` — provisional-only two-body
    (uniform-density center-body box + 4 motor point masses) parallel-axis
    estimate; raises rather than silently defaulting if geometry/motor
    fields are missing; never invoked for the measured config.
  * `dynamics/mjcf_builder.py` — programmatic MJCF from
    `UAVPhysicalParams` (mass/CG/inertia via explicit `<inertial>` so
    geoms never duplicate mass; only the 4 landing-gear sphere geoms have
    collision enabled; body/arm/motor geoms are visual-only).
  * `dynamics/identified_inner_loop.py` — pure math: attitude-error->rate
    surrogate (reuses existing `LandingConfig.attitude_time_constant_s`
    cascade unchanged), `omega_dot_des=(K*omega_sp_delayed-omega)/tau` per
    axis, `tau_body=J*omega_dot_des+omega×(J*omega)`, first-order thrust
    response, and the pure `r×F` helper for a future rotor-level model.
  * `dynamics/actuation_model.py` — `ActuationModel` Protocol,
    `IdentifiedWrenchActuation` (v0, owns per-axis `DelayBuffer`s for
    `T_delay` + the persistent thrust-actuator scalar), `RotorLevelActuation`
    stub (`raises NotImplementedError` — no rotor coefficients exist).
  * `dynamics/mujoco_dynamics.py` — `MuJoCoDynamics`: per physics-substep,
    reads attitude/rates back FROM MuJoCo (never Euler-integrates them in
    Python), computes torque via the identified law + measured/provisional
    `J`, rotates body->world via MuJoCo's live `xmat`, sums
    thrust+drag+wind+process-noise into one world force, writes
    `xfrc_applied`, calls `mujoco.mj_step`. Ground-effect factor reuses the
    existing `LandingConfig` fields (gain defaults to 0.0 = disabled,
    matching the task's "ground_effect_enabled=False" initial requirement).
  * `dynamics/contact.py` — classification only (soft/hard/rough/bounce),
    reusing `landing_rl.contact.contact_model.ContactState`/`ContactResult`
    dataclasses; reads MuJoCo's own resolved contact/velocity, never
    writes pos/vel.
  * `envs/mujoco_landing_env.py` — `MujocoLandingEnv(gym.Env)`; reuses
    `BaselineController`, `DisturbanceModel`, `LoopTiming`, `ActionLatency`,
    `ObsLatency`, `TargetMeasurementModel`, `ObservationNoiseSampler`,
    `InitialStateSampler` UNCHANGED from `landing_rl`; reward/success/
    failure formulas ported with identical coefficients/thresholds;
    `deterministic_physics=True` zeroes every stochastic channel (wind,
    all noise stds, dropout/stale/outlier probs, dt jitter, delay ranges,
    init-condition ranges) via `deterministic_overrides()`.
  * `check_parameter_set.py` — CLI, `--parameter-set
    {tarot680b_reference,ugrp_vehicle_measured}`, `--run-smoke-episode`.
  * `evaluation/compare_old_vs_mujoco.py` — 5 of the 8 suggested scenarios
    (hover, north/east velocity-error, climb, descent; roll/pitch-transient
    and full landing-approach not yet added — same `Scenario`
    mechanism, not a redesign); saves N/E/D position+velocity plots and
    prints peak-tilt/peak-speed/peak-accel/final-state comparisons.
    `compare_flightlog_vs_mujoco.py` NOT built (no real UGRP `.ulg` exists).

VALIDATION: `python3 -m unittest discover -s landing_mujoco/tests -p
"test_*.py"` → **40 passed, 0 failures** (2026-09-13). Covers: coordinate
round-trips + gravity sign (`test_coordinate_transforms.py`); param-schema
missing-field fail-fast with the exact expected field set + auto-inertia
(`test_param_schema.py`); free-fall / hover-equilibrium / thrust-to-weight
margin / CG force-moment (`r×F`) / raw-torque-vs-`J` inertia check
(`test_physical_validation.py`); roll/pitch/yaw/vertical sign conventions
end-to-end through the full stack (`test_sign_conventions.py`); identified
rate/thrust step-response steady-state gain + observed delay + rise time
vs. analytic first-order prediction (`test_identified_response.py`); 16-D
obs / 3-D action contract + no-NaN-over-episode (`test_observation_action
_contract.py`); same-seed reproducibility (`test_reset_determinism.py`);
ground-contact classification + motor-cutoff latch
(`test_ground_contact.py`). Also re-verified: `landing_rl/tests` and
`mujoco_rl/` files are byte-unchanged by this work (`git status --short`
outside `landing_mujoco/` is identical to before this entry).
`landing_mujoco/evaluation/compare_old_vs_mujoco.py` ran successfully;
sample result (hover scenario, 80 steps): old final pos
`[0,0,-1.77]`/vel `[0,0,0.341]` vs MuJoCo `[0,0,-1.78]`/`[0,0,0.341]` —
close agreement under these benign, low-tilt scenarios (expected, since
gains/setpoints are shared and the provisional response `tau`s are
comparable in magnitude to the legacy alpha-response time constants).

`[INTERPRETATION]` The close old-vs-MuJoCo agreement under hover/velocity/
climb/descent scenarios is evidence the plumbing (coordinate transforms,
actuation cascade, force/torque application) is wired correctly — it is
NOT evidence the physical model is accurate, since every physical
parameter behind it is provisional/synthetic (CLAUDE.md §34's caution
about not over-interpreting simulator behavior applies with equal force to
"looks reasonable" as to "PPO performs poorly").

`[CAUTION]` **A genuine, unresolved physical-model interaction was found,
not fixed** (see the new §0.9/§0.10 bullets and `MUJOCO_MODEL.md`): the
legacy contact model implicitly assumed zero landing-gear standoff
(CG snaps to `pos[2]=ground_z_m`); MuJoCo's real landing-gear geometry
(`ground_clearance_m`) means the CG rests at `z_error≈ground_clearance_m`,
which can exceed `success_altitude_m=0.05m` even on a clean soft landing.
Confirmed empirically: a zero-action descent from 1m altitude reaches
`touchdown_quality=="soft"` with `contact_count>=1` but never satisfies
`kinematic_success` before truncation. `success_altitude_m` was NOT
retuned — this is flagged as an open `[DECISION]` for the user, not
silently patched, per the structural-freeze rule.

`[OPEN]` Whether/how to redefine the altitude success check once real
landing-gear geometry is measured (redefine relative to gear contact point
vs. accept/adjust the CG-based threshold). Whether boundary A (attitude+
thrust setpoint into a MuJoCo-side controller) should still be pursued as
an alternative to boundary C now that C has been built end-to-end and
tests pass — not revisited this session since the user's directive was to
proceed with C.

`[TODO]` Fill in `landing_mujoco/configs/ugrp_vehicle_measured.yaml` once
real measurements exist (pure YAML edit per the architecture's own
requirement — no dynamics code should need to change). Add the 3 remaining
comparison scenarios. Build `compare_flightlog_vs_mujoco.py` once a real
UGRP `.ulg` exists. Resolve the landing-gear/`success_altitude_m`
interaction above. Commit this package (currently fully uncommitted, along
with the two prior same-day entries) — at least 2, probably 3, separate
commits per CLAUDE.md §24 (entry-point migration; backend-interface seam;
`landing_mujoco/` package).

GIT STATE: branch `refactor/landing-rl-architecture`, HEAD `9a154df`,
working tree dirty — this entry adds `landing_mujoco/` (new, ~20 files),
`MUJOCO_MODEL.md` (new), and this `PROJECT_HANDOVER.md` edit; zero changes
under `landing_rl/`, `mujoco_rl/`, or any other existing path from this
entry's work.

---

### 2026-09-13 — landing-gear-standoff fix + all 8 comparison scenarios + full regression re-run

`[FACT]` Per explicit user instruction ("resolve the landing-gear standoff
semantic issue and finish the deterministic validation harness... Do NOT
solve the landing-gear issue by simply increasing success_altitude_m"),
resolved the landing-gear-standoff finding flagged in the prior same-day
entry, finished all 8 `compare_old_vs_mujoco.py` scenarios, and re-ran both
the full legacy regression suite and all `landing_mujoco` tests.

`[DECISION]` (user-directed): implement the exact `pad_target_ned` /
`vehicle_touchdown_target_ned` distinction and derivation
(`p_landing^N = p_vehicle^N + R_B^N r_landing^B`) the user specified,
scoped ENTIRELY to `landing_mujoco/` — `success_altitude_m` stays literally
`0.05` (verified: `grep success_altitude_m landing_rl/envs/landing_env.py`
unchanged), and `landing_rl/envs/landing_env.py` /
`mujoco_rl/envs/env_prototype.py` are untouched (verified: `git diff
--stat -- landing_rl/ mujoco_rl/` identical to before this entry).

IMPLEMENTATION:
  * `landing_mujoco/configs/param_schema.py::landing_reference_point_body_m`
    — new pure function: mean of the actual per-leg
    `geometry.landing_gear_points_body_m` (the SAME points
    `mjcf_builder` already uses for MuJoCo contact geoms), falling back to
    `[0,0,ground_clearance_m]` only when per-leg points aren't given —
    explicitly NOT assumed equal to `ground_clearance_m` per the user's
    instruction.
  * `landing_mujoco/envs/mujoco_landing_env.py` — new `_touchdown_offset_ned`
    (`R_desired @ r_landing_body`, `R_desired` = level attitude at the
    episode's fixed `target_yaw`) and `_vehicle_touchdown_target` helpers;
    `_get_obs` (dx,dy,dz), `_potential` (progress shaping), and `step`'s
    `control_target`/`true_error` all now go through
    `_vehicle_touchdown_target(...)` instead of the raw pad
    (`target_true`/`obs_target`) directly. `_altitude_agl` redefined as the
    landing-gear reference point's clearance above ground at the vehicle's
    CURRENT attitude (a physical-clearance quantity, deliberately decoupled
    from the touchdown-target-based `z_error`). New info fields
    `pad_target_ned`, `vehicle_touchdown_target_ned`, and `thrust_accel`
    (specific-force units, matching the legacy info field for direct
    comparability) added.
  * `landing_mujoco/dynamics/mujoco_dynamics.py` — ground-effect trigger
    switched from raw CG height to the same gear-clearance quantity
    (`_gear_altitude_agl`, using CURRENT attitude) — a physically-motivated
    MuJoCo-side refinement, computed independently from the env's copy via
    the same `landing_reference_point_body_m` function against the shared
    `UAVPhysicalParams.geometry`.
  * `landing_mujoco/tests/test_landing_reference_point.py` — new, 8 tests:
    level touchdown gives `z_error~=0` + `ground_contact=True` + `success`;
    `vehicle_touchdown_target` correctly offset from the pad; doubling/
    halving gear length changes required resting altitude while
    `success_altitude_m` stays literally `0.05` in both cases; a tilted
    vehicle at the exact target z shows `z_error~=0` but `altitude_agl>0.03`
    (demonstrating the two are genuinely different concepts); the OLD
    `landing_rl.envs.landing_env.LandingEnv` still reaches `pos[2] ==
    ground_z_m` exactly on contact (zero-standoff behavior unchanged);
    observation shape `(16,)` / action shape `(3,)` unchanged.
  * `landing_mujoco/evaluation/compare_old_vs_mujoco.py` — rewritten:
    all 8 scenarios now present (added `roll_transient`, `pitch_transient`
    via nonzero initial attitude with zero position/velocity error;
    `landing_approach` via an offset+altitude initial condition run to
    400 steps); trace now also records `body_rates`, `thrust_accel`,
    `v_cmd`; two plots per scenario (`*_kinematics.png`:
    position/velocity/acceleration N/E/D; `*_attitude_thrust_command.png`:
    roll/pitch/yaw, body rates, thrust_accel, v_cmd); added a generic
    `analyze_transient` (rise time 10-90%, settling time within 5% band,
    overshoot) applied to each scenario's dominant position-error axis,
    plus touchdown speed for `landing_approach`.

VALIDATION:
  * `python3 -m unittest discover -s landing_mujoco/tests -p "test_*.py"`
    → **48 passed, 0 failures** (40 prior + 8 new
    `test_landing_reference_point.py`, 2026-09-13).
  * `python3 -m unittest discover -s landing_rl/tests -p "test_*.py"` (full
    legacy regression suite, re-run from scratch this entry) → **218
    passed, 0 failures, 226.8s** — byte-identical pass count to §0.5/the
    first 2026-09-13 entry, confirming zero regression from any of this
    session's `landing_mujoco/`-only work.
  * `compare_old_vs_mujoco.py` ran end-to-end for all 8 scenarios; concrete
    illustration of the fix in the `landing_approach` scenario: old final
    `pos=[-0.003, 0.002, 0.0]` (CG exactly at `ground_z_m`, zero standoff)
    vs. MuJoCo final `pos=[-0.004, 0.002, -0.14]` (CG resting at the gear
    standoff) — both now report `z_error`/`touchdown_speed` correctly
    relative to their OWN model's physically-correct resting pose, which is
    exactly what the fix was for; the two models are NOT tuned to match
    (task instruction), and this difference is left visible, not erased.
  * Manual check (zero-action descent from 1m, reference config):
    `z_error=0.0195`, `xy_error~=0`, `touchdown_quality="soft"`,
    `success=True` — the concrete failure mode from the prior entry
    (soft landing, `kinematic_success` never true) no longer occurs.

`[INTERPRETATION]` The fix is a pure task-semantics correction (what
`z_error`/dx,dy,dz are measured against) plus one physically-motivated
MuJoCo-side ground-effect refinement (gear-clearance instead of CG-height
trigger) — no reward coefficient, success/failure threshold, PID gain, or
observation dimension changed. `success_altitude_m` remains exactly `0.05`
in every config.

`[OPEN]` Whether `r_landing_body`'s "mean of per-leg points" definition
remains the right nominal reference once real (possibly asymmetric)
UGRP gear geometry is measured, or whether a different reference point
(e.g. the single lowest leg) is more appropriate — not decided, left as a
config-time question since the function already derives from whatever real
per-leg points are supplied.

`[TODO]` Same as the prior entry: fill in
`landing_mujoco/configs/ugrp_vehicle_measured.yaml` once real measurements
exist. Commit this branch's three logically separate concerns (see §0.10
and the commit proposal reported to the user this session) — proposed, not
performed, pending explicit permission.

GIT STATE: branch `refactor/landing-rl-architecture`, HEAD `9a154df`,
working tree dirty — this entry modifies only files already inside
`landing_mujoco/` (plus this `PROJECT_HANDOVER.md` and `MUJOCO_MODEL.md`
edit); zero changes under `landing_rl/`, `mujoco_rl/`, or any other
existing path (re-verified: `git status --short` outside `landing_mujoco/`
and the two `.md` files is identical to before this entry).

`[CORRECTION 2026-09-14]` The GIT STATE lines of the 2026-09-13 entries
above were written before the three commits were created. The A/B/C work
was subsequently committed as `53e1ce6` (entry-point migration), `e3b408f`
(VehicleDynamicsBackend seam) and `7b231ef` (provisional MuJoCo pipeline,
including this handover update); HEAD at the start of 2026-09-14 was
`7b231ef` with a clean tree apart from the new untracked vendor copy.

---

### 2026-09-14 — x500 visual shell (visualization only), UNCOMMITTED

`[FACT]` Per explicit user instruction, integrated the PX4/Gazebo x500
airframe appearance into the MuJoCo plant as a purely cosmetic visual shell.
User-approved decisions this session (recorded as `[DECISION]`): use a
uniformly rescaled visual assembly (`uniform_scale: auto` =
target/source wheelbase, currently 0.737 / 0.49214631970583705 =
1.4975221199266826), configurable in YAML; conversion tooling
(`trimesh`, `pycollada`, `Pillow`) only in a disposable venv, never a
runtime dependency; preserve 9 frame components separately with source
materials and the CF.png texture; preserve source-authored normals; 8-decimal
OBJ text precision; drop the 48 FMUK66 COLLADA line primitives; accept the
sub-texel trimesh UV discrepancy on untextured components; do not convert
OakD-Lite; keep native reconstruction and UGRP display scaling as separate
layers.

IMPLEMENTATION (all under `landing_mujoco/`, plus docs):
  * `vendor/px4_gz_x500/` — user-supplied immutable upstream copy (x500,
    x500_base, x500_depth, OakD-Lite); read only, not modified.
  * `tools/convert_x500_visual_assets.py` — reproducible converter
    (conversion env: Python 3.10.12, trimesh 5.1.0, pycollada 0.9.3,
    Pillow 12.3.0). Writes OBJ directly from pycollada index streams
    (per-corner v/vt/vn), bakes COLLADA node transforms (normals via
    inverse-transpose + renormalization when non-identity, e.g. the 5010
    parts' Y-up→Z-up rotation × 0.01 scale), enforces hard gates (position,
    bbox, UV, normal ≤ 1e-7; triangle counts equal; textured UV and texture
    pixels vs trimesh), and records everything in `conversion_report.json`.
  * `assets/x500_visual/` — 12 OBJ components (9 frame, 1 motor base, 2 motor
    bell), 2 byte-copied prop STLs, byte-copied `textures/cf.png`,
    `x500_native_assembly.json` (SDF visual poses/scales transcribed
    programmatically, no physics keys), `LICENSE` (verbatim BSD-3-Clause),
    `THIRD_PARTY_NOTICES.md`, `README.md`, `conversion_report.json`
    (gates_passed: true). Re-running the converter reproduces identical
    output hashes.
  * `configs/visualization_config.py`, `configs/x500_visualization.yaml` —
    visualization-only config, separate from `UAVPhysicalParams`.
  * `visualization/x500_shell.py` — native assembly + UGRP alignment,
    emitted as nested compile-time `<frame>` elements (no bodies added);
    shell geoms `contype=0 conaffinity=0 group=2`.
  * `coordinates/transforms.py` — additive helpers only
    (`SDF_MODEL_FLU_TO_MUJOCO_BODY`, `sdf_pose_matrix`,
    `frd_body_offset_to_mujoco_body`, `rotation_matrix_to_mujoco_quaternion`).
  * `dynamics/mjcf_builder.py`, `dynamics/mujoco_dynamics.py`,
    `envs/mujoco_landing_env.py` — optional `visual_shell` / `visualization`
    argument, default `None`.
  * `tools/view_x500_visual.py` — interactive viewer / offscreen renders.

VALIDATION:
  * Visual-OFF MJCF sha256 `e2c5cb7a…` is byte-identical to the MJCF recorded
    immediately before integration (pinned in `test_x500_visual_shell.py`).
  * MuJoCo reconstruction of every component vs its OBJ, per face corner:
    positions ≤ 1e-7 m, UV ≤ 1e-6 (MuJoCo's internal OBJ V-flip verified
    against a ground-truth quadrant texture), authored normals ≤ 1e-6;
    NXP-HGD-CF union bbox matches the approved acceptance reference.
  * Native (scale 1) placement matches the SDF: motor bases at
    (±0.174, ±0.174, 0.032) yaw −0.45, frame z +0.025 yaw π, prop hubs on the
    rotor axes, bells at rotor link −0.032 m. Display layer satisfies
    p_display = t + R (s·p_native) for every visual vertex. Scaled wheelbase
    = 0.737000000000 m.
  * Physics invariance (`test_visual_physics_invariance.py`): exact equality
    of body mass/inertia/ipos/iquat, DoF/joint/actuator structure, options,
    and all physics geoms; ON vs OFF trajectories for free fall, hover, known
    body torque, landing/contact, deterministic and seeded stochastic env
    episodes pass `rtol=0, atol=1e-12` and are in fact bit-identical.
  * `python3 -m unittest discover -s landing_mujoco/tests -p "test_*.py"` →
    82 passed, 0 failed (48 prior + 34 new).
    `python3 -m unittest discover -s landing_rl/tests -p "test_*.py"` →
    218 passed, 0 failed (171.6 s), unchanged baseline.
  * Offscreen renders (OSMesa) of native and scaled shells match the vendor
    Gazebo thumbnail; interactive GLFW viewer could not be exercised in this
    headless session (`gladLoadGL error`).

`[CAUTION]` Prop RGBA `0.175 0.175 0.175 1` is an ASSUMPTION for the SDF's
`Gazebo/DarkGrey` script material (script file not in the vendor tree, not
verified locally). Three decorative SDF plane decals are not integrated.

`[OPEN]` Visual-only alignment conflict: with the default zero translation
the visual rotor hubs match the physical motor positions horizontally
(≤ 4.4e-5 m), but the scaled x500 landing gear bottom is at body z = −0.341 m
versus the physical contact bottom at −0.140 m, so at physical rest the
visual skids render ≈ 0.20 m below the ground plane. Options: set
`translation_body_m` FRD z ≈ −0.2008 (skids on ground, visual rotor plane
≈ 0.29 m above the physical motor plane), keep the default, or wait for
measured gear geometry. Not decided.

`[OPEN]` Observation only (no physics effect today): the x500 visual prop
handedness follows PX4 quad-X (front-right CCW), which is opposite to the
provisional `tarot680b_reference.yaml` spin assumption (M1 front-right CW).
Spin direction is not consumed by the v0 dynamics; confirm against the real
vehicle next week.

GIT STATE: branch `refactor/landing-rl-architecture`, HEAD `7b231ef`,
uncommitted: new `landing_mujoco/{vendor,assets,tools,visualization}/`,
new visualization config/tests, additive edits to `transforms.py`,
`mjcf_builder.py`, `mujoco_dynamics.py`, `mujoco_landing_env.py`,
`MUJOCO_MODEL.md`, and this file. No changes under `landing_rl/` or
`mujoco_rl/`.

---

### 2026-09-14 (continued) — pinned upstream provenance, split-shell landing gear, vendor copy removed, UNCOMMITTED

`[FACT]` Upstream provenance, verified this session with read-only git
commands against the local PX4 checkout (not just transcribed):
`/home/qntmdghkss/drone_stack/px4/PX4-Autopilot` HEAD
`85df8c2281c2466b30a121b22b0bf33dc69bcfe4` records gitlink
`Tools/simulation/gz` → `d754381a1cecdd7f17050acd72bf5bf1327bced6`, origin
`https://github.com/PX4/PX4-gazebo-models.git`; `git status` / `git diff`
for `models/{x500_base,x500,x500_depth,OakD-Lite}` are empty (the submodule
is dirty only in unrelated `models/mono_cam`, `models/standard_vtol`,
`worlds/psi_vtol_world.sdf`). All 42 files of the former
`landing_mujoco/vendor/px4_gz_x500/` copy were byte-identical to the git
blobs at `d754381a…`, and the file sets matched exactly.

`[DECISION]` (user-selected via explicit choice, 2026-09-14): **split
shell** — the x500 visual shell contains only the upper structure (frame
plates/arms, metal, FMU, rails, antenna holder, motors, props); the x500
landing gear is excluded and the landing gear shown in the viewer is the
PHYSICAL contact geometry. Supersedes the earlier same-day `[OPEN]`
alignment item (skids rendering ≈0.20 m below ground). Translation/rotation
stay at the approved defaults `[0,0,0]`; scale stays auto (1.4975221199266826).

`[DECISION]` (user): do not commit the duplicated vendor copy, provided the
derived bundle is complete, hash-recorded, reproducible from an external
PX4-gazebo-models tree at the pinned commit, and runtime does not depend on
`vendor/`. All four conditions were verified (below) and the untracked copy
was then removed.

IMPLEMENTATION:
  * `tools/convert_x500_visual_assets.py` v1.1.0: required `--source`
    (PX4-gazebo-models root or `models/`); every input verified against a
    pinned sha256 of the git blob at `d754381a…` (abort on mismatch; a one-byte
    tamper test aborted with exit 2 and wrote nothing); git HEAD / origin /
    path cleanliness recorded when the source is itself the checkout (a bug
    that recorded an *enclosing* repository's git state for nested copies was
    found and fixed this session); report/manifest paths are upstream-relative
    (`models/x500_base/...`), no local or vendor path in the runtime
    manifest; tool-owned `meshes/` and `textures/` are cleared on regeneration.
  * Landing-gear partition: whole shared-vertex-connected pieces classified as
    gear if the COLLADA component is `Landing*` or the piece reaches below
    −0.10 m (native DAE frame). Result: `frame_landing_{foam,rubber,plastic}`
    excluded entirely; carbon fiber 28,332 → 26,956 triangles (4 pieces: 2 leg
    tubes, 2 skid tubes, 1,376 triangles). Margins: lowest upper piece −0.0716 m,
    highest z-rule gear piece −0.2057 m. No triangle modified. Recorded per
    piece in `conversion_report.json` → `landing_gear_partition`.
  * `dynamics/mjcf_builder.py`: with a shell, the collision-inert placeholder
    markers (`body_box`, `arm_*`, `motor_*`) get `group="3"` (hidden by
    default); ground and leg contact geoms untouched. Without a shell the MJCF
    is still byte-identical to the pre-integration fingerprint `e2c5cb7a…`.
  * `tools/view_x500_visual.py`: default render groups {0, 2};
    `--physics-overlay` adds group 3.
  * Bundle README / THIRD_PARTY_NOTICES / YAML comments / `MUJOCO_MODEL.md`
    updated for pinned provenance and the split shell.

VALIDATION:
  * Canonical bundle regenerated from
    `PX4-Autopilot/Tools/simulation/gz` (source verified, git HEAD pinned,
    clean): 9 OBJ + 2 STL + `cf.png`, gates passed. Regeneration from a plain
    (non-git) copy and again after the vendor removal produced byte-identical
    runtime files (meshes, textures, LICENSE, manifest).
  * Runtime independence: a scratch copy of `landing_mujoco` + `landing_rl`
    with no `vendor/` ran all 37 visualization tests (OK) and the viewer
    (landed `success=True`); its render was pixel-identical to the in-repo
    render; no runtime code/config references `vendor`.
  * After removing `landing_mujoco/vendor/` (untracked, never committed; audit
    list of the 42 removed files and their sha256 kept in the session
    scratchpad): `landing_mujoco` suite 85 passed, 0 failed (48 prior + 37
    visualization); `landing_rl` suite 218 passed, 0 failed (173.5 s).
  * Physics invariance unchanged: structural exact equality (placeholder
    markers differ only in `geom_group`) and bit-identical ON/OFF trajectories.

RESULT (size): new/changed content 22.5 MB raw (bundle 22.2 MB), ≈5.6 MB as
git zlib objects; 34 MB vendor copy not committed; landing-gear exclusion cut
mesh bundle from ≈32 MB to 20.5 MB raw.

`[CAUTION]` Prop RGBA remains the documented `Gazebo/DarkGrey` assumption;
decals and OakD-Lite remain excluded.

GIT STATE: branch `refactor/landing-rl-architecture`, HEAD `7b231ef`,
uncommitted: `landing_mujoco/{assets,tools,visualization}/`, visualization
config + 4 test files, additive edits to `transforms.py`, `mjcf_builder.py`,
`mujoco_dynamics.py`, `mujoco_landing_env.py`, `MUJOCO_MODEL.md`, this file.
`landing_mujoco/vendor/` no longer exists. No changes under `landing_rl/` or
`mujoco_rl/`.

---

### 2026-09-14 — SESSION CLOSEOUT / HANDOFF (x500 visualization accepted)

`[FACT]` Final read-only audit before commit: every modified/untracked file
belongs to the x500 MuJoCo visualization feature (6 modified tracked files,
27 new files); no unrelated change; `landing_mujoco/vendor/` absent.

REPOSITORY STATE
  * worktree: `/home/qntmdghkss/drone_stack_rl_refactor`
  * branch: `refactor/landing-rl-architecture`
  * previous MuJoCo baseline: `7b231ef` ("Add provisional MuJoCo landing
    simulation pipeline")
  * new visualization commit: the commit that introduces this entry, subject
    "Add x500 visual shell to MuJoCo landing environment", parent `7b231ef`
    (a file cannot contain its own commit hash; resolve with
    `git log -1 --format=%H -- PROJECT_HANDOVER.md`). Not pushed.

VALIDATED STATE (final pre-commit run, 2026-09-14)
  * `landing_rl`: 218 passed, 0 failed
  * `landing_mujoco`: 85 passed, 0 failed
  * visualization (`test_visualization_config`, `test_x500_visual_assets`,
    `test_x500_visual_shell`, `test_visual_physics_invariance`): 37 passed,
    0 failed

VISUALIZATION DESIGN (accepted)
  * PX4 x500 assets are a visual shell only; UGRP physical config alone
    defines physics.
  * visual scale ≈ 1.497522 (exactly 1.4975221199266826 = 0.737 /
    0.49214631970583705); visual translation `[0, 0, 0]`; rotation `[0, 0, 0]`.
  * split-shell landing gear: x500 upper airframe / motors / props + the
    physical UGRP landing-gear contact geoms. The scaled x500 skids are not
    used and the shell is not shifted vertically.
  * visual ON/OFF physics is bit-identical (structural equality + trajectory
    tests).

ASSET PROVENANCE
  * repository: `https://github.com/PX4/PX4-gazebo-models.git`
  * commit: `d754381a1cecdd7f17050acd72bf5bf1327bced6` (pinned by
    PX4-Autopilot `85df8c2281c2466b30a121b22b0bf33dc69bcfe4` as
    `Tools/simulation/gz`)
  * The duplicated vendor tree is intentionally NOT committed. Committed
    instead: the derived runtime assets (`landing_mujoco/assets/x500_visual/`,
    every source/output sha256 in `conversion_report.json`) and the
    reproducible conversion tool
    (`landing_mujoco/tools/convert_x500_visual_assets.py --source <checkout>`,
    conversion-only dependencies trimesh/pycollada/Pillow, inputs verified
    against pinned sha256).

KNOWN REMAINING ISSUES / INTENTIONALLY UNRESOLVED
  1. `[OPEN]` Real UGRP vehicle parameters are still unmeasured. Required:
     `total_mass_kg`, `cg_body_m`, `Ixx`, `Iyy`, `Izz`,
     `motor_positions_body_m`, `max_collective_thrust_n`, `tau_roll_s`,
     `tau_pitch_s`, `tau_thrust_s`, `actuator_delay_s`
     (`ugrp_vehicle_measured.yaml` still fails fast listing exactly these).
  2. `[OPEN]` Landing-gear geometry is still provisional (synthetic
     `ground_clearance_m` / `landing_gear_points_body_m`) and must be replaced
     with measured geometry.
  3. `[CAUTION]` The interactive MuJoCo viewer was not validated in the
     headless Claude environment (GLFW `gladLoadGL error`); offscreen
     (OSMesa) rendering is validated.
  4. `[OPEN]` x500 propeller spin handedness (PX4 quad-X, front-right CCW)
     differs from the current provisional UGRP spin guess (M1 front-right CW).
     Do NOT change physics from the x500 model; verify the real vehicle's motor
     index, location and CW/CCW direction later.
  5. `[DECISION — standing]` Do not train PPO yet.

NEXT-SESSION STARTING POINT
Real vehicle parameter measurement / injection — not additional
visualization work. Recommended order: mass → CG → inertia → motor positions
→ landing-gear geometry → max collective thrust → roll/pitch/thrust response
identification → actuator delay → real-flight vs MuJoCo validation.

---

### 2026-09-14 (continued) — interactive MuJoCo viewer SIGSEGV on WSLg: native GPU-driver fault, no repository change

Scope: inspection/diagnosis only, requested by the user so they could
visually inspect the x500 model themselves. No source, config, asset, model,
dependency or package was changed; nothing was committed. Evidence files
(matrix log, gdb backtraces, offscreen PNGs) lived only in the session
scratchpad and are NOT preserved; the key facts are transcribed here.

CONTEXT (user report)
  * An earlier GLX-context failure was cleared by the user removing
    `LIBGL_ALWAYS_INDIRECT=1` from their shell. After that,
    `python3 landing_mujoco/tools/view_x500_visual.py --pose ground
    --physics-overlay` printed the shell summary and
    `settled: ground_contact=True touchdown_quality=soft success=True
    z_error=0.0200 altitude_agl=0.0200`, then died with
    `Segmentation fault (core dumped)` and no Python traceback.

`[FACT]` Crash location (code reading + faulthandler + gdb):
  * Without `--offscreen`/`--rollout`, `view_x500_visual.py:147` calls
    `mujoco.viewer.launch(model, data)` → `viewer.py:525`
    `simulate.render_loop()` (native `_simulate` module, main thread; the
    physics side thread was at `viewer.py:315` `simulate.load`).
  * gdb, main thread at crash time: `_simulate.so` → GLFW
    `glfwSwapBuffers`/`swapBuffersGLX` → `libGLX_mesa.so.0` → Mesa DRI
    driver, i.e. the first frame swap.
  * gdb, faulting thread (a driver worker thread, not a Python thread):
    `libd3d12core.so` → `libigd12um64xel.so` → `libigc.so` →
    `libLLVM-9.so` (`llvm::FPPassManager::runOnFunction`) → `libigc.so` →
    `libstdc++.so.6` `std::string::_M_assign` (SIGSEGV). Intel driver files
    loaded from
    `/usr/lib/wsl/drivers/iigd_dch.inf_amd64_77ca5c827a04e39b/`.
  * Not implicated: geom-group setup, camera setup, data sync, or any
    repository callback.

`[FACT]` A–F isolation matrix (all with `LIBGL_ALWAYS_INDIRECT` unset,
`/usr/bin/python3 -X faulthandler`, 40 s SIGKILL timeout). Every case printed
a faulthandler `Fatal Python error: Segmentation fault` dump:

| case | flags | SIGSEGV dump | exit code |
|---|---|---|---|
| A | (none) | yes, +7.4 s | 139 |
| B | `--pose ground` | yes, +14.3 s | 139 |
| C | `--physics-overlay` | yes, +5.5 s | 137 |
| D | `--pose ground --physics-overlay` | yes, +4.3 s | 139 |
| E | `--no-visual` | yes, +4.5 s | 139 |
| F | `--no-visual --pose ground` | yes, +5.5 s | 139 |

  Observation: exit code is not a reliable crash marker here — WSL crash
  capture (`core_pattern = |/wsl-capture-crash …`) kept the process alive for
  >20 s after the fault, so C was SIGKILLed (137) before it could exit 139.
  The faulthandler dump is the reliable signal.

`[FACT]` Offscreen rendering (repo script, case-D flags, `--offscreen` to a
scratch dir):
  * default backend (GLFW) and `MUJOCO_GL=egl`: SIGSEGV at
    `mujoco/renderer.py:89` (`MjrContext` creation), rc 139, no PNG.
  * `MUJOCO_GL=osmesa`: rc 0, 9 s, wrote `D_{iso,top,front,side}.png`; the
    iso render shows the x500 shell, placeholder overlay markers and
    physical contact geoms.

`[FACT]` Repository code excluded:
  * A trivial inline plane + free box MJCF (no `landing_mujoco` / `landing_rl`
    import) also segfaults in `mujoco.viewer.launch()`, and in offscreen
    `mujoco.Renderer` with default/EGL backends (`renderer.py:243`); gdb on
    the EGL case shows the same Intel `libigd12um64xel.so` / `libigc.so` /
    `libLLVM-9.so` → `libstdc++` chain. OSMesa renders it (rc 0).
  * `glxgears -info` on the default adapter reports
    `GL_RENDERER = D3D12 (Intel(R) Iris(R) Xe Graphics)` and then dumps core
    (rc 139). `glxinfo -B` succeeds only because it never draws.
  * `dmesg`: `python3: potentially unexpected fatal signal 11`, followed by
    `WSL (… CaptureCrash): Capturing crash for pid …`. `ulimit -c` = 0;
    `coredumpctl` not installed; gdb used live instead.

`[FACT]` Graphics environment at diagnosis time:
  * `DISPLAY=:0`, `WAYLAND_DISPLAY=wayland-0`, `XDG_RUNTIME_DIR=/run/user/1000/`,
    `MUJOCO_GL` unset, `LIBGL_ALWAYS_SOFTWARE` unset.
  * `LIBGL_ALWAYS_INDIRECT=1` is still exported by `~/.bashrc:120`, so every
    new shell re-inherits it; with it set, `glxinfo -B` fails with
    `BadValue … X_GLXCreateNewContext`. With it unset: direct rendering Yes,
    accelerated, `D3D12 (Intel(R) Iris(R) Xe Graphics)`, OpenGL 4.1 core,
    Mesa `23.2.1-1ubuntu3.1~22.04.4`.
  * Versions: Python 3.10.12 (`/usr/bin/python3`, no venv); MuJoCo 3.4.0,
    pyGLFW 2.10.0 (bundled `glfw/x11/libglfw.so`, native
    `3.4.0 X11 GLX Null EGL OSMesa`), PyOpenGL 3.1.10 — all in
    `~/.local/lib/python3.10/site-packages`; Mesa packages
    (`libgl1-mesa-dri`, `libglx-mesa0`, `libegl-mesa0`, `libosmesa6`)
    23.2.1-1ubuntu3.1~22.04.4; WSL 2.5.9.0, kernel 6.6.87.2-1, WSLg 1.0.66,
    Direct3D 1.611.1, Windows 10.0.26200.9168.
  * Host GPUs (Windows `Win32_VideoController`): Intel Iris Xe, driver
    `31.0.101.4032`, dated 2022-12-21 (Mesa D3D12 default adapter);
    NVIDIA GeForce RTX 3050 4GB Laptop GPU, driver `32.0.15.9579`,
    dated 2026-03-04.

`[INTERPRETATION]` Classification: **native graphics-stack bug (GLFW/WSLg/
OpenGL layer), specifically the Intel D3D12 user-mode driver's shader
compiler crashing on the first real draw**. Not a repository/model bug, not
an x500 asset bug, not a physics-overlay bug, not a MuJoCo-viewer bug. The
2022-12-21 Intel driver age is the most likely contributing factor; the exact
internal mechanism (driver bug vs. C++-runtime interaction) is NOT VERIFIED.
Confidence: high that the repository is not the cause and that the fault is
Intel-adapter-specific; low on the internal mechanism.

WORKAROUND PROBES (per-command environment variables only; nothing installed)
  * `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`: `glxinfo -B` renderer
    `D3D12 (NVIDIA GeForce RTX 3050 4GB Laptop GPU)`; `glxgears` 146 frames in
    5.0 s (≈29 FPS; not investigated); trivial `launch()` and repo case D ran
    until SIGKILL at 20 s / 30 s with **no** SIGSEGV dump; trivial
    `launch_passive()` loop (mj_step + `viewer.sync()` for 15 s) exited
    normally, rc 0.
  * `LIBGL_ALWAYS_SOFTWARE=1`: renderer `llvmpipe (LLVM 15.0.7, 256 bits)`;
    `glxgears` ≈437 FPS; trivial `launch()` and repo case D: no SIGSEGV dump
    until SIGKILL; trivial passive loop 15 s, rc 0.

`[CAUTION]` What the probes do NOT prove:
  * No human looked at any window; there is no frame-level evidence that the
    x500 model renders correctly in the interactive viewer under either
    workaround.
  * The repo viewer (`view_x500_visual.py`, blocking `launch()`) was only
    shown not to crash within 30 s; it was never run to a clean exit.
  * Nothing here touches or validates mass, CG, inertia, motor positions,
    landing-gear geometry, contact, visual scale, controllers, reward, or the
    observation/action contract — all unchanged.

`[CORRECTION]` (2026-09-14)
  * Old statements: §9 "x500 visual shell" entry — "interactive GLFW viewer
    could not be exercised in this headless session (`gladLoadGL error`)";
    SESSION CLOSEOUT item 3 — "The interactive MuJoCo viewer was not
    validated in the headless Claude environment (GLFW `gladLoadGL error`)".
  * Corrected statement: in this diagnosis the Claude shell for this worktree
    was NOT headless (WSLg `DISPLAY=:0`, `WAYLAND_DISPLAY=wayland-0`). The
    interactive viewer has two independent local-environment failure modes,
    neither of which is "no display": (1) with `LIBGL_ALWAYS_INDIRECT=1`
    (inherited from `~/.bashrc:120`) GLX context creation fails; (2) with it
    unset, SIGSEGV in the Intel Iris Xe D3D12 driver. Whether the earlier
    `gladLoadGL error` was failure mode (1) is NOT VERIFIED (that session's
    environment variables were not recorded). The status "interactive viewer
    not validated" remains true.
  * Evidence: this entry's environment check, glxinfo results, A–F matrix and
    gdb backtraces.

`[FACT]` Untracked `MUJOCO_LOG.TXT` at the repository root (mtime
2026-09-14 17:10:46 +0900, content `ERROR: could not create window`). None of
this entry's diagnostic runs reported that error (all ran with
`LIBGL_ALWAYS_INDIRECT` unset and failed with SIGSEGV instead);
`[INTERPRETATION]` it is left over from an earlier viewer attempt under the
GLX-indirect setting. Left untouched.

`[TODO]` (user-side, no repository change needed):
  1. Remove or comment out `export LIBGL_ALWAYS_INDIRECT=1` at
     `~/.bashrc:120` (otherwise every new terminal reintroduces the GLX error).
  2. Open the viewer on the NVIDIA adapter and perform the pending visual
     inspection, e.g.
     `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA python3
     landing_mujoco/tools/view_x500_visual.py --pose ground --physics-overlay`
     (fallback: `LIBGL_ALWAYS_SOFTWARE=1`). Record the visual feedback before
     any model/visual change is considered.
  3. `[PROPOSAL]` (not approved) Update the Windows Intel Iris Xe driver as a
     longer-term fix; untested.

`[OPEN]` §0.10 is unchanged: real UGRP parameter measurement remains the
immediate next step; this diagnosis does not change the project state or
the validated baseline (`landing_rl` 218 / `landing_mujoco` 85 /
visualization 37 passed, recorded at `6306699`; not re-run in this
diagnosis because no code changed).

GIT STATE: branch `refactor/landing-rl-architecture`, HEAD
`6306699871122cd985a3e35ebd1b93ce575aeaa4`; working tree: `PROJECT_HANDOVER.md`
modified (this entry, §0.9 item, header), `MUJOCO_LOG.TXT` untracked
(pre-existing). Nothing staged or committed. Not pushed.

---

### 2026-09-16 — first real UGRP vehicle measurements recorded (storage / provenance only, no physics)

Scope: **parameter storage, provenance and validation only.** Explicitly out
of scope and not touched: MuJoCo physics implementation, MJCF authoring,
actuator/motor/thrust-curve modelling, `LegacyVehicleDynamics` and
`InnerLoopCommandModel` equations, contact model, reward, PPO, VecNormalize,
checkpoints. No inertia placeholder was injected into any physics path.

`[FACT]` New measurements supplied by the user and recorded:
mass 6.408 kg; motor radial distance (center→motor center) 0.360 m; motor
spin layout FL/RR = CW, FR/RL = CCW on an X-frame; CG, motor-plane and
battery-center z in a raw measurement datum (0.241 / 0.334 / 0.220 m);
component model designations for motor (T-MOTOR MN501-S IP45 KV360), ESC
(HOBBYWING Skywalker V2 60A), battery (Poly-Tronics 8th Gen 10000 mAh
22.2 V 6S1P 75C+ XT90-S) and propeller (T-MOTOR MS1704). Full table with
per-value provenance: §0.11.1.

`[FACT]` Parameter-ownership audit. `landing_mujoco/configs/*.yaml` +
`param_schema.py` is the only structured owner of mass/CG/inertia/geometry/
motor/thrust/component data. **`landing_rl` legacy dynamics holds no mass,
inertia, CG or geometry whatsoever** — it is mass-normalized (thrust carried
as `thrust_accel`, m/s²). `mujoco_rl/envs/env_prototype.py` duplicates the
same mass-normalized `LandingConfig` fields (legacy, read-only). No
URDF/SDF/MJCF describes this vehicle (`models/`, `px4/` hold other
projects' airframes). `landing_rl/configs/vehicle/` is an empty untracked
placeholder. Details: §0.11.4.

`[FACT]` Discrepancy found, **not** reconciled: gravity is 9.8065 in
`landing_rl` (`LandingConfig.gravity_mps2`) and hardcoded 9.80665 in
`landing_mujoco` (`actuation_model.py`, MJCF, tests). Reconciling either is
a behavior change and was left alone. Recorded as `[OPEN]` in §0.11.4.

`[FACT]` Discrepancy found: the measured spin layout (M1 CCW, M2 CW,
M3 CCW, M4 CW under MUJOCO_MODEL.md §D's numbering) is the **inverse
polarity** of `tarot680b_reference.yaml`'s assumed M1 CW, M2 CCW, M3 CW,
M4 CCW — both preserve "diagonals match". No behavior impact: spin
direction is consumed by no dynamics code (MUJOCO_MODEL.md §E). The
reference config was **not** edited (different airframe, its value is an
explicitly-tagged assumption).

IMPLEMENTATION
  * `landing_mujoco/configs/ugrp_vehicle_measured.yaml` — populated the
    datum-independent measurements and component specs; added two new
    blocks: `raw_measurements:` (as-measured numbers in the unresolved
    datum, consumed by nothing) and `provisional_geometry:`
    (derived-under-assumption motor XY). Added a provenance vocabulary
    header (MEASURED / USER_PROVIDED_SPEC / DERIVED / NOT_MEASURED …).
  * `landing_mujoco/configs/param_schema.py` — added `RawMeasurements` and
    `ProvisionalGeometry` dataclasses + parsers, the
    `motor_xy_from_radial_distance()` geometric helper, and a `_vec2`
    parser. The two new `UAVPhysicalParams` fields are defaulted, so
    `tarot680b_reference.yaml` (which omits both blocks) parses unchanged.
    **`_REQUIRED_FIELDS` was deliberately not touched** — fail-fast
    semantics are unchanged.
  * `landing_mujoco/tests/test_ugrp_measured_params.py` — new, 29 tests
    across 5 cases: measured values, derived values, unknowns-stay-unknown,
    datum separation, and reference-config-unaffected. Provenance *tags*
    are asserted too, so promoting an observation to MEASURED breaks a test.
  * `landing_mujoco/tests/test_param_schema.py` — updated the two tests that
    encoded "mass is still null".

PHYSICAL / CONTROL MEANING
Nothing is applied. The measured mass is recorded next to a mass-normalized
simulator that does not consume mass; the MuJoCo path still refuses to load
this config. Connecting real parameters to runtime physics is a separate,
future, behavior-changing step.

`[OPEN]` **Datum ambiguity is the blocking issue.** The three measured
z-coordinates share one datum whose origin and +Z direction were not
recorded, so `motor_z_relative_CG` is undetermined in sign — only
`|motor_plane − CG| = 0.093 m` and `|CG − battery| = 0.021 m` are known.
Six body-frame fields are held `null` and a test enforces it. See §0.11.2.

`[OPEN]` The ±45° arm angle was **not** confirmed by any repository source,
so the derived motor XY stays in `provisional_geometry:` and is not
promoted into `motors.positions_body_m`. See §0.11.3.

VALIDATION
  * `python3 -m pytest landing_rl/tests landing_mujoco/tests -q`
    → baseline measured at the START of this session, before any edit:
    **303 passed** (218 landing_rl + 85 landing_mujoco) in 222.22 s,
    matching the figure recorded at `6306699`.
  * `python3 -m pytest landing_rl/tests landing_mujoco/tests system_id/tests
    -q` → after this session: **399 passed** in 229.00 s
    (218 landing_rl unchanged + 114 landing_mujoco [85 unchanged + 29 new]
    + 67 system_id). **No test changed status from pass to fail.** The 29
    new tests are all in `test_ugrp_measured_params.py`; the 2 edited tests
    in `test_param_schema.py` still pass.
  * `system_id/tests` was added to the final run to confirm independence;
    it was not in the baseline command, so its 67 is a new measurement, not
    a delta. Grep confirms `system_id/` and `mujoco_rl/` contain **zero**
    references to `param_schema`/`landing_mujoco`, so this session's edits
    could not reach them.
  * `python3 -m landing_mujoco.check_parameter_set --parameter-set
    ugrp_vehicle_measured` → still exits 1, now listing 10 missing fields
    (`total_mass_kg` correctly dropped off; `Ixx/Iyy/Izz`,
    `motor_positions_body_m`, `cg_body_m`, `max_collective_thrust_n`,
    `tau_*`, `actuator_delay_s` remain).
  * `--parameter-set tarot680b_reference` → still exits 0, unchanged
    summary (mass 3.57 kg, T_max 70.02 N).

`[CAUTION]` What this does NOT establish: nothing here validates the
measurements themselves (no independent re-weigh, no CAD cross-check), the
vehicle configuration at weighing is not recorded, the component specs are
user-supplied rather than datasheet-verified, and no simulator has been
shown to predict this vehicle's behavior.

`[TODO]` In order:
  1. Measure **Ixx / Iyy / Izz** (bifilar/trifilar pendulum or HELIX) and
     enter them directly — `inertia_estimation_mode` stays `manual` so the
     geometric auto-estimate can never fill them.
  2. Define the raw measurement datum (origin + +Z direction), record it,
     convert CG / motor-plane / battery into FRD, and flip
     `raw_measurements.datum_status` to `RESOLVED`.
  3. Measure the actual arm angles; only then promote motor XY into
     `motors.positions_body_m`.
  4. Separately, and only after 1–3: MuJoCo physical-model parameter
     injection.

GIT STATE: branch `refactor/landing-rl-architecture`, HEAD
`6306699871122cd985a3e35ebd1b93ce575aeaa4` (unchanged — nothing committed
this session). Working tree: `PROJECT_HANDOVER.md` (this entry + §0.10/§0.11
+ header; **also still carrying the uncommitted 2026-09-14 viewer-diagnosis
revision, which was preserved, not overwritten**),
`landing_mujoco/configs/ugrp_vehicle_measured.yaml`,
`landing_mujoco/configs/param_schema.py`,
`landing_mujoco/tests/test_param_schema.py` modified;
`landing_mujoco/tests/test_ugrp_measured_params.py` untracked (new);
`MUJOCO_LOG.TXT` untracked (pre-existing). Nothing staged or committed.
Not pushed.

`[CAUTION]` CLAUDE.md §23 lists the default write scope as `landing_rl/`,
`system_id/` and `PROJECT_HANDOVER.md`; `landing_mujoco/` postdates that
list and is not in §22's protected-infrastructure set. This session wrote
under `landing_mujoco/configs/` and `landing_mujoco/tests/` because the
canonical vehicle-parameter source demonstrably lives there. Root-level
`MUJOCO_MODEL.md` was **not** modified — its §H parameter table still
describes only the provisional reference config. `[PROPOSAL]` (not
approved) add a pointer from `MUJOCO_MODEL.md` §H to §0.11.

---

### 2026-09-16 (second pass) — coordinate blockers resolved; motor/CG/battery geometry promoted to confirmed body-frame values

Scope: **parameter recording, schema, tests and documentation only** —
unchanged from the first pass. No MuJoCo physics injection, no MJCF, no
actuator/thrust model, no change to `LegacyVehicleDynamics`,
`InnerLoopCommandModel`, contact, reward, PPO, VecNormalize or checkpoints.
Ixx/Iyy/Izz were **not** filled with any placeholder.

`[DECISION — user, 2026-09-16]` Both blockers raised by the first pass were
resolved by the user:

1. **Measurement datum defined.** The raw z-values are heights measured from
   the **floor**, **+Z up**. Combined with the body frame being FRD (+Z
   down) with origin at the CG, the conversion is
   `z_body_FRD = cg_raw_z − raw_z`.
2. **Frame symmetry confirmed.** The vehicle is an **exact symmetric 45°
   X-frame**, R = 0.360 m ⇒ `a = R/√2 = 0.2545584412271571 m`.

`[SUPERSEDED]` These two decisions supersede the first pass's `[OPEN]`
items "datum ambiguity is the blocking issue" and "the ±45° arm angle was
not confirmed" (see the preceding §9 entry, left intact as history) and the
corresponding `[OPEN]`/`[CAUTION]` text formerly in §0.11.2 / §0.11.3, which
have been rewritten in place per CLAUDE.md §9.

`[FACT]` Promoted to confirmed body-frame values:

| Field | Value (FRD) | Basis |
|---|---|---|
| `mass_properties.cg_body_m` | `[0, 0, 0]` | body origin **is** the CG; now reconciled, not a default |
| `motors.positions_body_m` | FL `[+a,−a,−0.093]`, FR `[+a,+a,−0.093]`, RL `[−a,−a,−0.093]`, RR `[−a,+a,−0.093]` | XY from arm length + confirmed 45°; Z from the datum conversion |
| `battery.position_body_m` | `[0, 0, +0.021]` | datum conversion |
| `geometry.arm_angle_deg` | `45.0` | USER_CONFIRMED (new schema field) |

PHYSICAL MEANING: motors sit **93 mm above** the CG (negative in FRD);
the battery sits **21 mm below** it (positive in FRD).

IMPLEMENTATION
  * `landing_mujoco/configs/ugrp_vehicle_measured.yaml` — filled the three
    body-frame fields; added `geometry.arm_angle_deg: 45.0`;
    `raw_measurements` now carries `datum_status: RESOLVED`,
    `datum_origin: floor`, `datum_up_axis: "+z_up"`,
    `datum_resolved_date`, and its `unresolved_signs` block was **renamed**
    `resolved_separations` (magnitudes kept, signs added) rather than
    dropped, so the history stays legible. Raw values preserved verbatim.
  * `landing_mujoco/configs/param_schema.py` — added
    `Geometry.arm_angle_deg`; added the datum fields plus
    `RawMeasurements.body_z_from_raw_height()`, the single definition of the
    conversion, which **raises** rather than guessing if the datum is
    unresolved or not `+z_up`. Kept the `abs_*` accessors as an independent
    cross-check (they read the raw pair, so they can disconfirm a stored
    body-frame magnitude; a signed accessor derived from the same conversion
    could not). **Removed** `ProvisionalGeometry`, its parser, the
    `provisional_geometry:` YAML block and the now-unused `_vec2` helper.
  * `landing_mujoco/configs/param_schema.py` — also added `meta` to
    `PropellerSpec`, `BatterySpec` and `ESCSpec` and wired it into their
    parsers. `[CORRECTION]` those three dataclasses had **no** `meta` field,
    so the propeller/battery/ESC provenance blocks written in the first pass
    were being silently discarded at load time. Caught by a test asserting
    the battery's provenance tag. The first pass's claim that every
    populated value carries a provenance tag was true of the YAML but not of
    the parsed object; it is now true of both.
  * `landing_mujoco/tests/test_ugrp_measured_params.py` — 29 → 35 tests.
    `DatumSeparationTest` became `DatumConversionTest`: the tests that
    asserted "datum pending / fields null / arm angle unconfirmed" were
    **inverted rather than deleted**, so the new invariant is pinned with the
    same strength as the old one.
  * `landing_mujoco/tests/test_param_schema.py` — the missing-field set
    updated a second time (11 → 10 → 8).

VALIDATION
  * `python3 -m pytest landing_rl/tests landing_mujoco/tests system_id/tests
    -q` → **405 passed** in 239.39 s, versus 399 at the end of the first
    pass. No test changed status from pass to fail.
  * Per-suite collection counts: **`landing_rl` = 218 — identical to the
    session baseline**, which is the decisive evidence for UNCHANGED since
    that is the suite covering the legacy dynamics, controller, contact and
    reward. `landing_mujoco` 120 (85 unchanged + 35 in the new file),
    `system_id` 67 (unchanged).
  * `check_parameter_set --parameter-set ugrp_vehicle_measured` → still
    exits 1, now listing **8** missing: `Ixx`, `Iyy`, `Izz`,
    `max_collective_thrust_n`, `tau_roll_s`, `tau_pitch_s`, `tau_thrust_s`,
    `actuator_delay_s`. `cg_body_m` and `motor_positions_body_m` correctly
    dropped off.
  * `--parameter-set tarot680b_reference` → still exits 0, unchanged.
  * Explicitly re-verified that `load_uav_params()` still raises on the
    measured config: `view_x500_visual.py` calls it **without**
    `validate=False`, so populating `positions_body_m` did **not** open a
    path for these numbers to reach MuJoCo. Missing inertia keeps that shut.

`[CAUTION]` What this does NOT establish: the datum definition and frame
symmetry are user assertions, not independently re-measured in this session;
no CAD or metrology cross-check was performed; battery MASS is still
unknown, so the battery position cannot yet contribute to an inertia
estimate; and nothing here validates that any simulator predicts this
vehicle's behavior.

`[OPEN]` `geometry.wheelbase_m` deliberately left `null` — see the
`[DECISION]` in §0.11.3a. `[OPEN]` `ground_clearance_m`,
`landing_gear_points_body_m`, `frame_height_m` are NOT implied by the datum
resolution and remain unmeasured.

`[TODO]` In order:
  1. Measure **Ixx / Iyy / Izz** (bifilar/trifilar pendulum or HELIX) and
     enter them directly — `inertia_estimation_mode` stays `manual`, and a
     test asserts it, so the geometric auto-estimate can never fill them.
     This is now the ONLY geometry/mass blocker.
  2. Separately, and only after 1: MuJoCo physical-model parameter
     injection (a behavior change needing its own commit and regression
     pass). `max_collective_thrust_n` and the identified response are still
     required before that config will load.

GIT STATE: branch `refactor/landing-rl-architecture`, HEAD
`6306699871122cd985a3e35ebd1b93ce575aeaa4` (unchanged — nothing committed).
Working tree: `PROJECT_HANDOVER.md`,
`landing_mujoco/configs/ugrp_vehicle_measured.yaml`,
`landing_mujoco/configs/param_schema.py`,
`landing_mujoco/tests/test_param_schema.py` modified;
`landing_mujoco/tests/test_ugrp_measured_params.py` untracked (new);
`MUJOCO_LOG.TXT` untracked (pre-existing). The 2026-09-14 viewer-diagnosis
revision of this file is still carried, uncommitted and unmodified. Nothing
staged or committed. Not pushed.

### 2026-09-19 — measured mass / inertia / geometry injected into the MuJoCo rigid body (physical-inertia injection complete; propulsion calibration pending)

Scope: **`landing_mujoco` MuJoCo physical-model parameters only.** No
observation / action / reward / termination / perception / latency /
`BaselineController` / PPO / VecNormalize / checkpoint / curriculum change,
and `LegacyVehicleDynamics` is untouched. No propulsion or motor model was
invented; nothing was tuned.

`[FACT]` Working-tree state found at session start (all UNCOMMITTED, carried
from the 2026-09-16 sessions — classified, none reverted): `M
PROJECT_HANDOVER.md`, `M landing_mujoco/configs/{param_schema.py,
ugrp_vehicle_measured.yaml}`, `M landing_mujoco/tests/test_param_schema.py`,
`?? landing_mujoco/tests/test_ugrp_measured_params.py`, `?? MUJOCO_LOG.TXT`
(pre-existing viewer log). Baseline: `landing_mujoco` 120 tests OK.

`[FACT — user-provided, 2026-09-19]` Diagonal inertia by bifilar
suspension on the full flight configuration: Ixx 0.153184, Iyy 0.126285,
Izz 0.149050 kg·m² (per-axis D, L, raw periods, std, variance, n, and the
8.48 s yaw outlier exclusion are recorded in the YAML; table in §0.11.5).
`[USER_CONFIRMED]` the 6.408 kg total mass is the full flight configuration
(battery, ~1.8 kg ballast brick, motors, ESCs, frame, electronics, landing
gear, wiring), so **none of it may be added again**. `Ixy/Ixz/Iyz` recorded
as `ASSUMED_ZERO_FOR_V0`; R² recorded as `N/A`.

`[FACT]` Independent reproduction: recomputing from the raw periods with
`I = m·g·D²·T²/(16π²L)` and averaging per-trial I reproduces every reported
statistic exactly at g = 9.8065 (inferred, not stated by the user), and the
with-outlier Izz (0.150874 / 0.005984). Recorded as provenance; the
canonical values remain the user's literals.

IMPLEMENTATION
  * `landing_mujoco/configs/ugrp_vehicle_measured.yaml` — `ixx/iyy/izz`
    filled; `meta.inertia_kgm2` rewritten `NOT_MEASURED → MEASURED`
    (method, reduction, per-axis raw periods and statistics, outlier policy,
    `r2: "N/A"`, `uncertainty_use: RECORDED_ONLY_NOT_WIRED_INTO_RANDOMIZATION`);
    added `meta.inertia_products_kgm2` (`ASSUMED_ZERO_FOR_V0`) and
    `meta.included_hardware` (ballast/battery, `contributes_*_to_mujoco:
    false`); mass note updated; header/vocabulary updated; stale
    hover-thrust rounding in the closing comment corrected (15.7102 N /
    1.6021 kgf → 15.7103 N / 1.6020 kgf at g = 9.80665).
    `inertia_estimation_mode` stays `manual`.
  * `landing_mujoco/configs/param_schema.py` — **docstring only**. No schema
    field or logic added: provenance lives in the free-form `meta` dict,
    which nothing consumes. `_REQUIRED_FIELDS` deliberately NOT relaxed.
  * `landing_mujoco/dynamics/mjcf_builder.py` — (1) comment documenting why
    diagonal inertia needs no FRD→FLU conversion and that products would;
    (2) motor/arm marker coordinates AND `leg_*` contact-geom coordinates now
    use `_coord()`: the legacy 4-dp text when it is lossless, full-precision
    `repr` otherwise. Without this the measured `a = 0.2545584412…` was
    rounded to 0.2546 in the compiled model (radius 0.36006 m) — required by
    the task's own "motor positions exactly matching" check. Markers are
    massless and collision-disabled. The `leg_*` change is preventive: the
    measured config has no gear points yet, but the touchdown reference reads
    the full-precision Python array, so future measured gear must not be
    silently rounded in the collision geoms (reference gear points are 4-dp
    exact, so nothing changes today).
    **Existing configs are byte-identical**: the reference-config MJCF hash
    pin `PRE_VISUAL_PHYSICS_MJCF_SHA256` in `test_x500_visual_shell.py`
    still passes unchanged. (A first draft using plain `repr` changed that
    hash; it was replaced rather than re-pinning the freeze gate.)
  * Tests: `test_measured_mjcf_injection.py` (new, 27) — compiled `mjModel`
    vs YAML (mass, inertia, `body_ipos`/`body_iquat`, compiler attributes,
    single `<inertial>`, single non-zero-mass body, shell adds no mass),
    motor geometry (radius 0.360, z, 45°, uniqueness, FRD round-trip),
    `_coord` and gear-coordinate precision (synthetic test point), and
    force/torque-level sanity (free fall, hover-equivalent
    force, τ→α = τ/I per axis, inertia bound to body axes). Plus
    `test_ugrp_measured_params.py` 35 → 50 (inertia values, provenance,
    raw-period re-reduction, outlier policy, ownership, YAML duplicate-key
    guard) and `test_param_schema.py` updated (missing set 8 → 5). The
    obsolete "inertia is still unmeasured" assertions were inverted or moved
    into the new inertia test class, not silently deleted.

PHYSICAL / CONTROL MEANING: the MuJoCo rigid body now has the real vehicle's
mass and principal inertia about its CG. `landing_rl` (mass-normalized) does
not consume any of it.

VALIDATION
  * `python3 -m unittest discover -s landing_mujoco/tests -t .` → **162 OK**
    (baseline 120 + 42).
  * `python3 -m unittest discover -s landing_rl/tests -p "test_*.py"` →
    **218 OK**, unchanged (incl. the structural-freeze matrix).
  * `python3 -m unittest discover -s system_id/tests -t .` → **67 OK**.
  * Mutation check: swapping Ixx/Iyy in the builder made 5 tests fail;
    reverted.
  * Old-vs-new reference-config MJCF, compiled: no `mjModel` array differs.
  * `check_parameter_set --parameter-set ugrp_vehicle_measured` → exit 1,
    Missing: `max_collective_thrust_n`, `tau_roll_s`, `tau_pitch_s`,
    `tau_thrust_s`, `actuator_delay_s` (Ixx/Iyy/Izz dropped off).
    `tarot680b_reference` → exit 0, unchanged.

`[INTERPRETATION]` The wiring YAML → MJCF → `mjModel` is correct and exact
for mass, inertia, inertial frame and motor geometry.

`[CAUTION]` Not established: any propulsion or closed-loop fidelity; any
ground-contact behavior of this vehicle (no gear geometry ⇒ no contact
geoms); that the off-diagonal products are small; that the manual-stopwatch
inertia is accurate beyond its recorded scatter (R² N/A; the Ixx/Iyy ≈ 1.21
ratio is recorded, not explained). No real-flight or SI validation.

`[OPEN]` `MuJoCoDynamics` / `MujocoLandingEnv` still cannot be built from the
measured config (`max_collective_thrust_n`, `tau_roll/pitch/thrust`,
`actuator_delay_s`, and gear geometry). `MUJOCO_MODEL.md` (root-level,
outside this session's write scope) still describes the provisional
reference values and the §D `a` wording error — needs a user-approved edit.
`[RESOLVED — 2026-09-19, later pass]` The user authorized that edit and
`MUJOCO_MODEL.md` was rewritten (documentation only — no YAML, Python, MJCF
builder or test change): measured vs reference configurations separated
(§H.1 / §H.2), `a` vs `R` corrected, measured inertia / geometry / compiled
`mjModel` values added, the landed-pose ground plane (body z = +0.241 m) and
the `ground_clearance_m` / leg-sphere-radius finding documented, x500 shell
scale caveat (0.737 vs measured 0.720 m) recorded, stale test list refreshed.
The gravity constants 9.8065 (`landing_rl`) vs 9.80665 (`landing_mujoco`) are
still un-reconciled.

`[TODO]` Minimum propulsion/actuator inputs, in order: (1) propulsion bench
for the MN501-S + MS1704 pair → `max_collective_thrust_n` (and per-motor
max); (2) landing-gear contact points or `ground_clearance_m`; (3) after PX4
gains are stabilized (CLAUDE.md §3): `tau_roll_s`, `tau_pitch_s`,
`tau_thrust_s`, `actuator_delay_s` from closed-loop SI. `k_f` / `k_m` / motor
time constant only if a rotor-level backend is later chosen.

GIT STATE: branch `refactor/landing-rl-architecture`, HEAD
`6306699871122cd985a3e35ebd1b93ce575aeaa4` (unchanged — nothing staged or
committed). Modified: `PROJECT_HANDOVER.md`, `MUJOCO_MODEL.md` (added by the
later documentation-only pass, see the `[RESOLVED]` note above),
`landing_mujoco/configs/
{param_schema.py, ugrp_vehicle_measured.yaml}`,
`landing_mujoco/dynamics/mjcf_builder.py`,
`landing_mujoco/tests/test_param_schema.py`. Untracked:
`landing_mujoco/tests/{test_ugrp_measured_params.py,
test_measured_mjcf_injection.py}`, `MUJOCO_LOG.TXT`. Not pushed.

### 2026-09-19 (third pass) — landed-pose geometry: physical ground clearance 0.241 m vs. simulation gear sphere (radius 0.020 m, centre offset 0.221 m)

Scope: **`landing_mujoco` physical-model representation only.** No
observation / action / reward / termination / perception / latency /
`BaselineController` / PPO / VecNormalize / checkpoint change; `landing_rl`,
`mujoco_rl`, `system_id` untouched (`git diff` empty for them). No propulsion
value invented or tuned. Per-leg gear x/y were **not** estimated.

`[DECISION — user, 2026-09-19]` In the normal landed pose (gear on the ground)
the CG is 0.241 m above the ground: `cg_body_m = [0,0,0]`, ground plane at body
z = **+0.241 m** (FRD), and `ground_clearance_m = 0.241` is a **physical**
vehicle parameter. The MuJoCo gear feet are spheres of radius 0.020 m; a sphere
centre at the physical contact depth would rest the CG 0.020 m too high, so the
centre must sit at `ground_clearance_m − radius = 0.221 m`. Parameter ownership:
PHYSICAL = `ground_clearance_m` 0.241 / contact-point z +0.241; SIMULATION =
radius 0.020 / centre z 0.221. **0.221 is never stored as a physical
clearance.** This closes the `[OPEN]` convention question of the earlier
2026-09-19 entries.

`[FACT]` Before the change (read-only simulation, no file changed): reference
gear point 0.120 ⇒ resting CG 0.1399 m; measured + synthetic gear 0.241 ⇒
0.2609 m, 0.221 ⇒ 0.2409 m.

IMPLEMENTATION
  * `landing_mujoco/configs/param_schema.py` — `Geometry.landing_gear_points_
    semantics` (`geom_center` legacy default | `physical_contact`), validated in
    `__post_init__`, parsed from YAML; constants exported; docstrings of
    `Geometry.ground_clearance_m` / `landing_reference_point_body_m` clarified.
    **`landing_reference_point_body_m` logic unchanged.**
  * `landing_mujoco/dynamics/mjcf_builder.py` — new pure function
    `landing_gear_sphere_centers_body_m(geometry, radius)`: `physical_contact` ⇒
    `centre_z = point_z − radius` (copy, never mutates the stored points);
    `geom_center` ⇒ points unchanged; `None` ⇒ no geoms. `build_mjcf` builds the
    collision spheres from it. `LEG_CONTACT_RADIUS_M = 0.02` stays a builder
    constant, documented as a simulation representation.
  * `landing_mujoco/configs/ugrp_vehicle_measured.yaml` — `ground_clearance_m:
    0.241` (MEASURED, dated, `derived_from` the raw CG height);
    `landing_gear_points_body_m` stays `null` (x/y UNKNOWN);
    `landing_gear_points_semantics: physical_contact` declared explicitly;
    header/`datum_note` updated with the landed-pose confirmation (the stale "not
    stated" wording is gone). **`tarot680b_reference.yaml` untouched.**
  * Tests — `test_ugrp_measured_params.py` 50 → 62 (new
    `LandedPoseGroundClearanceTest`, incl. a forward guard that future physical
    contact points must have z = `ground_clearance_m` — vacuous while they are null; the two assertions that pinned
    `ground_clearance_m` as null / NOT_MEASURED were inverted, not deleted);
    `test_measured_mjcf_injection.py` 27 → 44 (new `PhysicalVsSimulationGearTest`,
    `LandedPoseSettleTest`; the precision test now covers both readings).
  * `MUJOCO_MODEL.md` — ground-plane section rewritten around the three
    quantities, verification table and tolerance basis; §H.1 / header / §G /
    Running / Not-done synced.

VALIDATION
  * `python3 -m unittest discover -s landing_mujoco/tests -t .` → **191 OK**
    (162 + 29), including the reference-config MJCF hash pin
    (`PRE_VISUAL_PHYSICS_MJCF_SHA256`) **unchanged**.
  * `... -s landing_rl/tests -p "test_*.py"` → **218 OK** (unchanged);
    `... -s system_id/tests -t .` → **67 OK**.
  * `check_parameter_set --parameter-set ugrp_vehicle_measured` → exit 1, the
    same 5 missing fields (`max_collective_thrust_n`, `tau_roll/pitch/thrust_s`,
    `actuator_delay_s`); `tarot680b_reference` → exit 0.
  * Mutation checks: skipping the `− radius` conversion, and flipping its sign,
    each made 13 tests fail; builder restored and byte-compared.

EXPERIMENT (dynamic landed pose; `LandedPoseSettleTest`): measured mass and
inertia + YAML `ground_clearance_m`; ground plane world z = 0; gravity −9.80665;
**synthetic** per-leg x/y (square ±0.25 / rectangle ±0.15×±0.30 / triangle
r = 0.25 m — test data only); released level, from rest, 0.5 m up, no thrust;
3 s at `physics_dt` 0.002.

RESULT
  | layout | legs | CG world z | error | sphere bottom world z |
  |---|---|---|---|---|
  | square | 4 | 0.24092 m | −0.079 mm | −0.079 mm |
  | rectangle | 4 | 0.24092 m | −0.079 mm | −0.079 mm |
  | triangle | 3 | 0.24089 m | −0.105 mm | −0.105 mm |

  Sphere centre world z ≈ 0.0199 m; landing reference point (CG − 0.241)
  reaches the ground; unchanged for `physics_dt` {0.001, 0.002, 0.005} and drop
  {0.3, 0.5} m (spread < 0.01 mm). Negative control (0.241 read as the sphere
  centre): CG 0.2609 m, +19.9 mm. Tolerance ±2 mm: penetration is linear in
  contact load (≈ 5 µm/N, `n × pen` = 0.316 mm for 4 and 3 legs), a single
  contact carrying the full 62.8 N would sink ≈ 0.32 mm ⇒ ±2 mm ≈ 6× that and
  ≈ 10× below the 20 mm error to be caught.

`[INTERPRETATION]` The landed height is independent of the unknown gear x/y,
which is why synthetic x/y are acceptable in the test; the representation is
now consistent with the measured 0.241 m.

`[CAUTION]` `[FACT]` The reference config's `ground_clearance_m = 0.12` is its
legacy `geom_center` gear-point z (a sphere centre); its actual CG-to-ground
distance is 0.14 m, so it does not satisfy the physical definition adopted for
the measured vehicle. Synthetic, unread there (gear points exist), left
untouched. See `MUJOCO_MODEL.md`.

`[CAUTION]` Not established: real per-leg x/y, tip-over / touchdown behaviour
of the real gear, the real foot geometry (a sphere is a proxy), any propulsion
or closed-loop fidelity. `physical_contact` points assume a level landing and a
vertical foot axis. `landing_gear_points_body_m` and `ground_clearance_m` are
two descriptions of one fact with no automatic consistency check yet.
`landing_reference_point_body_m()` on the measured config returns
`[0, 0, 0.241]` with x/y **assumed** centred (a fallback, not a measurement).

`[OPEN]` `MuJoCoDynamics` / `MujocoLandingEnv` still cannot be built from the
measured config: it now stops at `IdentifiedWrenchActuation` (thrust, τ, delay
`null`), no longer at the landing reference point.

`[TODO]` (1) measure per-leg contact x/y and fill
`landing_gear_points_body_m` as PHYSICAL contact points (z = 0.241 for a level
landing); (2) propulsion bench ⇒ `max_collective_thrust_n`; (3) closed-loop SI
(after PX4 gain stabilization) ⇒ `tau_roll/pitch/thrust`, `actuator_delay_s`.

GIT STATE: branch `refactor/landing-rl-architecture`, HEAD
`6306699871122cd985a3e35ebd1b93ce575aeaa4` (unchanged — nothing staged or
committed). Modified: `PROJECT_HANDOVER.md`, `MUJOCO_MODEL.md`,
`landing_mujoco/configs/{param_schema.py, ugrp_vehicle_measured.yaml}`,
`landing_mujoco/dynamics/mjcf_builder.py`,
`landing_mujoco/tests/test_param_schema.py`. Untracked:
`landing_mujoco/tests/{test_ugrp_measured_params.py,
test_measured_mjcf_injection.py}`, `MUJOCO_LOG.TXT`. Not pushed.

### 2026-09-19 (fourth pass) — skid-type landing gear footprint recorded; four contact spheres in the measured MuJoCo model; landed and ±5° tilted contact verified

Scope: **landing-gear geometry only** (measured YAML, tests, documentation). No
change to propulsion, controller, RL contract, mass / inertia, motor geometry,
CG, ground clearance or `landing_rl` / `mujoco_rl` / `system_id`. **No builder or
schema code change this session** — the existing physical→simulation conversion
(`landing_gear_sphere_centers_body_m`, `landing_gear_points_semantics`) was
reused. Nothing was tuned (no contact stiffness / friction / restitution change).

`[FACT — USER_CONFIRMED, 2026-09-19]` The real gear is helicopter-style: two
continuous parallel skid bars whose long direction is body X (the roll axis).
Skid effective length **0.300 m**, centre-line spacing **0.310 m**; CG → ground
**0.241 m** (already MEASURED). `[DERIVED]` half length 0.150 m, half spacing
0.155 m.

`[DECISION — user, 2026-09-19]` For v0, approximate the two continuous bars by
**four representative physical contact points at the skid endpoints**, centred
and symmetric about the CG (FRD, +Y right): left-front `[+0.150, −0.155, +0.241]`,
left-rear `[−0.150, −0.155, +0.241]`, right-front `[+0.150, +0.155, +0.241]`,
right-rear `[−0.150, +0.155, +0.241]`, read as `physical_contact`. The sphere
centre (0.241 − 0.020 = 0.221 m) is derived by the builder and never stored.
`[CAUTION]` Provenance is **`DERIVED_FROM_MEASURED_DIMENSIONS` +
`CENTERED_SYMMETRY_ASSUMPTION`**, not "fully measured point coordinates": the
centring on the CG (the actual per-leg offset from the CG) is an assumption.

IMPLEMENTATION
  * `landing_mujoco/configs/ugrp_vehicle_measured.yaml` — `landing_gear_points_
    body_m` filled with the four points; `meta.landing_gear_points_body_m`
    rewritten `NOT_MEASURED → DERIVED_FROM_MEASURED_DIMENSIONS` with
    `assumptions: [CENTERED_SYMMETRY_ASSUMPTION]`, `skid_geometry` (length,
    spacing = USER_CONFIRMED; half values = DERIVED; centering = assumption;
    point order; v0 approximation text); header + provenance vocabulary updated.
    `ground_clearance_m` stays 0.241; `landing_gear_points_semantics:
    physical_contact` unchanged. **`tarot680b_reference.yaml` untouched.**
  * Tests — `test_ugrp_measured_params.py` 62 → 77 (new `MeasuredSkidFootprintTest`:
    count, contact z, x/y extents, centred symmetry, semantics, 0.221 conversion,
    no-contamination, skid direction, provenance, recompute-from-dimensions,
    reference point, "other parameters unchanged"); `test_measured_mjcf_injection.py`
    44 → 59 (new `MeasuredSkidCompiledGeometryTest`, `TiltedContactTest`;
    `PhysicalVsSimulationGearTest` / `LandedPoseSettleTest` moved from SYNTHETIC
    x/y to the canonical footprint; ground friction now the environment's 0.55).
    The assertions that pinned the per-leg points as null / NOT_MEASURED were
    inverted, not deleted.
  * `MUJOCO_MODEL.md` — new "Landing-gear skid footprint" subsection, dynamic
    verification rewritten with the canonical results, limitation stated.

VALIDATION
  * `python3 -m unittest discover -s landing_mujoco/tests -t .` → **221 OK**
    (191 + 30), including the reference-config MJCF hash pin
    (`PRE_VISUAL_PHYSICS_MJCF_SHA256`) **unchanged**.
  * `... -s landing_rl/tests -p "test_*.py"` → **218 OK** (unchanged);
    `... -s system_id/tests -t .` → **67 OK**.
  * `check_parameter_set --parameter-set ugrp_vehicle_measured` → exit 1, the
    same 5 missing propulsion fields; `tarot680b_reference` → exit 0.
  * Mutation checks (YAML restored byte-identical afterwards): a flipped y sign
    (38 failures), a contact z entered as the sphere centre 0.221 (34), a skid
    length off by 5 mm (14), the builder skipping the radius offset (19) and the
    builder bypassing the FRD→FLU helper (27).

EXPERIMENT (landed): measured mass 6.408 kg and inertia; canonical skid
footprint; ground plane world z = 0; gravity −9.80665; ground friction 0.55 (the
environment's `LandingConfig.ground_friction_xy`); released level from rest 0.5 m
up; 3 s at `physics_dt` 0.002.

RESULT (landed)
  * final CG world z **0.24098 m** (error −0.022 mm vs 0.241); sphere centre world
    z 0.01998 m; lowest sphere bottom −0.0215 mm (= penetration); 4 contacts;
    residual speed ~1e-15. Negative control (0.241 read as the sphere centre):
    0.2610 m, +20 mm.
  * Tolerance ±2 mm: settled penetration is linear in contact load at fixed
    friction and depends on μ (0.0054 / 0.0215 / 0.0795 / 0.184 mm at μ = 0.3 /
    0.55 / 0.9 / 1.2); the worst single-contact extrapolation (0.74 mm at μ = 1.2)
    is < ½ of the tolerance.

RESULT (tilted, 5°, lowest sphere bottom 20 mm above ground, 3 s): the low-side
skids touch first (roll+ → right, pitch+ → rear; equals the prediction from the
repository's NED rotation); net contact torque at first contact is restoring
(roll ±5°: τx ∓15.65 N·m; pitch ±5°: τy ∓13.19 N·m; roll+pitch: τx −20.06,
τy −17.92); no NaN or solver warning; peak body rate ≤ 1.91 rad/s; peak
penetration 4.31 mm (single axis; level control 4.32 mm) / 7.79 mm (roll+pitch,
one skid takes the impact) — always below the 20 mm sphere radius; the vehicle
settles level at CG 0.24098 m with all bottoms −0.022 mm; friction slide 2.2 cm
(0.8 cm combined), yaw −0.5° (combined). Static tip-over angle of the footprint
(derived, not tested): 32.7° roll, 31.9° pitch.

`[CORRECTION]` The third-pass entry above quoted settled penetrations of
0.079 / 0.105 mm and "≈ 5 µm/N". Those were measured at the **builder's default**
ground friction μ = 0.9, not at the environment's **0.55** (the environment passes
`LandingConfig.ground_friction_xy` to `build_mjcf`). At μ = 0.55 the 4-leg value is
0.0215 mm. The ±2 mm tolerance holds in both cases; tests now use 0.55.

`[INTERPRETATION]` With a real footprint, the current contact model is sane for
level and ±5° landings on this geometry, and the measured 0.241 m clearance is
reproduced to 0.02 mm.

`[CAUTION]` Not established: the real per-leg offset from the CG (centring is an
assumption); behaviour on a continuous skid (four endpoint spheres cannot
represent line contact or an intermediate contact point); the real foot shape (a
sphere is a proxy); contact stiffness / friction / restitution against real
touchdowns (defaults, untuned); anything about propulsion or closed-loop
fidelity. The 5° cases start 20 mm above the ground — larger impact speeds give
larger transient penetration in the default soft contact (level drop, μ = 0.55:
2.1 mm at 0.31 m/s, 4.3 mm at 0.63 m/s, 6.5 mm at 0.99 m/s, 9.4 mm at 1.4 m/s,
16 mm at 2.2 m/s). The 12 mm bound in `TiltedContactTest` applies only to its
20 mm-lift scenario.

`[OPEN]` `MuJoCoDynamics` / `MujocoLandingEnv` still cannot be built from the
measured config (thrust, τ, delay `null`). Landing-gear geometry is no longer a
blocker for that.

`[TODO]` (1) propulsion bench ⇒ `max_collective_thrust_n`; (2) closed-loop SI
after PX4 gain stabilization ⇒ `tau_roll/pitch/thrust`, `actuator_delay_s`;
(3) optionally measure the real skid-footprint offset from the CG and replace the
centring assumption.

GIT STATE: branch `refactor/landing-rl-architecture`, HEAD
`6306699871122cd985a3e35ebd1b93ce575aeaa4` (unchanged — nothing staged or
committed). Modified: `PROJECT_HANDOVER.md`, `MUJOCO_MODEL.md`,
`landing_mujoco/configs/{param_schema.py, ugrp_vehicle_measured.yaml}`,
`landing_mujoco/dynamics/mjcf_builder.py`,
`landing_mujoco/tests/test_param_schema.py`. Untracked:
`landing_mujoco/tests/{test_ugrp_measured_params.py,
test_measured_mjcf_injection.py}`, `MUJOCO_LOG.TXT`. Not pushed.

---

## 2026-09-25 — RL Baseline v0.1 packaging (repository engineering; no research-behaviour change)

Detail lives in §0.12; this entry preserves the provenance.

[FACT] Trigger: a read-only audit earlier the same day concluded NOT READY FOR TEAM BASELINE, mainly for
reproducibility: fresh-clone PID-only evaluation failed (`FileNotFoundError` for the PPO `.zip`), the checkpoint gate
hard-coded a developer's home path, no root dependency definition or RL quickstart existed, and the measured-vehicle
work and its 136 tests (`test_ugrp_measured_params.py` 77, `test_measured_mjcf_injection.py` 59) were uncommitted.

IMPLEMENTATION: see the commit table in §0.12 (`6c4fd81`, `a666ab8`, `7d65074`, `b633024`, `9dd5563`). New:
`landing_rl/evaluation/{__init__,artifacts,smoke}.py`, `landing_rl/tests/{test_evaluation_artifacts,test_smoke}.py`,
`BASELINE.md`, `MODEL_ARTIFACTS.md`, `requirements.txt`. Modified: `README.md`, `mujoco_rl/eval_compare_v2.py`,
`mujoco_rl/eval_robustness_paper.py`, `mujoco_rl/eval_env_stage0.py`, `mujoco_rl/scripts/{plot_trajectory,smoke_test}.py`,
`landing_rl/tests/{test_checkpoint_compatibility,test_entry_point_migration}.py`. `eval_env_stage0.py` keeps its CRLF
line endings. One structural finding while editing: `test_entry_point_migration.py` (test C) pinned the literal
`./runs/...` strings in the eval scripts; it now pins the pair-of-record identity through `artifacts.py`, and its
`eval_env_stage0.py` literals were updated to the stage-1 pair.

PHYSICAL / CONTROL MEANING: none. No physics, controller, reward, observation, action, randomization or timing change.

VALIDATION (exact commands, artifacts via `UGRP_RL_ARTIFACT_DIR=<maintainer runs dir>`, strict mode):
`python3 -m unittest discover -s landing_rl/tests -p "test_*.py"` (238 OK, 190 s);
`python3 -m unittest discover -s landing_mujoco/tests -t . -p "test_*.py"` (221 OK, 55.7 s);
`python3 -m unittest discover -s system_id/tests -t . -p "test_*.py"` (67 OK, 8.8 s).
Also: `python3 -m landing_rl.evaluation.smoke`; skip and strict-failure behaviour of the checkpoint gate checked directly.

EXPERIMENT (reference reproduction, seeds 5000-5199 and 7000-7199, 200 episodes per cell):
`python3 -m mujoco_rl.eval_compare_v2` and `python3 -m mujoco_rl.eval_robustness_paper --output-dir <scratch>` with the
v4 pair; PID-only from another working directory with no artifacts via
`python3 mujoco_rl/eval_compare_v2.py --policy pid`.

RESULT: `eval_compare_v2` output identical to the archived file (PID 190, random 183, PPO 191 of 200); the three robustness
CSVs byte-identical to the archived ones; PID-only 190/200 reproduced without artifacts.

[INTERPRETATION] The refactored evaluation path and the current environment code reproduce the archived paper results
exactly, so the packaging changes did not alter evaluation behaviour.

[CAUTION] Reproducibility is not performance evidence (§0.12). Only the evaluation of the pair of record was
reproduced; training from scratch remains unreproducible. `requirements.txt` was resolved with `pip install --dry-run`
in a fresh venv, not installed from scratch.

[OPEN] Push approval; artifact hosting; missing stage-0 / stage-2-v3 scripts; baseline branch choice; deployed-PID vs
simulated-PID gap.

GIT STATE: branch `refactor/landing-rl-architecture`; 5 packaging/measured-vehicle commits on top of `6306699` plus this
handover commit (`git log -1 -- PROJECT_HANDOVER.md`); remote `origin/refactor/landing-rl-architecture` is 6 commits
behind before this session's commits; nothing pushed. Untracked: `MUJOCO_LOG.TXT` (not committed by design).

## 2026-09-28 — Clean-room reproduction fixes (repository engineering; no research-behaviour change)

[FACT] A clean-room reproduction was performed on a separate fresh Ubuntu / Python 3.10.12 environment against
commit `553fdfd` (before this session's fix). Results: `landing_rl` 238 PASS with the canonical PPO/VecNormalize
artifacts present; `landing_mujoco` 221 PASS; `system_id` 67 PASS, 6 skipped; `python -m landing_rl.evaluation.smoke`
PASS; PID reference reproduced at 190/200 = 95.0%; PPO reference reproduced at 191/200 = 95.5%; the PPO model and
VecNormalize files matched their recorded SHA256 in `MODEL_ARTIFACTS.md`.

[FACT] Two setup issues were found: (1) `README.md`'s quick start did not mention the OS `python3-venv`
prerequisite needed on a fresh Debian/Ubuntu install before `python3 -m venv` works; (2) `matplotlib` is imported
at module scope by `mujoco_rl/scripts/plot_trajectory.py`, which the mandatory `landing_rl` regression suite
(`test_entry_point_migration.py`) loads, but `requirements.txt` listed `matplotlib` as optional.

IMPLEMENTATION: both issues fixed on `refactor/landing-rl-architecture` (`README.md`, `requirements.txt`,
`BASELINE.md`). No RL, MuJoCo, or system_id behaviour source changed.

[CAUTION] The trained PPO baseline remains the analytical `landing_rl`-environment policy; it is still not trained
on the measured-UGRP MuJoCo model (§0.4, `BASELINE.md`).

[OPEN] Shared artifact hosting for the PPO/VecNormalize pair is still unresolved. A `ResourceWarning` (unclosed file)
in `system_id/tests/test_loader.py` remains unresolved; it is non-blocking (the suite still passes).
