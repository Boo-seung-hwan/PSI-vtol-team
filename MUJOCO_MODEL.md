# MuJoCo UAV Model — `landing_mujoco/`

This document describes the MuJoCo-backed precision-landing plant added in
`landing_mujoco/`. It does not replace `landing_rl/` or `mujoco_rl/`, which
remain the frozen legacy/live reference (see `PROJECT_HANDOVER.md`).

Status as of this writing: **PROVISIONAL_REFERENCE only.** No measured UGRP
vehicle parameter exists yet. See Section H for exactly which numbers are
real Tarot TL680B manufacturer data, which are geometric derivations, and
which are synthetic placeholders — none of them describe the actual
assembled UGRP UAV.

---

## A. Coordinate convention

Policy/control convention (unchanged): **NED** world (+X North, +Y East,
+Z Down), **FRD** body (+X Forward, +Y Right, +Z Down).

MuJoCo convention chosen: **NWU** world (+X North, +Y West, +Z Up), **FLU**
body (+X Forward, +Y Left, +Z Up). Both are related to NED/FRD by the
identical fixed matrix `REFLECT = diag(1, -1, -1)`, which is a proper
rotation (a 180-degree rotation about the shared North/Forward axis, i.e.
`R_x(pi)`), not a reflection — it is its own inverse, so ONE matrix converts
both directions for both position/velocity/vectors and (via conjugation)
attitude. All of this lives in exactly one file:
`landing_mujoco/coordinates/transforms.py`. No other file in
`landing_mujoco` contains a sign flip, axis swap, or frame-specific
rotation.

A useful consequence: MuJoCo's native gravity option (`0 0 -9.80665`, down
in a Z-up world) is exactly the NED gravity vector `[0,0,+g]` passed through
`REFLECT` — no separate gravity bookkeeping exists anywhere in the dynamics
code.

## B. Body-frame definition

The MuJoCo `vehicle` body's local frame origin is defined to coincide with
the **vehicle CG**. Mass/inertia are attached via an explicit
`<inertial pos="0 0 0" mass="..." diaginertia="Ixx Iyy Izz"/>` at that
origin (see `landing_mujoco/dynamics/mjcf_builder.py`). Because an explicit
`<inertial>` is present, MuJoCo does NOT infer mass/inertia from the
visual/collision geoms — this is what prevents the measured mass/inertia
from ever being silently duplicated by geometry.

## C. CG definition

`mass_properties.cg_body_m` is nominally "CG in the vehicle reference
frame," but because the body origin IS the CG (B, above), all other
body-frame vectors in the config (`motors.positions_body_m`,
`geometry.landing_gear_points_body_m`, `battery.position_body_m`) are
stored **already relative to CG** — matching how a real measurement of
`r_Mi - r_CG` is naturally reported. `cg_body_m` itself is not used as a
translation offset anywhere in `mjcf_builder.py`; it exists in the schema
for provenance/documentation and for a future model where the body-frame
origin and CG are allowed to differ.

For the current `PROVISIONAL_REFERENCE` config, `cg_body_m = [0,0,0]`
(perfect symmetry assumption, tagged `temporary_symmetry_assumption`, low
confidence).

## D. Motor numbering

Square-X layout, radial distance `a = wheelbase / (2*sqrt(2))`:

| Motor | Position (body/FRD, relative to CG) |
|---|---|
| M1 | `[+a, +a, 0]` |
| M2 | `[-a, +a, 0]` |
| M3 | `[-a, -a, 0]` |
| M4 | `[+a, -a, 0]` |

(Provisional reference config: `a = 0.2606 m`.)

## E. Motor spin directions

Provisional reference config (derived assumption, NOT verified against real
wiring): M1 CW, M2 CCW, M3 CW, M4 CCW (standard alternating X pattern).

Spin direction is currently **not consumed by any dynamics code** — v0 uses
`IdentifiedWrenchActuation` (collective force + body torque only, see
Section G/H of the task spec and `landing_mujoco/dynamics/actuation_model
.py`), which has no per-rotor decomposition. Spin direction is recorded for
a future `RotorLevelActuation` only.

## F. Force / torque directions

* **Thrust force**: applied along `-body_z_in_ned` (body Z is down in FRD,
  so `-body_z` is up when level), scaled by the identified thrust response
  (`landing_mujoco/dynamics/identified_inner_loop.py::thrust_response_step`)
  times ground-effect factor times vehicle mass, converted to MuJoCo's
  world frame and written into `xfrc_applied`'s linear part.
* **Body torque**: `tau_body = J*omega_dot_des + omega x (J*omega)`
  (`identified_inner_loop.py::body_torque_from_rate_response`), computed
  entirely in the FRD/policy convention, then rotated body->world using
  MuJoCo's own live orientation (`data.xmat`) at the point of application.
  Because the Coriolis term `omega x (J*omega)` is included in the applied
  torque, MuJoCo's own free-rigid-body equations make the ACTUAL angular
  acceleration equal `omega_dot_des` — the identified response IS the
  vehicle's real dynamics, not something fighting MuJoCo's own terms.

MuJoCo integrates position/velocity/attitude/angular-velocity itself
(`mujoco.mj_step`); no Python code Euler-integrates rigid-body state.

## G. Torque directions / inertia validation

Verified in `landing_mujoco/tests/test_physical_validation.py
::InertiaResponseTest`: applying a known raw torque about each axis (no
identified-response layer involved) and reading `data.qacc` immediately
after `mj_forward` reproduces `alpha = tau / J_axis` to within `1e-5`,
confirming the MJCF mass/inertia wiring is correct.

## H. Parameter table

Values below are from `landing_mujoco/configs/tarot680b_reference.yaml`
(**PROVISIONAL_REFERENCE** — NOT the measured UGRP vehicle). The
`ugrp_vehicle_measured.yaml` counterpart has every research-critical field
`null` until measured; loading it raises `MissingMeasurementError` listing
exactly what's missing.

| Parameter | Value | Unit | Category | Used in |
|---|---|---|---|---|
| `mass_kg` | 3.57 | kg | 2. MANUFACTURER SPEC | mass_properties, MJCF `<inertial>` |
| `cg_body_m` | [0,0,0] | m | 5. ASSUMED | body-frame origin convention |
| `Ixx` | 0.05828 | kg·m² | 4. DERIVED (parallel-axis, provisional) | MJCF `<inertial>` |
| `Iyy` | 0.05828 | kg·m² | 4. DERIVED (provisional) | MJCF `<inertial>` |
| `Izz` | 0.05650 | kg·m² | 4. DERIVED (provisional) | MJCF `<inertial>` |
| `wheelbase_m` | 0.737 | m | 2. MANUFACTURER SPEC | motor-position derivation |
| motor positions (a) | 0.2606 | m | 4. DERIVED (from wheelbase, symmetric-X assumption) | MJCF geoms, inertia estimate |
| `ground_clearance_m` | 0.12 | m | 5. ASSUMED (SYNTHETIC_TEST) | landing-gear geometry |
| `T_max` | 70.02 | N | 5. ASSUMED (SYNTHETIC_TEST, T/W=2.0) | thrust saturation |
| `T_max/(mg)` | 2.00 | - | 4. DERIVED | thrust-margin check |
| `tau_roll_s` | 0.15 | s | 5. ASSUMED (SYNTHETIC_TEST) | identified rate response |
| `tau_pitch_s` | 0.15 | s | 5. ASSUMED (SYNTHETIC_TEST) | identified rate response |
| `tau_thrust_s` | 0.20 | s | 5. ASSUMED (SYNTHETIC_TEST) | identified thrust response |
| `T_delay` (`delay_s`) | 0.02 | s | 5. ASSUMED (SYNTHETIC_TEST) | actuator delay buffers |
| `K_roll`,`K_pitch`,`K_thrust` | 1.0 | - | 5. ASSUMED (SYNTHETIC_TEST, unity gain) | identified response |

Categories, per task spec: **1. DIRECTLY MEASURED** (none exist yet — that
is the point of this week's pipeline-only scope), **2. MANUFACTURER
SPECIFICATION** (Tarot TL680B published figures), **3. SYSTEM IDENTIFIED**
(none exist yet), **4. DERIVED** (computed from another value in this
table, e.g. inertia from geometry+mass, motor position from wheelbase),
**5. ASSUMED** (no measurement or derivation basis at all — synthetic or
symmetry placeholders, all tagged `replace_before_real_training: true` in
the YAML).

---

## Landing-gear standoff: `pad_target_ned` vs. `vehicle_touchdown_target_ned` (RESOLVED)

**Previously flagged as an open, unresolved finding; now fixed without
touching `success_altitude_m`.** The legacy analytic contact model
(`landing_rl.contact.contact_model.ContactModel`) snaps the vehicle's CG
directly to `pos[2] = ground_z_m` on contact — i.e. it implicitly assumes
zero landing-gear standoff, and `target_true` (the pad position) doubled as
the vehicle's own resting-CG target. MuJoCo's real landing-gear geometry
means the CG actually rests at `z ~= ground_z_m - (gear standoff)`, so
comparing CG position directly against the pad target made `z_error`
converge to the standoff distance, not zero, even on a clean landing.

**Fix**: `landing_mujoco.envs.mujoco_landing_env.MujocoLandingEnv` now
explicitly distinguishes:

    pad_target_ned                 -- the physical landing-pad position
                                       (what perception measures; same role
                                       as legacy target_true/obs_target)
    vehicle_touchdown_target_ned   -- the desired VEHICLE-REFERENCE (CG)
                                       position at a correct level touchdown

related by (`MujocoLandingEnv._vehicle_touchdown_target`):

    p_landing^N = p_vehicle^N + R_B^N r_landing^B
    =>  vehicle_touchdown_target_ned = pad_target_ned - R_desired @ r_landing_body

where `r_landing_body` (`landing_mujoco.configs.param_schema
.landing_reference_point_body_m`) is the nominal landing-gear contact point
relative to CG — the mean of the actual per-leg `landing_gear_points_body_m`
geometry, **explicitly never assumed equal to** `ground_clearance_m` — and
`R_desired` is the level (roll=pitch=0, yaw=target_yaw) touchdown attitude.
dx/dy/dz (observation), `xy_error`/`z_error` (reward/success), and the
potential-based shaping term are all computed against
`vehicle_touchdown_target_ned`, not the raw pad position.
`success_altitude_m` (`0.05 m`) is **byte-unchanged** — only what `z_error`
is measured against changed.

`altitude_agl` is kept as a SEPARATE, purely physical clearance quantity:
the landing-gear reference point's height above the ground plane using the
vehicle's CURRENT (not desired) attitude — it reaches zero at actual
physical contact independently of `z_error` (e.g. a tilted vehicle sitting
exactly at the target z has `z_error~=0` but `altitude_agl>0`). Ground
effect (in `MuJoCoDynamics`) was also switched to trigger off this same
gear-clearance quantity rather than raw CG height, for physical consistency
— a deliberate MuJoCo-side refinement, not a legacy behavior change.

**Verified** (`landing_mujoco/tests/test_landing_reference_point.py`, 8
tests): a clean zero-action descent now reaches `z_error ≈ 0.02 m < 0.05 m`
with `ground_contact=True` and `success=True`; doubling/halving the gear
standoff changes the required CG resting altitude by the corresponding
amount while `success_altitude_m` stays literally `0.05` in both cases; a
tilted vehicle at the exact target z shows `altitude_agl` and `z_error`
diverging as expected; the legacy `landing_rl.envs.landing_env.LandingEnv`
is confirmed unchanged (CG still reaches `ground_z_m` exactly on contact).

---

## Software structure

```
landing_mujoco/
    configs/        param_schema.py, simulation_config.py,
                    tarot680b_reference.yaml, ugrp_vehicle_measured.yaml
    coordinates/    transforms.py
    dynamics/       mjcf_builder.py, inertia_estimation.py, delay_buffer.py,
                    identified_inner_loop.py, actuation_model.py,
                    mujoco_dynamics.py, contact.py
    envs/           mujoco_landing_env.py
    evaluation/     compare_old_vs_mujoco.py
    tests/          test_coordinate_transforms.py, test_param_schema.py,
                    test_physical_validation.py, test_sign_conventions.py,
                    test_identified_response.py,
                    test_observation_action_contract.py,
                    test_reset_determinism.py, test_ground_contact.py
    check_parameter_set.py   (CLI: --parameter-set {tarot680b_reference,ugrp_vehicle_measured})
```

Reused UNCHANGED from `landing_rl`: `BaselineController`, `DisturbanceModel`,
`LoopTiming`, `ActionLatency`, `ObsLatency`, `TargetMeasurementModel`,
`ObservationNoiseSampler`, `InitialStateSampler`, `InnerLoopCommandModel`,
`VehicleState`, `ContactState`/`ContactResult` (dataclasses only), and the
reward/success/failure formulas (ported with identical coefficients/
thresholds into `MujocoLandingEnv.step`).

Replaced by MuJoCo: `LegacyVehicleDynamics.advance_free_flight` (Python
Euler integration) -> `MuJoCoDynamics` (MuJoCo rigid-body integration);
`ContactModel`'s manual bounce math -> MuJoCo native contact +
`landing_mujoco.dynamics.contact.classify_touchdown` (classification only).

## Running

```bash
python3 -m unittest discover -s landing_mujoco/tests -p "test_*.py"
python3 -m landing_mujoco.check_parameter_set --parameter-set tarot680b_reference --run-smoke-episode
python3 -m landing_mujoco.check_parameter_set --parameter-set ugrp_vehicle_measured   # fails fast, lists missing measurements
python3 landing_mujoco/evaluation/compare_old_vs_mujoco.py
```

## Explicitly NOT done in this pass (see CLAUDE.md / task spec restrictions)

* No PPO training or retraining.
* No rotor-level actuation (`RotorLevelActuation` raises `NotImplementedError`).
* No ground effect tuning (`ground_effect_gain = 0.0`, inherited unchanged
  from `LandingConfig` — effectively disabled, matching "ground_effect_enabled
  = False" for this phase).
* No real UGRP numeric parameter anywhere — every non-informational value in
  `tarot680b_reference.yaml` is manufacturer/derived/synthetic and tagged
  `replace_before_real_training: true`.
* All 8 comparison scenarios (hover, north/east velocity, climb, descent,
  roll/pitch transient, landing approach) are implemented in
  `compare_old_vs_mujoco.py`, each reporting position/velocity/acceleration/
  attitude/body-rates/thrust/command traces plus rise time/settling time/
  overshoot/peak tilt/peak accel/steady-state speed/touchdown speed.
* `compare_flightlog_vs_mujoco.py` (real `.ulg` vs MuJoCo) is not built yet
  — deferred until a real UGRP flight log exists (the only ULogs in this
  repo are tagged `OTHER_PROJECT_SAMPLE` / `PIPELINE_VALIDATION_ONLY`, per
  CLAUDE.md §0/§19 they may not be used for this).
