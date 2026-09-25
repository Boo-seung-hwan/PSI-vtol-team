# MuJoCo UAV Model — `landing_mujoco/`

This document describes the MuJoCo-backed precision-landing plant added in
`landing_mujoco/`. It does not replace `landing_rl/` or `mujoco_rl/`, which
remain the frozen legacy/live reference (see `PROJECT_HANDOVER.md`).

**Two physical-parameter configurations exist, and this document keeps them
strictly apart** (last reviewed against the source and both YAML files:
2026-09-19):

| | **MEASURED UGRP vehicle** | Reference / historical |
|---|---|---|
| File | `landing_mujoco/configs/ugrp_vehicle_measured.yaml` | `landing_mujoco/configs/tarot680b_reference.yaml` |
| `parameter_set` | `MEASURED_VEHICLE` | `PROVISIONAL_REFERENCE` |
| Describes | the actual assembled UGRP UAV, full flight configuration, **6.408 kg** | a **different, unmeasured** airframe (Tarot TL680B, 3.57 kg): manufacturer / derived / synthetic values used only to exercise the software pipeline |
| Documented in | §B, §C, §D, §E, §H.1, "Measured vehicle: ground plane…" | §H.2, and notes explicitly labelled "reference" |
| Can build `MuJoCoDynamics` / `MujocoLandingEnv`? | **No** — `load_uav_params()` raises `MissingMeasurementError` (propulsion fields, below) | Yes |

Current state of the measured vehicle, stated exactly:

    physical-inertia injection complete
    propulsion calibration pending

The measured mass, CG, motor / battery geometry and diagonal inertia tensor
reach the **compiled `mjModel`** (§H.1). The propulsion / closed-loop side is
**not** part of the measured physical model, because it has not been measured
or identified: `max_collective_thrust_n`, `tau_roll_s`, `tau_pitch_s`,
`tau_thrust_s`, `actuator_delay_s` (and, only if a rotor-level backend is ever
built, `k_f`, `k_m`, motor time constant, ESC delay). The vertical
CG-to-ground distance is recorded (`ground_clearance_m = 0.241 m`) and the
skid-type landing gear is recorded as four physical contact points **derived
from the measured skid length / spacing under a centred-symmetry assumption**
(so the measured MJCF has four contact spheres). Only the rigid-body MJCF can
be compiled from the measured config today (see "Running").

**Reading rule:** the following are **reference-config numbers** and say
nothing about the UGRP vehicle: 3.57 kg, 0.058285 / 0.056498 kg·m², R = 0.3685 m,
a = 0.2606 m, 0.12 m gear point, 70.02 N, τ ≈ 0.15 / 0.20 / 0.30 s, 0.02 s delay,
and the 0.737 m wheelbase. They appear in §H.2 and in paragraphs explicitly
labelled *reference*. The UGRP vehicle's numbers — 6.408 kg, 0.153184 / 0.126285
/ 0.149050 kg·m², R = 0.360 m, a = 0.2545584412271571 m, and the datum values
0.241 / 0.334 / 0.220 m — appear in §B–§E, §H.1 and "Measured vehicle: ground
plane…", each labelled as measured or derived.

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

**Mass / inertia ownership (measured vehicle): one rigid body, no double
counting.** The measured 6.408 kg and the measured Ixx / Iyy / Izz describe
the **whole assembled flight configuration** — battery, the ~1.8 kg ballast
brick, motors, ESCs, frame, electronics, landing gear, wiring and other
installed hardware. They already contain all of it. Therefore:

* MuJoCo carries **exactly one** inertial body, `vehicle`, with mass 6.408 kg
  and the measured principal inertia.
* battery / brick / motors / electronics **must NOT be added again as
  independent inertial masses** when the measured whole-vehicle mass and
  inertia tensor are used. Doing so would count them twice.
* **Visual geometry and inertial ownership are deliberately decoupled.**
  Everything drawn — the placeholder arm / motor markers and the x500 visual
  shell — is massless and collision-disabled. Component data in the YAML
  (`battery.position_body_m`, the brick's approximate mass / size / position
  under `mass_properties.meta.included_hardware`, motor positions) is visual /
  provenance / sanity-check information only
  (`contributes_mass_to_mujoco: false`, `contributes_inertia_to_mujoco:
  false`). The component masses `battery.mass_kg` and
  `motors.mass_kg_each` stay `null` and are never summed in.
* Verified on the compiled measured model: exactly one non-zero-mass body,
  `Σ body_mass == mj_getTotalmass == 6.408`, and enabling the x500 visual
  shell (9 → 31 geoms) leaves `body_mass / body_inertia / body_ipos /
  body_iquat / body_subtreemass` bit-identical
  (`tests/test_measured_mjcf_injection.py`).

## C. CG definition and measurement datum

`mass_properties.cg_body_m` is nominally "CG in the vehicle reference
frame," but because the body origin IS the CG (B, above), all other
body-frame vectors in the config (`motors.positions_body_m`,
`geometry.landing_gear_points_body_m`, `battery.position_body_m`) are
stored **already relative to CG** — matching how a real measurement of
`r_Mi - r_CG` is naturally reported. `cg_body_m` itself is not used as a
translation offset anywhere in `mjcf_builder.py`; it exists in the schema
for provenance/documentation and for a future model where the body-frame
origin and CG are allowed to differ.

**Measured UGRP vehicle: `cg_body_m = [0, 0, 0]`** (DERIVED, high confidence —
the body origin *is* the measured CG). The reference config also has
`cg_body_m = [0,0,0]`, but there it is a perfect-symmetry assumption (tagged
`temporary_symmetry_assumption`, low confidence) and no measurement datum
exists for it.

### Raw measurement vs. body frame (do not confuse them)

The physical measurements were made with the vehicle in its **normal landed
pose, landing gear resting on the ground**. Heights were measured **upward
from the ground plane (ground = 0, +up)** — USER_CONFIRMED. The simulation
body frame is **FRD** (+X forward, +Y right, +Z **down**) with its origin at
the CG, so `z_body = z_CG_raw − z_raw` (an origin shift to the CG plus a sign
flip):

| Quantity | Raw (ground = 0, +up) | Body FRD (CG origin, +Z down) |
|---|---|---|
| CG | +0.241 m (above ground) | **[0, 0, 0]** — the origin, by definition |
| motor plane | +0.334 m | **z = −0.093 m** (9.3 cm **above** the CG) |
| battery centre | +0.220 m | **z = +0.021 m** (2.1 cm **below** the CG) |
| ground plane (landed pose) | 0 | **z = +0.241 m** (24.1 cm **below** the CG) |

**Wording guard.** The CG is *not* at "body z = +0.241 m". 0.241 m is the CG's
raw **height above the ground**. In the body frame the CG is the origin, and
it is the **ground plane** that sits at z = +0.241 m. Writing "CG body z =
+0.241" is wrong.

The raw numbers are preserved verbatim in the YAML's `raw_measurements`
block; `RawMeasurements.body_z_from_raw_height()` is the single definition of
the conversion. (The YAML records the datum origin as `floor` — that floor
is this ground plane; the landed-pose confirmation is written into its
`raw_measurements.datum_note` and header, and `geometry.ground_clearance_m`
stores the body-frame ground plane, 0.241 m. See "Measured vehicle: ground
plane…".)

## D. Motor numbering and geometry

Square-X layout. Two different lengths are involved — **do not confuse them**:

| Symbol | Meaning | Reference config | **Measured UGRP vehicle** |
|---|---|---|---|
| `R` | **radial distance**, centre → motor centre (`geometry.arm_length_m`) | 0.3685 m (= wheelbase / 2) | **0.360 m** (MEASURED) |
| `a` | the **x / y component** of each motor position, `a = R / √2` (exact 45° X-frame) | 0.2606 m (0.26057) | **0.2545584412271571 m** (DERIVED) |
| opposite-motor span (wheelbase) | `2R`, the max pairwise horizontal motor distance | 0.737 m (manufacturer) | 0.720 m (DERIVED; `geometry.wheelbase_m` is deliberately `null` so there is one source of truth) |

`a` is **not** the radial distance: `a = R/√2 = wheelbase/(2√2)`, and
`hypot(a, a) = R`. *(Corrected 2026-09-19: earlier text called
`a = wheelbase/(2√2)` the "radial distance"; the radial distance is
`wheelbase/2 = R`. Applying that formula with R = 0.360 m as a "wheelbase"
would give a = 0.127 m, off by a factor of 2.)*

### Measured UGRP vehicle — FRD body frame, relative to CG

The frame is an **exact symmetric 45° X-frame** (USER_CONFIRMED). With
`a = 0.2545584412271571 m` and the motor plane 9.3 cm above the CG:

| Motor | Position (FRD: +x fwd, +y right, +z down) | Spin |
|---|---|---|
| FL | `[+a, −a, −0.093]` | CW |
| FR | `[+a, +a, −0.093]` | CCW |
| RL | `[−a, −a, −0.093]` | CCW |
| RR | `[−a, +a, −0.093]` | CW |

Every motor satisfies `hypot(x, y) = 0.360 m`. The measured config keys
motors by position (`FL/FR/RL/RR`). In the compiled MJCF the same points
appear in the MuJoCo FLU body frame (`y` and `z` sign-flipped by
`coordinates/transforms.py`), e.g. FL = `[+a, +a, +0.093]` — see §H.1.

### Reference config — M-numbering (historical)

The reference config keys motors `M1..M4`, motor plane coplanar with the CG
(`z = 0`, an assumption for the pipeline-test model):

| Motor | Position (body/FRD, relative to CG) |
|---|---|
| M1 | `[+a, +a, 0]` |
| M2 | `[-a, +a, 0]` |
| M3 | `[-a, -a, 0]` |
| M4 | `[+a, -a, 0]` |

(Reference config: `a = 0.2606 m`, `R = 0.3685 m`.) Correspondence to the
measured vehicle's names: **M1 = FR, M2 = RR, M3 = RL, M4 = FL**.

## E. Motor spin directions

**Measured UGRP vehicle** (USER_REPORTED_OBSERVATION — a reported build
observation, not an instrument reading, and not yet checked against the PX4
motor assignment): FL CW, FR CCW, RL CCW, RR CW; in M-numbering that is
M1 (FR) CCW, M2 (RR) CW, M3 (RL) CCW, M4 (FL) CW.

**Reference config** (derived assumption, NOT verified against real wiring):
M1 CW, M2 CCW, M3 CW, M4 CCW (standard alternating X pattern).

The two are of **inverse polarity**; both keep diagonally opposed motors on
the same direction.

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

`J` in the torque law and the vehicle mass in the thrust law are the active
config's `Ixx/Iyy/Izz` and `mass_kg` (`IdentifiedWrenchActuation` reads them
from `params.mass_properties`). For the measured UGRP vehicle these
equations **cannot run yet**: the thrust saturation (`max_collective_thrust_n`),
the rate / thrust time constants and the actuator delay they need are still
pending (§H.1).

## G. Torque directions / inertia validation

**Reference config.** `landing_mujoco/tests/test_physical_validation.py
::InertiaResponseTest`: applying a known raw torque about each axis (no
identified-response layer involved) and reading `data.qacc` immediately
after `mj_forward` reproduces `alpha = tau / J_axis` to within `1e-5`,
confirming the MJCF mass/inertia wiring is correct (through
`MuJoCoDynamics`, so reference-config only).

**Measured UGRP vehicle.** Because `MuJoCoDynamics` cannot be built from the
measured config, `landing_mujoco/tests/test_measured_mjcf_injection.py`
validates the rigid-body MJCF directly
(`load_uav_params(validate=False) → build_mjcf → MjModel`), at force/torque
level with no propulsion model:

* compiled `body_mass / body_inertia / body_ipos / body_iquat` equal the YAML
  exactly (and the YAML read directly, bypassing the parser); the `<compiler>`
  carries no `balanceinertia / boundmass / boundinertia / inertiafromgeom`, so
  MuJoCo cannot have rewritten the tensor;
* `τ = 0.5 N·m` about each body axis gives `qacc = τ / I_axis` (rtol 1e-9);
* rolling the vehicle +90° and applying a world-z torque accelerates the
  **body y** rate by `τ / Iyy` — the inertia is bound to body axes;
* free fall gives `−g`; a hover-equivalent force `m·g` at the CG gives zero
  acceleration.

These prove the **wiring**, not that the real vehicle can produce `m·g`
(`max_collective_thrust_n` is unknown). Ground contact is exercised by the
landed-pose settle test and the ±5° tilted-contact tests in "Measured vehicle:
ground plane…", which use the measured skid footprint on top of the measured
mass, inertia and `ground_clearance_m` (no propulsion model involved).

**Frame note for the inertia tensor.** The builder writes `Ixx / Iyy / Izz`
into `<inertial diaginertia>` with **no FRD→FLU conversion**. That is correct
only because the tensor is diagonal: for `R = diag(1, −1, −1)`,
`R I Rᵀ = I`. Products of inertia would flip sign (`Ixy → −Ixy`,
`Ixz → −Ixz`, `Iyz → +Iyz`) and would need converting; the measured
off-diagonals are `ASSUMED_ZERO_FOR_V0` and are not passed to MuJoCo.

## H. Parameter tables

### H.1 MEASURED UGRP vehicle — `landing_mujoco/configs/ugrp_vehicle_measured.yaml`

`parameter_set: MEASURED_VEHICLE`. Status vocabulary (asserted by tests, so
promoting a tag is a visible change): **MEASURED** (physically measured on the
assembled vehicle) · **USER_CONFIRMED** (a property the user explicitly
confirmed) · **DERIVED** (computed from other recorded values) ·
**USER_PROVIDED_SPEC** (model designation / nameplate figure; no datasheet held
in this repository) · **USER_REPORTED_OBSERVATION** (a reported build
observation, weaker than MEASURED) · **ASSUMED_ZERO_FOR_V0** (modelling
assumption, *not* a measurement) · **UNKNOWN**.

| Parameter | Value | Unit | Status | Runtime owner |
|---|---|---|---|---|
| total mass | **6.408** | kg | MEASURED — full flight configuration | `body_mass[vehicle]` (MJCF `<inertial mass>`) |
| CG (body) | **[0, 0, 0]** | m | DERIVED (body origin = CG; raw CG is 0.241 m above ground) | `<inertial pos>` → `body_ipos` |
| Ixx (roll) | **0.153184** | kg·m² | MEASURED (bifilar) | `body_inertia[vehicle][0]` |
| Iyy (pitch) | **0.126285** | kg·m² | MEASURED (bifilar) | `body_inertia[vehicle][1]` |
| Izz (yaw) | **0.149050** | kg·m² | MEASURED (bifilar; 8.48 s trial excluded) | `body_inertia[vehicle][2]` |
| Ixy, Ixz, Iyz | **0, 0, 0** | kg·m² | **ASSUMED_ZERO_FOR_V0** — *not* a measured zero | not passed to MuJoCo (`diaginertia` only); `mass_properties.meta.inertia_products_kgm2` |
| radial distance R | 0.360 | m | MEASURED | `geometry.arm_length_m` → motor positions |
| arm angle | 45 | ° | USER_CONFIRMED (exact symmetric X) | `geometry.arm_angle_deg` |
| `a = R/√2` | 0.2545584412271571 | m | DERIVED | motor marker x/y |
| motor plane z | −0.093 | m | DERIVED (0.241 − 0.334) | motor marker z (FLU +0.093) |
| motor positions | §D table | m | DERIVED | `motor_*` / `arm_*` geoms — massless, collision-disabled |
| motor spin | FL/RR CW, FR/RL CCW | — | USER_REPORTED_OBSERVATION | none (not consumed) |
| battery centre z | +0.021 | m | DERIVED (0.241 − 0.220) | none (record-only) |
| ballast brick | ≈ 1.8 kg; ≈ 0.09 × 0.21 × 0.06 m (x, y, z as reported); centre ≈ +0.08 m in body +Z (≈ 8 cm below CG) | — | USER_PROVIDED_APPROXIMATE — **already inside** mass and inertia | none (`contributes_*_to_mujoco: false`) |
| **physical ground clearance** `ground_clearance_m` (CG → ground, landed pose) | **0.241** (ground plane at body z = **+0.241**) | m | MEASURED (raw CG height above ground); landed-pose datum USER_CONFIRMED 2026-09-19 | `geometry.ground_clearance_m` — PHYSICAL; fallback `r_landing^B = [0, 0, 0.241]`; creates no contact geom |
| gear sphere radius | 0.020 | m | SIMULATION representation (not measured, **not stored in the YAML**) | `LEG_CONTACT_RADIUS_M` in `mjcf_builder.py` |
| sphere-centre offset (body z) | 0.221 (= 0.241 − 0.020) | m | DERIVED — never stored | `landing_gear_sphere_centers_body_m()` in `mjcf_builder.py` |
| skid effective length | **0.300** | m | USER_CONFIRMED | `landing_gear_points_body_m` provenance (`meta`); skid direction = body X |
| skid centre-line spacing | **0.310** | m | USER_CONFIRMED | same |
| skid half length / half spacing | 0.150 / 0.155 | m | DERIVED (length / 2, spacing / 2) | same |
| **landing-gear contact points** (FRD) | LF [+0.150, −0.155, +0.241] · LR [−0.150, −0.155, +0.241] · RF [+0.150, +0.155, +0.241] · RR [−0.150, +0.155, +0.241] | m | **DERIVED_FROM_MEASURED_DIMENSIONS + CENTERED_SYMMETRY_ASSUMPTION** — *not* four measured coordinates | `geometry.landing_gear_points_body_m` → `leg_0..leg_3` contact spheres (centre z derived to 0.221) |
| gear-point reading | `physical_contact` | — | USER_CONFIRMED (modelling decision) | `geometry.landing_gear_points_semantics` |
| hover thrust `m·g` | 62.8401 (g = 9.8065) / 62.8410 (g = 9.80665) | N | DERIVED — **not** a maximum | none; ≈ 15.71 N ≈ 1.602 kgf per motor **assuming equal sharing** |
| components | motor T-MOTOR MN501-S IP45 KV360 · ESC HOBBYWING Skywalker V2 60A · battery Poly-Tronics 6S1P 10000 mAh 22.2 V 75C+ XT90-S · propeller T-MOTOR MS1704 | — | USER_PROVIDED_SPEC | none |

**Compiled `mjModel` (measured config, `physics_dt` 0.002, gravity
`[0, 0, −9.80665]`)** — read back from `MjModel.from_xml_string(build_mjcf(…))`:

| Field | Value |
|---|---|
| `body_mass[vehicle]` | 6.408 |
| `body_inertia[vehicle]` | [0.153184, 0.126285, 0.14905] |
| `body_ipos[vehicle]` | [0, 0, 0] |
| `body_iquat[vehicle]` | [1, 0, 0, 0] |
| `motor_FL` `geom_pos` (FLU) | [+0.2545584412271571, +0.2545584412271571, +0.093] |
| `motor_FR` `geom_pos` (FLU) | [+0.2545584412271571, −0.2545584412271571, +0.093] |
| `motor_RL` `geom_pos` (FLU) | [−0.2545584412271571, +0.2545584412271571, +0.093] |
| `motor_RR` `geom_pos` (FLU) | [−0.2545584412271571, −0.2545584412271571, +0.093] |
| motor radius (all four) | 0.360 m (machine precision); converting back to FRD recovers §D exactly |
| `leg_0` … `leg_3` `geom_pos` (FLU), radius 0.020 | [+0.15, +0.155, −0.221], [−0.15, +0.155, −0.221], [+0.15, −0.155, −0.221], [−0.15, −0.155, −0.221] (bottom at z = −0.241) |
| contact-enabled geoms | `ground`, `leg_0` … `leg_3` (markers and shell are collision-disabled) |

**Inertia measurement** (bifilar suspension, **manual stopwatch** periods,
mass used 6.408 kg; full raw periods live in `mass_properties.meta.inertia_kgm2`):

| Axis | mean | sample std | n | bifilar D / L | T_mean (single period) |
|---|---|---|---|---|---|
| Roll — Ixx | 0.153184 kg·m² | 0.004362 | 10 | 0.090 m / 0.427 m | 4.50433 s |
| Pitch — Iyy | 0.126285 kg·m² | 0.003041 | 10 | 0.185 m / 0.424 m | 1.98267 s |
| Yaw — Izz | 0.149050 kg·m² | 0.001698 | 9 | 0.163 m / 0.505 m | 2.66815 s |

* Yaw: the **8.48 s** trial was judged an outlier candidate and **excluded**;
  the nominal Izz is the outlier-excluded value. Including it would give
  0.150874 kg·m² (std 0.005984, n = 10). The excluded trial is preserved in the
  YAML, not deleted.
* **R² = N/A.** These are stopwatch period measurements; no damped-sinusoid
  regression exists and none was invented. An R² can only be added if a
  HELIX angle–time trajectory `θ(t) = A e^{−λt} cos(ω_d t + φ) + θ₀` is fitted.
* Reduction (reproduced from the raw periods, not merely quoted):
  `I = m g D² T² / (16 π² L)` per trial, then the **mean of per-trial I**.
  This reproduces every reported mean / std at **g = 9.8065**; the user did
  not state g, so 9.8065 is an *inference*. The canonical values are the
  user's literals, not recomputed.
* The standard deviations are **recorded only**; they are not wired into any
  domain randomization.
* Observation (not an explanation): Ixx/Iyy ≈ 1.21 on a nominally symmetric
  X-frame. The brick's footprint points the same way if x/y/z are body axes,
  but nothing here demonstrates that it accounts for the difference; the values
  are not adjusted toward symmetry.

**Pending — propulsion calibration** (UNKNOWN / NOT MEASURED / NOT IDENTIFIED;
none of these is part of the measured physical model):

* required by the runtime and currently `null` (they are exactly what
  `check_parameter_set --parameter-set ugrp_vehicle_measured` lists):
  `max_collective_thrust_n`, `tau_roll_s`, `tau_pitch_s`, `tau_thrust_s`,
  `actuator_delay_s` (`identified_response.delay_s`);
* also unknown: individual motor max thrust for the MN501-S + MS1704 pair,
  `tau_yaw_s`, `K_*` (default to unity gain — a neutral default, not an
  identified value), and — only if a rotor-level backend is ever built —
  `k_f`, `k_m`, motor time constant, ESC delay, propeller aerodynamics;
* exact per-leg offset of the skid footprint from the CG (centred symmetry is an *assumption*), and a continuous-skid contact model (see "Measured vehicle: ground plane…");
* Ixy / Ixz / Iyz (assumed zero, not measured).

State: **physical-inertia injection complete; propulsion calibration
pending.**

### H.2 Reference / historical configuration — `landing_mujoco/configs/tarot680b_reference.yaml`

`parameter_set: PROVISIONAL_REFERENCE` — a **different, unmeasured airframe**
(Tarot TL680B). **None of these values describes the UGRP vehicle.** It is
kept to exercise the software pipeline (coordinate transforms, actuator
plumbing, contact, observation contract) and is the config the existing
regression tests run on. Every non-informational field is tagged
`replace_before_real_training: true` in its YAML.

| Parameter | Value | Unit | Category | Used in |
|---|---|---|---|---|
| `mass_kg` | 3.57 | kg | 2. MANUFACTURER SPEC | mass_properties, MJCF `<inertial>` |
| `cg_body_m` | [0,0,0] | m | 5. ASSUMED | body-frame origin convention |
| `Ixx` | 0.058285 | kg·m² | 4. DERIVED (parallel-axis, provisional) | MJCF `<inertial>` |
| `Iyy` | 0.058285 | kg·m² | 4. DERIVED (provisional) | MJCF `<inertial>` |
| `Izz` | 0.056498 | kg·m² | 4. DERIVED (provisional) | MJCF `<inertial>` |
| `wheelbase_m` | 0.737 | m | 2. MANUFACTURER SPEC | motor-position derivation |
| `arm_length_m` (R) | 0.3685 | m | 4. DERIVED (wheelbase / 2) | radial distance |
| motor positions (a) | 0.2606 | m | 4. DERIVED (from wheelbase, symmetric-X assumption) | MJCF geoms, inertia estimate |
| `ground_clearance_m` | 0.12 | m | 5. ASSUMED (SYNTHETIC_TEST) | landing-gear geometry. Legacy `geom_center` gear-point z (a sphere centre): the actual CG-to-ground distance is 0.14 m |
| `T_max` | 70.02 | N | 5. ASSUMED (SYNTHETIC_TEST, T/W=2.0) | thrust saturation |
| `T_max/(mg)` | 2.00 | - | 4. DERIVED | thrust-margin check |
| `tau_roll_s` | 0.15 | s | 5. ASSUMED (SYNTHETIC_TEST) | identified rate response |
| `tau_pitch_s` | 0.15 | s | 5. ASSUMED (SYNTHETIC_TEST) | identified rate response |
| `tau_thrust_s` | 0.20 | s | 5. ASSUMED (SYNTHETIC_TEST) | identified thrust response |
| `tau_yaw_s` | 0.30 | s | 5. ASSUMED (SYNTHETIC_TEST) | identified yaw response |
| `T_delay` (`delay_s`) | 0.02 | s | 5. ASSUMED (SYNTHETIC_TEST) | actuator delay buffers |
| `K_roll`,`K_pitch`,`K_thrust`,`K_yaw` | 1.0 | - | 5. ASSUMED (SYNTHETIC_TEST, unity gain) | identified response |

Categories, per task spec: **1. DIRECTLY MEASURED** (none in this reference
config — the measured vehicle's values are in H.1), **2. MANUFACTURER
SPECIFICATION** (Tarot TL680B published figures), **3. SYSTEM IDENTIFIED**
(none exist yet for either configuration), **4. DERIVED** (computed from
another value in the table, e.g. inertia from geometry+mass, motor position
from wheelbase), **5. ASSUMED** (no measurement or derivation basis at all —
synthetic or symmetry placeholders).

---

## Landing-gear standoff: `pad_target_ned` vs. `vehicle_touchdown_target_ned` (RESOLVED)

*Scope: the task-semantics fix described here is config-independent, but every
number in this section (0.12 m standoff, 0.02 m `z_error`, the 8 tests) comes
from the **reference** config's synthetic gear, read with the legacy
`geom_center` semantics. The measured UGRP vehicle reads its gear points as
physical contact points (CG-to-ground 0.241 m, four skid-endpoint points derived
from the measured skid dimensions) — see "Measured vehicle: ground plane…"
below.*

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
with `ground_contact=True` and `success=True` (the 0.02 m is the leg contact
sphere's radius, `LEG_CONTACT_RADIUS_M`: under the reference config's legacy
`geom_center` reading the gear *point* is the sphere centre, so the CG rests
0.02 m above `gear_point_z`; the measured vehicle's `physical_contact` reading
removes that offset — see the next section); doubling/halving the gear
standoff changes the required CG resting altitude by the corresponding
amount while `success_altitude_m` stays literally `0.05` in both cases; a
tilted vehicle at the exact target z shows `altitude_agl` and `z_error`
diverging as expected; the legacy `landing_rl.envs.landing_env.LandingEnv`
is confirmed unchanged (CG still reaches `ground_z_m` exactly on contact).

---

## Measured vehicle: ground plane, `ground_clearance_m` and the landing reference point

Status: **implemented 2026-09-19** on the user's decision. The vehicle's
*physical* contact geometry and its *MuJoCo representation* are separate
quantities with separate owners. The skid-type gear footprint is recorded as
four physical contact points derived from the confirmed skid dimensions (see
"Landing-gear skid footprint").

### Three different quantities — never interchange them

| Quantity | Value | Kind | Owner |
|---|---|---|---|
| **physical ground clearance** `ground_clearance_m` | **0.241 m** | PHYSICAL (MEASURED): vertical CG → ground plane in the normal landed pose; the ground plane sits at body z = **+0.241** (FRD) | `ugrp_vehicle_measured.yaml` → `geometry.ground_clearance_m` |
| **gear sphere radius** | **0.020 m** | SIMULATION representation (not a measured vehicle parameter) | `LEG_CONTACT_RADIUS_M` in `mjcf_builder.py`; **not** in any physical YAML |
| **derived sphere-centre offset** | **0.221 m** (= 0.241 − 0.020) | DERIVED — never stored | `landing_gear_sphere_centers_body_m()` in `mjcf_builder.py` |

**0.221 m is not the physical ground clearance.** It must never be written to
`ground_clearance_m` or to a physical contact point; the YAML test suite
checks that no numeric leaf of the measured YAML equals 0.221 **or 0.020**.
Both are intended: 0.221 is a derived simulation value, and the 0.020 m radius
is owned by the builder, not by the physical YAML. A provenance note that needs
to mention them must write them in prose (a string), not as a YAML number;
relaxing this check should be a deliberate decision, not a convenience.

### Parameter ownership

```
PHYSICAL (config)                          SIMULATION REPRESENTATION (builder)
  ground_clearance_m = 0.241                 gear sphere radius        = 0.020
  landing contact point z = +0.241           gear sphere centre z (FRD) = 0.221
  (each contact point, level landing;          = contact point z - radius
   x/y derived from skid length/spacing)
```

Landed pose, world frame (ground plane at z = 0):

```
CG world z              = 0.241 m
sphere centre world z   = 0.020 m
sphere bottom world z   = 0
```

### Landing-gear skid footprint

**Physical gear:** two continuous, parallel, helicopter-style skid bars; their
long direction is parallel to the roll axis, i.e. **body X**.
**MuJoCo v0 approximation:** four representative contact points at the skid
**endpoints**, each a contact sphere — a deliberate simplification, **not** a
continuous line-contact model.

| Quantity | Value | Unit | Status | Owner |
|---|---|---|---|---|
| CG → ground (vertical) | 0.241 | m | MEASURED | `geometry.ground_clearance_m` |
| skid effective length | 0.300 | m | USER_CONFIRMED | `meta` of `landing_gear_points_body_m` |
| skid centre-line spacing | 0.310 | m | USER_CONFIRMED | same |
| half length (x) | 0.150 | m | DERIVED (length / 2) | same; the x of the points |
| half spacing (y) | 0.155 | m | DERIVED (spacing / 2) | same; the y of the points |
| footprint centred on the CG in x and y | — | — | **MODELLING ASSUMPTION** (`CENTERED_SYMMETRY_ASSUMPTION`) | same |
| contact sphere radius | 0.020 | m | SIMULATION representation | `LEG_CONTACT_RADIUS_M` (builder) |
| sphere-centre z | 0.221 | m | DERIVED (0.241 − 0.020), never stored | `landing_gear_sphere_centers_body_m()` |

Physical contact points, FRD body frame (+Y right, so the left skid is
negative y); `landing_gear_points_semantics: physical_contact`:

| Index / geom | Point | FRD contact point [m] | Compiled sphere centre, MuJoCo FLU [m] |
|---|---|---|---|
| 0 / `leg_0` | left-front | [+0.150, −0.155, +0.241] | [+0.150, +0.155, −0.221] |
| 1 / `leg_1` | left-rear | [−0.150, −0.155, +0.241] | [−0.150, +0.155, −0.221] |
| 2 / `leg_2` | right-front | [+0.150, +0.155, +0.241] | [+0.150, −0.155, −0.221] |
| 3 / `leg_3` | right-rear | [−0.150, +0.155, +0.241] | [−0.150, −0.155, −0.221] |

**Provenance wording — these four coordinates are not "fully measured point
coordinates".** What is known: skid length, skid spacing, ground clearance and
the skid direction. What is *derived*: `x = ±length/2`, `y = ±spacing/2`,
`z = ground_clearance_m`. What is *assumed*: that the footprint is centred and
symmetric about the CG in x and y — the actual per-leg offset from the CG was
not measured. Provenance tag: `DERIVED_FROM_MEASURED_DIMENSIONS` +
`CENTERED_SYMMETRY_ASSUMPTION`.

**Representation limitation.** Two continuous skid bars → four endpoint
spheres. Between the endpoints there is no contact, so the model cannot represent
a skid lying along the ground over its length, nor contact at an intermediate
point. If needed later, a continuous skid could be modelled with a capsule or
cylinder geom; that is **not** done here.

### What the code does (source-verified)

* `Geometry.landing_gear_points_semantics` (`param_schema.py`) says how
  `landing_gear_points_body_m` is read:
  * `physical_contact` — the points are where the real gear touches the ground.
    `landing_gear_sphere_centers_body_m()` derives each sphere centre one radius
    above the contact point, along body z: `centre_z(FRD) = point_z − radius`;
    x/y are untouched. **`ugrp_vehicle_measured.yaml` declares this** (explicitly,
    so a future measured point set cannot silently inherit the legacy reading).
  * `geom_center` (default when the field is absent) — the legacy reading: each
    point *is* the sphere centre. **`tarot680b_reference.yaml` keeps this**, so
    its MJCF is byte-identical to before and its CG still rests one radius
    *above* its declared 0.12 m gear point (0.14 m); see the standoff section.
* `build_mjcf()` builds the collision spheres from the derived centres. It does
  not modify the stored points.
* With **no** per-leg points the builder creates **no** contact geoms — it does
  not invent any from `ground_clearance_m`.
* `landing_reference_point_body_m()` (logic unchanged) returns the mean of the
  stored points — for `physical_contact` that is the physical touchdown point,
  so `altitude_agl` reaches 0 at true contact — and otherwise falls back to
  `[0, 0, ground_clearance_m]`.

### What `ground_clearance_m` means in this codebase

`Geometry.ground_clearance_m`: a positive scalar, measured vertically from the
body origin (CG) **down** to the ground / landing-contact plane in the landed
pose. It is **not** the conventional "underside of the airframe to the ground"
clearance. Its only reader is the fallback branch of
`landing_reference_point_body_m()`.

**The reference config does not satisfy this definition.** Its YAML comment
claims the same wording ("distance from CG down to the lowest contact point when
resting on the pad"), but its `ground_clearance_m = 0.12` is the legacy
`geom_center` gear-point z — i.e. the sphere *centre* — so its actual
CG-to-ground distance is **0.14 m** (0.12 + the 0.02 m radius; measured this
session). The field is synthetic (`SYNTHETIC_TEST`) and unread there (gear
points exist, so the fallback branch never fires), so it was left untouched. The
two configs therefore mean different things by the same field name, which is
exactly why `landing_gear_points_semantics` is an explicit tag.

### What `landing_reference_point_body_m()` requires — and what is now solved

It needs either non-empty `landing_gear_points_body_m` (x/y from the points) or
`ground_clearance_m` (then `r_landing^B = [0, 0, ground_clearance_m]` with x/y
**assumed centred under the CG**); otherwise it raises `ValueError`.

* **Solved:** the measured config returns `[0, 0, 0.241]` (the mean of the four
  physical contact points — by the centred symmetry also equal to the
  `ground_clearance_m` fallback). The vertical datum is fully resolved and the
  contact spheres now exist.
* **Assumption, not measurement:** the x/y of that reference point (0, 0) and of
  the four points come from the centred-symmetry assumption. The real per-leg
  offset from the CG was not measured. The points were derived from the skid
  dimensions, **not** inferred from the motor positions (the reference config's
  "one leg under each motor arm" is an assumption about a different airframe).
* `MuJoCoDynamics` / `MujocoLandingEnv` for the measured config therefore no
  longer stop at the reference point; they stop at `IdentifiedWrenchActuation`
  because `max_collective_thrust_n`, `tau_*` and `actuator_delay_s` are still
  `null` (and `load_uav_params()` still raises `MissingMeasurementError`).
* `ground_clearance_m` **alone creates no contact geoms** (leg geoms are built
  from `landing_gear_points_body_m` only). The measured MJCF has contact spheres
  because the four points are now present.

### Dynamic verification (`tests/test_measured_mjcf_injection.py`)

**Landed pose** (`LandedPoseSettleTest`). Measured mass 6.408 kg and inertia
0.153184 / 0.126285 / 0.149050 kg·m²; the canonical skid footprint from the
YAML; ground plane at world z = 0; gravity on; ground friction as the
environment applies it (`LandingConfig.ground_friction_xy = 0.55`); vehicle
released level, from rest, 0.5 m up, no thrust; 3 s at `physics_dt` 0.002
(settling takes ≈ 0.5 s).

| Quantity | Result |
|---|---|
| final CG world z | **0.24098 m** (error vs 0.241: **−0.022 mm**) |
| sphere centre world z | 0.01998 m (CG − centre = 0.221 m) |
| lowest sphere bottom world z | **−0.0215 mm** (= the penetration) |
| contacts / residual speed | 4 / ~1e-15 |

The landing reference point (CG − 0.241 m) reaches the ground (`altitude_agl` ≈ 0);
the result is unchanged for `physics_dt` ∈ {0.001, 0.002, 0.005} and drop heights
{0.3, 0.5} m. **Negative control:** reading 0.241 as the sphere *centre* (the
legacy `geom_center` reading, same points) rests the CG at 0.2610 m, **+20 mm**
too high — the trap the physical/simulation separation removes.

**Tolerance ±2 mm — basis (measured, not assumed).** MuJoCo contacts are soft,
so the settled sphere sinks slightly into the plane. At fixed friction the
sinking is linear in load per contact (`n_legs × penetration` is constant:
4 canonical legs vs a 3-leg synthetic layout agree to < 1 %). It also depends
on the friction coefficient (MuJoCo's pyramidal-cone regularisation):

| ground friction μ | 0.3 | **0.55 (environment)** | 0.9 (builder default) | 1.2 |
|---|---|---|---|---|
| settled penetration, 4 legs [mm] | 0.0054 | **0.0215** | 0.0795 | 0.184 |
| one contact carrying all 62.8 N [mm] | 0.021 | 0.086 | 0.32 | 0.74 |

The worst case in the table (0.74 mm) is < ½ of the ±2 mm tolerance, and
±2 mm is ~10× below the 20 mm error the test must catch. *(Correction: the
earlier version of this section quoted 0.079 / 0.105 mm and "≈ 5 µm/N"; those
were measured at the builder's default μ = 0.9, not at the environment's 0.55.)*
No contact stiffness or friction was tuned.

**Tilted contact** (`TiltedContactTest`). Attitude via the repository helper
(FRD: positive roll = right side down, positive pitch = nose up). Released at
rest, tilted 5°, lowest sphere bottom 20 mm above the ground (impact ≈ 0.63 m/s),
3 s. Level control from the same lift: peak penetration 4.32 mm.

| roll / pitch | first-contact skids | net contact torque at first contact (FRD, N·m) | peak penetration | final |
|---|---|---|---|---|
| +5° / 0 | right (`leg_2`,`leg_3`) | τx = −15.65 (restoring) | 4.31 mm | level, CG 0.24098, slide 2.2 cm |
| −5° / 0 | left (`leg_0`,`leg_1`) | τx = +15.65 (restoring) | 4.31 mm | same |
| 0 / +5° | rear (`leg_1`,`leg_3`) | τy = −13.19 (restoring) | 4.31 mm | same |
| 0 / −5° | front (`leg_0`,`leg_2`) | τy = +13.19 (restoring) | 4.31 mm | same |
| +5° / +5° | right-rear (`leg_3`) | τx = −20.06, τy = −17.92 (both restoring) | 7.79 mm | level, CG 0.24098, yaw −0.50°, slide 0.8 cm |

In every case: no NaN, no solver warning, peak body rate ≤ 1.91 rad/s, the
low-side skids touch first (matching an independent prediction from the
repository's NED rotation), the contact force points up (+322 … +398 N at first
contact), the torque opposes the tilt, the sphere never crosses the plane
(peak penetration ≤ 7.8 mm < the 20 mm radius; test bound 12 mm), and the
vehicle settles level at the physical clearance with all four bottoms within
0.03 mm of the ground. The 2 cm slide is friction acting during the righting
transient (not tuned). For reference, the static tip-over angle of the
footprint is atan(0.155/0.241) = 32.7° in roll and atan(0.150/0.241) = 31.9° in
pitch (derived, not tested).

### Still unknown / not done

* **The exact per-leg offset of the footprint from the CG is not measured.** The
  centred, symmetric footprint is a v0 *assumption*; a real offset would move
  the CG relative to the support polygon (tip-over margins, touchdown attitude).
* **Continuous skids are approximated by four endpoint spheres** (see "Landing-
  gear skid footprint"); no line contact, no capsule / cylinder geom.
* `physical_contact` points assume a **level** landing and a **vertical** foot
  axis (the sphere sits directly above the contact point along body z).
* The 0.020 m radius is a simulation choice, not measured skid geometry; a real
  skid bar is not a sphere.
* Contact behaviour (stiffness, friction, restitution) is MuJoCo's default plus
  the environment's `ground_friction_xy`; it was **not** tuned or validated
  against real touchdowns.

---

## x500 visual shell (visualization only)

The MuJoCo viewer can show the PX4/Gazebo **x500** airframe (frame, motor
bases, motor bells, propellers) instead of the simple placeholder geoms. It
is **purely cosmetic**:

* derived assets: `landing_mujoco/assets/x500_visual/` (BSD-3-Clause, see
  its `LICENSE`, `THIRD_PARTY_NOTICES.md`, `README.md`,
  `conversion_report.json`); upstream source (not kept in this repository):
  `https://github.com/PX4/PX4-gazebo-models.git` @
  `d754381a1cecdd7f17050acd72bf5bf1327bced6`, `models/x500_base/` — the
  commit PX4-Autopilot `85df8c2281c2466b30a121b22b0bf33dc69bcfe4` pins as
  `Tools/simulation/gz`; regenerate with
  `landing_mujoco/tools/convert_x500_visual_assets.py --source <checkout>`
  (conversion-only environment; inputs verified against pinned sha256);
* config: `landing_mujoco/configs/x500_visualization.yaml` (separate from the
  physical parameter YAML; nothing in it is a physical parameter);
* runtime: `landing_mujoco/visualization/x500_shell.py`, enabled by passing
  `visualization=load_visualization_config(...)` to `MujocoLandingEnv` /
  `MuJoCoDynamics` (default `None` = physics-only MJCF, byte-identical to
  before the shell existed);
* viewer: `landing_mujoco/tools/view_x500_visual.py` (interactive or
  `--offscreen` PNG renders).

Every shell geom is `contype=0 conaffinity=0`, geom group 2, attached to a
body with an explicit `<inertial>`; no body, joint, actuator, option, or
collision geom is added or changed. `tests/test_visual_physics_invariance.py`
checks exact equality of mass/inertia/DoF/actuator/option/physics-geom data
and ON vs OFF trajectories (free fall, hover, known body torque,
landing/contact, deterministic and seeded-stochastic env episodes) with
`rtol=0, atol=1e-12`; they are currently bit-identical.

Transform layers are kept separate (and mirrored as nested compile-time
MJCF `<frame>` elements):

```
converted mesh (COLLADA node transforms baked by the converter)
  -> x500 SDF link pose + visual pose + mesh scale   = native x500 visual
  -> UGRP display alignment: p_body = t + R (s * p_native)
```

`uniform_scale: auto` = `target_wheelbase_m / source_wheelbase_m`
= 0.737 / 0.49214631970583705 = **1.4975221199266826**, applied consistently
to mesh dimensions and all component offsets; `translation_body_m` and
`rotation_rpy_rad` (FRD body frame) default to zero.

**Reference-derived target — not the measured vehicle.** `target_wheelbase_m:
0.737` in `x500_visualization.yaml` is a numeric literal (origin `config`): the
provisional Tarot TL680B visual target. The shell is therefore scaled to
0.737 m **also when the measured config is loaded** (checked: the resolved
`uniform_scale` is 1.4975221199 for both configs). The measured opposite-motor
span is **0.720 m** (max pairwise horizontal distance of
`motors.positions_body_m`); `target_wheelbase_m: from_physical_params` would
give 0.720 / 0.49214631970583705 = 1.46298, i.e. the shell renders ≈ 2.4 %
larger than the measured airframe. Visual only; not changed here (no YAML edit
in this pass).

**Landing gear: split shell (decision 2026-09-14).** *(All gear numbers in this
paragraph — 0.341 m, 0.140 m = 0.12 m gear point + 0.02 m leg radius — are
reference-config figures. For the measured config the viewer shows the four
skid-endpoint contact spheres (geom group 0, radius 0.020 m) instead; the x500
skids themselves are not drawn, and this trade-off would have to be recomputed
for the measured footprint.)* Uniform scaling matches
the wheelbase exactly (visual rotor hubs within 4.4e-5 m of the physical
motor positions horizontally), but the scaled x500 landing gear would reach
0.341 m below the body origin while the physical contact bottom is 0.140 m
below it, i.e. the x500 skids would render ~0.20 m below the ground at rest.
The shell therefore contains only the x500 **upper structure**; the landing
gear shown in the viewer is the **physical contact geometry** (geom group
0), so what you see touching the ground is exactly what MuJoCo collides.
The converter excludes whole connected landing-gear pieces (all `Landing*`
COLLADA components plus the carbon-fiber leg and skid tubes: pieces reaching
below −0.10 m in the native DAE frame; margins −0.072 m / −0.206 m); no
triangle is modified. With the shell enabled, the placeholder markers (body
box, arm capsules, motor spheres; all collision-inert) move to hidden geom
group 3 — the only change to non-shell MJCF elements, and it affects viewer
visibility only.

---

## Software structure

```
landing_mujoco/
    configs/        param_schema.py, simulation_config.py,
                    ugrp_vehicle_measured.yaml   (MEASURED_VEHICLE: the real UGRP UAV),
                    tarot680b_reference.yaml     (PROVISIONAL_REFERENCE: a different airframe),
                    visualization_config.py, x500_visualization.yaml   (visual-only)
    coordinates/    transforms.py
    dynamics/       mjcf_builder.py, inertia_estimation.py, delay_buffer.py,
                    identified_inner_loop.py, actuation_model.py,
                    mujoco_dynamics.py, contact.py
    envs/           mujoco_landing_env.py
    evaluation/     compare_old_vs_mujoco.py
    visualization/  x500_shell.py                     (visual-only)
    assets/         x500_visual/                      (derived visual bundle; upstream not vendored)
    tools/          convert_x500_visual_assets.py (conversion-only env),
                    view_x500_visual.py
    tests/          _helpers.py (loads the REFERENCE config),
                    test_coordinate_transforms.py, test_sign_conventions.py,
                    test_param_schema.py, test_physical_validation.py,
                    test_identified_response.py, test_ground_contact.py,
                    test_landing_reference_point.py, test_reset_determinism.py,
                    test_observation_action_contract.py,
                    test_ugrp_measured_params.py     (measured YAML: values, provenance, inertia),
                    test_measured_mjcf_injection.py  (measured YAML -> compiled mjModel),
                    test_visualization_config.py, test_x500_visual_assets.py,
                    test_x500_visual_shell.py, test_visual_physics_invariance.py
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
python3 -m landing_mujoco.check_parameter_set --parameter-set ugrp_vehicle_measured   # fails fast (exit 1), see below
python3 landing_mujoco/evaluation/compare_old_vs_mujoco.py
python3 landing_mujoco/tools/view_x500_visual.py                       # interactive viewer, x500 shell
MUJOCO_GL=osmesa python3 landing_mujoco/tools/view_x500_visual.py --offscreen out/   # headless renders
```

For the measured vehicle the fail-fast output currently lists exactly
`max_collective_thrust_n`, `tau_roll_s`, `tau_pitch_s`, `tau_thrust_s`,
`actuator_delay_s` — the propulsion / closed-loop fields still pending. Mass,
CG, inertia and motor geometry no longer appear in that list. The measured
rigid-body MJCF can still be compiled and inspected without a propulsion model
(this is how the values in §H.1 were read back):

```python
import mujoco
from landing_mujoco.configs.param_schema import load_uav_params
from landing_mujoco.dynamics.mjcf_builder import build_mjcf

params = load_uav_params("landing_mujoco/configs/ugrp_vehicle_measured.yaml", validate=False)
model = mujoco.MjModel.from_xml_string(build_mjcf(params, physics_dt=0.002))
```

`validate=False` is for inspection only; it does not make the config runnable:
`MuJoCoDynamics` / `MujocoLandingEnv` still need the pending propulsion fields
(the landing reference point and the four skid contact spheres are already in
place).

## Explicitly NOT done in this pass (see CLAUDE.md / task spec restrictions)

* No PPO training or retraining.
* No rotor-level actuation (`RotorLevelActuation` raises `NotImplementedError`).
* No ground effect tuning (`ground_effect_gain = 0.0`, inherited unchanged
  from `LandingConfig` — effectively disabled, matching "ground_effect_enabled
  = False" for this phase).
* No propulsion / closed-loop calibration for the measured UGRP vehicle
  (`max_collective_thrust_n`, `tau_roll/pitch/thrust`, `actuator_delay_s`, and
  `k_f` / `k_m` / motor time constant if ever needed) — see §H.1. The skid gear
  is a four-endpoint-sphere approximation of two continuous bars, with an
  assumed (not measured) centring on the CG. The measured mass, CG, inertia and motor
  geometry ARE recorded and reach the compiled model; every
  non-informational value in `tarot680b_reference.yaml` remains
  manufacturer/derived/synthetic and tagged `replace_before_real_training:
  true`.
* All 8 comparison scenarios (hover, north/east velocity, climb, descent,
  roll/pitch transient, landing approach) are implemented in
  `compare_old_vs_mujoco.py`, each reporting position/velocity/acceleration/
  attitude/body-rates/thrust/command traces plus rise time/settling time/
  overshoot/peak tilt/peak accel/steady-state speed/touchdown speed.
* `compare_flightlog_vs_mujoco.py` (real `.ulg` vs MuJoCo) is not built yet
  — deferred until a real UGRP flight log exists (the only ULogs in this
  repo are tagged `OTHER_PROJECT_SAMPLE` / `PIPELINE_VALIDATION_ONLY`, per
  CLAUDE.md §0/§19 they may not be used for this).
