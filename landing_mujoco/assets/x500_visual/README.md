# x500 visual shell (visualization only)

Derived, MuJoCo-loadable visual assets for the upper structure of the PX4
Gazebo **x500** airframe: frame, motor bases, motor bells and propellers. Their
only purpose is to make the MuJoCo viewer look like a quadrotor.

**These assets have no physical meaning.** Mass, CG, inertia, motor
positions, landing-gear contact geometry, thrust, torque, contact, the
controller, and the RL observation/action contract all come exclusively from
the `landing_mujoco/configs/*.yaml` physical parameters. Every geom created
from this bundle is `contype="0" conaffinity="0"` (collision-inert), sits in
geom group 2, and is attached to a body that already has an explicit
`<inertial>`, so it contributes no mass or inertia. Visual ON and visual OFF
are tested to give identical physics
(`landing_mujoco/tests/test_visual_physics_invariance.py`; currently
bit-identical).

## Landing gear

The x500 landing gear is deliberately **not** in this bundle. Uniformly
scaling the x500 to the vehicle's wheelbase makes its legs much longer than
the vehicle's physical landing gear, so in the viewer the landing gear is
shown by the **physical contact geoms** (geom group 0). With the shell
enabled, the simple placeholder markers (body box, arm capsules, motor
spheres) move to hidden group 3.

Whole connected mesh pieces are classified by a recorded rule: a
`Landing*` COLLADA component, or reaching below −0.10 m in the native DAE
frame. See `landing_gear_partition` in `conversion_report.json`.

## Provenance

- Upstream: `https://github.com/PX4/PX4-gazebo-models.git` @
  `d754381a1cecdd7f17050acd72bf5bf1327bced6`, path `models/x500_base/`.
- PX4-Autopilot `85df8c2281c2466b30a121b22b0bf33dc69bcfe4` pins that commit
  as `Tools/simulation/gz`.
- License: BSD-3-Clause, Rudis Laboratories. See `LICENSE` and
  `THIRD_PARTY_NOTICES.md`.

The repository keeps no copy of the upstream sources; they are reproducible
from the pinned commit.

## Contents

| Path | What |
|---|---|
| `meshes/frame_*.obj` | 6 frame upper-structure components (carbon-fiber plates/arms, metal, FMU, FMU rubber, rails rubber, antenna holder) |
| `meshes/motor_base_5010_stator.obj`, `meshes/motor_bell_5010_{side,head}.obj` | Motor visuals |
| `meshes/prop_1345_{ccw,cw}.stl` | Propeller visuals (byte copies) |
| `textures/cf.png` | Carbon-fiber texture, used only by `frame_carbon_fiber_upper` |
| `x500_native_assembly.json` | Native x500 visual assembly: per-instance SDF link pose, visual pose, mesh scale, component materials |
| `conversion_report.json` | Provenance and source verification, tool versions, precision, acceptance gates, landing-gear partition, dropped primitives, known tolerances, all source/output sha256 |
| `LICENSE`, `THIRD_PARTY_NOTICES.md` | Upstream license and attribution |

## Transform layers

Kept explicitly separate:

```
source mesh coordinates
  -> COLLADA node transforms            (baked into the OBJ files by the converter)
  -> x500 SDF link/visual pose + mesh scale   (x500_native_assembly.json)
  = native x500 visual                  (x500_base model frame, FLU, meters)
  -> UGRP display alignment             (landing_mujoco/configs/x500_visualization.yaml)
  = MuJoCo vehicle body frame
```

`uniform_scale: auto` = `target_wheelbase_m / source_wheelbase_m`. The source
wheelbase comes from the rotor link poses in the manifest
(2·√(0.174² + 0.174²) m). Changing the target is a YAML edit only.

## Material provenance

- Frame and motor colors: exact COLLADA effect diffuse colors.
- Carbon fiber: `textures/cf.png` via the source UV set.
- Propellers: the SDF references the OGRE script material `Gazebo/DarkGrey`,
  which is not part of PX4-gazebo-models. The RGBA `0.175 0.175 0.175 1` is
  an **assumption**.

## Regenerating

Conversion needs `pycollada`, `trimesh` and `Pillow`. These are
**conversion-only** packages, never runtime or training dependencies:

```bash
git clone https://github.com/PX4/PX4-gazebo-models.git /tmp/PX4-gazebo-models
git -C /tmp/PX4-gazebo-models checkout d754381a1cecdd7f17050acd72bf5bf1327bced6
# (or use an existing PX4-Autopilot@85df8c22.../Tools/simulation/gz checkout)

python3 -m venv /tmp/x500_mesh_convert
source /tmp/x500_mesh_convert/bin/activate
pip install trimesh pycollada Pillow
python3 landing_mujoco/tools/convert_x500_visual_assets.py --source /tmp/PX4-gazebo-models
deactivate
```

The script aborts if any input differs from its pinned sha256. It
re-validates every output against the source per face corner, exits non-zero
if any acceptance gate fails, and regenerates `meshes/`, `textures/`,
`LICENSE`, `x500_native_assembly.json` and `conversion_report.json`
deterministically. This README and `THIRD_PARTY_NOTICES.md` are hand-written
and never overwritten.
