# Third-party notices — `landing_mujoco/assets/x500_visual/`

## PX4 Gazebo x500 visual model

- **Upstream repository:** PX4-gazebo-models,
  `https://github.com/PX4/PX4-gazebo-models.git`
- **Upstream commit:** `d754381a1cecdd7f17050acd72bf5bf1327bced6`
  (the commit pinned by PX4-Autopilot
  `85df8c2281c2466b30a121b22b0bf33dc69bcfe4` as its `Tools/simulation/gz`
  submodule)
- **Upstream paths used:** `models/x500_base/` (as included by `models/x500/`)
- **Model author:** Benjamin Perseghetti (Rudis Laboratories), per the
  upstream `model.config`
- **Model description (upstream):** NXP HoverGames drone development kit
  (KIT-HGDRONEK66)
- **License:** BSD 3-Clause License, Copyright (c) 2022, Rudis Laboratories.
  The full license text is reproduced verbatim in `LICENSE` in this
  directory (byte-identical to upstream `models/x500_base/LICENSE`, sha256
  `cee4ef94e73cd38fb2886f5d7e7a04d6488f54b1e70cd8b1b29d74f38bfdba5b`).

In accordance with the license, the copyright notice, conditions and
disclaimer are retained with these derived files. The names of Rudis
Laboratories and its contributors are not used to endorse or promote this
project.

Every input file was verified against its sha256 at the pinned commit, and
every derived file's sha256 is recorded in `conversion_report.json`.

### Files derived from the upstream model

| Bundle file | Upstream source | Modification |
|---|---|---|
| `meshes/frame_carbon_fiber_upper.obj` | `models/x500_base/meshes/NXP-HGD-CF.dae` (`CarbonFiber`) | Format conversion COLLADA → OBJ; the 4 connected landing-gear pieces (2 leg tubes, 2 skid tubes) excluded; remaining triangles unmodified |
| `meshes/frame_metal.obj`, `frame_rails_rubber.obj`, `frame_fmuk66.obj`, `frame_antenna_holder.obj`, `frame_fmu_rubber.obj` | same DAE, one file per COLLADA component | Format conversion; for `frame_fmuk66`, 48 internal CAD line primitives (0–0.65 mm) dropped as non-triangular |
| `meshes/motor_base_5010_stator.obj` | `models/x500_base/meshes/5010Base.dae` | Format conversion; node transform (Y-up→Z-up rotation, 0.01 scale) baked; normals by inverse-transpose |
| `meshes/motor_bell_5010_side.obj`, `motor_bell_5010_head.obj` | `models/x500_base/meshes/5010Bell.dae` | Same as above, one file per component |
| `meshes/prop_1345_ccw.stl`, `prop_1345_cw.stl` | `models/x500_base/meshes/1345_prop_ccw.stl`, `1345_prop_cw.stl` | None (byte copies) |
| `textures/cf.png` | `models/x500_base/meshes/CF.png` | None (byte copy) |
| `x500_native_assembly.json` | `models/x500_base/model.sdf` | Visual poses and mesh scales transcribed programmatically; all collision, inertial, sensor, joint and plugin data omitted |

All OBJ conversions bake COLLADA node transforms, preserve authored normals
and UVs, and use 8-decimal text precision. No geometry was decimated,
simplified, reshaped, or synthesized.

### Not included

- The x500 landing gear (`LandingFoam`, `LandingRubber`, `LandingPlastic`
  components and the carbon-fiber leg/skid tubes): in this project the
  MuJoCo viewer shows the vehicle's physical landing-gear contact geoms
  instead.
- `OakD-Lite` camera model; the `x500_depth` composition.
- The three decorative SDF `<plane>` decals and their textures (`nxp.png`,
  `rd.png`).
- Any physical, motor, thrust, contact, or sensor parameter from the upstream
  SDF files.
