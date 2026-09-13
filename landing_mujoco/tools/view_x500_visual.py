#!/usr/bin/env python3
"""View the MuJoCo landing plant with (or without) the x500 visual shell.

Interactive (needs a working OpenGL display)::

    python3 landing_mujoco/tools/view_x500_visual.py                 # static pose
    python3 landing_mujoco/tools/view_x500_visual.py --rollout       # PID-only landing

Offscreen PNG renders (headless; e.g. MUJOCO_GL=osmesa)::

    MUJOCO_GL=osmesa python3 landing_mujoco/tools/view_x500_visual.py --offscreen out/

Geom groups with the shell enabled: 0 = ground + PHYSICAL landing-gear
contact geoms (the x500 landing gear is intentionally not part of the shell),
2 = x500 visual shell, 3 = simple placeholder markers (hidden by default;
``--physics-overlay`` shows them). Without the shell all physics-model geoms
are in group 0. The shell is purely cosmetic; the physical model is identical
with or without it.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from landing_mujoco.configs.param_schema import load_uav_params  # noqa: E402
from landing_mujoco.configs.visualization_config import load_visualization_config  # noqa: E402
from landing_mujoco.envs.mujoco_landing_env import MujocoLandingEnv  # noqa: E402
from landing_rl.envs.landing_env import LandingConfig  # noqa: E402

PARAMETER_SETS = {
    "tarot680b_reference": REPO_ROOT / "landing_mujoco/configs/tarot680b_reference.yaml",
    "ugrp_vehicle_measured": REPO_ROOT / "landing_mujoco/configs/ugrp_vehicle_measured.yaml",
}
VIEWS = {"iso": (135.0, -22.0), "top": (90.0, -89.9), "front": (180.0, -4.0), "side": (90.0, -4.0)}


def write_png(path: Path, img: np.ndarray) -> None:
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag, payload):
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def build_env(args) -> MujocoLandingEnv:
    params = load_uav_params(PARAMETER_SETS[args.parameter_set])
    vis = None
    if not args.no_visual:
        vis = load_visualization_config(args.visualization_config)
        if args.uniform_scale is not None:
            vis = replace(vis, uniform_scale=float(args.uniform_scale))
    cfg = LandingConfig(init_altitude_min_m=args.altitude, init_altitude_max_m=args.altitude)
    return MujocoLandingEnv(params, config=cfg, deterministic_physics=True, visualization=vis)


def settle_on_ground(env: MujocoLandingEnv, max_steps: int = 600) -> dict:
    info = {}
    for _ in range(max_steps):
        _, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        if term or trunc:
            break
    return info


def render_offscreen(env: MujocoLandingEnv, out_dir: Path, tag: str, groups: tuple, distance: float) -> list:
    model, data = env.mujoco_dynamics.model, env.mujoco_dynamics.data
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 960
    renderer = mujoco.Renderer(model, 960, 1280)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    for g in groups:
        opt.geomgroup[g] = 1
    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.xpos[env.mujoco_dynamics.body_id]
    written = []
    for view, (az, el) in VIEWS.items():
        cam.azimuth, cam.elevation, cam.distance = az, el, distance
        renderer.update_scene(data, camera=cam, scene_option=opt)
        path = out_dir / f"{tag}_{view}.png"
        write_png(path, renderer.render())
        written.append(path)
    renderer.close()
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parameter-set", choices=sorted(PARAMETER_SETS), default="tarot680b_reference")
    ap.add_argument("--visualization-config", type=Path,
                    default=REPO_ROOT / "landing_mujoco/configs/x500_visualization.yaml")
    ap.add_argument("--no-visual", action="store_true", help="physics-only model (no x500 shell)")
    ap.add_argument("--uniform-scale", type=float, default=None,
                    help="override the display scale (e.g. 1.0 for the native x500 reconstruction)")
    ap.add_argument("--altitude", type=float, default=1.5, help="initial altitude [m]")
    ap.add_argument("--pose", choices=["air", "ground"], default="air")
    ap.add_argument("--physics-overlay", action="store_true",
                    help="also draw the placeholder physics markers (body box, arms, motor positions)")
    ap.add_argument("--offscreen", type=Path, default=None, help="write PNG renders here instead of a window")
    ap.add_argument("--tag", default="x500")
    ap.add_argument("--rollout", action="store_true", help="interactive: run a PID-only landing")
    args = ap.parse_args(argv)

    env = build_env(args)
    env.reset(seed=0)
    shell = env.mujoco_dynamics.visual_shell
    print(json.dumps({"parameter_set": env.physical_params.parameter_set.value,
                      "visual_shell": shell.summary() if shell else None}, indent=2))

    info = settle_on_ground(env) if args.pose == "ground" else {}
    if info:
        print(f"settled: ground_contact={info['ground_contact']} touchdown_quality={info['touchdown_quality']} "
              f"success={info['success']} z_error={info['z_error']:.4f} altitude_agl={info['altitude_agl']:.4f}")

    if args.offscreen is not None:
        args.offscreen.mkdir(parents=True, exist_ok=True)
        # 0: ground + physical landing-gear contact geoms (+ placeholder markers
        #    when no shell), 2: x500 shell, 3: placeholder markers with shell.
        groups = {0, 2} if shell else {0}
        if args.physics_overlay and shell:
            groups |= {3}
        groups = tuple(sorted(groups))
        scale = shell.alignment.uniform_scale if shell else 1.0
        for p in render_offscreen(env, args.offscreen, args.tag, groups, distance=0.85 * max(scale, 1.0) + 0.2):
            print("wrote", p)
        return 0

    import mujoco.viewer

    model, data = env.mujoco_dynamics.model, env.mujoco_dynamics.data
    if not args.rollout:
        mujoco.viewer.launch(model, data)
        return 0
    import time

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            _, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
            viewer.sync()
            time.sleep(float(info["dt"]))
            if term or trunc:
                env.reset(seed=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
