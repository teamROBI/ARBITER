#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Render one axis's scene so it can be looked at, not just asserted about.

Every other check on the scene is programmatic -- USD prim APIs, sidecar geometry, layout JSON
keys. Those catch what they were written for and are blind to everything else. Three
composition bugs passed every one of them and were caught by a single image: a barrier rotated
90 degrees so it ran parallel to the arm's approach, a head camera aimed at the robot's
shoulder, and a robot floating 18 cm off the table. **Render after any scene change.**

This is a thin wrapper over ``arbiter.sim.env.scene.build_scene`` and holds no scene
construction of its own. It used to, and that duplication produced a wrong render immediately:
ORDER came out showing one cube when the axis is *about* the sequence between two, because the
renderer's private copy spawned a single hardcoded object.

Cameras come from the axis's layout, so what is captured is what the policy will see.

One axis per launch: building a second SimulationContext in the same process deadlocks. Use
``scripts/bench/render_all_axes.sh`` for a full sweep.

Usage:
    arbiter/sim/tools/render_scene.py --axis topology --barrier-h 0.11
    arbiter/sim/tools/render_scene.py --axis position --object-at 0.62,0.20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
_parser.add_argument("--axis", default="topology", help="which axis layout to render")
_parser.add_argument("--all", action="store_true",
                     help="not supported in one process; points at the driver script")
_parser.add_argument("--barrier-h", type=float, default=0.11,
                     help="TOPOLOGY: barrier height. Default demands the AROUND class "
                          "(h* = 0.08), the harder of the two.")
_parser.add_argument("--barrier-offset", type=float, default=0.0,
                     help="TOPOLOGY: barrier lateral offset")
_parser.add_argument("--object-at", default=None,
                     help="override the object position, 'x,y'. Use workspace corners to check "
                          "the head camera frames the whole sweep.")
_parser.add_argument("--settle", type=int, default=60,
                     help="control steps holding the home pose before capture")
_parser.add_argument("--out", default="data/output/media/renders")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(_parser)
args_cli = _parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402

from arbiter.sim.env.isaac_backend import IsaacMotionBackend  # noqa: E402
from arbiter.sim.env.scene import (  # noqa: E402
    HOME_JOINT_POS,
    build_scene,
    place_object,
    place_pads_for,
    reset_condition_scene,
)
from arbiter.suites.spec import AXES, object_keys_for  # noqa: E402


def _representative_condition(axis_name: str):
    """A real condition to draw, preferring a held-out one.

    For TOPOLOGY it must match the barrier this launch actually built, or the render shows a
    condition the stage does not represent -- the same mismatch assert_condition_realised exists
    to catch during collection.
    """
    from arbiter.suites.spec import enumerate_conditions, object_keys_for

    keys = object_keys_for(axis_name)
    for splits in (("test",), ("train", "test")):
        cs = list(enumerate_conditions(AXES[axis_name], keys, splits=splits))
        if axis_name == "topology":
            cs = [c for c in cs
                  if abs(float(c.params["barrier_h"]) - float(args_cli.barrier_h)) < 1e-6
                  and abs(float(c.params["barrier_offset"])
                          - float(args_cli.barrier_offset)) < 1e-6]
        if cs:
            return cs[0]
    raise ValueError(
        f"{axis_name}: no condition matches the built stage "
        f"(barrier_h={args_cli.barrier_h}, offset={args_cli.barrier_offset})"
    )


def _place_axis_objects(scene, sim, info, axis_name: str) -> None:
    """Put the scene into a real condition, through the SAME call the collector uses.

    This used to compute its own object position, and for TOPOLOGY it hardcoded the start offset
    as ``cx - 0.12``. When the spec moved the start to -0.18 to stop the barrier occluding the
    object, the renderer kept drawing the old position -- so the picture used to CHECK
    observability was the one picture not showing what gets collected. Placement geometry lives
    in the spec; tools ask for it.
    """
    cond = _representative_condition(axis_name)
    print(f"[RENDER] condition: {cond.key()}  split={cond.split}", flush=True)

    if args_cli.object_at:
        # Explicit override, for probing a spot no condition uses. Pads still come from the
        # condition so the goal shown is a real one.
        xy = tuple(float(v) for v in args_cli.object_at.split(","))
        place_pads_for(scene, sim, info, cond)
        place_object(scene, sim, info, info["object_keys"][0], xy)
        return

    reset_condition_scene(scene, sim, info, cond)


def save(scene, cam: str, path: Path) -> bool:
    rgb = scene[cam].data.output.get("rgb")
    if rgb is None:
        return False
    img = rgb[0, ..., :3].detach().cpu().numpy()
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    try:
        import cv2
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except ImportError:
        from PIL import Image
        Image.fromarray(img).save(str(path))
    return True


def render(axis_name: str, out_dir: Path) -> list[Path]:
    # object_keys explicitly, exactly as the collector passes them. Left to default, build_scene
    # falls back to the layout's object_pool, which carries spare cubes no condition uses: they
    # spawn at their parking slots, and the parking slots are INSIDE the head camera's frustum,
    # so a render showed a cube floating beside the table that collection never contains. The
    # renderer is the view used to verify the scene, so it must not be the one view that differs.
    scene, sim, info = build_scene(
        axis_name, device=args_cli.device, object_keys=object_keys_for(axis_name),
        barrier_h=args_cli.barrier_h, barrier_offset=args_cli.barrier_offset,
        with_cameras=True,
    )
    backend = IsaacMotionBackend(scene, sim)
    backend.reset_to(HOME_JOINT_POS)
    _place_axis_objects(scene, sim, info, axis_name)

    # Hold the home pose while things settle: write_joint_state_to_sim sets state, not the
    # position target, so without commanding it the arm drifts back off the home pose.
    for _ in range(args_cli.settle):
        backend.step()
    for _ in range(30):                       # let RTX load textures and converge
        sim.render()
        scene.update(0.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    cams = sorted(k for k in scene.keys() if k.startswith("cam"))
    if not cams:
        print(f"[RENDER] !! axis '{axis_name}' declared no cameras", flush=True)
        return []

    written: list[Path] = []
    for cam in cams:
        p = out_dir / f"{axis_name}_{cam.replace('cam_', '')}.png"
        if save(scene, cam, p):
            written.append(p)
            print(f"[RENDER] wrote {p}", flush=True)
    return written


def main() -> None:
    if args_cli.all:
        print("[RENDER] --all cannot work in one process: a second SimulationContext "
              "deadlocks. Use scripts/bench/render_all_axes.sh, which launches once per axis.",
              flush=True)
        import os
        os._exit(2)

    out = Path(args_cli.out)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    written = render(args_cli.axis, out)
    print(f"[RENDER] {len(written)} image(s)", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
