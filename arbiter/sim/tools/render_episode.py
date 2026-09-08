#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Render a scripted-expert episode as a filmstrip, from the policy's own camera.

A static scene render shows the setup; it does not show that the setup *works*. The
achievability gate proves that numerically, but a number is not inspectable — a gate can pass
while the expert reaches the success predicate by some route nobody intended, and that is
precisely the failure the TOPOLOGY around-route had (it flew *over* the barrier for several
runs while reporting success on an axis that had stopped distinguishing its two classes).

So this captures frames through an episode and tiles them. The frames come from the layout's
head camera, so what is shown is what the policy will be trained on, not a convenient
third-person view.

One episode per launch: a second SimulationContext in the same process deadlocks.

Usage:
    render_episode.py --axis topology --barrier-h 0.11
    render_episode.py --axis direction --frames 8
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
_parser.add_argument("--axis", required=True)
_parser.add_argument("--barrier-h", type=float, default=None)
_parser.add_argument("--barrier-offset", type=float, default=0.0)
_parser.add_argument("--split", default="test", choices=("train", "test"),
                     help="which split to draw the condition from; test by default, since the "
                          "held-out configuration is the interesting one")
_parser.add_argument("--frames", type=int, default=6, help="frames in the filmstrip")
_parser.add_argument("--camera", default="cam_head",
                     help="which stream to capture. Defaults to the head camera so the strip "
                          "shows what the policy sees.")
_parser.add_argument("--max-steps", type=int, default=3000)
_parser.add_argument("--out", default="data/output/media/episodes")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(_parser)
args_cli = _parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402

from arbiter.collect.context import (  # noqa: E402
    ObjectDropped,
    PhaseTimeout,
    TaskContext,
    TaskFailed,
)
from arbiter.collect.tasks import OrderTask, task_for  # noqa: E402
from arbiter.sim.env.isaac_backend import IsaacMotionBackend  # noqa: E402
from arbiter.sim.env.scene import (  # noqa: E402
    HOME_JOINT_POS,
    assert_condition_realised,
    build_scene,
    place_object,
    reset_condition_scene,
)
from arbiter.suites.spec import (  # noqa: E402
    AXES,
    WORKSPACE,
    enumerate_conditions,
    object_keys_for,
    object_start_xy,
    object_start_yaw,
    order_slot_objects,
)
from arbiter.suites.layout_gen import ORDER_OBJECT_POS, ORDER_TARGET_POS  # noqa: E402


def pick_condition(axis_name: str, object_keys: list[str]):
    """One condition to film. For TOPOLOGY it must match the loaded prop."""
    ax = AXES[axis_name]
    conds = enumerate_conditions(ax, object_keys, splits=(args_cli.split,))
    if axis_name == "topology" and args_cli.barrier_h is not None:
        h, off = float(args_cli.barrier_h), float(args_cli.barrier_offset)
        conds = [c for c in conds
                 if abs(float(c.params["barrier_h"]) - h) < 1e-6
                 and abs(float(c.params["barrier_offset"]) - off) < 1e-6]
    if not conds:
        raise SystemExit(f"[EPISODE] no {args_cli.split} conditions match for {axis_name}")
    # Farthest from support: the hardest configuration is the one worth looking at.
    return max(conds, key=lambda c: c.sweep_coord)


def grab(scene, cam: str) -> np.ndarray | None:
    rgb = scene[cam].data.output.get("rgb")
    if rgb is None:
        return None
    img = rgb[0, ..., :3].detach().cpu().numpy()
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return img


def main() -> int:
    axis_name = args_cli.axis
    object_keys = object_keys_for(axis_name)

    scene, sim, info = build_scene(
        axis_name, device=args_cli.device, object_keys=object_keys,
        barrier_h=args_cli.barrier_h, barrier_offset=args_cli.barrier_offset,
        with_cameras=True,
    )
    if args_cli.camera not in scene.keys():
        raise SystemExit(f"[EPISODE] axis '{axis_name}' has no camera '{args_cli.camera}'. "
                         f"Available: {[k for k in scene.keys() if k.startswith('cam')]}")

    backend = IsaacMotionBackend(scene, sim)
    cond = pick_condition(axis_name, object_keys)
    assert_condition_realised(axis_name, cond.params, info["built"])
    print(f"[EPISODE] {axis_name} {cond.split} d={cond.sweep_coord:.2f}  {cond.key()}",
          flush=True)
    print(f"[EPISODE] instruction: {cond.instruction!r}", flush=True)

    backend.reset_to(HOME_JOINT_POS)
    reset_condition_scene(scene, sim, info, cond)

    ctx = TaskContext(backend)
    ctx.object_name = info["instance_name"][cond.object_key]
    task = (OrderTask(cond,
                      objects=[info["instance_name"][k] for k in order_slot_objects()],
                      targets=[tuple(t) for t in ORDER_TARGET_POS])
            if axis_name == "order" else task_for(cond)(cond))

    # Warm the renderer before the first capture: RTX needs a few frames to load textures, and
    # an un-warmed first frame comes back flat grey.
    for _ in range(30):
        sim.render()
        scene.update(0.0)

    frames: list[tuple[int, str, np.ndarray]] = []
    ok, reason, n = True, "", 0
    try:
        gen = task.run(ctx)
        for _ in gen:
            backend.step()
            n += 1
            if n % 12 == 0:
                sim.render()
                scene.update(0.0)
                img = grab(scene, args_cli.camera)
                if img is not None:
                    frames.append((n, ctx.last_phase, img))
    except (PhaseTimeout, ObjectDropped, TaskFailed, TimeoutError) as e:
        ok, reason = False, f"{type(e).__name__}: {e}"

    print(f"[EPISODE] {'SUCCESS' if ok else 'FAILED'} after {n} control steps"
          + ("" if ok else f" -- {reason}"), flush=True)
    print(f"[EPISODE] phases: {' -> '.join(p.label for p in ctx.phases)}", flush=True)

    if not frames:
        print("[EPISODE] no frames captured", flush=True)
        sys.stdout.flush()
        import os
        os._exit(1)

    # Even spread across the episode, always keeping the first and last.
    k = max(2, min(args_cli.frames, len(frames)))
    idx = [round(i * (len(frames) - 1) / (k - 1)) for i in range(k)]
    picked = [frames[i] for i in dict.fromkeys(idx)]

    # Grid, not a single row. Six 672x376 frames side by side is 4032x376 -- an aspect ratio
    # so extreme the strip is unreadable at any sane display width, which defeats the point of
    # rendering it for a human. Near-square tiling keeps every frame legible.
    import math as _math

    cols = max(1, int(_math.ceil(_math.sqrt(len(picked)))))
    rows_of: list[np.ndarray] = []
    h, w = picked[0][2].shape[:2]
    for r in range(0, len(picked), cols):
        chunk = [f[2] for f in picked[r:r + cols]]
        if len(chunk) < cols:                      # pad the last row so concatenate lines up
            chunk += [np.zeros((h, w, 3), dtype=chunk[0].dtype)] * (cols - len(chunk))
        rows_of.append(np.concatenate(chunk, axis=1))
    strip = np.concatenate(rows_of, axis=0)
    out = Path(args_cli.out)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    tag = axis_name + (f"_h{int(round(args_cli.barrier_h * 1000)):04d}"
                       if args_cli.barrier_h else "")
    path = out / f"{tag}_{args_cli.split}_strip.png"
    try:
        import cv2
        cv2.imwrite(str(path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    except ImportError:
        from PIL import Image
        Image.fromarray(strip).save(str(path))

    print(f"[EPISODE] wrote {path}  ({len(picked)} frames: "
          + ", ".join(f"{s}@{ph}" for s, ph, _ in picked) + ")", flush=True)
    sys.stdout.flush()
    import os
    os._exit(0 if ok else 1)


def _hard_exit(code: int) -> None:
    """Terminate for real.

    Isaac's app holds ~200 non-daemon threads, so a bare ``raise`` or ``SystemExit`` unwinds the
    stack and then hangs the process indefinitely instead of exiting -- observed as a run
    spinning at 116%% CPU for 2h50m past its own ``timeout``, having printed nothing. Every exit
    path after ``AppLauncher`` therefore goes through ``os._exit``.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        _hard_exit(int(e.code) if isinstance(e.code, int) else 1)
    except BaseException:
        import traceback
        traceback.print_exc()
        _hard_exit(1)
