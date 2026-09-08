#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Collect scripted-expert demonstrations for one axis into an HDF5 archive.

Phase 2 of the v2 plan. The achievability gate already proved the expert can reach every
condition; this runs the same experts again and *keeps* what they produce, with the condition's
full parameter vector on every episode so the split is recoverable downstream.

Three properties worth stating, because each is a way this could silently produce a corrupt
dataset rather than fail:

**Observation before step.** At the point a task generator yields, ``movep`` has been called
but the physics step has not run -- so the simulator still holds the state the expert looked
at. Capturing there pairs action ``a_t`` with the observation ``o_t`` it was computed from.
Capturing after the step would pair every action with the state it already produced, which
trains a policy to describe the present instead of choosing the future, and would look
perfectly fine in every summary statistic.

**Only success advances the queue.** ``ConditionQueue.record_failure`` moves the cursor without
decrementing, so a hard condition gets retried later rather than quietly ending up with fewer
demonstrations than its neighbours. For a benchmark whose entire subject is coverage, a
collection process that thins out exactly where the task is hard would bias the measurement.

**One axis, one launch.** A second ``SimulationContext`` in one process deadlocks, and TOPOLOGY
additionally needs one launch per barrier height because the barrier is baked into the stage at
build time. ``--barrier-h`` filters the queue to match what was actually built.

Usage:
    collect_axis.py --axis position --demos 5
    collect_axis.py --axis topology --barrier-h 0.05 --demos 5
    collect_axis.py --axis direction --demos 5 --partition-idx 0 --partition-total 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
_parser.add_argument("--axis", required=True)
_parser.add_argument("--demos", type=int, default=5,
                     help="successful demonstrations required per condition")
_parser.add_argument("--splits", default="train",
                     help="comma-separated splits to collect; train by default")
_parser.add_argument("--barrier-h", type=float, default=None,
                     help="TOPOLOGY only: the barrier height built into this stage. The queue "
                          "is filtered to conditions that match it.")
_parser.add_argument("--barrier-offset", type=float, default=0.0)
_parser.add_argument("--partition-idx", type=int, default=0)
_parser.add_argument("--partition-total", type=int, default=1)
_parser.add_argument("--max-attempts", type=int, default=3,
                     help="attempts per required demonstration before giving up on the axis")
_parser.add_argument("--max-steps", type=int, default=3000)
_parser.add_argument("--no-images", action="store_true",
                     help="record actions and joint states only. For timing the loop without "
                          "paying for rendering; the result is NOT trainable.")
_parser.add_argument("--no-jitter", action="store_true",
                     help="start every episode from the exact home pose. Makes repeated demos "
                          "of one condition byte-identical, so only useful with --demos 1.")
_parser.add_argument("--out", default="data/collect")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(_parser)
args_cli = _parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = not args_cli.no_images

_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402

from arbiter.collect.context import (  # noqa: E402
    ObjectDropped,
    PhaseTimeout,
    TaskContext,
    TaskFailed,
)
from arbiter.collect.recorder import (  # noqa: E402
    EpisodeRecorder,
    archive_summary,
    next_episode_index,
)
from arbiter.collect.task import ConditionQueue  # noqa: E402
from arbiter.collect.tasks import OrderTask, task_for  # noqa: E402
from arbiter.sim.env.isaac_backend import IsaacMotionBackend  # noqa: E402
from arbiter.sim.env.scene import (  # noqa: E402
    HOME_JOINT_POS,
    assert_condition_realised,
    build_scene,
    episode_seed,
    jittered_home,
    place_object,
    reset_condition_scene,
)
from arbiter.suites.spec import order_slot_objects, axis as get_axis  # noqa: E402
from arbiter.suites.spec import object_keys_for, object_start_yaw  # noqa: E402
from arbiter.suites.spec import object_start_xy  # noqa: E402
from arbiter.suites.layout_gen import ORDER_OBJECT_POS, ORDER_TARGET_POS  # noqa: E402


def grab(scene, cams: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for cam in cams:
        rgb = scene[cam].data.output.get("rgb")
        if rgb is None:
            continue
        img = rgb[0, ..., :3].detach().cpu().numpy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        out[cam] = img
    return out


def main() -> int:
    axis_name = args_cli.axis
    ax = get_axis(axis_name)
    object_keys = object_keys_for(axis_name)

    scene, sim, info = build_scene(
        axis_name, device=args_cli.device, object_keys=object_keys,
        barrier_h=args_cli.barrier_h, barrier_offset=args_cli.barrier_offset,
        with_cameras=not args_cli.no_images,
    )
    cams = ([] if args_cli.no_images
            else [k for k in scene.keys() if k.startswith("cam")])
    backend = IsaacMotionBackend(scene, sim)

    # TOPOLOGY's barrier is stage geometry, so only the matching conditions are collectable
    # from this launch. Everything else takes the whole axis.
    predicate = None
    if axis_name == "topology" and args_cli.barrier_h is not None:
        h, off = float(args_cli.barrier_h), float(args_cli.barrier_offset)

        def predicate(c):  # noqa: F811
            return (abs(float(c.params["barrier_h"]) - h) < 1e-6
                    and abs(float(c.params["barrier_offset"]) - off) < 1e-6)

    queue = ConditionQueue(
        ax, object_keys,
        demos_per_condition=args_cli.demos,
        splits=tuple(s.strip() for s in args_cli.splits.split(",") if s.strip()),
        partition_idx=args_cli.partition_idx, partition_total=args_cli.partition_total,
        predicate=predicate,
    )

    out_dir = Path(args_cli.out)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    # The archive name must identify the *whole* stage configuration, offset included. Tagging
    # by height alone made all three lateral offsets of a height share one file. Sequentially
    # that merely hid the offset from the filename; in parallel two workers opened the same
    # HDF5 for writing and hit `BlockingIOError: unable to lock file`, and where they did not,
    # they raced on next_episode_index and overwrote each other -- observed as archives holding
    # 22, 24, 18 and 15 episodes instead of 45, each missing an offset entirely. One writer per
    # file, enforced by the name.
    tag = axis_name + (f"_h{int(round(args_cli.barrier_h * 1000)):04d}"
                       f"_o{int(round(args_cli.barrier_offset * 1000)):+05d}"
                       if args_cli.barrier_h is not None else "")
    if args_cli.partition_total > 1:
        tag += f"_p{args_cli.partition_idx}of{args_cli.partition_total}"
    archive = out_dir / f"{tag}.hdf5"

    ep_index = next_episode_index(archive)
    recorder = EpisodeRecorder(image_keys=cams)
    backend.attach_recorder(recorder)

    print(f"[COLLECT] {queue.summary()}", flush=True)
    print(f"[COLLECT] cameras={cams or 'none'} -> {archive} (resuming at episode {ep_index})",
          flush=True)

    # Warm the renderer: RTX needs several frames to load textures and the first captures come
    # back flat grey otherwise.
    if cams:
        for _ in range(30):
            sim.render()
            scene.update(0.0)

    budget = queue.total_episodes * max(1, args_cli.max_attempts)
    attempts = n_ok = n_fail = 0
    # Attempts so far per condition. The jitter seed advances on every attempt rather than on
    # every success: seeding by success index would retry a failed episode with exactly the
    # jitter that just failed, so a marginal condition would fail identically until it burned
    # the whole retry budget. Deterministic either way, so a rerun still reproduces the archive.
    attempt_of: dict[str, int] = {}
    t0 = time.time()

    while not queue.done and attempts < budget:
        cond = queue.next_condition()
        if cond is None:
            break
        attempts += 1

        assert_condition_realised(axis_name, cond.params, info["built"])
        rep = attempt_of.get(cond.key(), 0)
        attempt_of[cond.key()] = rep + 1
        seed = episode_seed(cond.key(), rep)
        home = HOME_JOINT_POS if args_cli.no_jitter else jittered_home(seed)
        backend.reset_to(home)
        # Target pads first: they are kinematic and collisionless, so they neither fall
        # nor disturb the object settle, and placing them before the object means they
        # are present in every recorded frame. One call, because five tools each used to
        # carry their own copy of this branch.
        reset_condition_scene(scene, sim, info, cond)

        ctx = TaskContext(backend)
        ctx.object_name = info["instance_name"][cond.object_key]
        task = (OrderTask(cond,
                          objects=[info["instance_name"][k] for k in order_slot_objects()],
                          targets=[tuple(t) for t in ORDER_TARGET_POS])
                if axis_name == "order" else task_for(cond)(cond))

        recorder.reset()
        ok, reason, n = True, "", 0
        try:
            for _ in task.run(ctx):
                # Capture BEFORE stepping: the sim still holds the observation this action was
                # computed from. See the module docstring.
                if cams:
                    sim.render()
                    scene.update(0.0)
                recorder.step(
                    action=backend.last_action(),
                    joint_pos=backend.getj() + [backend.gripper_pos()],
                    images=grab(scene, cams) if cams else None,
                )
                backend.step()
                n += 1
                if n >= args_cli.max_steps:
                    raise TimeoutError(f"exceeded {args_cli.max_steps} control steps")
        except (PhaseTimeout, ObjectDropped, TaskFailed, TimeoutError) as e:
            ok, reason = False, f"{type(e).__name__}: {e}"
            if isinstance(e, TaskFailed) and not e.retriable:
                # A geometry bug, not an unlucky attempt. Retrying forever would hide it.
                print(f"[COLLECT] FATAL non-retriable on {cond.key()}: {e}", flush=True)
                break

        if recorder.n_frames:
            attrs = cond.to_attrs()
            # Recorded so a trajectory can be reproduced from the archive alone.
            attrs["episode_seed"] = int(seed)
            attrs["attempt"] = int(rep)
            recorder.write(archive, ep_index, attrs, success=ok)
            ep_index += 1

        if ok:
            n_ok += 1
            queue.confirm_success(cond)
        else:
            n_fail += 1
            queue.record_failure(cond)

        if attempts % 10 == 0 or not ok:
            rate = (time.time() - t0) / max(attempts, 1)
            left = (queue.total_episodes - queue.collected) * rate
            print(f"[COLLECT] {queue.collected}/{queue.total_episodes} kept "
                  f"({n_fail} failed) {rate:.1f}s/ep  eta {left / 60:.0f}m"
                  + (f"  last-fail {cond.key()}: {reason}" if not ok else ""), flush=True)

    dt = time.time() - t0
    done = queue.done
    print(f"[COLLECT] {'COMPLETE' if done else 'INCOMPLETE'} "
          f"{queue.collected}/{queue.total_episodes} in {dt / 60:.1f}m "
          f"({n_ok} ok, {n_fail} failed, {attempts} attempts)", flush=True)
    print(f"[COLLECT] archive {archive}: {archive_summary(archive)}", flush=True)

    sys.stdout.flush()
    import os
    os._exit(0 if done else 1)


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
