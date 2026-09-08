#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Drive every scripted expert against a kinematics stub — no Isaac, no h5py.

    python arbiter/collect/tests/test_tasks.py

Running the experts without a simulator catches the class of bug that is most expensive to
find in-sim: geometry derived wrongly from a condition's parameter vector. In Isaac that looks
like "the expert fails on some conditions", takes minutes per episode to reproduce, and is
easy to misread as physics. Here it is deterministic and instant.

The grasp maths gets its own checks because it is where a silent sign error hides: a frame
that is not right-handed, or a quaternion conversion that loses precision exactly at the
top-down pose, produces demonstrations that look plausible and grasp the wrong way round.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arbiter.collect.context import TaskContext, TaskFailed  # noqa: E402
from arbiter.collect.grasp import (  # noqa: E402
    canonical_azimuth,
    grasp_frame,
    grasp_poses,
    grasp_quat,
    mat_to_quat,
)
from arbiter.collect.tasks import (  # noqa: E402
    LiftTask,
    OrderTask,
    RouteTask,
    TransportTask,
    task_for,
)
from arbiter.collect.tests.test_context import FakeArm, drive  # noqa: E402
from arbiter.suites.spec import (  # noqa: E402
    AXES,
    ORDER_OBJECT_POS,
    ORDER_TARGET_POS,
    WORKSPACE,
    enumerate_conditions,
    expected_homotopy,
    object_keys_for,
    order_move_sequence,
    order_slot_objects,
)

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  {'ok   ' if cond else 'FAIL '} {label}")
    if not cond:
        FAILURES.append(label)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _col(m, j):
    return [m[0][j], m[1][j], m[2][j]]


# ──────────────────────────────────────────────────────────────────────────────

def test_grasp_frame_orthonormal() -> None:
    print("\n[grasp frame is a proper rotation]")
    worst_orth = 0.0
    worst_norm = 0.0
    worst_det = 0.0
    for az in range(0, 360, 15):
        for tilt in (0.0, 12.0, 30.0):
            m = grasp_frame(az, tilt)
            x, y, z = _col(m, 0), _col(m, 1), _col(m, 2)
            worst_orth = max(worst_orth, abs(_dot(x, y)), abs(_dot(y, z)), abs(_dot(x, z)))
            for v in (x, y, z):
                worst_norm = max(worst_norm, abs(math.sqrt(_dot(v, v)) - 1.0))
            # Right-handed: x cross y == z
            cx = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]]
            worst_det = max(worst_det, max(abs(a - b) for a, b in zip(cx, z)))
    check(worst_orth < 1e-9, f"axes mutually orthogonal (worst {worst_orth:.2e})")
    check(worst_norm < 1e-9, f"axes unit length (worst {worst_norm:.2e})")
    check(worst_det < 1e-9, f"right-handed, x cross y == z (worst {worst_det:.2e})")


def test_topdown_approach_points_down() -> None:
    print("\n[top-down grasp approaches downward]")
    for az in (0, 45, 90, 137, 180, 271):
        z = _col(grasp_frame(az, 0.0), 2)
        check(abs(z[2] + 1.0) < 1e-9,
              f"az={az}: approach is straight down (z={[round(c, 3) for c in z]})")
    # A tilted approach leans toward the azimuth heading but still descends.
    z = _col(grasp_frame(0.0, 30.0), 2)
    check(z[2] < 0.0 and z[0] > 0.0,
          f"tilt leans toward the heading while still descending ({[round(c, 3) for c in z]})")


def test_jaw_axis_follows_azimuth() -> None:
    print("\n[jaw axis follows the azimuth]")
    for az in (0, 30, 60, 90, 120):
        x = _col(grasp_frame(az, 0.0), 0)
        want = [math.cos(math.radians(az)), math.sin(math.radians(az)), 0.0]
        check(max(abs(a - b) for a, b in zip(x, want)) < 1e-9,
              f"az={az}: jaw axis is the azimuth heading")
    # What that heading *is* was ambiguous and cost a 0/16 gate: it is the direction the
    # fingers lie along, not the axis they separate on. So a bar is grasped across its width
    # when the azimuth equals the bar's own yaw. Measured at yaw=0: phi=0 lifts, phi=90 stalls.
    from arbiter.suites.spec import approach_phi
    check(abs(approach_phi(0.0) - 0.0) < 1e-9,
          "a bar at yaw=0 is grasped at phi=0, not phi=90")
    check(abs(approach_phi(30.0) - 30.0) < 1e-9, "and the grasp azimuth tracks the bar's yaw")


def test_quat_conversion() -> None:
    print("\n[quaternion conversion]")
    q = mat_to_quat([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    check(abs(q[0] - 1.0) < 1e-9 and max(abs(c) for c in q[1:]) < 1e-9,
          "identity matrix gives the identity quaternion")
    for az in range(0, 360, 30):
        qq = grasp_quat(az, 0.0)
        n = math.sqrt(sum(c * c for c in qq))
        check(abs(n - 1.0) < 1e-9, f"az={az}: quaternion is unit length (|q|={n:.9f})")
        check(all(math.isfinite(c) for c in qq), f"az={az}: quaternion is finite")
    # The top-down pose is a 180-degree rotation, exactly where the naive trace formula
    # degenerates — so it is the case most worth pinning.
    q180 = mat_to_quat([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    check(abs(math.sqrt(sum(c * c for c in q180)) - 1.0) < 1e-9,
          "180-degree rotation converts without precision loss")


def test_azimuth_period() -> None:
    print("\n[azimuth is periodic at 180, not 360]")
    check(canonical_azimuth(200.0) == 20.0, "200 folds to 20")
    check(canonical_azimuth(180.0) == 0.0, "180 folds to 0")
    check(canonical_azimuth(-30.0) == 150.0, "negative folds into range")
    # A parallel jaw at az and az+180 is the same physical grasp: the jaw axis flips sign.
    a = _col(grasp_frame(30.0), 0)
    b = _col(grasp_frame(210.0), 0)
    check(max(abs(x + y) for x, y in zip(a, b)) < 1e-9,
          "az and az+180 give opposite jaw axes, i.e. the same grasp")


def test_grasp_poses_geometry() -> None:
    print("\n[grasp / pregrasp geometry]")
    obj = [0.5, 0.1, 0.027]
    grasp, pregrasp, _ = grasp_poses(obj, azimuth_deg=0.0)
    check(abs(grasp[2] - 0.027) < 1e-9,
          "grasp target sits at the object CENTRE: v2 assets are centre-origin, so adding a "
          "half-height would put the jaws on the top face and close them on air")
    base_origin, _, _ = grasp_poses([0.5, 0.1, 0.0], origin_to_centre_z=0.025)
    check(abs(base_origin[2] - 0.025) < 1e-9,
          "origin_to_centre_z still supports base-origin assets")
    check(abs(grasp[0] - obj[0]) < 1e-9 and abs(grasp[1] - obj[1]) < 1e-9,
          "top-down grasp is directly over the object in XY")
    check(pregrasp[2] > grasp[2],
          f"pregrasp stands off above the grasp ({pregrasp[2]:.3f} > {grasp[2]:.3f})")
    check(abs(pregrasp[0] - grasp[0]) < 1e-9 and abs(pregrasp[1] - grasp[1]) < 1e-9,
          "top-down pregrasp has no lateral offset")

    _, pre_tilt, _ = grasp_poses(obj, azimuth_deg=0.0, tilt_deg=30.0)
    check(abs(pre_tilt[0] - grasp[0]) > 1e-3,
          "a tilted pregrasp IS offset laterally, along the approach")


# ──────────────────────────────────────────────────────────────────────────────

def _arm_for(cond, objects=("cube",)) -> FakeArm:
    """A stub arm with the target object placed where the condition says it should be."""
    cx, cy = WORKSPACE.centre
    x = float(cond.params.get("x", cx))
    y = float(cond.params.get("y", cy))
    arm = FakeArm(pos=(x, y, 0.30), tracking=0.5)
    arm.objects = {name: [x, y, 0.0] for name in objects}
    return arm


def _run_task(cond, cls=None, **kw) -> tuple[TaskContext, FakeArm]:
    objects = kw.pop("objects", ("cube",))
    arm = _arm_for(cond, objects)
    ctx = TaskContext(arm)
    ctx.object_name = objects[0]
    task = (cls or task_for(cond))(cond, **kw)
    drive(task.run(ctx))
    return ctx, arm


def test_lift_task() -> None:
    print("\n[LiftTask]")
    for axis_name in ("position", "approach"):
        ax = AXES[axis_name]
        cond = next(c for c in enumerate_conditions(ax, ["cube"], splits=("train",)))
        ctx, arm = _run_task(cond)
        check(ctx.reached("grasp"), f"{axis_name}: grasp phase reached")
        check(ctx.reached("lift"), f"{axis_name}: lift phase reached")
        check(ctx.reached("retract"), f"{axis_name}: retract phase reached")
        check(task_for(cond) is LiftTask, f"{axis_name}: dispatches to LiftTask")


def test_transport_task() -> None:
    print("\n[TransportTask]")
    for axis_name in ("direction", "extent", "factorial"):
        ax = AXES[axis_name]
        ok = 0
        checked = 0
        for cond in enumerate_conditions(ax, ["cube"])[:14]:
            checked += 1
            try:
                ctx, _ = _run_task(cond)
                if ctx.reached("carry") and ctx.reached("lower"):
                    ok += 1
            except TaskFailed as e:
                # Only a non-retriable failure means the geometry itself is wrong.
                if not e.retriable:
                    print(f"        {axis_name} {cond.key()}: {e}")
        check(ok == checked, f"{axis_name}: all {checked} sampled conditions transported "
                             f"({ok} ok)")
        check(task_for(cond) is TransportTask, f"{axis_name}: dispatches to TransportTask")


def test_transport_targets_inside_workspace() -> None:
    """A target derived off the table is a geometry bug, and must not be silently retried."""
    print("\n[transport targets stay on the table]")
    for axis_name in ("direction", "extent", "factorial"):
        ax = AXES[axis_name]
        outside = []
        for cond in enumerate_conditions(ax, ["cube"]):
            try:
                _run_task(cond)
            except TaskFailed as e:
                if not e.retriable:
                    outside.append(cond.key())
            except Exception:
                pass
        check(not outside,
              f"{axis_name}: no condition puts its target off the table "
              f"({len(outside)} bad" + (f", e.g. {outside[0]}" if outside else "") + ")")


def test_route_task_uses_axis_homotopy() -> None:
    print("\n[RouteTask follows the axis's homotopy class]")
    ax = AXES["topology"]
    seen = {"over": 0, "around": 0}
    for cond in enumerate_conditions(ax, ["cube"])[:40]:
        h = float(cond.params["barrier_h"])
        want = expected_homotopy(h)
        ctx, arm = _run_task(cond)
        labels = [p.label for p in ctx.phases]
        has_over = any("over_apex" in lbl for lbl in labels)
        has_around = any("around_out" in lbl for lbl in labels)
        got = "over" if has_over else "around" if has_around else "none"
        if got != want:
            check(False, f"h={h:.2f}: expected {want}, expert did {got}")
            return
        seen[want] += 1
    check(seen["over"] > 0 and seen["around"] > 0,
          f"both classes exercised ({seen['over']} over / {seen['around']} around)")
    check(task_for(next(iter(enumerate_conditions(ax, ['cube'])))) is RouteTask,
          "topology dispatches to RouteTask")


def test_order_task() -> None:
    print("\n[OrderTask]")
    ax = AXES["order"]
    conds = enumerate_conditions(ax, object_keys_for("order"))
    names = [f"cube_{i}" for i in range(len(order_slot_objects()))]
    targets = [tuple(t) for t in ORDER_TARGET_POS]
    start = {n: [float(xy[0]), float(xy[1]), 0.0]
             for n, xy in zip(names, ORDER_OBJECT_POS)}

    seqs = {}
    for cond in conds:
        arm = FakeArm(pos=(0.40, 0.0, 0.30), tracking=0.5)
        arm.objects = dict(start)
        ctx = TaskContext(arm)
        ctx.object_name = names[0]
        drive(OrderTask(cond, objects=names, targets=targets).run(ctx))
        seqs[str(cond.params["seq"])] = [
            p.label for p in ctx.phases if p.label.endswith("_grasp")
        ]

    check(len(seqs) == len(conds), f"every ordering ran ({len(seqs)}/{len(conds)})")
    check(all(len(v) == len(names) for v in seqs.values()),
          f"each ordering records exactly {len(names)} grasp phases")

    # The load-bearing property: identical scene, different executed trajectory. Two orderings
    # that differ only in the instruction must diverge early, or the axis measures nothing.
    # Two orderings whose FIRST object differs. 012 and 021 share a first object and so are
    # identical for the whole first sub-task by design -- comparing those would fail for a
    # correct expert.
    a = conds[0]
    a_first = order_move_sequence(a)[0]
    b = next(c for c in conds if order_move_sequence(c)[0] != a_first)
    cmds = []
    for cond in (a, b):
        arm = FakeArm(pos=(0.40, 0.0, 0.30), tracking=0.5)
        arm.objects = dict(start)
        ctx = TaskContext(arm)
        ctx.object_name = names[0]
        drive(OrderTask(cond, objects=names, targets=targets).run(ctx))
        cmds.append([c[0][1] for c in arm.commands[:120]])
    check(max(abs(x - y) for x, y in zip(*cmds)) > 0.02,
          f"{a.params['seq']} and {b.params['seq']} diverge early despite an identical scene")


def test_order_instruction_matches_execution() -> None:
    """The wording must name the block the expert actually moves first.

    ORDER's two conditions are identical in every observable way at t=0 and differ *only* by
    the instruction, so the instruction is the entire signal. If it names the wrong block the
    axis is not merely mislabelled -- it is inverted, and a policy that obeys the words is
    scored wrong while one that ignores them is scored right. Nothing else in the stack would
    notice, because the expert is driven by `params["order"]` and never reads the text.
    """
    print("\n[OrderTask instruction]")
    ax = AXES["order"]
    targets = [tuple(t) for t in ORDER_TARGET_POS]

    conds = enumerate_conditions(ax, object_keys_for("order"))
    for cond in conds:
        # Slot order is fixed and never tracks the sequence -- that is what stops a positional
        # rule ("always move the left one first") from solving the axis without reading the
        # instruction.
        objects = order_slot_objects()
        colour_of = {o: o.split("_")[-1] for o in objects}
        arm = FakeArm(pos=(0.40, 0.0, 0.30), tracking=0.5)
        arm.objects = {o: [float(xy[0]), float(xy[1]), 0.0]
                       for o, xy in zip(objects, ORDER_OBJECT_POS)}
        ctx = TaskContext(arm)
        ctx.object_name = objects[0]
        drive(OrderTask(cond, objects=objects, targets=targets).run(ctx))

        seq = order_move_sequence(cond)
        text = cond.instruction
        tag = str(cond.params["seq"])
        # EVERY object must be named, in the order the expert actually moves them.
        names = [colour_of[objects[i]] for i in seq]
        for nm in names:
            check(nm in text, f"{tag}: instruction names {nm!r}")
        positions = [text.index(nm) for nm in names if nm in text]
        check(positions == sorted(positions),
              f"{tag}: instruction names {names} in execution order: {text!r}")

    # The property that makes the axis learnable AND kills positional shortcuts: across the
    # trained sequences every object appears at every position. Without it a policy can score
    # 100% on train with a fixed habit and 0% on test by construction -- which is what the old
    # single-pair design did, and is the confound the plan levels at v1's Inversion suite.
    train_seqs = [order_move_sequence(c) for c in conds if c.split == "train"]
    n_obj = len(order_slot_objects())
    for pos in range(n_obj):
        seen = sorted({sq[pos] for sq in train_seqs})
        check(seen == list(range(n_obj)),
              f"every object appears at sequence position {pos} in training (got {seen})")
    held = [c for c in conds if c.split == "test"]
    check(len(held) >= 3, f"more than one held-out ordering ({len(held)})")

    texts = {c.instruction for c in conds}
    check(len(texts) == len(conds),
          f"all {len(conds)} conditions have distinct instructions: {len(texts)}")
    check(not any("first block" in t or "second block" in t for t in texts),
          "no positional wording survives -- 'the first block' never said which block, so a "
          "failure could not separate sequence composition from referring-expression grounding")


def test_every_axis_has_an_expert() -> None:
    print("\n[coverage]")
    missing = []
    for name in AXES:
        try:
            cond = next(iter(enumerate_conditions(AXES[name], ["cube"])))
            task_for(cond)
        except KeyError:
            missing.append(name)
    check(not missing, f"every axis has a registered expert (missing: {missing})")
    try:
        class Fake:
            axis = "nonexistent"
        task_for(Fake())
        check(False, "an unregistered axis raises rather than silently doing nothing")
    except KeyError:
        check(True, "an unregistered axis raises rather than silently doing nothing")


def main() -> int:
    test_grasp_frame_orthonormal()
    test_topdown_approach_points_down()
    test_jaw_axis_follows_azimuth()
    test_quat_conversion()
    test_azimuth_period()
    test_grasp_poses_geometry()
    test_lift_task()
    test_transport_task()
    test_transport_targets_inside_workspace()
    test_route_task_uses_axis_homotopy()
    test_order_task()
    test_order_instruction_matches_execution()
    test_every_axis_has_an_expert()

    print()
    if FAILURES:
        print(f"[FAIL] {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("[PASS] all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
