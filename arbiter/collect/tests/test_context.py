#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the collection primitives against a kinematics stub — no Isaac, no h5py.

    python arbiter/collect/tests/test_context.py

The primitives are where scripted-collection bugs actually live: a rate cap that lets the
expert command a step the policy cannot reproduce, a timeout budget that starves a long
transit, a stall detector that fires on a legitimate slow phase, a drop check that trips on a
deliberate place. All of that is pure logic over floats, so it is testable without a simulator
— and it has to be, because in-sim these failures look like "the expert is flaky" rather than
like a bug.

``FakeArm`` is a first-order EEF: it moves a fraction of the way toward whatever pose is
commanded each step. That is enough to exercise convergence, budgets and stalls, and it can be
made deliberately unreachable to test the failure paths.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arbiter.collect import constants as K  # noqa: E402
from arbiter.collect.backend import (  # noqa: E402
    dist,
    movement_timeout_steps,
    quat_angle_between,
    quat_normalize,
    quat_slerp,
    step_orientation_towards,
    step_position_towards,
)
from arbiter.collect.context import (  # noqa: E402
    ObjectDropped,
    PhaseTimeout,
    TaskContext,
    TaskFailed,
)
from arbiter.collect.task import ConditionQueue, Task  # noqa: E402
from arbiter.suites.spec import DIRECTION, TOPOLOGY, expected_homotopy  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  {'ok   ' if cond else 'FAIL '} {label}")
    if not cond:
        FAILURES.append(label)


# ──────────────────────────────────────────────────────────────────────────────

class FakeArm:
    """First-order EEF tracker standing in for differential IK.

    ``tracking`` < 1 makes convergence take several steps, which is what the budget and stall
    logic must cope with. ``blocked`` freezes motion to exercise the failure paths.
    """

    def __init__(
        self,
        pos=(0.4, 0.0, 0.3),
        quat=(1.0, 0.0, 0.0, 0.0),
        *,
        tracking: float = 0.35,
        blocked: bool = False,
    ) -> None:
        self.pos = list(pos)
        self.quat = list(quat)
        self.grip = 0.04
        self.tracking = float(tracking)
        self.blocked = bool(blocked)
        self.steps = 0
        self.commands: list[tuple[list[float], list[float], float]] = []
        self.cache_cleared = 0
        self.objects: dict[str, list[float]] = {"cube": [0.5, 0.0, 0.05]}
        self.attached: str | None = None
        self.attach_offset: list[float] | None = None

    # MotionBackend
    def getp(self):
        return list(self.pos), list(self.quat)

    def getj(self):
        return [0.0] * 7

    #: Below this gripper width the jaws are considered closed.
    GRASP_WIDTH = 0.02
    #: EEF must be this close to an object to pick it up.
    GRASP_REACH_M = 0.05

    def movep(self, pos_b, quat_b, gripper):
        self.commands.append((list(pos_b), list(quat_b), float(gripper)))
        self.steps += 1
        if self.blocked:
            return
        t = self.tracking
        self.pos = [c + t * (d - c) for c, d in zip(self.pos, pos_b)]
        self.quat = quat_slerp(self.quat, list(quat_b), t)
        self.grip = self.grip + t * (float(gripper) - self.grip)
        self._update_attachment()

    def _update_attachment(self):
        """Model pick-up and release, so a carried object actually moves.

        Without this the stub holds every object fixed, and any task ending in
        ``check_placed`` fails no matter how correct it is — the transport experts would look
        broken while being fine. Deliberately crude: a real grasp depends on contact and
        friction, but for exercising task *logic* "jaws closed near an object means carrying
        it" is the behaviour that matters.
        """
        if self.grip <= self.GRASP_WIDTH:
            if self.attached is None:
                for name, p in self.objects.items():
                    if math.dist(self.pos, p) <= self.GRASP_REACH_M:
                        self.attached = name
                        self.attach_offset = [o - e for o, e in zip(p, self.pos)]
                        break
        else:
            if self.attached is not None:
                # Released: settle straight down onto the table.
                self.objects[self.attached][2] = 0.0
            self.attached = None
            self.attach_offset = None

        if self.attached is not None and self.attach_offset is not None:
            self.objects[self.attached] = [
                e + o for e, o in zip(self.pos, self.attach_offset)
            ]

    def gripper_pos(self):
        return self.grip

    def object_pos_world(self, name):
        return list(self.objects[name])

    def object_pos_in_base(self, name):
        return list(self.objects[name])

    def step(self):
        pass

    def clear_recording_cache(self):
        self.cache_cleared += 1


def drive(gen) -> int:
    """Run a primitive generator to completion, counting control steps."""
    n = 0
    for _ in gen:
        n += 1
        if n > 5000:
            raise AssertionError("generator did not terminate")
    return n


# ──────────────────────────────────────────────────────────────────────────────

def test_rate_caps() -> None:
    print("\n[rate caps — the expert must not command more than one step's motion]")
    cur = [0.0, 0.0, 0.0]
    far = [1.0, 0.0, 0.0]
    nxt = step_position_towards(cur, far)
    check(abs(dist(cur, nxt) - K.MAX_TRANSLATION_STEP_M) < 1e-9,
          f"far target steps exactly {K.MAX_TRANSLATION_STEP_M} m")

    near = [0.001, 0.0, 0.0]
    check(step_position_towards(cur, near) == near,
          "target inside one step snaps exactly, so the expert converges rather than dithering")

    diag = step_position_towards(cur, [1.0, 1.0, 1.0])
    check(abs(dist(cur, diag) - K.MAX_TRANSLATION_STEP_M) < 1e-9,
          "cap applies to the 3D norm, not per-axis")

    q0 = [1.0, 0.0, 0.0, 0.0]
    q1 = quat_normalize([0.0, 1.0, 0.0, 0.0])          # 180 deg away
    stepped = step_orientation_towards([0.0] * 3, q0, [0.0] * 3, q1)
    arc = quat_angle_between(q0, stepped) * 0.5
    check(arc <= K.MAX_ORIENTATION_STEP_RAD + 1e-6,
          f"orientation step capped at {K.MAX_ORIENTATION_STEP_RAD} rad (got {arc:.4f})")

    # While position is still far, orientation must not race ahead of translation.
    slow = step_orientation_towards([0.0] * 3, q0, [1.0, 0.0, 0.0], q1)
    check(quat_angle_between(q0, slow) * 0.5 <= K.MAX_ORIENTATION_STEP_RAD + 1e-6,
          "orientation stays capped when position is far too")


def test_slerp_shortest_arc() -> None:
    print("\n[slerp takes the shortest arc]")
    q = quat_normalize([0.7071, 0.0, 0.7071, 0.0])
    neg = [-c for c in q]
    a = quat_slerp([1.0, 0.0, 0.0, 0.0], q, 0.5)
    b = quat_slerp([1.0, 0.0, 0.0, 0.0], neg, 0.5)
    check(quat_angle_between(a, b) < 1e-6,
          "q and -q are the same rotation, so they interpolate identically")
    ident = quat_slerp([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 0.5)
    check(quat_angle_between(ident, [1.0, 0.0, 0.0, 0.0]) < 1e-6,
          "identical rotations interpolate to themselves (no NaN from the sin(0) path)")
    check(all(math.isfinite(c) for c in a), "slerp output is finite")


def test_timeout_budget() -> None:
    print("\n[timeout budget scales with distance]")
    near = movement_timeout_steps(0.001)
    far = movement_timeout_steps(0.5)
    check(near == K.PHASE_TIMEOUT_STEPS, f"tiny move gets the floor budget ({near})")
    check(far > near, f"long move gets more budget ({far} > {near})")
    check(far <= K.MOVEMENT_TIMEOUT_MAX_STEPS, f"budget clamped at the ceiling ({far})")
    check(movement_timeout_steps(100.0) == K.MOVEMENT_TIMEOUT_MAX_STEPS,
          "absurd distance saturates rather than running forever")
    # A pure rotation still needs a budget.
    check(movement_timeout_steps(0.0, math.pi) > K.PHASE_TIMEOUT_STEPS,
          "orientation-only move gets a budget from the rotation, not just the translation")


def test_move_to_converges() -> None:
    print("\n[move_to]")
    arm = FakeArm()
    ctx = TaskContext(arm)
    target = [0.45, 0.05, 0.25]
    n = drive(ctx.move_to(target, [1.0, 0.0, 0.0, 0.0], label="approach"))
    check(dist(arm.pos, target) <= K.PHASE_ADVANCE_TOLERANCE_M,
          f"converged inside the advance tolerance in {n} steps")
    check(ctx.reached("approach"), "phase recorded as reached")
    check(ctx.phases[-1].steps == n, "phase step count matches yields")
    check(len(arm.commands) == n, "exactly one command per yielded step")

    biggest = max(
        dist(arm.commands[i][0], arm.commands[i - 1][0]) for i in range(1, len(arm.commands))
    )
    check(biggest <= K.MAX_TRANSLATION_STEP_M + 1e-6,
          f"no commanded step exceeded the rate cap (max {biggest:.4f} m)")

    # The command is accumulated, not recomputed from actual. Recomputing creates a fixed point
    # where the rate-cap step is cancelled by tracking lag, and the arm asymptotes short of the
    # target however large the budget is -- measured at 0.032 m against a 0.030 m threshold.
    cmds = [c[0] for c in arm.commands]
    errs = [dist(c, target) for c in cmds]
    check(all(errs[i] <= errs[i - 1] + 1e-9 for i in range(1, len(errs))),
          "commanded pose approaches the target MONOTONICALLY")
    check(errs[-1] < K.MAX_TRANSLATION_STEP_M,
          f"the command actually reaches the target rather than asymptoting short "
          f"(final command error {errs[-1]:.5f} m)")


def test_move_to_timeout_and_soft_accept() -> None:
    print("\n[move_to failure paths]")
    arm = FakeArm(blocked=True)
    ctx = TaskContext(arm)
    try:
        drive(ctx.move_to([0.9, 0.4, 0.5], label="transit", check_stall=False))
        check(False, "blocked arm raises PhaseTimeout")
    except PhaseTimeout as e:
        check("timeout" in str(e), f"blocked arm raises PhaseTimeout ({e})")
    check(ctx.phases[-1].ok is False, "failed phase recorded as not ok")

    # Soft accept: stop just outside the tight tolerance but inside the loose one.
    arm2 = FakeArm(pos=(0.4, 0.0, 0.3))
    ctx2 = TaskContext(arm2)
    target = [0.4 + 0.025, 0.0, 0.3]        # 2.5 cm: > advance (2 cm), < position (3 cm)
    arm2.blocked = True
    try:
        drive(ctx2.move_to(target, label="nudge", check_stall=False))
        check(True, "a miss inside POSITION_TOLERANCE_M is accepted, not discarded")
    except PhaseTimeout:
        check(False, "a miss inside POSITION_TOLERANCE_M is accepted, not discarded")


def test_stall_detection() -> None:
    print("\n[stall detection]")
    arm = FakeArm(blocked=True)
    ctx = TaskContext(arm)
    try:
        drive(ctx.move_to([0.9, 0.4, 0.5], label="grasp"))
        check(False, "stalled grasp raises")
    except PhaseTimeout as e:
        steps = ctx.phases[-1].steps
        check("stall" in str(e), f"stalled grasp raises a stall error ({e})")
        check(steps <= K.STALL_WINDOW_STEPS + 2,
              f"failed at ~{K.STALL_WINDOW_STEPS} steps rather than the full budget "
              f"(got {steps})")

    # Transit phases must not stall-check: they legitimately hold near-constant error.
    arm2 = FakeArm(blocked=True)
    ctx2 = TaskContext(arm2)
    try:
        drive(ctx2.move_to([0.9, 0.4, 0.5], label="grasp", check_stall=False))
    except PhaseTimeout as e:
        check("stall" not in str(e), "stall check is suppressible for transit phases")


def test_drop_detection() -> None:
    print("\n[drop detection needs BOTH conditions]")
    arm = FakeArm()
    ctx = TaskContext(arm)
    ctx.object_name = "cube"
    ctx.snapshot_object_z()          # rest z = 0.05

    arm.objects["cube"] = [0.5, 0.0, 0.04]
    try:
        drive(ctx.hold(3))
        check(True, "near floor but barely moved: not a drop")
    except ObjectDropped:
        check(False, "near floor but barely moved: not a drop")

    # A deliberate place: relative drop is large but the object is not near the floor.
    ctx2 = TaskContext(arm)
    ctx2.object_name = "cube"
    arm.objects["cube"] = [0.5, 0.0, 0.9]
    ctx2.snapshot_object_z()
    arm.objects["cube"] = [0.5, 0.0, 0.6]
    try:
        drive(ctx2.hold(3))
        check(True, "large relative drop but high up: not a drop")
    except ObjectDropped:
        check(False, "large relative drop but high up: not a drop")

    arm.objects["cube"] = [0.5, 0.0, 0.02]
    try:
        drive(ctx2.hold(3))
        check(False, "near floor AND far below start: IS a drop")
    except ObjectDropped:
        check(True, "near floor AND far below start: IS a drop")


def test_gripper_and_hold() -> None:
    print("\n[gripper + hold]")
    arm = FakeArm()
    ctx = TaskContext(arm)
    n = drive(ctx.close_gripper())
    check(n == K.CLOSE_HOLD_STEPS + K.POST_CLOSE_SETTLE_STEPS,
          f"close runs close+settle steps ({n})")
    check(ctx.settled_gripper is not None, "settled gripper width recorded")
    grips = [c[2] for c in arm.commands]
    biggest = max(abs(grips[i] - grips[i - 1]) for i in range(1, len(grips)))
    check(biggest <= K.MAX_GRIPPER_STEP + 1e-9,
          f"gripper respects its rate cap (max {biggest:.5f})")
    check(all(grips[i] <= grips[i - 1] + 1e-9 for i in range(1, len(grips))),
          "gripper command is MONOTONIC: re-reading the actual position would cap grip force "
          "at one step once the fingers contact the object")
    check(abs(grips[-1] - ctx.closed_value) < 1e-6,
          f"command reaches fully closed and is held there (final {grips[-1]:.4f})")

    arm2 = FakeArm()
    ctx2 = TaskContext(arm2)
    n2 = drive(ctx2.hold(7, discard=True))
    check(n2 == 7, "hold yields exactly the requested steps")
    check(arm2.cache_cleared == 1, "discard clears the recording cache once")
    ctx3 = TaskContext(arm2)
    drive(ctx3.hold(3))
    check(arm2.cache_cleared == 1, "a non-discard hold does not clear the cache")


def test_route_homotopy() -> None:
    print("\n[route — the new v2 primitive]")
    for homotopy in ("over", "around"):
        arm = FakeArm(pos=(0.35, 0.0, 0.10))
        ctx = TaskContext(arm)
        goal = [0.60, 0.0, 0.10]
        drive(ctx.route(goal, [1.0, 0.0, 0.0, 0.0],
                        barrier_top_z=0.18, barrier_y_extent=(-0.15, 0.15),
                        homotopy=homotopy))
        zs = [c[0][2] for c in arm.commands]
        ys = [c[0][1] for c in arm.commands]
        if homotopy == "over":
            check(max(zs) > 0.18, f"over: apex clears the barrier top (max z {max(zs):.3f})")
            check(max(abs(y) for y in ys) < 0.15,
                  "over: stays within the barrier's lateral span")
        else:
            check(max(abs(y) for y in ys) > 0.15,
                  f"around: detours past the barrier edge (max |y| {max(abs(y) for y in ys):.3f})")
            check(max(zs) < 0.18, f"around: stays below the barrier top (max z {max(zs):.3f})")
        check(dist(arm.pos, goal) <= K.POSITION_TOLERANCE_M,
              f"{homotopy}: reaches the goal")

    # The two classes must be geometrically distinguishable, or TOPOLOGY measures nothing.
    paths = {}
    for homotopy in ("over", "around"):
        arm = FakeArm(pos=(0.35, 0.0, 0.10))
        ctx = TaskContext(arm)
        drive(ctx.route([0.60, 0.0, 0.10], [1.0, 0.0, 0.0, 0.0],
                        barrier_top_z=0.18, barrier_y_extent=(-0.15, 0.15), homotopy=homotopy))
        paths[homotopy] = [c[0] for c in arm.commands]
    sep = max(
        dist(paths["over"][min(i, len(paths["over"]) - 1)], paths["around"][i])
        for i in range(len(paths["around"]))
    )
    check(sep > 0.10, f"the two homotopy classes are far apart in space ({sep:.3f} m)")

    # Nearer-edge selection: starting off-centre should detour that way.
    arm = FakeArm(pos=(0.35, 0.10, 0.10))
    ctx = TaskContext(arm)
    drive(ctx.route([0.60, 0.10, 0.10], [1.0, 0.0, 0.0, 0.0],
                    barrier_top_z=0.18, barrier_y_extent=(-0.15, 0.15), homotopy="around"))
    check(max(c[0][1] for c in arm.commands) > 0.15,
          "around: detours past the nearer edge, not across the whole face")

    try:
        arm = FakeArm()
        drive(TaskContext(arm).route([0.5, 0, 0.1], [1, 0, 0, 0], barrier_top_z=0.1,
                                     barrier_y_extent=(-0.1, 0.1), homotopy="sideways"))
        check(False, "an unknown homotopy class is rejected")
    except ValueError:
        check(True, "an unknown homotopy class is rejected")


def test_route_matches_axis() -> None:
    print("\n[route class agrees with the TOPOLOGY axis]")
    trained = [p for p in TOPOLOGY.grid() if TOPOLOGY.in_train(p)]
    # The route primitive has to be exercised in BOTH classes during collection, because both
    # are now trained: 0.02/0.05 demonstrate OVER and 0.12/0.13 demonstrate AROUND. Previously
    # every trained height was OVER, so the collector never ran the around route at all and the
    # policy could not have learned the class existed.
    cls = {expected_homotopy(float(p["barrier_h"])) for p in trained}
    check(cls == {"over", "around"},
          f"collection demonstrates both route classes (got {sorted(cls)})")
    for want in ("over", "around"):
        held = [p for p in TOPOLOGY.grid()
                if not TOPOLOGY.in_train(p)
                and expected_homotopy(float(p["barrier_h"])) == want]
        check(len(held) > 0, f"held-out {want.upper()} conditions exist ({len(held)})")


def test_success_checks() -> None:
    print("\n[success checks]")
    arm = FakeArm()
    ctx = TaskContext(arm)
    ctx.object_name = "cube"

    # No baseline -> no meaning. Silently skipping the rise check is how the weak predicate
    # survived, so this is a hard error rather than a pass.
    arm.objects["cube"] = list(arm.pos)
    try:
        ctx.check_lift()
        check(False, "check_lift without snapshot_object_z() must refuse")
    except TaskFailed as e:
        check(not e.retriable, "missing baseline is a bug, not an unlucky attempt")

    # Baseline taken with the object on the table, then lifted with the gripper: passes.
    arm.objects["cube"] = [arm.pos[0], arm.pos[1], 0.0]
    ctx.snapshot_object_z()
    arm.objects["cube"] = [arm.pos[0], arm.pos[1], 0.10]
    try:
        ctx.check_lift()
        check(True, "risen object under the gripper passes check_lift")
    except TaskFailed as e:
        check(False, f"risen object under the gripper passes check_lift ({e})")

    # THE false positive the strict predicate exists to catch: the gripper closed on nothing,
    # so the object sits at its spawn pose -- still under the gripper, never lifted. The old
    # check passed this, which would have credited a policy for a failed grasp and inflated
    # the in-distribution control every radius is measured against.
    arm.objects["cube"] = [arm.pos[0], arm.pos[1], 0.0]
    try:
        ctx.check_lift()
        check(False, "an object that never rose fails check_lift")
    except TaskFailed as e:
        check(e.retriable, "a failed grasp is retriable")

    # Risen, but not in the gripper: knocked away rather than carried.
    arm.objects["cube"] = [arm.pos[0] + 0.5, arm.pos[1], 0.10]
    try:
        ctx.check_lift()
        check(False, "object far from the gripper fails check_lift")
    except TaskFailed as e:
        check(e.retriable, "check_lift failure is retriable (an unlucky attempt, not a bad "
                           "condition)")

    arm.objects["cube"] = [0.5, 0.1, 0.05]
    try:
        ctx.check_placed((0.5, 0.1))
        check(True, "object at target passes check_placed")
    except TaskFailed:
        check(False, "object at target passes check_placed")
    try:
        ctx.check_placed((0.9, -0.3))
        check(False, "object away from target fails check_placed")
    except TaskFailed:
        check(True, "object away from target fails check_placed")


def test_condition_queue() -> None:
    print("\n[condition queue — advances only on confirmed success]")
    q = ConditionQueue(DIRECTION, ["cube_a", "cube_b"], demos_per_condition=3)
    total = q.total_episodes
    check(total > 0, f"queue non-empty ({total} episodes)")

    c = q.next_condition()
    check(c is not None and c.split == "train", "only train conditions are collected")

    before = q.collected
    for _ in range(5):
        q.record_failure(q.next_condition())
    check(q.collected == before, "failures do NOT advance progress")

    q.confirm_success(q.next_condition())
    check(q.collected == before + 1, "a confirmed success advances progress by one")

    # Round-robin: an interrupted run should span the axis, not a prefix of it. Draw exactly
    # one pass over the conditions -- more than that must repeat, by pigeonhole.
    q2 = ConditionQueue(DIRECTION, ["cube_a"], demos_per_condition=2)
    n_conds = len(q2._order)
    seen = []
    for _ in range(n_conds):
        cc = q2.next_condition()
        seen.append(cc.key())
        q2.confirm_success(cc)
    check(len(set(seen)) == n_conds,
          f"first {n_conds} episodes cover all {n_conds} conditions once "
          f"(round-robin, not drain-one-first)")

    # Drain fully.
    q3 = ConditionQueue(DIRECTION, ["cube_a"], demos_per_condition=1)
    guard = 0
    while not q3.done and guard < 10000:
        q3.confirm_success(q3.next_condition())
        guard += 1
    check(q3.done, "queue drains to done")
    check(q3.next_condition() is None, "an exhausted queue returns None")
    check(q3.collected == q3.total_episodes, "collected == total when done")

    # Partitioning must cover everything exactly once.
    parts = [ConditionQueue(DIRECTION, ["cube_a"], demos_per_condition=1,
                            partition_idx=i, partition_total=3) for i in range(3)]
    keys = [k for p in parts for k in p._remaining]
    whole = ConditionQueue(DIRECTION, ["cube_a"], demos_per_condition=1)
    check(sorted(keys) == sorted(whole._remaining), "3 partitions tile the axis exactly once")
    check(len(set(keys)) == len(keys), "no condition appears in two partitions")
    try:
        ConditionQueue(DIRECTION, ["cube_a"], partition_idx=3, partition_total=3)
        check(False, "an out-of-range partition is rejected")
    except ValueError:
        check(True, "an out-of-range partition is rejected")


def test_task_base() -> None:
    print("\n[task base]")
    c = ConditionQueue(DIRECTION, ["cube_a"]).next_condition()

    class Lift(Task):
        def run(self, ctx):
            yield from ctx.hold(2)

    t = Lift(c)
    check(t.instruction == c.instruction,
          "instruction comes from the condition, so it matches the axis")
    check(drive(t.run(TaskContext(FakeArm()))) == 2, "task generator drives")
    try:
        drive(Task(c).run(TaskContext(FakeArm())))
        check(False, "the base Task refuses to run unimplemented")
    except NotImplementedError:
        check(True, "the base Task refuses to run unimplemented")


def main() -> int:
    test_rate_caps()
    test_slerp_shortest_arc()
    test_timeout_budget()
    test_move_to_converges()
    test_move_to_timeout_and_soft_accept()
    test_stall_detection()
    test_drop_detection()
    test_gripper_and_hold()
    test_route_homotopy()
    test_route_matches_axis()
    test_success_checks()
    test_condition_queue()
    test_task_base()

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
