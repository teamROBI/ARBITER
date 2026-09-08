# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted expert tasks, one behaviour per axis family.

Four behaviours cover all seven axes, because several axes differ only in which parameter is
swept rather than in what the arm does:

===========  =====================================  ==============================
Behaviour    Axes                                   What it demonstrates
===========  =====================================  ==============================
``LiftTask``      POSITION, APPROACH                pick up and lift
``TransportTask`` DIRECTION, EXTENT, FACTORIAL      pick up, carry, place
``RouteTask``     TOPOLOGY                          carry past a barrier
``OrderTask``     ORDER                             two transports in a stated order
===========  =====================================  ==============================

Each task reads its geometry **from the condition's parameter vector**, never from a
hardcoded table. That is the point of the v2 schema: the expert, the axis definition, and the
evaluation split all derive from one source, so a collection run cannot quietly demonstrate
something the axis does not describe. v1 could — its conditions were named spawn areas and
the expert's geometry was written out separately.

Every task ends by asserting the outcome it was supposed to produce, and raises
``TaskFailed(retriable=True)`` when it does not. The queue then re-attempts the condition
rather than banking a bad demonstration, which is what keeps the collected distribution equal
to the declared one.
"""

from __future__ import annotations

import math
from typing import Generator

from arbiter.collect import constants as K
from arbiter.collect.context import TaskContext, TaskFailed
from arbiter.collect.grasp import grasp_poses
from arbiter.collect.task import Task
from arbiter.suites.spec import (
    ORDER_OBJECTS,
    order_move_sequence,
    WORKSPACE,
    barrier_half_width,
    expected_homotopy,
    forced_bypass_side,
    topology_endpoints,
    transport_target,
)

Vec3 = list[float]


def _param(cond, name: str, default=None) -> float:
    v = cond.params.get(name, default)
    if v is None:
        raise TaskFailed(f"condition {cond.key()} is missing parameter {name!r}")
    return float(v)


def _polar_target(cx: float, cy: float, radius: float, theta_deg: float) -> tuple[float, float]:
    """Point at ``radius`` and ``theta_deg`` from a centre, in table coordinates."""
    t = math.radians(theta_deg)
    return cx + radius * math.cos(t), cy + radius * math.sin(t)


class LiftTask(Task):
    """Pick up the target object and lift it clear of the table.

    Covers POSITION (object position swept) and APPROACH (grasp azimuth swept). The azimuth
    comes from the condition when the axis sweeps it, and defaults to zero otherwise, so one
    behaviour serves both without a branch on axis name.
    """

    def run(self, ctx: TaskContext) -> Generator:
        cond = self.condition
        azimuth = float(cond.params.get("phi_deg", 0.0))

        obj = ctx.b.object_pos_in_base(ctx.object_name)
        grasp, pregrasp, quat = grasp_poses(obj, azimuth_deg=azimuth)

        ctx.snapshot_object_z()
        yield from ctx.hold(K.DISCARD_HOLD_STEPS, discard=True)
        yield from ctx.hold(K.RECORD_HOLD_STEPS)

        yield from ctx.move_to(pregrasp, quat, gripper="open", label="pregrasp")
        yield from ctx.move_to(grasp, quat, gripper="open", label="grasp")
        yield from ctx.close_gripper()
        yield from ctx.lift(K.LIFT_HEIGHT_M)
        yield from ctx.retract()
        yield from ctx.hold(K.SETTLE_HOLD_STEPS, label="settle")

        ctx.check_lift()


class TransportTask(Task):
    """Pick up the object, carry it to a target, and release.

    Covers DIRECTION (azimuth of travel), EXTENT (distance travelled) and FACTORIAL (both,
    crossed). The object always starts at the workspace centre for DIRECTION, which is what
    removes v1 Reverse's terminal-state confound — there the centre appeared only as an
    episode's *final* state, so a test episode beginning with the object at the centre showed
    the policy what it had learned to treat as task completion.
    """

    def run(self, ctx: TaskContext) -> Generator:
        cond = self.condition
        cx, cy = WORKSPACE.centre

        # From the spec, not recomputed here. The expert, the success predicate and the policy
        # evaluator all need this point, and a second copy drifts silently -- the symptom being
        # a policy graded against a target the demonstrations never used.
        target = transport_target(cond)
        if target is None:
            raise TaskFailed(f"{cond.key()}: no transport target derivable from parameters")
        tx, ty = target

        if not WORKSPACE.contains(tx, ty):
            # A geometry bug, not an unlucky attempt: do not let the queue retry it forever.
            raise TaskFailed(
                f"{cond.key()}: target ({tx:.3f}, {ty:.3f}) is outside the workspace",
                retriable=False,
            )

        obj = ctx.b.object_pos_in_base(ctx.object_name)
        grasp, pregrasp, quat = grasp_poses(obj)
        half_h = grasp[2] - obj[2]

        ctx.snapshot_object_z()
        yield from ctx.hold(K.DISCARD_HOLD_STEPS, discard=True)
        yield from ctx.hold(K.RECORD_HOLD_STEPS)

        yield from ctx.move_to(pregrasp, quat, gripper="open", label="pregrasp")
        yield from ctx.move_to(grasp, quat, gripper="open", label="grasp")
        yield from ctx.close_gripper()

        carry_z = grasp[2] + K.LIFT_HEIGHT_M
        yield from ctx.move_to([grasp[0], grasp[1], carry_z], quat, label="lift")
        yield from ctx.move_to([tx, ty, carry_z], quat, label="carry", check_stall=False)
        yield from ctx.move_to([tx, ty, obj[2] + half_h], quat, label="lower")
        yield from ctx.open_gripper()
        yield from ctx.retract()
        yield from ctx.hold(K.SETTLE_HOLD_STEPS, label="settle")

        ctx.check_placed((tx, ty))


class RouteTask(Task):
    """Carry the object past a barrier, via the homotopy class the barrier height dictates.

    TOPOLOGY's whole construction: start, goal and object are fixed while the barrier height
    varies, so the observation changes smoothly and the required trajectory changes
    discontinuously at ``h*``. Which class applies comes from
    :func:`arbiter.suites.spec.expected_homotopy`, so the barrier asset, the axis definition and
    this expert cannot disagree about it.
    """

    def run(self, ctx: TaskContext) -> Generator:
        cond = self.condition
        h = _param(cond, "barrier_h")
        offset = float(cond.params.get("barrier_offset", 0.0))
        homotopy = expected_homotopy(h)

        # start_xy is realised by the scene placing the object there; the expert only needs the
        # goal, but both come from one spec function so they cannot disagree.
        (_start_x, cy), (goal_x, _goal_y) = topology_endpoints()
        barrier_y = cy + offset
        barrier_half_w = barrier_half_width()      # from the spec, never restated
        barrier_top = h

        obj = ctx.b.object_pos_in_base(ctx.object_name)
        grasp, pregrasp, quat = grasp_poses(obj)
        half_h = grasp[2] - obj[2]

        ctx.snapshot_object_z()
        yield from ctx.hold(K.DISCARD_HOLD_STEPS, discard=True)
        yield from ctx.hold(K.RECORD_HOLD_STEPS)

        yield from ctx.move_to(pregrasp, quat, gripper="open", label="pregrasp")
        yield from ctx.move_to(grasp, quat, gripper="open", label="grasp")
        yield from ctx.close_gripper()
        yield from ctx.move_to([grasp[0], grasp[1], grasp[2] + K.LIFT_HEIGHT_M], quat,
                               label="lift")

        # DETOUR: when the condition carries a blocker, the lane `route()`'s tie-break would
        # take is physically sealed, so the expert must detour to the complement. The side comes
        # from the spec's `forced_bypass_side`, which mirrors that tie-break -- the blocker
        # placement in `props.blocker_xy` reads the same function, so the sealed lane and
        # the avoided lane cannot disagree. Restating the choice here is precisely how the
        # barrier offset came to be honoured in one place and ignored in another.
        side = forced_bypass_side(offset) if float(cond.params.get("blocker", 0.0)) else None

        goal = [goal_x, cy, obj[2] + half_h + K.LIFT_HEIGHT_M]
        yield from ctx.route(
            goal, quat,
            barrier_top_z=barrier_top,
            barrier_y_extent=(barrier_y - barrier_half_w, barrier_y + barrier_half_w),
            homotopy=homotopy,
            side=side,
        )

        yield from ctx.move_to([goal_x, cy, obj[2] + half_h], quat, label="lower")
        yield from ctx.open_gripper()
        yield from ctx.retract()
        yield from ctx.hold(K.SETTLE_HOLD_STEPS, label="settle")

        ctx.check_placed((goal_x, cy))


class OrderTask(Task):
    """Transport two objects to their targets in the order the instruction states.

    ORDER instantiates the cell nothing in v1 covered: identical pixels at ``t=0``, different
    required trajectory. Both objects and both targets sit at fixed positions, so the only
    thing distinguishing the two conditions is the sequence — which means a failure here
    cannot be explained by anything the policy can see at the first frame.

    Object and target positions come from the caller rather than the parameter vector, because
    they are fixed scene furniture for this axis rather than swept quantities.
    """

    def __init__(self, condition, *, objects: list[str], targets: list[tuple[float, float]]):
        super().__init__(condition)
        n = len(ORDER_OBJECTS)
        if len(objects) != n or len(targets) != n:
            raise ValueError(
                f"OrderTask needs exactly {n} objects and {n} targets, "
                f"got {len(objects)} and {len(targets)}"
            )
        self.objects = list(objects)
        self.targets = list(targets)

    def run(self, ctx: TaskContext) -> Generator:
        # order_move_sequence owns the mapping from the condition to an execution order; the
        # evaluator asks the same function, so the demonstrated sequence and the scored
        # sequence cannot drift apart.
        sequence = order_move_sequence(self.condition)
        if sorted(sequence) != list(range(len(self.objects))):
            raise TaskFailed(
                f"sequence {sequence} is not a permutation of "
                f"{list(range(len(self.objects)))}", retriable=False)

        yield from ctx.hold(K.DISCARD_HOLD_STEPS, discard=True)
        yield from ctx.hold(K.RECORD_HOLD_STEPS)

        for slot, idx in enumerate(sequence):
            name = self.objects[idx]
            tx, ty = self.targets[idx]
            ctx.object_name = name
            ctx.snapshot_object_z()

            obj = ctx.b.object_pos_in_base(name)
            grasp, pregrasp, quat = grasp_poses(obj)
            half_h = grasp[2] - obj[2]
            carry_z = grasp[2] + K.LIFT_HEIGHT_M
            tag = f"obj{slot}"

            yield from ctx.move_to(pregrasp, quat, gripper="open", label=f"{tag}_pregrasp")
            yield from ctx.move_to(grasp, quat, gripper="open", label=f"{tag}_grasp")
            yield from ctx.close_gripper(label=f"{tag}_close")
            yield from ctx.move_to([grasp[0], grasp[1], carry_z], quat, label=f"{tag}_lift")
            yield from ctx.move_to([tx, ty, carry_z], quat, label=f"{tag}_carry",
                                   check_stall=False)
            yield from ctx.move_to([tx, ty, obj[2] + half_h], quat, label=f"{tag}_lower")
            yield from ctx.open_gripper(label=f"{tag}_open")
            ctx.check_placed((tx, ty))
            # Lift straight up before the loop traverses to the next block. Without this the
            # gripper went from the release pose (~4 cm) directly to the next pregrasp, sweeping
            # laterally at cube height right through the block it had just placed: measured, the
            # red block was knocked from y=-0.097 to y=-0.048, 4.9 cm off its target, and stayed
            # there. The episode still scored success only because the place margin was 8 cm.
            yield from ctx.move_to([tx, ty, carry_z], quat, label=f"{tag}_clear")

        yield from ctx.retract()
        yield from ctx.hold(K.SETTLE_HOLD_STEPS, label="settle")


#: Which behaviour collects which axis. One place to look, so a new axis cannot be added
#: without deciding how it gets demonstrated.
TASK_FOR_AXIS: dict[str, type[Task]] = {
    "position": LiftTask,
    "approach": LiftTask,
    "direction": TransportTask,
    "extent": TransportTask,
    "factorial": TransportTask,
    "topology": RouteTask,
    "order": OrderTask,
}


def task_for(condition) -> type[Task]:
    axis = condition.axis
    if axis not in TASK_FOR_AXIS:
        raise KeyError(
            f"no scripted expert registered for axis '{axis}'. "
            f"Known: {', '.join(sorted(TASK_FOR_AXIS))}"
        )
    return TASK_FOR_AXIS[axis]
