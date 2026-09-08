# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Success predicates, shared by the scripted expert and the policy evaluator.

One definition, because the two must agree. The expert's success rate is what certifies a
condition as achievable; the policy's success rate on the same condition is the measurement.
If those are computed by different code, the benchmark is comparing two different questions
and the retention ratio `SR(test)/SR(ID)` means nothing.

Pure functions of pose, stdlib only: no Isaac, no `TaskContext`. That keeps them importable by
the evaluator, which runs in a different process from the collector, and testable without a
simulator.

**`lifted` checks that the object actually rose.** The original expert-side check verified only
that the object was horizontally near the gripper after the retract, and
`SUCCESS_LIFT_DELTA_M` was defined but never referenced. That admits a false positive: a grasp
that closes on nothing leaves the object at its spawn pose, and if that pose is near where the
arm retracts to, the check passes. For the expert that is mostly harmless -- it does grasp --
but a policy evaluated under it would be credited for failures, inflating the in-distribution
control and deflating every radius measured against it.
"""

from __future__ import annotations

import math

from arbiter.collect import constants as K

Vec3 = list[float] | tuple[float, ...]


def _xy_dist(a: Vec3, b: Vec3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def lifted(*, object_z_world: float, rest_z_world: float,
           object_xy: Vec3, eef_xy: Vec3,
           min_rise: float = K.SUCCESS_LIFT_DELTA_M,
           max_xy: float = K.SUCCESS_RETRACT_DIST_M + K.POSITION_TOLERANCE_M
           ) -> tuple[bool, str]:
    """Object raised clear of the table and still in the gripper.

    Both halves are load-bearing. The rise alone would accept an object knocked into the air;
    the proximity alone accepts an object that never moved.

    Keyword-only, with the frame in every name, because the two halves live in *different*
    frames and mixing them is silent. `TaskContext.rest_object_z` is a **world** z, while
    `object_pos_in_base` is base-frame; those differ by the robot base offset (~0.75 m), so a
    rise computed across them is wrong by that amount and the check fails always. That is
    exactly the mistake this signature exists to prevent -- `object_pos`/`rest_z` positionally
    gave no hint that one was world and one was not.
    """
    rise = object_z_world - rest_z_world
    if rise < min_rise:
        return False, f"object did not rise ({rise * 100:.1f} cm < {min_rise * 100:.1f} cm)"
    d = _xy_dist(object_xy, eef_xy)
    if d > max_xy:
        return False, f"object not in gripper (eef-obj XY {d:.3f} m > {max_xy:.3f} m)"
    return True, ""


def placed(object_pos_world: Vec3, target_xy: tuple[float, float], *,
           margin: float = K.POSITION_TOLERANCE_M) -> tuple[bool, str]:
    """Object came to rest near a target, in world XY.

    The margin is POSITION_TOLERANCE_M, which is also the target pad's half-width, so the
    marker the policy can see is exactly the region this accepts. It was 0.08 -- more than twice
    the pad -- in BOTH collection and evaluation, which broke two things at once. A block could
    come to rest visibly off its pad and still score success; and on EXTENT, whose trained
    lengths are only 4 cm apart, an 8 cm tolerance means a policy emitting one habitual distance
    passes most of the test grid, so the axis measured the tolerance rather than any
    generalization limit.
    """
    d = _xy_dist(object_pos_world, target_xy)
    if d > margin:
        return False, f"object not at target (XY {d:.3f} m > {margin:.3f} m)"
    return True, ""


def dropped(object_pos_world: Vec3, rest_z_world: float) -> tuple[bool, str]:
    """Object fell off the table.

    Both conditions must hold. Either alone gives false positives: a low table trips the floor
    test, and a deliberate place trips the relative-drop test.
    """
    near_floor = object_pos_world[2] < K.FALL_FLOOR_Z_THRESHOLD_M
    fell_far = (rest_z_world - object_pos_world[2]) > K.FALL_RELATIVE_DROP_M
    if near_floor and fell_far:
        return True, (f"object dropped (z {object_pos_world[2]:.3f} m, "
                      f"{rest_z_world - object_pos_world[2]:.3f} m below rest)")
    return False, ""


#: Which predicate decides each axis, and what it needs. The evaluator reads this rather than
#: branching on axis name in its own code -- the same reason the axis owns its grid and split.
SUCCESS_KIND = {
    "position": "lift",
    "approach": "lift",
    "extent": "place",
    "direction": "place",
    "factorial": "place",
    "topology": "place",
    "order": "place_both",
}


def success_kind(axis_name: str) -> str:
    if axis_name not in SUCCESS_KIND:
        raise KeyError(
            f"no success predicate registered for axis '{axis_name}'. "
            f"Known: {', '.join(sorted(SUCCESS_KIND))}"
        )
    return SUCCESS_KIND[axis_name]
