# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""The narrow interface a scripted expert needs from the simulator, plus the motion maths.

The collection stack this was adapted from routed its task primitives through a controller
class built for a *mobile dual-arm* robot with teleop, a web UI, and an LLM planner — roughly
two dozen methods, most of them irrelevant to a fixed single-arm Franka. Depending on that
surface would have made the primitives untestable without booting Isaac Sim.

Instead: ``MotionBackend`` is the eight-method protocol the primitives actually use, and every
rate-limiting / error / timeout computation lives here as a plain function over lists of
floats. That means the logic where bugs actually hide — step capping, slerp rate limits,
timeout budgets, stall detection — is testable against a kinematics stub with no simulator
present. The Isaac implementation is a thin adapter over ``RobotPrimitiveMixin``
(``arbiter/sim/robot/primitives.py``), which already provides ``getp`` / ``movep`` / ``getj`` /
gripper control over differential IK.

Single-arm by design: no ``arm_side`` parameter anywhere. v2 is a single-arm Franka benchmark
and vestigial dual-arm plumbing is how the v1 confusion started. ``arbiter/sim/robot/FFW/``
stays in the repo for the separate dual-arm paper.

Quaternions are ``(w, x, y, z)``, matching the rest of the repo. Positions are in the robot
base frame unless a name says ``_world``.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from arbiter.collect.constants import (
    MAX_ORIENTATION_STEP_RAD,
    MAX_TRANSLATION_STEP_M,
    MAX_WRIST_STEP_RAD,
    MOVEMENT_TIMEOUT_GAIN,
    MOVEMENT_TIMEOUT_MAX_STEPS,
    PHASE_TIMEOUT_STEPS,
)

Vec3 = list[float]
Quat = list[float]


@runtime_checkable
class MotionBackend(Protocol):
    """What a scripted expert needs from the world. Implemented for Isaac; stubbed in tests."""

    def getp(self) -> tuple[Vec3, Quat]:
        """End-effector pose in the robot base frame, as ``(pos_xyz, quat_wxyz)``."""
        ...

    def getj(self) -> list[float]:
        """Arm joint positions."""
        ...

    def movep(self, pos_b: Vec3, quat_b: Quat, gripper: float) -> None:
        """Command an EEF pose target and a gripper width. One control step."""
        ...

    def gripper_pos(self) -> float:
        """Current gripper joint position."""
        ...

    def object_pos_world(self, name: str) -> Vec3:
        """Object origin in world coordinates."""
        ...

    def object_pos_in_base(self, name: str) -> Vec3:
        """Object origin in the robot base frame."""
        ...

    def step(self) -> None:
        """Advance the simulation by one control step."""
        ...

    def clear_recording_cache(self) -> None:
        """Drop buffered frames — used for camera settle frames that must not be recorded."""
        ...


# ──────────────────────────────────────────────────────────────────────────────
# quaternion helpers
# ──────────────────────────────────────────────────────────────────────────────

def quat_normalize(q: Quat) -> Quat:
    n = math.sqrt(sum(c * c for c in q))
    if n < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [c / n for c in q]


def quat_angle_between(a: Quat, b: Quat) -> float:
    """Angular distance in radians. Sign-insensitive: q and -q are the same rotation."""
    dot = min(abs(sum(x * y for x, y in zip(a, b))), 1.0)
    return 2.0 * math.acos(dot)


def quat_slerp(cur: Quat, tgt: Quat, fraction: float) -> Quat:
    """Shortest-arc slerp, falling back to lerp when the rotations are nearly identical.

    Taking the shortest arc matters: without the sign flip the expert can rotate the wrist the
    long way round, producing a demonstration that looks like a deliberate 300-degree twist.
    """
    tgt = list(tgt)
    dot = sum(x * y for x, y in zip(cur, tgt))
    if dot < 0.0:
        tgt = [-c for c in tgt]
        dot = -dot
    dot = min(dot, 1.0)

    if dot > 0.9995:
        return quat_normalize([c + fraction * (t - c) for c, t in zip(cur, tgt)])

    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    s0 = math.sin((1.0 - fraction) * theta) / sin_theta
    s1 = math.sin(fraction * theta) / sin_theta
    return quat_normalize([s0 * c + s1 * t for c, t in zip(cur, tgt)])


# ──────────────────────────────────────────────────────────────────────────────
# rate-limited stepping
# ──────────────────────────────────────────────────────────────────────────────

def dist(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def step_position_towards(current: Vec3, desired: Vec3) -> Vec3:
    """One rate-capped translation step from ``current`` toward ``desired``.

    Snaps to the target once inside one step's reach, so the expert converges exactly rather
    than oscillating around the goal.
    """
    delta = [d - c for c, d in zip(current, desired)]
    distance = math.sqrt(sum(x * x for x in delta))
    if distance <= MAX_TRANSLATION_STEP_M:
        return list(desired)
    scale = MAX_TRANSLATION_STEP_M / max(distance, 1e-8)
    return [c + d * scale for c, d in zip(current, delta)]


def step_orientation_towards(
    current_pos: Vec3, current_quat: Quat, desired_pos: Vec3, desired_quat: Quat
) -> Quat:
    """One rate-capped orientation step, coupled to how far the position still has to travel.

    Two behaviours worth preserving, both learned the hard way upstream:

    - While the position is still far, the orientation fraction is tied to the *position* step
      rate, so translation and rotation finish together instead of the wrist snapping early.
    - Once the position has converged, the fraction opens to 1.0 but the absolute cap
      ``MAX_ORIENTATION_STEP_RAD`` takes over. Without that cap the wrist snaps as position
      error shrinks, and the IK controller drifts in position while chasing it.
    """
    pos_distance = dist(current_pos, desired_pos)
    if pos_distance <= MAX_TRANSLATION_STEP_M:
        fraction = 1.0
    else:
        fraction = MAX_TRANSLATION_STEP_M / max(pos_distance, 1e-8)

    theta = quat_angle_between(current_quat, desired_quat) * 0.5   # slerp arc, not the 2x angle
    if theta > 1e-8:
        fraction = min(fraction, MAX_ORIENTATION_STEP_RAD / theta)
    return quat_slerp(current_quat, desired_quat, min(max(fraction, 0.0), 1.0))


def movement_timeout_steps(
    pos_error: float, orient_error: float | None = None
) -> int:
    """Step budget for a phase, from how far it has to travel.

    A fixed budget is wrong in both directions: too small for a long transit, and wastefully
    large for a 2 cm approach. Scaling by distance and then multiplying by a slack gain gives
    a phase room for IK transients without letting a stuck one run forever.
    """
    pos_steps = math.ceil(pos_error / max(MAX_TRANSLATION_STEP_M, 1e-6))
    orient_steps = 0
    if orient_error is not None:
        orient_steps = math.ceil(orient_error / max(MAX_WRIST_STEP_RAD, 1e-6))
    required = max(pos_steps, orient_steps)
    budget = math.ceil(required * MOVEMENT_TIMEOUT_GAIN)
    return int(min(max(PHASE_TIMEOUT_STEPS, budget), MOVEMENT_TIMEOUT_MAX_STEPS))
