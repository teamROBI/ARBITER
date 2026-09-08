# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic grasp poses for the scripted expert.

No learned grasp predictor. The v1 collection used the same approach and the reason still
holds: for a coverage benchmark the expert must be *deterministic*, because the training
distribution has to match the declared one exactly. A sampling grasp predictor would put
variation into the demonstrations that the axis definition does not account for, and the
support-distance metric would then be measuring against a moving target.

Objects are small primitives on a table, so a top-down parallel-jaw grasp works for all of
them. The one free parameter is the **azimuth**: the yaw of the jaw-opening axis in the table
plane. That is exactly what the APPROACH axis sweeps, and what a fixture wall blocks.

Frame convention, matching Franka's ``panda_hand``:

- ``eef_z`` points *out* of the gripper, toward the object. Top-down means ``(0, 0, -1)``.
- ``eef_x`` is the jaw-opening axis.
- ``eef_y`` completes a right-handed frame.

Azimuth is periodic at 180 degrees, not 360: a parallel jaw closing along ``+x`` is the same
grasp as one closing along ``-x``. That is why ``APPROACH_TEST_PHI`` in
``arbiter/suites/spec.py`` stops at 180 rather than sweeping the full circle — beyond that the
sweep would revisit grasps it had already covered while reporting them as further from
support.
"""

from __future__ import annotations

import math

from arbiter.collect.constants import FINAL_APPROACH_EXTRA_M, PREGRASP_OFFSET_M

Vec3 = list[float]
Quat = list[float]

#: Distance from the ``panda_hand`` frame origin (the wrist) to the point between the
#: fingertips, along the approach axis. Matches the ee_frame offset IsaacLab's own Franka
#: manipulation environments use.
#:
#: Lives here because it is grasp geometry, and because getting it wrong is silent: an IK
#: target placed at the object's position without this offset overstates reach by 10 cm and
#: marks unreachable points reachable. It was duplicated in probe_reachability and then missing
#: from render_scene, which is exactly how a constant like this drifts.
GRASP_POINT_OFFSET_M = 0.1034


def mat_to_quat(r: list[Vec3]) -> Quat:
    """Rotation matrix (row-major 3x3) to quaternion ``(w, x, y, z)``.

    Branches on the largest diagonal term rather than always using the trace formula: the
    trace branch divides by ``sqrt(1 + trace)``, which loses precision and then blows up for
    rotations near 180 degrees — and a top-down grasp is a 180-degree rotation about a
    horizontal axis, so that degenerate case is the *common* one here, not an edge case.
    """
    m00, m01, m02 = r[0]
    m10, m11, m12 = r[1]
    m20, m21, m22 = r[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s]
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s]
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s]
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s]


def _normalize(v: Vec3) -> Vec3:
    n = math.sqrt(sum(c * c for c in v))
    if n < 1e-12:
        raise ValueError(f"cannot normalize a zero vector: {v}")
    return [c / n for c in v]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def grasp_frame(azimuth_deg: float, tilt_deg: float = 0.0) -> list[Vec3]:
    """Column-major-as-rows rotation matrix for a grasp at this azimuth and tilt.

    ``tilt_deg`` rotates the approach away from straight-down, toward the azimuth direction.
    Zero (straight down) is right for small primitives on a table; a nonzero tilt exists
    because a cluttered or fixture-bounded approach sometimes needs to come in at a slant.
    """
    phi = math.radians(azimuth_deg)
    tilt = math.radians(tilt_deg)

    # Approach direction: straight down, tilted toward the azimuth heading by `tilt`.
    heading = [math.cos(phi), math.sin(phi), 0.0]
    approach = _normalize([
        heading[0] * math.sin(tilt),
        heading[1] * math.sin(tilt),
        -math.cos(tilt),
    ])

    eef_z = approach
    # Jaw axis: the azimuth heading, projected off the approach so the frame stays orthogonal.
    proj = sum(h * z for h, z in zip(heading, eef_z))
    jaw = [h - proj * z for h, z in zip(heading, eef_z)]
    if math.sqrt(sum(c * c for c in jaw)) < 1e-8:
        # Degenerate only if the approach is parallel to the heading, i.e. tilt = 90 degrees.
        jaw = [0.0, 0.0, 1.0]
        proj = sum(j * z for j, z in zip(jaw, eef_z))
        jaw = [j - proj * z for j, z in zip(jaw, eef_z)]
    eef_x = _normalize(jaw)
    eef_y = _cross(eef_z, eef_x)

    # Rows of the returned matrix are the world-frame rows, columns are the EEF axes.
    return [
        [eef_x[0], eef_y[0], eef_z[0]],
        [eef_x[1], eef_y[1], eef_z[1]],
        [eef_x[2], eef_y[2], eef_z[2]],
    ]


def grasp_quat(azimuth_deg: float, tilt_deg: float = 0.0) -> Quat:
    """Quaternion ``(w, x, y, z)`` for a grasp at this azimuth and tilt."""
    return mat_to_quat(grasp_frame(azimuth_deg, tilt_deg))


def canonical_azimuth(azimuth_deg: float) -> float:
    """Fold an azimuth into ``[0, 180)``, the period of a parallel-jaw grasp."""
    return float(azimuth_deg) % 180.0


def grasp_poses(
    object_pos: Vec3,
    *,
    azimuth_deg: float = 0.0,
    tilt_deg: float = 0.0,
    origin_to_centre_z: float = 0.0,
    pregrasp_offset_m: float = PREGRASP_OFFSET_M,
) -> tuple[Vec3, Vec3, Quat]:
    """``(grasp_pos, pregrasp_pos, quat)`` for grasping an object at ``object_pos``.

    The grasp point goes at the object's *centre*, so the jaws straddle its body.

    ``origin_to_centre_z`` defaults to zero because v2's procedural assets
    (``create_assets.py``) put the USD origin at the object centre. It exists for assets
    whose origin is at the base, where the centre is half a height up.

    Getting this wrong is not a near miss, it is a total failure that looks like bad physics:
    with a 0.025 default against centre-origin cubes the target landed on the cube's top face,
    the jaws closed on air, and the achievability gate reported "object not in gripper after
    retract" on 7 of 8 conditions the kinematic probe had already passed.

    The pregrasp standoff is along the approach direction -- directly above the grasp for a
    top-down approach, offset laterally for a tilted one.
    """
    frame = grasp_frame(azimuth_deg, tilt_deg)
    approach = [frame[0][2], frame[1][2], frame[2][2]]      # third column = eef_z

    grasp = [
        float(object_pos[0]),
        float(object_pos[1]),
        float(object_pos[2]) + float(origin_to_centre_z),
    ]
    standoff = pregrasp_offset_m + FINAL_APPROACH_EXTRA_M
    pregrasp = [g - a * standoff for g, a in zip(grasp, approach)]
    return grasp, pregrasp, mat_to_quat(frame)
