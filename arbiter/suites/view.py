# Copyright 2026 ARBITER Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Head-camera projection, as pure geometry.

Exists because two questions about the arbitration cells are answerable analytically and were
previously answered by rendering and squinting:

1. **Is the visual cue actually in the picture?** A blocker the policy cannot see is not a
   vision channel, it is a null.
2. **Does the cue hide something the task needs?** If the prop occludes the object or the target,
   a behaviour change is confounded -- the policy might be failing to see the goal rather than
   re-routing.

Both reduce to projecting points into the head camera's frustum, which needs no simulator. That
matters more than convenience: TANGO's record is that *programmatic checks are blind to
composition* -- a barrier rotated 90 degrees, a camera aimed at the shoulder and a robot floating
18 cm off the table all passed every assertion and were caught by looking at an image. This
module does not replace that. It closes the narrower, quantitative questions so that looking at
an image is spent on the things arithmetic cannot see.

Stdlib only.

.. warning::
   The camera geometry here is **not** a free parameter. It reproduces LIBERO's ``agentview``
   (45 degrees vertical), and Phase 0 evaluates a checkpoint trained under exactly this view.
   Widening the FOV or raising the camera to make a prop fit would silently invalidate every
   borrowed checkpoint, and the resulting mismatch would be indistinguishable from a scene-copy
   error. Change the prop, never the camera.
"""

from __future__ import annotations

import math

from arbiter.suites.layout_gen import HEAD_CAM, HEAD_CAM_FOCAL_MM

#: IsaacLab's horizontal aperture, in millimetres. Paired with `HEAD_CAM_FOCAL_MM` this is what
#: makes the vertical FOV come out at LIBERO's 45 degrees on a 672x376 frame.
HORIZONTAL_APERTURE_MM = 20.955
FRAME_W, FRAME_H = 672, 376

Vec3 = tuple[float, float, float]


def _norm(v: Vec3) -> Vec3:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return (v[0] / n, v[1] / n, v[2] / n)


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def half_fov_deg() -> tuple[float, float]:
    """``(horizontal, vertical)`` half-angles of the head camera, in degrees."""
    ap_v = HORIZONTAL_APERTURE_MM * FRAME_H / FRAME_W
    return (
        math.degrees(math.atan(HORIZONTAL_APERTURE_MM / 2.0 / HEAD_CAM_FOCAL_MM)),
        math.degrees(math.atan(ap_v / 2.0 / HEAD_CAM_FOCAL_MM)),
    )


def camera_basis() -> tuple[Vec3, Vec3, Vec3]:
    """``(forward, right, up)`` for the head camera, in world coordinates.

    This is where the instruction vocabulary's frame convention comes from, and it is worth
    stating rather than trusting: the camera sits on the ``y = 0`` plane looking down the ``-x``
    axis at 45 degrees, so ``right`` comes out as ``+y`` and ``up`` as ``-x``. That is why the
    spec's compass words put image-right at ``+y``, and why "far"/"near" are used instead of
    "top"/"bottom". TANGO had every DIRECTION instruction 90 degrees out for weeks on exactly
    this point, and nothing the benchmark reported would have revealed it.
    """
    pos: Vec3 = tuple(HEAD_CAM["pos"])  # type: ignore[assignment]
    look: Vec3 = tuple(HEAD_CAM["look_at"])  # type: ignore[assignment]
    fwd = _norm(_sub(look, pos))
    right = _norm(_cross(fwd, (0.0, 0.0, 1.0)))
    up = _norm(_cross(right, fwd))
    return fwd, right, up


def camera_pos() -> Vec3:
    return tuple(HEAD_CAM["pos"])  # type: ignore[return-value]


def view_angles_deg(p: Vec3) -> tuple[float, float] | None:
    """Signed ``(horizontal, vertical)`` angles of ``p`` off the optical axis.

    ``None`` when the point is behind the camera. Angles are used rather than pixel coordinates
    so the result is independent of the projection convention (the head camera is ``"world"``,
    the wrist camera ``"ros"``), which is one fewer thing to get silently backwards.
    """
    fwd, right, up = camera_basis()
    v = _sub(p, camera_pos())
    depth = _dot(v, fwd)
    if depth <= 0.0:
        return None
    return (
        math.degrees(math.atan2(_dot(v, right), depth)),
        math.degrees(math.atan2(_dot(v, up), depth)),
    )


def in_frame(p: Vec3) -> bool:
    """Is ``p`` inside the head camera's frustum?"""
    a = view_angles_deg(p)
    if a is None:
        return False
    hh, hv = half_fov_deg()
    return abs(a[0]) < hh and abs(a[1]) < hv


def occluded_by_box(
    p: Vec3,
    centre: tuple[float, float],
    y_span: tuple[float, float],
    z_span: tuple[float, float],
    depth: float,
    samples: int = 512,
) -> bool:
    """Does the segment from the camera to ``p`` pass through an axis-aligned box?

    Sampled rather than solved analytically: the boxes here are thin slabs (2 cm deep), and a
    slab-intersection routine that is subtly wrong at an edge fails silently, whereas a dense
    sample that misses a 2 cm slab would have to skip 512 steps over a ~1 m ray. Cheap and hard
    to get wrong beats elegant and unverified for a check whose whole job is to be trusted.
    """
    cx = centre[0]
    x0, y0, z0 = camera_pos()
    x1, y1, z1 = p
    lo_y, hi_y = min(y_span), max(y_span)
    lo_z, hi_z = min(z_span), max(z_span)
    eps = 1e-6
    for i in range(1, samples):
        t = i / samples
        x = x0 + (x1 - x0) * t
        if abs(x - cx) > depth / 2.0 + eps:
            continue
        y = y0 + (y1 - y0) * t
        if not (lo_y - eps <= y <= hi_y + eps):
            continue
        z = z0 + (z1 - z0) * t
        if lo_z - eps <= z <= hi_z + eps:
            return True
    return False


def face_fraction_in_frame(
    centre: tuple[float, float],
    y_span: tuple[float, float],
    z_span: tuple[float, float],
    samples: int = 60,
) -> float:
    """Fraction of a box's camera-facing rectangle that lands inside the frustum.

    "Fully in frame" is deliberately *not* the criterion any cell asserts. What a vision channel
    requires is that the cue be unmistakably present and that it hide only what it is meant to
    hide; a prop whose top is cropped still reads as a wall filling a lane. Reporting the
    fraction keeps that a measured quantity rather than a yes/no with an arbitrary threshold.
    """
    lo_y, hi_y = min(y_span), max(y_span)
    lo_z, hi_z = min(z_span), max(z_span)
    inside = 0
    total = 0
    for i in range(samples + 1):
        z = lo_z + (hi_z - lo_z) * i / samples
        for j in range(samples + 1):
            y = lo_y + (hi_y - lo_y) * j / samples
            total += 1
            if in_frame((centre[0], y, z)):
                inside += 1
    return inside / total if total else 0.0
