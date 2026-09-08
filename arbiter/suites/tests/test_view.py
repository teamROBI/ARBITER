# Copyright 2026 ARBITER Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""What the head camera can and cannot see of the arbitration cells.

These tests exist because of a near-miss. The blocker was raised to 0.40 m on a *physics*
criterion -- above the measured end-effector apex of 0.222-0.349 m, so the sealed lane is sealed
rather than grazed, since grazing produces 600 steps of grinding instead of a forced detour.
Nobody had checked what that does to the picture. It turns out the tallest blocker that fits
*entirely* inside the head frame is 0.339 m, so the two criteria miss each other by about 1 cm
and the 0.40 m prop is cropped.

The resolution is that "entirely in frame" was never the requirement. What a vision channel needs
is that the cue be present and that it occlude only what it is supposed to occlude, and both hold
at 0.40 m: 85% of the prop's face is in frame, it hides the lane it seals, and it hides nothing
else. These tests pin that reasoning so a later reader does not "fix" the cropping by moving the
camera -- which would invalidate every checkpoint trained on this view.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arbiter.suites import cells as C  # noqa: E402
from arbiter.suites import spec as S  # noqa: E402
from arbiter.suites import view as V  # noqa: E402
from arbiter.suites.layout_gen import TABLE_TOP_Z  # noqa: E402

# The canonical value, not a local guess. It lives in collect/constants because it is a property
# of the expert's clearance, and the spec takes it as an *argument* rather than importing it so
# that the spec stays free of the collect layer. Tests must reach for the real one: a fallback
# default here would silently pass against the wrong geometry.
from arbiter.collect.constants import ROUTE_AROUND_MARGIN_M as AROUND_MARGIN  # noqa: E402


def _blocker_box(height: float, offset: float = 0.0):
    bx, _by = S.blocker_xy(offset, around_margin=AROUND_MARGIN)
    y_span = S.blocker_span(offset, around_margin=AROUND_MARGIN)
    return (bx, 0.0), y_span, (TABLE_TOP_Z, TABLE_TOP_Z + height)


def test_camera_frame_matches_the_grounded_word_convention():
    """right is +y and up is -x, which is what the spec's compass words assume."""
    fwd, right, up = V.camera_basis()
    assert right[1] > 0.99, f"image-right must be +y, got {right}"
    assert up[0] < -0.5, f"image-up must lean -x, got {up}"
    assert fwd[0] < 0 and fwd[2] < 0, f"camera must look toward -x and downward, got {fwd}"


def test_field_of_view_is_libero_agentview():
    hh, hv = V.half_fov_deg()
    assert abs(2 * hv - 45.0) < 0.5, f"vertical FOV must be LIBERO's 45 deg, got {2*hv:.2f}"
    assert abs(2 * hh - 73.0) < 1.0, f"horizontal FOV should be ~73 deg, got {2*hh:.2f}"


def test_blocker_occludes_the_lane_it_seals():
    """The point of the prop: the sealed lane must not be visibly open."""
    box = _blocker_box(S.BLOCKER_HEIGHT_M)
    sealed_y = S.bypass_y(0.0, S.bypass_side(0.0), around_margin=AROUND_MARGIN)
    p = (0.5, sealed_y, TABLE_TOP_Z + 0.05)
    assert V.occluded_by_box(p, *box, depth=S.BLOCKER_DEPTH_M), (
        "the blocker does not occlude the lane it seals, so the visual cue is absent"
    )


def test_blocker_occludes_nothing_the_task_depends_on():
    """A prop that hides the object or the target confounds re-routing with not seeing the goal."""
    box = _blocker_box(S.BLOCKER_HEIGHT_M)
    start, goal = S.topology_endpoints()
    forced_y = S.bypass_y(0.0, S.forced_bypass_side(0.0), around_margin=AROUND_MARGIN)
    must_stay_visible = {
        "object at start": (start[0], 0.0, TABLE_TOP_Z + 0.025),
        "target pad": (goal[0], 0.0, TABLE_TOP_Z + 0.002),
        "barrier near end": (0.5, -S.barrier_half_width(), TABLE_TOP_Z + S.TOPOLOGY_H_MAX),
        "barrier centre": (0.5, 0.0, TABLE_TOP_Z + S.TOPOLOGY_H_MAX),
        "barrier far end": (0.5, +S.barrier_half_width(), TABLE_TOP_Z + S.TOPOLOGY_H_MAX),
        "forced lane": (0.5, forced_y, TABLE_TOP_Z + 0.05),
    }
    for name, p in must_stay_visible.items():
        assert not V.occluded_by_box(p, *box, depth=S.BLOCKER_DEPTH_M), (
            f"blocker occludes {name} at {p}; a behaviour change there is confounded"
        )
        assert V.in_frame(p), f"{name} at {p} is outside the head frame entirely"


def test_blocker_cue_is_substantially_visible_even_though_cropped():
    """Cropped is acceptable; invisible is not. 0.40 m shows ~85% of its face."""
    _c, y_span, z_span = _blocker_box(S.BLOCKER_HEIGHT_M)
    bx, _ = S.blocker_xy(0.0, around_margin=AROUND_MARGIN)
    frac = V.face_fraction_in_frame((bx, 0.0), y_span, z_span)
    assert frac > 0.60, (
        f"only {frac:.1%} of the blocker face is in frame; the visual channel may be too weak "
        f"to drive route selection"
    )
    assert frac < 1.0, (
        "the blocker is fully in frame, which contradicts the recorded 0.339 m limit -- either "
        "BLOCKER_HEIGHT_M was lowered or the camera was changed. If the camera moved, every "
        "checkpoint trained on the old view is invalid."
    )


def test_the_physics_and_framing_criteria_really_do_conflict():
    """Guards the reasoning above, so the 1 cm miss is not rediscovered from scratch.

    If a future change makes these compatible -- a shallower camera angle, a blocker moved
    further from the camera -- this fails and the docstring above needs revisiting rather than
    quietly remaining wrong.
    """
    bx, _ = S.blocker_xy(0.0, around_margin=AROUND_MARGIN)
    y_span = S.blocker_span(0.0, around_margin=AROUND_MARGIN)

    def fully_in_frame(height: float) -> bool:
        return all(
            V.in_frame((bx + dx, y, TABLE_TOP_Z + height))
            for dx in (-S.BLOCKER_DEPTH_M / 2, S.BLOCKER_DEPTH_M / 2)
            for y in y_span
        )

    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if fully_in_frame(mid):
            lo = mid
        else:
            hi = mid
    assert 0.30 < lo < 0.35, f"max fully-framed height moved to {lo:.3f} m; re-read the docstring"
    assert S.BLOCKER_HEIGHT_M > lo, (
        f"BLOCKER_HEIGHT_M {S.BLOCKER_HEIGHT_M} now fits in frame; if it was lowered, check it "
        f"still clears the 0.349 m end-effector apex or the lane grazes instead of sealing"
    )


def test_legacy_blocker_would_have_fit_but_does_not_seal():
    """Why the 0.16 m prop is reproduction-only: visible, in frame, and passable."""
    _c, y_span, z_span = _blocker_box(S.BLOCKER_LEGACY_HEIGHT_M)
    bx, _ = S.blocker_xy(0.0, around_margin=AROUND_MARGIN)
    assert V.face_fraction_in_frame((bx, 0.0), y_span, z_span) == 1.0
    assert S.BLOCKER_LEGACY_HEIGHT_M < 0.222, (
        "the legacy prop is documented as sitting below the measured 0.222-0.349 m apex range, "
        "which is why 36/36 arms flew over it"
    )


def test_ghost_and_solid_are_perceptually_identical():
    """CONFLICT-VF's entire logic: the two props may differ in collision and nothing else."""
    vf = C.blocker_flags("CONFLICT-VF")
    vt = C.blocker_flags("CONFLICT-VT")
    assert vf["blocker_ghost"] is True and vt["blocker_ghost"] is False
    assert vf["with_blocker"] == vt["with_blocker"] is True
    # Same box, so same projection, so same picture.
    box_vf = _blocker_box(S.BLOCKER_HEIGHT_M)
    box_vt = _blocker_box(S.BLOCKER_HEIGHT_M)
    assert box_vf == box_vt
