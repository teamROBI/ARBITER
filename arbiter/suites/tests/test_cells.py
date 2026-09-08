# Copyright 2026 ARBITER Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the arbitration cell table.

The one that matters most is `test_every_spoken_phrase_is_grounded`. ARBITER's central claim
rests on nulls -- "the instruction named a lane and the policy did not take it" -- and a null is
only a finding if the instruction was a sentence the policy could have understood. TANGO has two
receipts for how easily that goes wrong: an *empty* instruction measured 0/6 with no obstacle in
the scene at all, and a scrambled one degraded toward no detour rather than toward the trained
lane. Both look like "language does not steer" and neither is.

So this asserts that every phrase the table speaks is byte-identical to one the spec emits for
some trained condition, by generating the whole grounded vocabulary and checking membership.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from arbiter.suites import cells as C  # noqa: E402
from arbiter.suites import spec as S  # noqa: E402


def _grounded_vocabulary() -> set[str]:
    """Every sub-task phrase the spec emits anywhere, over every axis and trained condition.

    Deliberately spans all axes rather than topology alone: V-AUTH's route-silent middle phrase
    is borrowed from EXTENT, and the point of borrowing rather than inventing is that some axis
    trains it.
    """
    out: set[str] = set()
    for ax_name in S.AXES:
        ax = S.axis(ax_name)
        for cond in S.enumerate_conditions(ax, list(S.object_keys_for(ax_name))):
            try:
                out.update(S.sub_instructions_from_attrs(cond.to_attrs()))
            except (KeyError, ValueError):
                # Axes whose phase texts need params this condition does not carry are not
                # part of the vocabulary this table draws on.
                continue
    return out


def _sample_params(c: C.Cell) -> dict[str, float]:
    return {"barrier_h": c.heights[0], "barrier_offset": C.DEFAULT_OFFSET}


def test_every_spoken_phrase_is_grounded():
    grounded = _grounded_vocabulary()
    assert grounded, "generated an empty vocabulary; the check would pass vacuously"
    for name, c in C.CELLS.items():
        for arm in c.arms:
            for h in c.heights:
                spoken = C.spoken_sub_tasks(
                    name, arm.name, {"barrier_h": h, "barrier_offset": C.DEFAULT_OFFSET}
                )
                for phrase in spoken:
                    assert phrase in grounded, (
                        f"{name}/{arm.name} at h={h} speaks {phrase!r}, which the spec never "
                        f"emits. An out-of-vocabulary instruction measures distribution shift, "
                        f"not arbitration."
                    )


def test_spoken_phrases_differ_only_in_the_middle():
    """Reach and release must be untouched, or the route intervention is confounded."""
    for name, c in C.CELLS.items():
        params = _sample_params(c)
        variants = [C.spoken_sub_tasks(name, a.name, params) for a in c.arms]
        assert all(len(v) == 3 for v in variants)
        assert len({v[0] for v in variants}) == 1, f"{name}: reach phase varies across arms"
        assert len({v[2] for v in variants}) == 1, f"{name}: release phase varies across arms"


def test_paired_arms_expect_opposite_lanes():
    """A cell with two naming arms must expect *different* behaviour, or it measures nothing."""
    for name, c in C.CELLS.items():
        naming = [a for a in c.arms if a.route_word is not None]
        if len(naming) < 2:
            continue
        params = _sample_params(c)
        expected = {C.expected(name, a.name, params) for a in naming}
        assert len(expected) == len(naming), (
            f"{name}: paired arms expect the same outcome {expected}, so the contrast cannot "
            f"separate obedience from habit"
        )


def test_side_naming_cells_reject_the_task_channel():
    for name, c in C.CELLS.items():
        speaks_lane = any(a.route_word is not None for a in c.arms)
        if speaks_lane:
            assert "task" not in c.valid_channels, name
            with pytest.raises(ValueError, match="not interpretable"):
                C.assert_interpretable(name, "task")
        C.assert_interpretable(name, "sub_task")  # must not raise


def test_neutral_middle_is_actually_route_silent():
    """V-AUTH's instruction must not leak the answer it is asking vision to supply."""
    mid = C.spoken_sub_tasks("V-AUTH", "silent", _sample_params(C.cell("V-AUTH")))[1]
    for leak in ("left", "right", "over the barrier", "around"):
        assert leak not in mid, f"V-AUTH middle phrase {mid!r} leaks the route via {leak!r}"


def test_v_auth_expectation_flips_at_h_star():
    below = [h for h in C.cell("V-AUTH").heights if h < S.TOPOLOGY_H_STAR]
    at_or_above = [h for h in C.cell("V-AUTH").heights if h >= S.TOPOLOGY_H_STAR]
    assert below and at_or_above, "V-AUTH must span h* or it measures no discontinuity"
    for h in below:
        assert C.expected("V-AUTH", "silent", {"barrier_h": h}) == ("route_class", "over")
    for h in at_or_above:
        assert C.expected("V-AUTH", "silent", {"barrier_h": h}) == ("route_class", "around")


def test_lane_cells_are_around_class_only():
    """Below h* there is no lane to choose, so a lateral expectation would be meaningless."""
    for name, c in C.CELLS.items():
        if not c.requires_around:
            continue
        for h in c.heights:
            assert S.expected_homotopy(h) == "around", (
                f"{name} includes h={h}, whose route class is "
                f"{S.expected_homotopy(h)!r}; a side score there is not interpretable"
            )


def test_blocker_seals_the_habitual_lane_and_leaves_the_other_open():
    for name, c in C.CELLS.items():
        sealed = C.sealed_side(name)
        if c.blocker == "none":
            assert sealed is None
            continue
        assert sealed == S.bypass_side(C.DEFAULT_OFFSET), (
            f"{name}: sealing anything but the tie-break lane removes nothing the policy was "
            f"going to do anyway"
        )
        assert C.forced_side() == S.forced_bypass_side(C.DEFAULT_OFFSET)
        assert sealed != C.forced_side()


def test_conflict_vf_contradicts_vision_and_vt_defers_to_it():
    """The two conflict cells must expect *opposite* resolutions of the same scene geometry."""
    params = _sample_params(C.cell("CONFLICT-VF"))
    vf = C.expected("CONFLICT-VF", "contradicts", params)
    vt = C.expected("CONFLICT-VT", "names_sealed", params)
    sealed = C.sealed_side("CONFLICT-VF")
    # VF: the prop has no collider, so the named (sealed-looking) lane is the right answer.
    assert vf == ("route_side", sealed)
    # VT: the prop is solid, so the named lane is infeasible and vision must win.
    assert vt == ("route_side", C.forced_side())
    assert vf != vt


def test_blocker_flags_never_select_the_legacy_prop():
    """The 0.16 m prop exists only to reproduce TANGO's v6/v7 grids; 36/36 arms flew over it."""
    for name, c in C.CELLS.items():
        flags = C.blocker_flags(name)
        assert flags["blocker_legacy"] is False, name
        assert flags["with_blocker"] == (c.blocker != "none")
        assert flags["blocker_ghost"] == (c.blocker == "ghost")


def test_conditions_are_non_empty_and_match_declared_heights():
    for name, c in C.CELLS.items():
        conds = C.conditions(name)
        got = {round(float(x.params["barrier_h"]), 6) for x in conds}
        assert got == {round(h, 6) for h in c.heights}, name
        assert all(x.axis == "topology" for x in conds)
        # Derived fields must be populated -- that is why conditions() delegates to the spec.
        assert all(x.instruction and x.split in ("train", "test") for x in conds)


def test_conditions_refuses_an_offset_that_matches_nothing():
    with pytest.raises(ValueError, match="zero conditions"):
        C.conditions("L-AUTH", offset=0.123)


def test_unknown_names_raise_with_the_options_listed():
    with pytest.raises(KeyError, match="unknown cell"):
        C.cell("NOPE")
    with pytest.raises(KeyError, match="has no arm"):
        C.cell("L-AUTH").arm("nope")


def test_headline_excludes_the_safety_cell():
    """Compliance in CONFLICT-VT is a collision, so it cannot sit in the controllability table."""
    assert "CONFLICT-VT" not in C.HEADLINE_CELLS
    assert set(C.HEADLINE_CELLS) == {"L-AUTH", "V-AUTH", "CONFLICT-VF"}


def test_no_heavy_imports():
    """The table must stay stdlib-only: the gate, collector and both venvs all read it."""
    src = (REPO_ROOT / "arbiter" / "suites" / "cells.py").read_text()
    tree = ast.parse(src)
    banned = {"numpy", "torch", "h5py", "isaaclab", "isaacsim", "pxr", "gymnasium", "pandas"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]]
        else:
            continue
        for m in mods:
            assert m not in banned, f"cells.py imports {m}, which not every consumer has"
