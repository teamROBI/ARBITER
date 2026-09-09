# Copyright 2026 ARBITER Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for per-phase instruction delivery.

These exist because the absence of this feature produced two uninterpretable evaluation runs.
A `sub_task` checkpoint was fed one constant sentence for a whole episode -- carry-text during
the reach in one attempt, the task-level sentence in the other -- and scored 0 success while
lifting over the barrier where it should have gone around. That looked like "language does not
steer the route" and was actually a policy out of distribution.

So the segmentation is pinned here: it must match how `spec.sub_task_spans` derived the training
spans, or the checkpoint is out of distribution again and the null is meaningless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arbiter.policy.phase_speaker import (  # noqa: E402
    PHASE_GRIPPER_CLOSED_BELOW,
    PhaseSpeaker,
)
from arbiter.suites import spec as S  # noqa: E402

OPEN, CLOSED = 0.04, 0.0
TEXTS = ["reach for the red block",
         "carry the red block around the left side of the barrier",
         "put the red block down on the target and let go"]


def _drive(seq: list[float], texts=None) -> tuple[PhaseSpeaker, list[int]]:
    sp = PhaseSpeaker(texts or TEXTS)
    trace = []
    for g in seq:
        sp.observe(g)
        trace.append(sp.phase)
    return sp, trace


def test_reach_carry_release_matches_the_training_segmentation():
    """Close ends a reach, open ends a carry -- the rule sub_task_spans used."""
    sp, trace = _drive([OPEN] * 40 + [CLOSED] * 90 + [OPEN] * 30)
    assert trace[:40] == [0] * 40
    assert trace[40:130] == [1] * 90
    assert trace[130:] == [2] * 30
    assert sp.text() == TEXTS[2]


def test_agrees_with_sub_task_spans_on_the_same_trace():
    """The authority is the spec. If these two disagree, one of them is wrong."""
    seq = [OPEN] * 25 + [CLOSED] * 60 + [OPEN] * 15
    closed = [g < PHASE_GRIPPER_CLOSED_BELOW for g in seq]
    spans = S.sub_task_spans(closed, 3)
    _sp, trace = _drive(seq)
    for phase_idx, (start, end) in enumerate(spans):
        assert set(trace[start:end]) == {phase_idx}, (
            f"span {phase_idx} = [{start},{end}) but the speaker reported "
            f"{sorted(set(trace[start:end]))}"
        )


def test_the_threshold_is_the_midpoint_of_the_commanded_rails():
    """Training thresholded at the episode midpoint; the command hits 0.0 and 0.04."""
    assert PHASE_GRIPPER_CLOSED_BELOW == pytest.approx(0.5 * (0.0 + 0.04))
    _sp, trace = _drive([0.021, 0.019])
    assert trace == [0, 1], "a value either side of the midpoint must count as a transition"


def test_phase_is_clamped_not_wrapped():
    """A policy that cycles the gripper more than the expert did has left the described phases.

    Repeating the final text is closer to what training showed than cycling back to
    "reach for the red block" while the object is already on the target.
    """
    sp, trace = _drive([OPEN] * 3 + [CLOSED] * 3 + [OPEN] * 3 + [CLOSED] * 3 + [OPEN] * 3)
    assert max(trace) == 2
    assert sp.text() == TEXTS[-1]
    assert sp.n_transitions == 4, "transitions are still counted, only the index is clamped"


def test_a_gripper_that_never_moves_stays_in_the_first_phase():
    """A policy that never grasps has not begun a carry, whatever else it did."""
    sp, trace = _drive([OPEN] * 50)
    assert set(trace) == {0}
    assert sp.text() == TEXTS[0]


def test_order_style_seven_phase_sequence():
    """ORDER emits 2 texts per block over 3 blocks plus a tail: 6 transitions, 7 spans."""
    texts = [f"phase{i}" for i in range(7)]
    seq = []
    for _ in range(3):
        seq += [OPEN] * 5 + [CLOSED] * 5
    seq += [OPEN] * 5
    sp, trace = _drive(seq, texts)
    assert sp.n_transitions == 6
    assert max(trace) == 6
    assert sp.text() == "phase6"


def test_empty_texts_are_refused():
    with pytest.raises(ValueError, match="at least one"):
        PhaseSpeaker([])


def test_a_single_text_never_advances():
    """The task channel's one-sentence-per-episode case, expressed through the same path."""
    sp, trace = _drive([OPEN] * 5 + [CLOSED] * 5, ["move the red block past the barrier"])
    assert set(trace) == {0}
    assert sp.text() == "move the red block past the barrier"
