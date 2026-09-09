# Copyright 2026 ARBITER Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-phase instruction delivery during a rollout.

Pure logic, no simulator and no numpy, so it can be unit-tested. That is the point: the three
worst bugs in this project so far all lived in code whose job was to *check* or *derive*
something, and each returned a confident wrong answer rather than an error. Anything in that
layer belongs somewhere it can be exercised directly.
"""

from __future__ import annotations

#: Commanded gripper below this counts as closed. Training thresholded at the *episode*
#: midpoint of the commanded gripper (`emit_sub_tasks.gripper_closed_trace`), which needs the
#: whole episode and so is not available online. The command is driven to both rails -- 0.0
#: closed, 0.04 open on the Franka finger joint -- so a fixed midpoint reproduces that trace
#: exactly for any episode whose command reaches both, which every completed one does.
PHASE_GRIPPER_CLOSED_BELOW = 0.02


class PhaseSpeaker:
    """Delivers per-phase instructions during a rollout, from the observed gripper command.

    Why this exists: before it, the evaluator had **no per-phase delivery at all**. `spoken` was
    the axis's task-level instruction, and `ARBITER_LANG_KEY` changed only the observation *key*.
    So a `sub_task` checkpoint -- trained on "reach for the red block" during the reach and
    "carry the red block around the left side" during the carry -- was fed one constant sentence
    on that key for the whole episode. Both of the first two attempts to evaluate one scored
    0 success and went `over` where `around` was required, because the policy was out of
    distribution, not because it ignored language. That is also why the per-phase checkpoint had
    sat unevaluated: it was not evaluable.

    Phases advance on transitions of the **commanded** gripper, which is what
    `spec.sub_task_spans` used to derive the training spans -- close ends a reach, open ends a
    carry. The command rather than the achieved width, deliberately: rate-limited primitives step
    a command toward a target and reading achieved state instead is a documented failure mode.

    **One deliberate deviation from training.** The loader gave each phase's text a half-horizon
    lead, so it appeared 8 frames (at horizon 16) *before* that phase began. Online that is
    anti-causal: the transition is something the policy commands, so it cannot be known before
    asking the policy for the chunk that commands it. Recovering it would mean requesting a chunk,
    inspecting its gripper lookahead, and re-requesting with the advanced text -- an extra forward
    pass at each boundary. Instead the text arrives up to 8 steps (0.27 s at 30 Hz) later than
    training would have delivered it. That is a boundary effect on phases lasting ~70-90 steps,
    and the phase this project cares about -- the carry, which carries the route -- is the long
    one, so a 9% boundary shift on it does not bear on the route decision. Recorded in the
    rollout provenance rather than left implicit.
    """

    def __init__(self, texts: list[str]):
        if not texts:
            raise ValueError("PhaseSpeaker needs at least one phase text")
        self.texts = list(texts)
        self.n_transitions = 0
        self._prev_closed: bool | None = None

    def observe(self, gripper_cmd: float) -> None:
        """Feed one executed step's commanded gripper value."""
        closed = float(gripper_cmd) < PHASE_GRIPPER_CLOSED_BELOW
        if self._prev_closed is not None and closed != self._prev_closed:
            self.n_transitions += 1
        self._prev_closed = closed

    @property
    def phase(self) -> int:
        """Phase index: the number of gripper transitions seen, clamped to the texts available.

        Clamped rather than wrapped: a policy that opens and closes more times than the expert
        did has not entered a phase the training set describes, and repeating the final text is
        closer to what training showed than cycling back to "reach for the red block".
        """
        return min(self.n_transitions, len(self.texts) - 1)

    def text(self) -> str:
        return self.texts[self.phase]
