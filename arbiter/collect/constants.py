# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Tuned constants for scripted demonstration collection.

Carried over from the collection stack this repo was split out of, where they were tuned
empirically against a differential-IK arm in Isaac Sim over many collection runs. They are
copied rather than referenced from TANGO -- ARBITER owns them now and diverges freely (see CLAUDE.md on
self-containment) — but they are *not* arbitrary, so change them with a measurement rather
than a guess.

Two groups are genuinely load-bearing:

**Rate caps** (``MAX_TRANSLATION_STEP_M``, ``MAX_ORIENTATION_STEP_RAD``). The scripted expert
does not teleport the end-effector to a target; it steps toward it at a capped rate so the IK
controller can track without drifting. Raising these produces demonstrations the policy cannot
reproduce, because the recorded actions become larger than one control step can achieve.

**Tolerances** (``PHASE_ADVANCE_TOLERANCE_M`` vs ``POSITION_TOLERANCE_M``). A phase advances at
the tighter tolerance, but a phase that *times out* is still accepted at the looser one. Both
are needed: the tight one keeps demonstrations clean, the loose one stops a near-miss from
discarding an otherwise good episode.
"""

from __future__ import annotations

# ── Rate caps: how fast the scripted expert may command the arm ──────────────
MAX_TRANSLATION_STEP_M = 0.010
MAX_WRIST_STEP_RAD = 0.05
MAX_ORIENTATION_STEP_RAD = 0.06        # ~3.4 deg per sim step; caps the orientation slerp rate

# ── Tolerances ───────────────────────────────────────────────────────────────
PHASE_ADVANCE_TOLERANCE_M = 0.02       # tight: advance to the next phase
POSITION_TOLERANCE_M = 0.03            # loose: accept on timeout rather than fail
ORIENTATION_TOLERANCE_RAD = 0.20

# ── Phase budgets ────────────────────────────────────────────────────────────
PHASE_TIMEOUT_STEPS = 60               # floor, for phases with almost no distance to cover
MOVEMENT_TIMEOUT_GAIN = 15.0           # budget = ceil(required_steps * gain)
MOVEMENT_TIMEOUT_MAX_STEPS = 600

# ── Stall detection ──────────────────────────────────────────────────────────
# A blocked approach shows as error that stops improving. Failing at ~90 steps beats burning
# the full 600-step budget on an episode that will not succeed.
STALL_WINDOW_STEPS = 90
STALL_THRESHOLD_M = 0.03

# ── Grasp geometry ───────────────────────────────────────────────────────────
PREGRASP_OFFSET_M = 0.06
FINAL_APPROACH_EXTRA_M = 0.02
LIFT_HEIGHT_M = 0.08

# ── Gripper ──────────────────────────────────────────────────────────────────
MAX_GRIPPER_STEP = 0.003
CLOSE_HOLD_STEPS = 20
POST_CLOSE_SETTLE_STEPS = 10
CLOSE_BLOCKED_EPS_M = 0.01
CLOSE_BLOCKED_FRACTION = 0.15
PLACE_OPEN_HOLD_STEPS = 15

# ── Drop detection ───────────────────────────────────────────────────────────
# Both must hold before an episode is called dropped: near the floor AND well below where the
# object started. Either alone gives false positives — a low table trips the first, a
# deliberate place trips the second.
FALL_FLOOR_Z_THRESHOLD_M = 0.15
FALL_RELATIVE_DROP_M = 0.20

# ── Episode framing ──────────────────────────────────────────────────────────
PREPLAN_SETTLE_STEPS = 45              # let the scene settle under gravity before acting
SETTLE_HOLD_STEPS = 30                 # hold after retract, before the success check
DISCARD_HOLD_STEPS = 10                # camera/PBR settle frames, excluded from the recording
RECORD_HOLD_STEPS = 30                 # still frames at episode start, kept

# ── Success thresholds ───────────────────────────────────────────────────────
SUCCESS_LIFT_DELTA_M = 0.01
SUCCESS_RETRACT_DIST_M = 0.03

# ── Route primitive (new in v2; no upstream equivalent) ──────────────────────
# The TOPOLOGY axis needs the expert to demonstrate two homotopy classes past a barrier.
# It only has to *demonstrate* them, not plan generally, so each is a fixed waypoint pattern.
#: Vertical clearance for the OVER apex, measured from the barrier top to the *end effector*.
#: It has to cover everything hanging below the EEF origin: the carried cube's centre sits ~15 mm
#: down and its underside ~40 mm down, with the fingertips lower still. At 0.05 the apex put the
#: cube's underside only 10 mm above the barrier, so the gripper rested on the barrier top and
#: the arm froze -- measured at h=0.07, where the EEF stuck at z=0.114 chasing an apex of 0.120.
#: Every over-class demonstration was grazing the barrier; only h=0.07 jammed hard enough to
#: stall, and nothing noticed because the route moves had stall checking disabled.
ROUTE_OVER_CLEARANCE_M = 0.12
#: Lateral clearance for the AROUND bypass, measured from the barrier edge to the *end
#: effector*. It must cover the hand's own half-width, not just the carried object's: at 0.06
#: the cube cleared the barrier end by 3.5 cm while the gripper body fouled the corner at
#: (0.49, -0.10), and the arm ground there for 300 control steps (~10 s) before slipping past.
#: The Franka hand is roughly 0.10 m across the fingers, so 0.06 was inside its own footprint.
ROUTE_AROUND_MARGIN_M = 0.12
ROUTE_TRANSIT_Z_M = 0.04               # floor on the around-route travel height, above the table
ROUTE_UNDER_MARGIN_M = 0.03            # how far BELOW the barrier top an around-route travels
