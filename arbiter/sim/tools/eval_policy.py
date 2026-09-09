#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Roll out a policy over one axis and report success per sweep coordinate.

Phase 3's measurement. The scripted expert certified every condition achievable; this asks how
much of that a trained policy reproduces, and how that decays with distance from support.

**Evaluate POSITION first.** It is the in-distribution control: at sweep coordinate 0 the policy
is being asked for a trajectory it was trained on. An instrument that cannot detect the success
it is calibrated for is not measuring anything, and every radius is a ratio against that
control -- so a low ID number invalidates the whole axis rather than being a weak result.

The policy runs in a separate process by necessity, not by design: Isaac Lab needs Python 3.11
and GR00T needs 3.10, so they cannot share an interpreter. GR00T's `PolicyServer` holds the
checkpoint in the gr00t venv and this client talks to it over ZMQ.

    # in the gr00t venv
    bash scripts/bench/run_policy_server.sh <checkpoint>
    # here, in the sim venv
    python arbiter/sim/tools/eval_policy.py --axis position --split train --episodes 3

`--policy random` needs no server and no checkpoint. That is the protocol's random floor, and
it doubles as the way to validate this harness: a floor that scores well means the success
predicate is broken, not that the policy is good.

One axis per launch, and one launch per (barrier height, offset) for TOPOLOGY.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
#: Not `required=True`: --cell implies the axis (every arbitration cell is built on the barrier
#: geometry), so demanding both would be redundant and would let a contradictory pair through.
#: Exactly one of --axis / --cell must be given; checked below, before the simulator boots.
_parser.add_argument("--axis", default=None)
_parser.add_argument("--split", default="train", choices=("train", "test", "all"),
                     help="train is the ID control; test is the sweep")
_parser.add_argument("--policy", default="server", choices=("server", "random", "hold"),
                     help="'random' and 'hold' are the protocol's floors and need no server")
_parser.add_argument("--host", default="localhost")
_parser.add_argument("--port", type=int, default=5555)
_parser.add_argument("--episodes", type=int, default=1,
                     help="rollouts per condition; >1 gives a per-condition rate")
_parser.add_argument("--max-steps", type=int, default=600,
                     help="cap per rollout. The expert needs 170-330 control steps, so this is "
                          "generous; a policy that has not finished by here has failed.")
_parser.add_argument("--barrier-h", type=float, default=None)
_parser.add_argument("--barrier-offset", type=float, default=0.0)
#: DETOUR: seal the bypass lane `route()`'s tie-break takes, so the policy must detour to the
#: complement. At offset 0 the tie-break goes -y and the complement is +y, and offset 0 is
#: already in TOPOLOGY_OFFSETS -- so the 90 collected lateral-bypass episodes ARE the training
#: set for this cell and the forced +y detour is a held-out trajectory at zero collection cost.
#:
#: Unlike `h*`, which is a convention about the expert and which nothing stops a policy from
#: ignoring by lifting over, this holdout is enforced by geometry: the sealed lane is a collider
#: and the blocker stands 0.16 m, above every barrier the arm was ever shown to clear. So the
#: success predicate suffices and no route classifier is needed to interpret the result.
_parser.add_argument("--max-conditions", type=int, default=0, help="0 = every condition")
_parser.add_argument("--seed", type=int, default=0)
#: Displace the target pad without changing the condition, to test whether the policy is reading
#: the marker or replaying a habitual endpoint. `lateral` moves it across the transport,
#: `transport` along it.
_parser.add_argument("--pad-offset", type=float, default=0.0,
                     help="metres to displace the target pad from the condition's target")
#: Speak a DIFFERENT condition's instruction while scoring stays on this one. If the policy reads
#: language it should now do the other thing -- and fail. If it ignores language, behaviour is
#: unchanged and it keeps succeeding. A success rate alone cannot tell those apart.
#: Same-axis only, and it raises on an unknown key rather than falling back.
_parser.add_argument("--swap-instruction", default=None,
                     help="condition key whose instruction to use instead (control)")
#: Change the instruction PART-WAY THROUGH a rollout: "step:key;step:key;...".
#:
#: Entries are separated by ';' and NOT ',' -- condition keys contain commas of their own
#: (`direction[radius=0.18,theta_deg=0]@arb_cube_red`), so a comma separator splits mid-key and
#: the lookup fails on a fragment. That is exactly how the first run of this control died.
#:
#: Scoring always stays on the episode's OWN condition, so "did the words change the trajectory"
#: is read from the recorded behaviour rather than from the success rate.
_parser.add_argument("--instruction-schedule", default=None,
                     help="mid-rollout instruction changes, ';'-separated: "
                          "\"0:order[seq=012]@arb_cube_red;60:order[seq=120]@arb_cube_red\"")
_parser.add_argument("--blocker", action="store_true",
                     help="seal route()'s tie-break bypass lane; the detour must be the "
                          "complement (TOPOLOGY only)")
#: Spawn the blocker with NO collider: the lane is passable but still looks closed. The control
#: that separates perception from physics. A policy re-routing here is reading the scene; one
#: driving straight through was only ever responding to contact.
_parser.add_argument("--blocker-ghost", action="store_true",
                     help="visual-only blocker (no collision); requires --blocker")
#: Use the ORIGINAL 0.16 m blocker instead of the 0.40 m default.
#:
#: Only for reproducing the v6/v7 DETOUR grids, which were measured against it. It does not seal
#: the lane: over 36 rollouts the end-effector apex ran 0.222-0.349 m, so every episode flew over
#: it and only the cube, carried ~4 cm lower, grazed the top -- two cleared it outright and scored
#: a lift-over as a success. The default is above every measured apex.
_parser.add_argument("--blocker-legacy", action="store_true",
                     help="use the original 0.16 m blocker (does NOT seal the lane; "
                          "reproduction only); requires --blocker")
#: Speak ARBITRARY text as the instruction, from step 0, instead of the condition's own.
#:
#: `--swap-instruction` can only utter a string the spec already generates for some condition on
#: the same axis, so it can test "does the model read the words" but NOT "can the words carry
#: information the task-level instruction never carries". Path guidance -- "around the right side
#: of the barrier" -- is not any condition's instruction, so it is unreachable through a key.
#:
#: Guidance is delivered at t=0, not mid-episode: the schedule measurements showed the policy
#: commits to its discrete choice between step 17 and step 45, well before the first observable
#: motion (~step 90), so guidance is only ever consumed from the initial observation anyway.
#:
#: An EMPTY string means "say nothing" and is honoured -- the check is `is not None`, because an
#: empty string is falsy and a truthiness test silently ran the normal condition instead, which
#: would have reported "no change" from an ablation that never applied.
#:
#: Scoring stays on the episode's own condition, exactly as with the other two flags.
_parser.add_argument("--speak", default=None,
                     help="raw instruction text to give the policy instead of the condition's "
                          "own (e.g. \"move the red block around the right side of the "
                          "barrier onto the target\")")
_parser.add_argument("--pad-offset-dir", default="lateral",
                     choices=("lateral", "transport"),
                     help="displace the pad across the transport, or along it")
_parser.add_argument("--hold-checks", type=int, default=3,
                     help="consecutive satisfied checks before a rollout is called a success. "
                          "A momentary success that immediately falls apart is not one, and the "
                          "hold is also what makes early exit safe.")
_parser.add_argument("--check-every", type=int, default=15,
                     help="control steps between success checks. Also the early-exit granularity")
_parser.add_argument("--save-video", default="none",
                     choices=("none", "fail", "success", "all"),
                     help="write an mp4 per rollout. 'fail' is usually what you want: a success "
                          "rate says how often the policy fails and the footage says what it "
                          "does instead, which is the difference between a number and a "
                          "diagnosis. Capturing everything on a full sweep writes hundreds of "
                          "files, so this is selective by outcome.")
_parser.add_argument("--video-fps", type=int, default=30,
                     help="the control rate, so 1 frame per control step plays at real time")
#: Run one arbitration cell instead of hand-assembling its flags.
#:
#: `--cell L-AUTH --cell-arm requested` sets the axis, the blocker configuration, the barrier
#: heights and the spoken instruction from `arbiter/suites/cells.py`, so the four-cell table
#: cannot drift out of step with what is actually run. Assembling a cell by hand is how the
#: sealed lane and the avoided lane come to disagree.
_parser.add_argument("--cell", default=None,
                     help="arbitration cell to run (L-AUTH, V-AUTH, CONFLICT-VF, CONFLICT-VT). "
                          "Sets axis, blocker flags, heights and spoken text from the cell "
                          "definition; incompatible with the manual --blocker/--speak flags.")
_parser.add_argument("--cell-arm", default=None,
                     help="which arm of the cell to speak (e.g. habitual/requested for L-AUTH). "
                          "Required with --cell: a single arm is never a result, so the arm has "
                          "to be named explicitly and the pair run as two invocations.")
#: Counterfactual Action Guidance scale. 1.0 is the plain conditional policy.
#:
#: The baseline this project has to beat. See `arbiter/policy/cag.py` for why the unconditional
#: branch has to be validated, and why one global scale cannot be correct on the whole table.
_parser.add_argument("--cag-omega", type=float, default=1.0,
                     help="CAG guidance scale: a = a_uncond + w*(a_cond - a_uncond). "
                          "1.0 (default) short-circuits to the conditional policy with a single "
                          "forward pass. Anything else costs two passes per chunk.")
_parser.add_argument("--cag-null-validated", action="store_true",
                     help="assert that the served checkpoint was fine-tuned with instruction "
                          "dropout, so the null instruction is in-distribution. Required for "
                          "--cag-omega != 1; without it a guidance sweep measures a difference "
                          "against out-of-distribution noise.")
#: Deliver per-phase text during the rollout instead of one constant sentence.
#:
#: Defaults on whenever the language channel is `sub_task`, because that is the only way such a
#: checkpoint is in distribution -- it was trained on phase-appropriate text and one constant
#: sentence on that key is a string it never saw. Pass --no-phase-language to measure that
#: mismatch deliberately rather than by accident.
_parser.add_argument("--phase-language", dest="phase_language", action="store_true",
                     default=None, help="advance the instruction on gripper transitions "
                                        "(default: on for the sub_task channel)")
_parser.add_argument("--no-phase-language", dest="phase_language", action="store_false",
                     help="send one constant sentence even on the sub_task channel")
_parser.add_argument("--out", default="data/output/eval")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(_parser)
args_cli = _parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

# Cell resolution and argument validation happen HERE, before the simulator boots.
#
# This module's own rule, and it was worth re-learning: an argument error after AppLauncher
# surfaces ~90 s into an Isaac boot, and because everything past this point must leave through
# os._exit, the failure is a bare exit with no result file -- which is how a broken control run
# once read as a silent success. The cell layer is stdlib-only precisely so it can run up here.
#
# `arbiter.suites.cells` and the language-channel constants are imported before AppLauncher for
# the same reason; neither touches Isaac.
from arbiter.suites import cells as ARB_CELLS  # noqa: E402
from arbiter.suites.spec import lang_obs_key as _lang_obs_key  # noqa: E402
from arbiter.suites.spec import sub_instructions_from_attrs  # noqa: E402

#: Which language channel the observation carries, from ARBITER_LANG_KEY. Load-bearing for the
#: cells: three of the four are only interpretable on `sub_task`, because topology's task-level
#: instruction never names a side.
LANGUAGE_CHANNEL = os.environ.get("ARBITER_LANG_KEY", "task").strip().lower()
LANGUAGE_KEY = _lang_obs_key(LANGUAGE_CHANNEL)

#: Whether to advance the instruction per phase. On by default for `sub_task`, because a
#: checkpoint trained on phase-appropriate text is out of distribution when fed one constant
#: sentence on that key -- which is what made the first two attempts to evaluate one score
#: 0 success. Off for `task`, whose training language really is one sentence per episode.
PHASE_LANGUAGE = (LANGUAGE_CHANNEL == "sub_task") if args_cli.phase_language is None \
    else bool(args_cli.phase_language)


def _resolve_cell_flags() -> None:
    """Rewrite args_cli from an arbitration cell definition.

    Everything a cell implies -- axis, blocker configuration, barrier heights, spoken text --
    comes from `arbiter.suites.cells`, so the table in the paper and the flags in the run cannot
    disagree. Conflicting manual flags are refused rather than silently overridden.

    **Why the middle phase is spoken from step 0**, rather than delivered on a phase schedule:
    the policy commits to its discrete route choice between step 17 and step 45, well before the
    first observable motion at ~step 90. That is inside what would be the *reach* phase, so route
    text that only arrives when the carry phase begins arrives after the decision it is meant to
    influence. Speaking the route phrase from the start is therefore required for the
    intervention to reach the decision point at all -- at the cost of the model hearing carry
    text during the reach, which is the smaller distortion of the two.
    """
    name = args_cli.cell
    if args_cli.cell_arm is None:
        raise SystemExit(
            f"[EVAL] --cell {name} needs --cell-arm. Arms for this cell: "
            f"{[a.name for a in ARB_CELLS.cell(name).arms]}. A single arm is not a result -- "
            f"compliance alone cannot separate obedience from habit -- so run the pair and "
            f"report the difference."
        )
    try:
        cell = ARB_CELLS.cell(name)
        arm = cell.arm(args_cli.cell_arm)
    except KeyError as e:
        raise SystemExit(f"[EVAL] {e}") from None

    # Refuse the cell outright if its wording is out of vocabulary on this language channel.
    try:
        ARB_CELLS.assert_interpretable(name, LANGUAGE_CHANNEL)
    except ValueError as e:
        raise SystemExit(f"[EVAL] {e}") from None

    if args_cli.axis is not None and args_cli.axis != "topology":
        raise SystemExit(
            f"[EVAL] --cell {name} is built on the barrier geometry, so it runs the 'topology' "
            f"axis; --axis {args_cli.axis!r} contradicts it. Drop --axis."
        )
    manual = [f for f in ("blocker", "blocker_ghost", "blocker_legacy")
              if getattr(args_cli, f)]
    if args_cli.speak is not None:
        manual.append("speak")
    if args_cli.swap_instruction:
        manual.append("swap_instruction")
    if manual:
        raise SystemExit(
            f"[EVAL] --cell sets {manual} itself; passing them too gives two answers to the "
            f"same question. Drop them, or drop --cell and assemble the cell by hand."
        )

    args_cli.axis = "topology"
    flags = ARB_CELLS.blocker_flags(name)
    args_cli.blocker = flags["with_blocker"]
    args_cli.blocker_ghost = flags["blocker_ghost"]
    args_cli.blocker_legacy = flags["blocker_legacy"]

    if args_cli.barrier_h is None:
        raise SystemExit(
            f"[EVAL] --cell {name} needs --barrier-h, because the barrier is baked into the "
            f"stage and one process realises one height. Heights for this cell: "
            f"{[round(h, 3) for h in cell.heights]}. Drive the sweep from a shell loop."
        )
    if round(float(args_cli.barrier_h), 6) not in {round(h, 6) for h in cell.heights}:
        raise SystemExit(
            f"[EVAL] barrier_h={args_cli.barrier_h} is not one of cell {name}'s heights "
            f"{[round(h, 3) for h in cell.heights]}. For the lane cells that is deliberate: "
            f"below h* the route class is 'over' and there is no lane to choose, so a side "
            f"score would not be interpretable."
        )

    params = {"barrier_h": float(args_cli.barrier_h),
              "barrier_offset": float(args_cli.barrier_offset)}
    phases = ARB_CELLS.spoken_sub_tasks(name, arm.name, params)
    # Keep the whole list. Collapsing it to the middle phase and speaking that from step 0 is
    # what made the first two evaluation attempts uninterpretable: the policy heard carry-text
    # during the reach, scored 0 success, and lifted over the barrier where it should have gone
    # around. PhaseSpeaker delivers them in order instead.
    args_cli.cell_phases = phases

    kind, want = ARB_CELLS.expected(name, arm.name, params)
    print(f"[EVAL] CELL {name} arm={arm.name} authority={cell.authority} "
          f"blocker={cell.blocker} channel={LANGUAGE_CHANNEL}", flush=True)
    print(f"[EVAL]   speaks : {args_cli.speak!r}", flush=True)
    print(f"[EVAL]   expects: {kind} == {want!r}"
          f"{' and route_class == around' if cell.requires_around else ''}", flush=True)
    if not cell.headline:
        print(f"[EVAL]   NOTE: {name} is safety arbitration and is reported SEPARATELY from "
              f"the controllability headline -- complying with the instruction here is a "
              f"collision.", flush=True)


if args_cli.cell:
    _resolve_cell_flags()
if args_cli.axis is None:
    raise SystemExit("[EVAL] one of --axis or --cell is required")
if args_cli.cag_omega != 1.0 and not args_cli.cag_null_validated:
    # Fails here rather than after the boot, and with the full explanation from cag.py.
    from arbiter.policy.cag import EMPTY_INSTRUCTION, require_validated_null  # noqa: E402
    try:
        require_validated_null(null_is_in_distribution=False, omega=args_cli.cag_omega,
                               null_instruction=EMPTY_INSTRUCTION)
    except ValueError as e:
        raise SystemExit(f"[EVAL] {e}") from None

_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402

from arbiter.collect import constants as K  # noqa: E402
from arbiter.collect import success as S  # noqa: E402
from arbiter.policy import cag as CAG  # noqa: E402
from arbiter.policy.phase_speaker import PhaseSpeaker  # noqa: E402
from arbiter.sim.env.isaac_backend import IsaacMotionBackend  # noqa: E402
from arbiter.sim.env.scene import (  # noqa: E402
    HOME_JOINT_POS,
    assert_condition_realised,
    build_scene,
    episode_seed,
    jittered_home,
    place_object,
    reset_condition_scene,
)
from arbiter.suites.spec import (  # noqa: E402
    AXES,
    TRAINING_EXCLUSIONS,
    bypass_side,
    enumerate_conditions,
    forced_bypass_side,
    is_withheld_from_training,
    object_keys_for,
    object_start_xy,
    object_start_yaw,
    order_move_sequence,
    target_xys,
    transport_target,
)
from arbiter.suites.layout_gen import ORDER_OBJECT_POS, ORDER_TARGET_POS  # noqa: E402

#: Must match arbiter/policy/gr00t/modality_config.py and each dataset's modality.json.
JOINT_KEYS = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7", "panda_finger_joint1",
]
CAM_TO_MODALITY = {"cam_head": "image", "cam_wrist": "wrist_image"}
#: Must match what the trained checkpoint's modality config declared, because GR00T resolves
#: language as `observation["language"][modality_keys[0]]` -- send the wrong key and the model
#: gets no instruction at all. Set ARBITER_LANG_KEY to the same value the run was trained with.
#: The mapping lives in the spec, which both this (sim venv) and the trainer (gr00t venv) can
#: import; they cannot import each other.


def grab_frames(scene, cams) -> dict:
    """One RGB frame per camera. Kept as uint8 arrays; encoding happens once at the end."""
    out = {}
    for cam in cams:
        rgb = scene[cam].data.output.get("rgb")
        if rgb is None:
            continue
        img = rgb[0, ..., :3].detach().cpu().numpy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        out[cam] = img
    return out


def _fmt_param(v) -> str:
    """Compact, filename-safe rendering of a condition parameter."""
    if isinstance(v, float):
        return f"{v:g}".replace("-", "m").replace(".", "p")
    return str(v).replace("-", "m").replace(".", "p")


def write_video(frames: list, cond, ep: int, ok: bool, why: str) -> None:
    """Head and wrist side by side, captioned with the condition and the outcome.

    Captioned in the pixels rather than the filename: these get looked at next to each other,
    and a folder of near-identical clips is unreadable without the condition on the frame.
    """
    import cv2

    out = Path(args_cli.out)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    vdir = out / "video"
    vdir.mkdir(parents=True, exist_ok=True)

    head_key = "cam_head" if "cam_head" in frames[0] else next(iter(frames[0]))
    h, w = frames[0][head_key].shape[:2]
    has_wrist = "cam_wrist" in frames[0]
    pane_w = w + (w // 2 if has_wrist else 0)
    canvas_h = h + 30                                  # caption strip

    # The sweep coordinate alone is NOT unique. TOPOLOGY's is |h - nearest trained height|, so
    # h=0.04 and h=0.11 both give d=1 and silently overwrote each other -- eight rollouts wrote
    # six files, and the two that survived were whichever finished last. The swept parameters go
    # in the name so a clip is identifiable from the filename alone.
    params = "_".join(f"{k}{_fmt_param(v)}" for k, v in sorted(cond.params.items()))
    tag = f"{cond.axis}_{'ok' if ok else 'FAIL'}_d{cond.sweep_coord:g}"
    if params:
        tag += f"_{params}"
    tag += f"_ep{ep}"
    path = vdir / f"{tag}.mp4"

    label = f"{cond.axis}  d={cond.sweep_coord:g}  {'SUCCESS' if ok else 'FAIL'}"
    sub = (why[:70] if not ok else cond.instruction[:70])

    def composite(i, fr):
        canvas = np.zeros((canvas_h, pane_w, 3), np.uint8)
        canvas[:h, :w] = cv2.cvtColor(fr[head_key], cv2.COLOR_RGB2BGR)
        if has_wrist:
            wr = cv2.resize(cv2.cvtColor(fr["cam_wrist"], cv2.COLOR_RGB2BGR), (w // 2, h))
            canvas[:h, w:] = wr
        colour = (110, 200, 130) if ok else (90, 90, 235)
        cv2.putText(canvas, label, (8, h + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1,
                    cv2.LINE_AA)
        cv2.putText(canvas, f"step {i + 1}/{len(frames)}  {sub}", (8, h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 170, 170), 1, cv2.LINE_AA)
        return canvas

    # Piped to ffmpeg/libx264 rather than cv2.VideoWriter. OpenCV builds usually ship without an
    # H.264 encoder for licensing reasons, so requesting the 'avc1' fourcc silently falls back to
    # 'mp4v' -- MPEG-4 Part 2, which no browser plays. The first 16 clips were written that way
    # and had to be re-encoded. isaaclab2lerobot.py already forces libx264+avc1 for exactly this
    # reason; this matches it.
    import shutil
    import subprocess

    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{pane_w}x{canvas_h}", "-r", str(args_cli.video_fps),
               "-i", "-", "-an",
               "-c:v", "libx264", "-tag:v", "avc1", "-pix_fmt", "yuv420p",
               "-preset", "veryfast", "-crf", "23",
               "-movflags", "+faststart", str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i, fr in enumerate(frames):
            proc.stdin.write(composite(i, fr).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            print(f"[EVAL] ffmpeg failed for {tag}", flush=True)
            return
    else:
        # No ffmpeg: write what we can, and say what it is rather than implying H.264.
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(args_cli.video_fps), (pane_w, canvas_h))
        for i, fr in enumerate(frames):
            vw.write(composite(i, fr))
        vw.release()
        print(f"[EVAL] WARNING no ffmpeg; wrote mp4v, which browsers cannot play", flush=True)

    print(f"[EVAL] video -> {path}", flush=True)


def _instruction_of(axis_name: str, key: str) -> str:
    """The instruction belonging to a condition key on this axis.

    Restricted to the same axis on purpose: a cross-axis instruction would change what the task
    IS, not how to do it, and the schedule is meant to test guidance rather than task swapping.
    """
    for c in enumerate_conditions(AXES[axis_name], object_keys_for(axis_name)):
        if c.key() == key:
            return c.instruction
    keys = [c.key() for c in enumerate_conditions(AXES[axis_name],
                                                  object_keys_for(axis_name))]
    raise SystemExit(f"no condition {key!r} on axis {axis_name!r}. Available:\n  "
                     + "\n  ".join(keys))


def _parse_schedule(axis_name: str, spec: str | None) -> list[tuple[int, str]]:
    """Parse "step:key,step:key" into a sorted (step, instruction) list."""
    if not spec:
        return []
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"--instruction-schedule: expected step:key, got {part!r}")
        step, key = part.split(":", 1)
        out.append((int(step), _instruction_of(axis_name, key.strip())))
    return sorted(out, key=lambda t: t[0])


def build_observation(scene, backend, cams, instruction: str) -> dict:
    """The (B=1, T=1, ...) observation dict the policy server expects."""
    joints = backend.getj() + [backend.gripper_pos()]
    state = {k: np.array([[[joints[i]]]], dtype=np.float32)
             for i, k in enumerate(JOINT_KEYS)}
    video = {}
    for cam in cams:
        key = CAM_TO_MODALITY.get(cam)
        if key is None:
            continue
        rgb = scene[cam].data.output.get("rgb")
        if rgb is None:
            continue
        img = rgb[0, ..., :3].detach().cpu().numpy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        video[key] = img[None, None, ...]
    return {"video": video, "state": state, "language": {LANGUAGE_KEY: [[instruction]]}}


def decode_chunk(chunk: dict, t: int) -> list[float]:
    """One timestep out of an action chunk, as 8 joint targets."""
    out = []
    for k in JOINT_KEYS:
        v = np.asarray(chunk[k])
        while v.ndim > 2:
            v = v[0]
        out.append(float(v[t][0] if v.ndim == 2 else v[t]))
    return out



class ZmqPolicyClient:
    """Minimal client for GR00T's PolicyServer, speaking its wire format directly.

    GR00T's own `PolicyClient` cannot be imported here: this process runs in the sim venv on
    Python 3.11 with Isaac Lab, and `gr00t` lives in a 3.10 venv. That split is the reason the
    policy is a server at all, so the client has to be re-expressed rather than reused.

    The protocol is small enough that this is safer than the alternatives -- msgpack, with
    ndarrays carried as `.npy` bytes under `__ndarray_class__`. Kept deliberately literal so a
    change on the server side fails loudly here rather than being silently reinterpreted.
    """

    def __init__(self, host: str, port: int, timeout_ms: int = 30000):
        import zmq
        self._zmq = zmq
        self.ctx = zmq.Context()
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self._connect()

    def _connect(self) -> None:
        self.sock = self.ctx.socket(self._zmq.REQ)
        self.sock.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            import io
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        return obj

    @staticmethod
    def _decode(obj):
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            import io
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    def _call(self, endpoint: str, data=None, requires_input: bool = True):
        import msgpack
        req = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        try:
            self.sock.send(msgpack.packb(req, default=self._encode))
            msg = self.sock.recv()
        except self._zmq.error.Again:
            self._connect()          # REQ is stuck awaiting a reply that never came
            raise
        if msg == b"ERROR":
            raise RuntimeError("policy server returned ERROR")
        resp = msgpack.unpackb(msg, object_hook=self._decode)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"policy server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self._call("ping", requires_input=False)
            return True
        except Exception:
            return False

    def get_action(self, observation: dict) -> dict:
        resp = self._call("get_action", {"observation": observation, "options": None})
        # The server may answer with the action dict directly or wrap it alongside info.
        if isinstance(resp, (list, tuple)):
            resp = resp[0]
        if isinstance(resp, dict) and "action" in resp and JOINT_KEYS[0] not in resp:
            resp = resp["action"]
        return resp


class RandomPolicy:
    """Uniform joint targets inside the soft limits. The protocol's random floor.

    Also the harness's own control: if this scores above chance the success predicate is
    admitting something it should not, and no policy number from this rig can be trusted.
    """

    def __init__(self, backend, seed: int):
        self.rng = np.random.default_rng(seed)
        lo = backend.robot.data.soft_joint_pos_limits[0, backend._arm.joint_ids, 0]
        hi = backend.robot.data.soft_joint_pos_limits[0, backend._arm.joint_ids, 1]
        self.lo = lo.cpu().numpy()
        self.hi = hi.cpu().numpy()

    def get_action(self, _obs) -> dict:
        arm = self.rng.uniform(self.lo, self.hi)
        grip = float(self.rng.uniform(0.0, 0.04))
        vals = list(arm) + [grip]
        return {k: np.array([[v]], dtype=np.float32) for k, v in zip(JOINT_KEYS, vals)}


class HoldPolicy:
    """Hold the start pose. The no-op floor: whatever this scores, no policy is credited for."""

    def __init__(self, home: list[float]):
        self.vals = list(home) + [0.04]

    def get_action(self, _obs) -> dict:
        return {k: np.array([[v]], dtype=np.float32) for k, v in zip(JOINT_KEYS, self.vals)}


def pick_conditions(axis_name: str, object_keys: list[str]) -> list:
    splits = ("train", "test") if args_cli.split == "all" else (args_cli.split,)
    conds = enumerate_conditions(AXES[axis_name], object_keys, splits=splits)
    if axis_name == "topology" and args_cli.barrier_h is not None:
        h, off = float(args_cli.barrier_h), float(args_cli.barrier_offset)
        conds = [c for c in conds
                 if abs(float(c.params["barrier_h"]) - h) < 1e-6
                 and abs(float(c.params["barrier_offset"]) - off) < 1e-6]
    conds.sort(key=lambda c: (c.sweep_coord, c.key()))
    if args_cli.max_conditions:
        # Stride, so a cap still spans the sweep instead of taking only near-support
        # conditions -- which would report a radius far larger than the truth.
        step = max(1, len(conds) // args_cli.max_conditions)
        conds = conds[::step][:args_cli.max_conditions]
    return conds


def evaluate(scene, sim, info, backend, policy, cond, cams, ep: int) -> dict:
    """One rollout. Returns the record; never raises for a policy failure."""
    axis_name = cond.axis
    if args_cli.blocker:
        cond.params["blocker"] = 1.0
    # A condition can be `in_train` by the axis's own rule and STILL never have been shown to the
    # policy -- `in_train` answers "is this part of the sweep holdout", TRAINING_EXCLUSIONS answers
    # "did the policy see it". Since v7 withholds the RIGHT-detour episodes, scoring one of those
    # as an ID control would report a low number for a condition the policy was never trained on,
    # and an ID control is exactly what every retention ratio is divided by. Warn loudly rather
    # than let it be read as competence loss.
    if cond.split == "train" and is_withheld_from_training(axis_name, cond.params):
        print(f"[EVAL] *** {cond.key()} is in_train for this axis but its episodes are WITHHELD "
              f"from the training set ({TRAINING_EXCLUSIONS.get(axis_name, '?')}). "
              f"This is NOT an ID control -- the policy has never seen it.", flush=True)
    assert_condition_realised(axis_name, cond.params, info["built"])

    seed = episode_seed(cond.key(), ep)
    home = jittered_home(seed)
    backend.reset_to(home)

    # Target pads first: they are kinematic and collisionless, so they neither fall
    # nor disturb the object settle, and placing them before the object means they
    # are present in every recorded frame.
    # Same call the collector uses, so a rollout is scored in the scene the demonstrations were
    # recorded in. Returning the used keys is what keeps ORDER's pair aligned between the
    # placement, the pads and the per-object scoring below.
    rest, used_keys = reset_condition_scene(scene, sim, info, cond)
    obj_names = [info["instance_name"][k] for k in used_keys]

    # Pad-decoupling control: move the marker off the true target after the scene is built.
    pad_xy = [list(t) for t in target_xys(cond)]
    if args_cli.pad_offset:
        from arbiter.sim.env.scene import place_pad
        from arbiter.suites.spec import pad_keys_for_condition
        import math as _m
        starts = [object_start_xy(cond)] if cond.axis != "order" else [
            (float(x), float(y)) for x, y in ORDER_OBJECT_POS]
        pad_xy = []
        for t, s0 in zip(target_xys(cond), starts * len(target_xys(cond))):
            if args_cli.pad_offset_dir == "transport":
                vx, vy = t[0] - s0[0], t[1] - s0[1]
                n = _m.hypot(vx, vy) or 1.0
                pad_xy.append([t[0] + args_cli.pad_offset * vx / n,
                               t[1] + args_cli.pad_offset * vy / n])
            else:
                pad_xy.append([t[0], t[1] + args_cli.pad_offset])
        for pkey, xy in zip(pad_keys_for_condition(cond), pad_xy):
            place_pad(scene, sim, info, pkey, xy)

    if cams:
        for _ in range(8):
            sim.render()
            scene.update(0.0)

    kind = S.success_kind(axis_name)
    frames: list = []
    want_video = args_cli.save_video != "none"

    start_xy = [backend.object_pos_world(n)[:2] for n in obj_names]
    first_move = [None] * len(obj_names)          # step each object first left its start

    def outcome() -> tuple[bool, str]:
        """The success predicate, evaluated from the live scene."""
        if kind == "lift":
            return S.lifted(
                object_z_world=backend.object_pos_world(obj_names[0])[2],
                rest_z_world=rest[0],
                object_xy=backend.object_pos_in_base(obj_names[0]),
                eef_xy=backend.getp()[0],
            )
        if kind == "place":
            tgt = transport_target(cond)
            if tgt is None:
                return False, "no target derivable"
            return S.placed(backend.object_pos_world(obj_names[0]), tgt)
        # ORDER: both placed AND in the sequence the instruction named.
        parts = [S.placed(backend.object_pos_world(n), tuple(t))
                 for n, t in zip(obj_names, ORDER_TARGET_POS)]
        if not all(p[0] for p in parts):
            return False, "; ".join(p[1] for p in parts if p[1])
        moved = [(t, i) for i, t in enumerate(first_move) if t is not None]
        if len(moved) < len(obj_names):
            return False, "not every object moved"
        observed = [i for _, i in sorted(moved)]
        # From the spec, not restated. The expert, the instruction wording and this check
        # must agree on which slot moves first; three copies of the mapping is how they
        # come to disagree.
        want = order_move_sequence(cond)
        if observed != want:
            return False, (f"wrong order: moved {observed} but the instruction asked for {want}")
        return True, ""

    horizon = 0
    chunk = None
    steps = 0
    held = 0
    # The words actually shown to the policy. Normally the condition's own instruction; under
    # --swap-instruction, a different axis-valid one; under --speak, arbitrary text. Scoring
    # stays on this condition in every case.
    spoken = cond.instruction
    if args_cli.swap_instruction:
        spoken = _instruction_of(axis_name, args_cli.swap_instruction)
    if args_cli.speak is not None:
        spoken = args_cli.speak

    # Per-phase delivery. The texts come from the cell when one is selected, otherwise from the
    # spec's own phase list for this condition -- never formatted here, so what is spoken is a
    # string the training set contains.
    speaker: PhaseSpeaker | None = None
    if PHASE_LANGUAGE:
        phase_texts = getattr(args_cli, "cell_phases", None)
        if phase_texts is None:
            phase_texts = sub_instructions_from_attrs(cond.to_attrs())
        speaker = PhaseSpeaker(phase_texts)
        spoken = speaker.text()
    # (step -> instruction), sorted. Empty unless --instruction-schedule was given, in which
    # case `spoken` is recomputed at each chunk boundary rather than fixed for the episode.
    schedule = _parse_schedule(axis_name, args_cli.instruction_schedule)
    spoken_log: list[tuple[int, str]] = []

    # End-effector path, subsampled. Needed because a success rate cannot see HOW the object got
    # to the target, and TOPOLOGY's whole claim is about the route: the axis declares a
    # `topology_correct_rate` mechanism metric that was never computed, so "100% at h=0.09" could
    # equally mean the policy selected the AROUND class or that it went over and the predicate
    # could not tell. Only the path distinguishes them.
    eef_path: list[list[float]] = []
    ok, why = False, "did not finish"
    t0 = time.time()
    for steps in range(1, args_cli.max_steps + 1):
        if cams:
            sim.render()
            scene.update(0.0)
        if chunk is None or horizon >= _chunk_len(chunk):
            # Re-resolved per chunk, which is the finest granularity that can matter: a change
            # mid-chunk cannot affect actions already decoded.
            if schedule:
                due = [t for t, _ in schedule if t <= steps]
                if due:
                    nxt = next(txt for t, txt in reversed(schedule) if t == max(due))
                    if nxt != spoken:
                        spoken = nxt
                        spoken_log.append((steps, spoken))
            if speaker is not None:
                nxt = speaker.text()
                if nxt != spoken:
                    spoken = nxt
                    spoken_log.append((steps, spoken))
            obs = build_observation(scene, backend, cams, spoken)
            chunk = policy.get_action(obs)
            horizon = 0
        _act = decode_chunk(chunk, horizon)
        backend.movej(*_split(_act))
        if speaker is not None:
            # The COMMANDED gripper, which is what sub_task_spans segmented on.
            speaker.observe(_act[-1])
        horizon += 1
        backend.step()
        if steps % 4 == 0:                      # ~7.5 Hz, plenty for a route classification
            eef_path.append([round(float(v), 4) for v in backend.getp()[0]])
        if want_video:
            frames.append(grab_frames(scene, cams))

        # Which object moved first is a temporal fact, so it has to be watched during the
        # rollout. Scoring ORDER from the end state alone cannot see it at all -- that is why
        # a policy doing the two placements in the WRONG order was scored 20/20.
        for i, n in enumerate(obj_names):
            if first_move[i] is None:
                p = backend.object_pos_world(n)[:2]
                if ((p[0]-start_xy[i][0])**2 + (p[1]-start_xy[i][1])**2) ** 0.5 > 0.02:
                    first_move[i] = steps

        if steps % args_cli.check_every == 0:
            good, reason = outcome()
            held = held + 1 if good else 0
            ok, why = good, reason
            if held >= args_cli.hold_checks:
                # Sustained success: stop. Running the full budget anyway quadrupled the cost
                # of every successful rollout for no information.
                break

    if not ok:
        ok, why = outcome()

    if want_video and frames:
        outcome = "ok" if ok else "fail"
        if args_cli.save_video == "all" or args_cli.save_video == outcome or (
                args_cli.save_video == "fail" and not ok) or (
                args_cli.save_video == "success" and ok):
            write_video(frames, cond, ep, ok, why)

    return {
        "key": cond.key(), "axis": axis_name, "split": cond.split,
        "sweep_coord": float(cond.sweep_coord), "params": cond.params,
        "episode": ep, "success": bool(ok), "reason": why,
        "steps": steps, "seconds": round(time.time() - t0, 2),
        "first_move_steps": list(first_move),
        "early_exit": steps < args_cli.max_steps,
        # Where each object actually ended, and how far that is from its target. The predicate
        # already decides success, but without these a reported rate cannot be audited after the
        # fact -- and this project has twice shipped a predicate that measured the wrong thing.
        # An EXTENT run at 98.5% is either real competence or a broken check, and the only way
        # to tell from the saved JSON is the distance.
        "final_object_xy": [[round(float(v), 4) for v in
                             backend.object_pos_world(n)[:2]] for n in obj_names],
        "target_xy": [[round(float(v), 4) for v in t] for t in target_xys(cond)],
        # Distance to the DISPLACED pad as well as to the true target. If the policy is
        # servoing to the marker these diverge: it lands on the pad and fails the predicate.
        "pad_xy": [[round(float(v), 4) for v in t] for t in pad_xy],
        "final_pad_dist": [
            round(float(((backend.object_pos_world(n)[0] - t[0]) ** 2
                         + (backend.object_pos_world(n)[1] - t[1]) ** 2) ** 0.5), 4)
            for n, t in zip(obj_names, pad_xy)
        ],
        "pad_offset": float(args_cli.pad_offset),
        "pad_offset_dir": args_cli.pad_offset_dir,
        # A schedule, not a string: with --instruction-schedule the episode hears several
        # instructions, and which one was live at which step is the whole measurement.
        "spoken_instruction": spoken,
        "spoken_schedule": [[t, txt] for t, txt in spoken_log],
        "own_instruction": cond.instruction,
        "eef_path": eef_path,
        "final_target_dist": [
            round(float(((backend.object_pos_world(n)[0] - t[0]) ** 2
                         + (backend.object_pos_world(n)[1] - t[1]) ** 2) ** 0.5), 4)
            for n, t in zip(obj_names, target_xys(cond))
        ],
    }


def _chunk_len(chunk: dict) -> int:
    v = np.asarray(chunk[JOINT_KEYS[0]])
    while v.ndim > 2:
        v = v[0]
    return int(v.shape[0])


def _split(vals: list[float]) -> tuple[list[float], float]:
    return vals[:7], vals[7]


def main() -> int:
    axis_name = args_cli.axis
    object_keys = object_keys_for(axis_name)

    # Validate the instruction arguments BEFORE the simulator starts. Resolving them per-episode
    # means a mistyped key fails ~90 s into an Isaac boot, and because anything after
    # AppLauncher must leave through os._exit, the failure is a bare exit with no result written
    # -- which is how the first run of this control looked like a silent success.
    if args_cli.instruction_schedule:
        sched = _parse_schedule(axis_name, args_cli.instruction_schedule)
        print(f"[EVAL] instruction schedule ({len(sched)} entries):", flush=True)
        for t, txt in sched:
            print(f"[EVAL]   step {t:4d} -> {txt!r}", flush=True)
    if args_cli.swap_instruction:
        print(f"[EVAL] swapped instruction: "
              f"{_instruction_of(axis_name, args_cli.swap_instruction)!r}", flush=True)
    if args_cli.speak is not None:
        # Mutually exclusive with the other two: three different answers to "what is spoken at
        # step 0" would otherwise resolve by order of assignment, silently.
        if args_cli.swap_instruction or args_cli.instruction_schedule:
            raise SystemExit("[EVAL] --speak cannot be combined with --swap-instruction or "
                             "--instruction-schedule: all three set the spoken text.")
        print(f"[EVAL] spoken text overridden: {args_cli.speak!r}", flush=True)
        print("[EVAL]   (scoring stays on each condition's own success predicate)", flush=True)

    if args_cli.blocker and axis_name != "topology":
        raise SystemExit("[EVAL] --blocker applies to the topology axis only")
    if args_cli.blocker_ghost and not args_cli.blocker:
        raise SystemExit("[EVAL] --blocker-ghost requires --blocker")
    if args_cli.blocker_legacy and not args_cli.blocker:
        raise SystemExit("[EVAL] --blocker-legacy requires --blocker")
    if args_cli.blocker_legacy and args_cli.blocker_ghost:
        raise SystemExit("[EVAL] --blocker-legacy and --blocker-ghost answer different "
                         "questions; pick one")
    scene, sim, info = build_scene(
        axis_name, device=args_cli.device, object_keys=object_keys,
        barrier_h=args_cli.barrier_h, barrier_offset=args_cli.barrier_offset,
        with_blocker=args_cli.blocker,
        blocker_ghost=args_cli.blocker_ghost,
        blocker_legacy=args_cli.blocker_legacy,
        with_cameras=True,
    )
    if args_cli.blocker:
        off = float(args_cli.barrier_offset)
        kind = "GHOST (visual only, passable)" if args_cli.blocker_ghost else "SOLID"
        print(f"[EVAL] BLOCKER {kind}: offset {off:+.2f} {'appears to seal' if args_cli.blocker_ghost else 'seals'} "
              f"the {bypass_side(off)} lane; the detour "
              f"{'need not' if args_cli.blocker_ghost else 'must'} be {forced_bypass_side(off)}",
              flush=True)
    cams = [k for k in scene.keys() if k.startswith("cam")]
    backend = IsaacMotionBackend(scene, sim)

    if args_cli.policy == "random":
        policy = RandomPolicy(backend, args_cli.seed)
        tag = "random"
    elif args_cli.policy == "hold":
        policy = HoldPolicy(HOME_JOINT_POS)
        tag = "hold"
    else:
        policy = ZmqPolicyClient(args_cli.host, args_cli.port)
        if not policy.ping():
            raise SystemExit(f"[EVAL] no policy server at {args_cli.host}:{args_cli.port}")
        tag = "server"

    if args_cli.cag_omega != 1.0:
        if args_cli.policy != "server":
            raise SystemExit(
                f"[EVAL] --cag-omega applies to a served policy; --policy={args_cli.policy} "
                f"ignores its instruction entirely, so guiding it would mix two identical "
                f"predictions and report the difference as zero."
            )
        try:
            policy = CAG.CagPolicy(
                policy, args_cli.cag_omega,
                null_is_in_distribution=args_cli.cag_null_validated,
            )
        except ValueError as e:
            raise SystemExit(f"[EVAL] {e}") from None
        tag = f"server+cag(w={args_cli.cag_omega})"
        print(f"[EVAL] CAG omega={args_cli.cag_omega}: two forward passes per chunk, null "
              f"instruction {CAG.EMPTY_INSTRUCTION!r}", flush=True)

    conds = pick_conditions(axis_name, object_keys)
    if not conds:
        raise SystemExit(f"[EVAL] no conditions for {axis_name} split={args_cli.split}")
    print(f"[EVAL] axis={axis_name} split={args_cli.split} policy={tag} "
          f"conditions={len(conds)} episodes/cond={args_cli.episodes} "
          f"cameras={cams}", flush=True)

    records = []
    for i, cond in enumerate(conds, 1):
        for ep in range(args_cli.episodes):
            r = evaluate(scene, sim, info, backend, policy, cond, cams, ep)
            records.append(r)
        n_ok = sum(1 for r in records[-args_cli.episodes:] if r["success"])
        if i % 5 == 0 or n_ok == 0:
            done = len(records)
            rate = sum(1 for r in records if r["success"]) / max(done, 1)
            print(f"[EVAL] {i}/{len(conds)} cond  d={cond.sweep_coord:6.2f}  "
                  f"{n_ok}/{args_cli.episodes} ok  running SR={100 * rate:.1f}%", flush=True)

    # Success rate per sweep coordinate: the curve a radius is fitted to.
    by_coord: dict[float, list[bool]] = {}
    for r in records:
        by_coord.setdefault(round(r["sweep_coord"], 4), []).append(r["success"])
    curve = [{"sweep_coord": c, "n": len(v), "n_success": sum(v),
              "success_rate": round(sum(v) / len(v), 4)}
             for c, v in sorted(by_coord.items())]

    total = len(records)
    n_ok = sum(1 for r in records if r["success"])
    print(f"\n[EVAL] ==== {axis_name} / {args_cli.split} / {tag} ====", flush=True)
    print(f"[EVAL] {n_ok}/{total} = {100 * n_ok / max(total, 1):.1f}%", flush=True)
    for row in curve:
        print(f"[EVAL]   d={row['sweep_coord']:7.2f}  "
              f"{row['n_success']:3d}/{row['n']:<3d}  {100 * row['success_rate']:5.1f}%",
              flush=True)
    if args_cli.split == "train":
        print(f"[EVAL] ID control: {100 * n_ok / max(total, 1):.1f}% — a low number here "
              f"invalidates the axis rather than being a weak result", flush=True)

    out = Path(args_cli.out)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    name = axis_name + (f"_h{int(round(args_cli.barrier_h * 1000)):04d}"
                        f"_o{int(round(args_cli.barrier_offset * 1000)):+05d}"
                        if args_cli.barrier_h is not None else "")
    if args_cli.cell:
        # Cell and arm go in the FILENAME, not just the payload: an omega sweep writes one file
        # per (cell, arm, height, omega) and `sweep_coord` is not unique across them, so a name
        # that omits them silently overwrites its own earlier cells.
        name = (f"{args_cli.cell.lower().replace('-', '')}_{args_cli.cell_arm}_{name}"
                + (f"_w{args_cli.cag_omega:g}" if args_cli.cag_omega != 1.0 else ""))
    path = out / f"{name}_{args_cli.split}_{tag}.json"

    #: Provenance, so a tradeoff curve cannot be read out of the context that makes it valid.
    #: The language channel matters most: three of the four cells are only interpretable on
    #: sub_task, and a reader who cannot see which channel produced a number cannot tell an
    #: arbitration result from a distribution-shift artifact.
    provenance = {
        "language_channel": LANGUAGE_CHANNEL,
        "language_obs_key": LANGUAGE_KEY,
        "phase_language": PHASE_LANGUAGE,
        "phase_lead_frames": 0,   # training used horizon//2 = 8; see PhaseSpeaker on why not here
        "cag_omega": args_cli.cag_omega,
        "cag_null_validated": bool(args_cli.cag_null_validated),
    }
    if args_cli.cell:
        _c = ARB_CELLS.cell(args_cli.cell)
        _params = {"barrier_h": float(args_cli.barrier_h),
                   "barrier_offset": float(args_cli.barrier_offset)}
        _kind, _want = ARB_CELLS.expected(args_cli.cell, args_cli.cell_arm, _params)
        provenance["cell"] = {
            "name": _c.name,
            "arm": args_cli.cell_arm,
            "authority": _c.authority,
            "blocker": _c.blocker,
            "headline": _c.headline,
            "requires_around": _c.requires_around,
            "expect_kind": _kind,
            "expect_value": _want,
            "sealed_side": ARB_CELLS.sealed_side(args_cli.cell, args_cli.barrier_offset),
            "forced_side": ARB_CELLS.forced_side(args_cli.barrier_offset),
        }
    if isinstance(policy, CAG.CagPolicy):
        provenance["cag"] = policy.describe()

    path.write_text(json.dumps({
        "axis": axis_name, "split": args_cli.split, "policy": tag,
        "episodes_per_condition": args_cli.episodes, "max_steps": args_cli.max_steps,
        "provenance": provenance,
        "n": total, "n_success": n_ok, "curve": curve, "records": records,
    }, indent=2) + "\n")
    print(f"[EVAL] wrote {path}", flush=True)
    return 0


def _hard_exit(code: int) -> None:
    """Isaac's non-daemon threads outlive a bare raise. See render_episode."""
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(code)


if __name__ == "__main__":
    try:
        _hard_exit(main())
    except SystemExit as e:
        _hard_exit(int(e.code) if isinstance(e.code, int) else 1)
    except BaseException:
        import traceback
        traceback.print_exc()
        _hard_exit(1)
