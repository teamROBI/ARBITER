#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the scripted expert on real episodes and report success per condition, per axis.

**This is the gate the whole benchmark rests on.** The central claim is that a held-out
condition is *achievable* -- that failure means the policy could not compose the behaviour, not
that the condition was impossible. v1 asserted that and never demonstrated it, which is the
single most damaging hole in the submission: with no achievability control, a large drop is
equally consistent with "policies cannot recombine factors" and "the uncharted condition is
simply a harder motor problem."

So: the expert must succeed at essentially every test point. Anything less and the axis is
measuring difficulty rather than composition, and the numbers have to be thrown out.

Strictly harder than ``probe_reachability``, which asks only whether IK can place the wrist
somewhere. This runs whole episodes with the table, the object, gravity and collisions present,
so it also catches a grasp that slips, a path that clips the barrier, and a place that does not
settle.

One axis per launch: building a second SimulationContext in the same process deadlocks (see
probe_reachability), and TOPOLOGY additionally needs one launch per barrier height because a
prop cannot be swapped once the stage is built.

Usage:
    achievability_gate.py --axis position --per-axis 24
    achievability_gate.py --axis topology --barrier-h 0.12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
_parser.add_argument("--axis", required=True, help="axis to gate")
_parser.add_argument("--per-axis", type=int, default=24,
                     help="conditions to attempt (stratified across the sweep)")
_parser.add_argument("--splits", default="train,test",
                     help="which splits to gate; both, because a training condition the "
                          "expert cannot demonstrate is a collection bug and a test condition "
                          "it cannot demonstrate invalidates the axis")
_parser.add_argument("--barrier-h", type=float, default=None,
                     help="TOPOLOGY only: barrier height for this launch")
_parser.add_argument("--barrier-offset", type=float, default=0.0,
                     help="TOPOLOGY only: barrier lateral offset for this launch. Like the "
                          "height, a prop cannot move after the stage is built, so one launch "
                          "covers one (height, offset) pair.")
_parser.add_argument("--objects", default="arb_cube_red",
                     help="comma-separated graspable assets")
_parser.add_argument("--max-steps", type=int, default=3000)
_parser.add_argument("--partition-idx", type=int, default=0)
_parser.add_argument("--partition-total", type=int, default=1,
                     help="split the axis across parallel launches by striding, so one axis "
                          "can use several GPUs. Strided rather than contiguous: a contiguous "
                          "split would give one worker only the far-from-support conditions, "
                          "so a crashed worker would bias coverage along the very axis being "
                          "measured.")
_parser.add_argument("--attempts", type=int, default=1,
                     help="attempts per condition, each from a differently jittered home pose. "
                          "1 measures achievability -- can the expert do this at all. >1 "
                          "measures reliability, which is what the protocol actually requires "
                          "of the expert at every test point, and which a single deterministic "
                          "attempt cannot report.")
_parser.add_argument("--exhaustive", action="store_true",
                     help="run EVERY condition of the axis instead of a stratified sample. "
                          "The default samples 4 conditions per distinct sweep coordinate, "
                          "which is strong coverage of every distance but not of every "
                          "condition -- POSITION alone declares 667. Slower, and the only "
                          "form of the gate whose pass rate is literally 'all of them'.")
_parser.add_argument("--seed", type=int, default=0)
#: Gate the DETOUR configuration: seal the tie-break bypass lane with a blocker so the expert
#: must detour to the complement, and check the arm can actually do it. This is the plan's
#: pre-condition for the axis -- the forced lane sits further out than anything TOPOLOGY
#: demonstrates, and at +/-0.06 it lands exactly ON the workspace boundary, so `DETOUR_OFFSETS`
#: excludes those. Collecting 180 episodes of an arm straining at its reach limit is the failure
#: this gate exists to prevent.
_parser.add_argument("--blocker", action="store_true",
                     help="seal route()'s tie-break lane and force the complement detour")
_parser.add_argument("--blocker-legacy", action="store_true",
                     help="use the original 0.16 m blocker (does NOT seal the lane; "
                          "reproduction only)")
_parser.add_argument("--no-props", action="store_true",
                     help="omit the axis's static props. Use to isolate whether a prop is "
                          "causing a failure rather than the axis geometry.")
_parser.add_argument("--out", default=None)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(_parser)
args_cli = _parser.parse_args()
args_cli.headless = True

_app = AppLauncher(args_cli).app

from arbiter.collect.context import (  # noqa: E402
    ObjectDropped,
    PhaseTimeout,
    TaskContext,
    TaskFailed,
)
from arbiter.collect.tasks import OrderTask, task_for  # noqa: E402
from arbiter.sim.env.isaac_backend import IsaacMotionBackend  # noqa: E402
from arbiter.sim.env.scene import (  # noqa: E402
    episode_seed,
    jittered_home,
    HOME_JOINT_POS,
    assert_condition_realised,
    build_scene,
    place_object,
    reset_condition_scene,
)
from arbiter.suites.spec import (  # noqa: E402
    order_slot_objects,
    AXES,
    WORKSPACE,
    axis as get_axis,
    object_keys_for,
    object_start_xy,
    object_start_yaw,
    sample_sweep,
)

_OBJECTS_DEFAULT = 'arb_cube_red'
from arbiter.suites.layout_gen import ORDER_OBJECT_POS, ORDER_TARGET_POS  # noqa: E402


def pick_conditions(axis_name: str, object_keys: list[str]) -> list:
    """Conditions to attempt, stratified across the sweep so both ends are represented.

    A gate that samples only near-support conditions would pass while the far end of the sweep
    was unachievable -- which is precisely the failure mode that would corrupt a radius fit.
    """
    ax = get_axis(axis_name)
    splits = tuple(s.strip() for s in args_cli.splits.split(","))
    if args_cli.exhaustive or axis_name in ("order", "factorial", "topology"):
        # Enumerate rather than sample for topology: the height filter below needs every
        # condition at the loaded height to still be present, and a stratified sample across
        # the whole sweep would have thrown most of them away first.
        from arbiter.suites.spec import enumerate_conditions
        conds = enumerate_conditions(ax, object_keys, splits=splits)  # type: ignore[arg-type]
    else:
        n_bins = max(4, args_cli.per_axis // 4)
        s = sample_sweep(ax, object_keys, n_bins=n_bins,
                         per_bin=max(1, args_cli.per_axis // n_bins), seed=args_cli.seed)
        conds = [c for c in s.conditions if c.split in splits]
    # TOPOLOGY: the scene holds ONE barrier asset, but the axis sweeps barrier_h. Running a
    # condition whose barrier_h differs from the loaded asset makes RouteTask compute its apex
    # for a barrier that is not there -- and when the condition's h is the smaller of the two,
    # the apex lands BELOW the real barrier top and the arm drives the object into it.
    #
    # That is exactly what produced 0/6 at every height: the failures were not a routing
    # limitation, they were the task routing around an imaginary barrier. Filter to the loaded
    # height. A prop cannot be swapped after the stage is built, so full coverage of the sweep
    # means one launch per height.
    if axis_name == "topology" and args_cli.barrier_h is not None:
        h = float(args_cli.barrier_h)
        off = float(args_cli.barrier_offset)
        conds = [c for c in conds
                 if abs(float(c.params["barrier_h"]) - h) < 1e-6
                 and abs(float(c.params["barrier_offset"]) - off) < 1e-6]
        if not conds:
            print(f"[GATE] no conditions at barrier_h={h}; the axis sweeps "
                  f"barrier_h so pass a height it actually contains", flush=True)
    if args_cli.partition_total > 1:
        pt, pi = args_cli.partition_total, args_cli.partition_idx
        if not (0 <= pi < pt):
            raise SystemExit(f"[GATE] bad partition {pi}/{pt}")
        conds = [c for i, c in enumerate(conds) if i % pt == pi]
    if args_cli.exhaustive:
        # Deliberately uncapped: --per-axis is a sampling budget, and applying it here would
        # silently truncate an "exhaustive" run back to a sample while still reporting it as
        # exhaustive.
        return conds
    return conds[: args_cli.per_axis]


def run_one(scene, sim, info, backend, cond, axis_name: str, attempt: int = 0) -> dict:
    """One episode. Returns the outcome plus the phase ladder the expert reached."""
    key = cond.object_key
    # Refuse to run a condition the stage does not actually represent.
    assert_condition_realised(axis_name, cond.params, info["built"])
    # Attempt 0 starts from the exact home pose, so a single-attempt run stays exactly what it
    # always was. Later attempts jitter the *starting configuration* -- the same nuisance
    # variation collection uses, and the only axis-orthogonal quantity available. Without it
    # every repeat would be byte-identical and a reliability figure would be meaningless.
    home = (HOME_JOINT_POS if attempt == 0
            else jittered_home(episode_seed(cond.key(), attempt)))
    backend.reset_to(home)

    # Target pads first: they are kinematic and collisionless, so they neither fall
    # nor disturb the object settle, and placing them before the object means they
    # are present in every recorded frame. One call, because five tools each used to
    # carry their own copy of this branch.
    reset_condition_scene(scene, sim, info, cond)

    ctx = TaskContext(backend)
    ctx.object_name = info["instance_name"][key]

    if axis_name == "order":
        task = OrderTask(cond,
                         objects=[info["instance_name"][k] for k in order_slot_objects()],
                         targets=[tuple(t) for t in ORDER_TARGET_POS])
    else:
        task = task_for(cond)(cond)

    out = {"key": cond.key(), "split": cond.split,
           "sweep_coord": cond.sweep_coord, "ok": False, "reason": "", "steps": 0}
    try:
        out["steps"] = backend.drive(task.run(ctx), max_steps=args_cli.max_steps)
        out["ok"] = True
    except (PhaseTimeout, ObjectDropped, TaskFailed, TimeoutError) as e:
        out["reason"] = f"{type(e).__name__}: {e}"
        # Recover the step count from the phase ladder: drive() raises rather than returning,
        # so reporting 0 for every failure hides how far the episode actually got.
        out["steps"] = sum(p.steps for p in ctx.phases)
    out["phases"] = [
        {"label": p.label, "ok": p.ok, "steps": p.steps,
         "err": None if p.final_error is None else round(p.final_error, 4)}
        for p in ctx.phases
    ]
    out["last_phase"] = ctx.last_phase
    return out


def main() -> int:
    axis_name = args_cli.axis
    if axis_name not in AXES:
        print(f"[GATE] unknown axis '{axis_name}'", flush=True)
        return 2
    object_keys = [k.strip() for k in args_cli.objects.split(",") if k.strip()]
    if not args_cli.objects or args_cli.objects == _OBJECTS_DEFAULT:
        # Let the axis choose unless the caller overrode it: ORDER needs a fixed pair and
        # APPROACH needs the bar, and hardcoding that here is how the four tools drift apart.
        object_keys = object_keys_for(axis_name)

    scene, sim, info = build_scene(
        axis_name, device=args_cli.device, object_keys=object_keys,
        barrier_h=args_cli.barrier_h, barrier_offset=args_cli.barrier_offset,
        with_blocker=args_cli.blocker,
        blocker_legacy=args_cli.blocker_legacy,
        with_cameras=False,
        with_props=not args_cli.no_props,
    )
    backend = IsaacMotionBackend(scene, sim)

    conds = pick_conditions(axis_name, object_keys)
    if args_cli.blocker:
        # Mark the conditions so RouteTask takes the forced side and assert_condition_realised
        # can cross-check the stage. Both read the side from the spec's `forced_bypass_side`,
        # so the sealed lane and the avoided lane come from one function.
        from arbiter.suites.spec import forced_bypass_side as _fbs
        for c in conds:
            c.params["blocker"] = 1.0
        off = float(args_cli.barrier_offset)
        print(f"[GATE] BLOCKER on: offset {off:+.2f} seals the "
              f"{'-y' if _fbs(off) == '+y' else '+y'} lane, expert must detour "
              f"{_fbs(off)}", flush=True)
    print(f"[GATE] axis={axis_name} conditions={len(conds)}"
          + (f" barrier_h={args_cli.barrier_h} offset={args_cli.barrier_offset}"
             if args_cli.barrier_h else ""), flush=True)

    n_att = max(1, args_cli.attempts)
    results = []
    per_cond: dict[str, list[bool]] = {}
    for i, cond in enumerate(conds):
        for attempt in range(n_att):
            r = run_one(scene, sim, info, backend, cond, axis_name, attempt=attempt)
            r["attempt"] = attempt
            r["axis"] = axis_name          # so results aggregate across files by axis
            results.append(r)
            per_cond.setdefault(cond.key(), []).append(bool(r["ok"]))
            tag = "ok  " if r["ok"] else "FAIL"
            suffix = f" a{attempt}" if n_att > 1 else ""
            print(f"[GATE] {i + 1:3d}/{len(conds)}{suffix} {tag} d={r['sweep_coord']:6.2f} "
                  f"{r['split']:5s} steps={r['steps']:4d} {r['key'][:56]}"
                  + ("" if r["ok"] else f"\n[GATE]        -> {r['reason'][:110]}"), flush=True)

    n_ok = sum(1 for r in results if r["ok"])
    by_split: dict[str, list] = {}
    for r in results:
        by_split.setdefault(r["split"], []).append(r)

    print(f"\n[GATE] ==== {axis_name} ====", flush=True)
    if n_att > 1:
        # Reliability is per *condition*, not per attempt: a condition that succeeds 2 of 3
        # times is a flaky test point, and averaging it into an overall rate hides exactly the
        # conditions a radius fit would be most distorted by.
        full = sum(1 for v in per_cond.values() if all(v))
        none = sum(1 for v in per_cond.values() if not any(v))
        flaky = {k: v for k, v in per_cond.items() if any(v) and not all(v)}
        print(f"[GATE] reliability over {n_att} attempts: {full}/{len(per_cond)} conditions "
              f"succeeded every time, {len(flaky)} flaky, {none} never", flush=True)
        for k, v in sorted(flaky.items())[:20]:
            print(f"[GATE]   flaky {sum(v)}/{len(v)}  {k[:70]}", flush=True)
    print(f"[GATE] {n_ok}/{len(results)} achievable "
          f"({100 * n_ok / max(len(results), 1):.1f}%)", flush=True)
    for split, rs in sorted(by_split.items()):
        k = sum(1 for r in rs if r["ok"])
        print(f"[GATE]   {split}: {k}/{len(rs)}", flush=True)

    # Where the expert stopped, when it stopped. A gate that only reports a rate does not say
    # whether the geometry, the grasp, or the routing is at fault.
    fails = [r for r in results if not r["ok"]]
    if fails:
        stops: dict[str, int] = {}
        for r in fails:
            stops[r["last_phase"]] = stops.get(r["last_phase"], 0) + 1
        print(f"[GATE]   failures by last phase: "
              f"{dict(sorted(stops.items(), key=lambda kv: -kv[1]))}", flush=True)
        worst = sorted(fails, key=lambda r: -r["sweep_coord"])[:3]
        for r in worst:
            print(f"[GATE]   d={r['sweep_coord']:.2f} {r['reason'][:100]}", flush=True)

    # An EMPTY run must not pass. `n_ok == len(results)` is trivially true at zero conditions,
    # so a filter that matched nothing -- a --barrier-h the axis does not sweep, or a
    # --barrier-offset outside TOPOLOGY_OFFSETS -- reported "VERDICT PASS: the axis is fully
    # achievable" having tested exactly nothing, and exited 0. That is the same shape as the
    # already-documented "Gate 3 check 8 is vacuous at --num-envs 2": a check that cannot fail
    # is worse than no check, because it reads as evidence.
    if not results:
        print("[GATE] VERDICT FAIL: no conditions matched the filters, so nothing was gated. "
              "Check --barrier-h is a height the axis sweeps and --barrier-offset is one of "
              "TOPOLOGY_OFFSETS.", flush=True)
        verdict = "FAIL"
    else:
        verdict = "PASS" if n_ok == len(results) else "FAIL"
    print(f"[GATE] VERDICT {verdict}: the axis {'is' if verdict == 'PASS' else 'IS NOT'} "
          f"fully achievable by the scripted expert", flush=True)

    if args_cli.out:
        out = Path(args_cli.out)
        out.mkdir(parents=True, exist_ok=True)
        tag = axis_name + (
            f"_h{int(round(args_cli.barrier_h * 1000)):04d}"
            f"_o{int(round(args_cli.barrier_offset * 1000)):+04d}"
            if args_cli.barrier_h else "")
        # Partitioned launches must not share a filename, or four workers each write the whole
        # file and the last one wins -- leaving a result that looks complete and covers a
        # quarter of the axis.
        if args_cli.partition_total > 1:
            tag += f"_p{args_cli.partition_idx}of{args_cli.partition_total}"
        (out / f"{tag}.json").write_text(json.dumps(
            {"axis": axis_name, "barrier_h": args_cli.barrier_h,
             "n": len(results), "n_ok": n_ok, "results": results}, indent=2) + "\n")
        print(f"[GATE] wrote {out}/{tag}.json", flush=True)

    sys.stdout.flush()
    import os
    os._exit(0 if verdict == "PASS" else 1)


def _hard_exit(code: int) -> None:
    """Terminate for real; Isaac's non-daemon threads outlive a bare raise. See render_episode."""
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        _hard_exit(int(e.code) if isinstance(e.code, int) else 1)
    except BaseException:
        import traceback
        traceback.print_exc()
        _hard_exit(1)
