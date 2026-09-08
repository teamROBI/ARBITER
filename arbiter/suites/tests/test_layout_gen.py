#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Check generated layouts are valid, reproducible, and reference assets that exist.

    python arbiter/suites/tests/test_layout_gen.py

The check that earns its keep is **props reference real assets**. A layout naming
``arb_barrier_h0120`` when the generator never wrote that file fails at episode-reset time,
deep inside a collection run, as an asset-load error with no hint that the layout and the asset
generator disagree. Catching it here costs milliseconds.

Two v1 conventions are asserted rather than trusted, because both were regressions once:
``session.scene.path`` must stay empty (a baked absolute path silently loaded the scene from
another project's tree), and the schema string must keep its inherited name.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arbiter.suites.spec import (  # noqa: E402
    AXES,
    TOPOLOGY_TRAIN_H,
    WORKSPACE,
    axis as get_axis,
    barrier_heights,
)
from arbiter.suites.layout_gen import HEAD_CAM, SCHEMA, TABLE_TOP_Z, build_layout  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  {'ok   ' if cond else 'FAIL '} {label}")
    if not cond:
        FAILURES.append(label)


def test_structure() -> None:
    print("\n[required keys]")
    required = {"camera", "object_pool", "object_spawn_areas", "props", "robot", "scene",
                "custom_cams", "workspace", "axis"}
    for name in sorted(AXES):
        s = build_layout(get_axis(name))["session"]
        missing = required - set(s)
        check(not missing, f"{name}: all session keys present"
                           + (f" (missing {sorted(missing)})" if missing else ""))


def test_v1_conventions_preserved() -> None:
    print("\n[v1 conventions that were regressions once]")
    for name in sorted(AXES):
        d = build_layout(get_axis(name))
        check(d["session"]["scene"]["path"] == "",
              f"{name}: scene.path is empty, forcing portable resolution")
        check(d["schema"] == SCHEMA, f"{name}: schema keeps its inherited name")
        check(d["session"]["scene"]["file_name"].endswith(".usd"),
              f"{name}: scene resolves by file_name")


def test_props_reference_real_assets() -> None:
    """The load-bearing check: a layout must not name an asset the generator never wrote."""
    print("\n[props reference assets that exist]")
    authored = {f"arb_barrier_h{int(round(h * 1000)):04d}" for h in barrier_heights()}
    authored |= {"arb_table", "arb_fixture_wall"}

    for name in sorted(AXES):
        s = build_layout(get_axis(name))["session"]
        unknown = [p["asset_name"] for p in s["props"] if p["asset_name"] not in authored]
        check(not unknown, f"{name}: every prop asset is authored"
                           + (f" (unknown: {unknown})" if unknown else ""))
        check(s["scene"]["name"] in authored,
              f"{name}: scene asset '{s['scene']['name']}' is authored")

    topo = build_layout(get_axis("topology"))["session"]
    heights = sorted(p["barrier_h_m"] for p in topo["props"])
    check(heights == sorted(TOPOLOGY_TRAIN_H),
          f"topology lists exactly the trained barrier heights ({heights})")
    check(all(p["swept_param"] == "barrier_h" for p in topo["props"]),
          "each barrier names the parameter that swaps it at runtime")


def test_spawn_areas_inside_workspace() -> None:
    print("\n[training positions are on the table]")
    for name in sorted(AXES):
        s = build_layout(get_axis(name))["session"]
        outside = [
            a["id"] for a in s["object_spawn_areas"]
            if not WORKSPACE.contains(a["pos"][0], a["pos"][1])
        ]
        check(not outside, f"{name}: all training positions inside the workspace"
                           + (f" (outside: {outside})" if outside else ""))
        check(all(abs(a["pos"][2] - TABLE_TOP_Z) < 1e-9 for a in s["object_spawn_areas"]),
              f"{name}: positions sit on the table surface")
        check(len(s["object_spawn_areas"]) > 0, f"{name}: has at least one training position")

    pos = build_layout(get_axis("position"))["session"]["object_spawn_areas"]
    check(len(pos) == 9, f"position lists its 9 training lattice cells (got {len(pos)})")
    fact = build_layout(get_axis("factorial"))["session"]["object_spawn_areas"]
    check(len(fact) == 4, f"factorial lists its 4 positions (got {len(fact)})")


def test_no_enumerated_sweep() -> None:
    """A layout must not enumerate swept positions — that duplicates the axis definition."""
    print("\n[layouts describe the scene, not the sweep]")
    for name in sorted(AXES):
        ax = get_axis(name)
        n_grid = len(list(ax.grid()))
        n_areas = len(build_layout(ax)["session"]["object_spawn_areas"])
        n_train = sum(1 for p in ax.grid() if ax.in_train(p))
        check(n_areas <= max(n_train, 1),
              f"{name}: {n_areas} areas listed, not {n_grid} grid points "
              f"(train positions: {n_train})")


def test_camera_symmetric() -> None:
    """v1's symmetric-kinematics argument did not cover the camera. This one does."""
    print("\n[camera is left-right symmetric]")
    for name in sorted(AXES):
        cams = build_layout(get_axis(name))["session"]["custom_cams"]
        head = [c for c in cams if c["name"] == "cam_head"]
        check(len(head) == 1, f"{name}: exactly one head camera")
        cam = head[0]
        check(abs(cam["pos"][1]) < 1e-9,
              f"{name}: head camera sits on y=0 (y={cam['pos'][1]})")
        check(abs(cam["look_at"][1]) < 1e-9,
              f"{name}: head camera looks at y=0 (y={cam['look_at'][1]})")
        # A wrist camera rides the hand, so it has no fixed world pose and cannot break
        # symmetry -- but it must declare what it is attached to.
        for w in [c for c in cams if c["name"] == "cam_wrist"]:
            check(w.get("parent_body") == "panda_hand",
                  f"{name}: wrist camera is attached to panda_hand")
            check("look_at" not in w,
                  f"{name}: wrist camera has no fixed look_at (it follows the hand)")
    check(abs(WORKSPACE.y_min + WORKSPACE.y_max) < 1e-9,
          "workspace is centred on y=0, so the whole scene mirrors cleanly")


def test_camera_streams() -> None:
    print("\n[camera streams]")
    from arbiter.suites.layout_gen import WRIST_CAM_AXES
    for name in sorted(AXES):
        cams = build_layout(get_axis(name))["session"]["custom_cams"]
        names = [c["name"] for c in cams]
        want_wrist = name in WRIST_CAM_AXES
        check(("cam_wrist" in names) == want_wrist,
              f"{name}: wrist camera {'present' if want_wrist else 'absent'} ({names})")
        check(all(c.get("width") and c.get("height") for c in cams),
              f"{name}: every camera declares a resolution")
        check(all(c.get("focal_mm") for c in cams),
              f"{name}: every camera declares a focal length")


def test_reproducible() -> None:
    """A generated file must be byte-identical across runs or it stops being reviewable."""
    print("\n[reproducible output]")
    for name in sorted(AXES):
        a = json.dumps(build_layout(get_axis(name)), indent=2, sort_keys=True)
        b = json.dumps(build_layout(get_axis(name)), indent=2, sort_keys=True)
        check(a == b, f"{name}: two builds are byte-identical")
    d = build_layout(get_axis("position"))
    check(d["saved_at"] == 0.0,
          "no wall-clock stamp, so regeneration is not a spurious diff")


def test_order_axis_extras() -> None:
    print("\n[ORDER carries its fixed two-object setup]")
    s = build_layout(get_axis("order"))["session"]
    check("order" in s, "order layout has an 'order' block")
    o = s["order"]
    # Two SLOTS, always. The slot geometry is fixed; which pair occupies it is a condition
    # parameter, and slot order must never track the sequence or "move the left one first"
    # would solve the axis without reading the instruction.
    n = len(o["objects_in_slot_order"])
    check(len(o["object_pos"]) == n and len(o["target_pos"]) == n,
          f"{n} start positions, {n} targets")
    check(len(o["objects_in_slot_order"]) == 3, "three objects in fixed slot order")
    # Every object must appear in every POSITION of some trained sequence. That is the
    # recoverability guarantee for this axis, and what kills any fixed positional habit:
    # without it the policy can score 100% on train with "always move the left one first".
    seqs = [tuple(int(c) for c in t) for t in o["train_sequences"]]
    for pos in range(3):
        seen = sorted({sq[pos] for sq in seqs})
        check(seen == [0, 1, 2],
              f"every object appears at sequence position {pos} in training (got {seen})")
    need = set(o["objects_in_slot_order"])
    have = {e["asset_name"] for e in s["object_pool"]}
    check(need <= have, f"object pool covers every pair's cubes (missing {sorted(need - have)})")
    inside = all(WORKSPACE.contains(x, y) for x, y in o["object_pos"] + o["target_pos"])
    check(inside, "ORDER object and target positions are on the table")
    check(o["object_pos"] != o["target_pos"], "objects must actually be moved somewhere")


def test_axis_metadata_matches() -> None:
    print("\n[axis metadata is copied, not restated]")
    for name in sorted(AXES):
        ax = get_axis(name)
        meta = build_layout(ax)["session"]["axis"]
        check(meta["name"] == ax.name and meta["kind"] == ax.kind
              and meta["unit"] == ax.unit and meta["sweep_param"] == ax.sweep_param
              and meta["mechanism_metric"] == ax.mechanism_metric,
              f"{name}: layout axis metadata matches the axis object")


def test_parking_slots_are_out_of_frame() -> None:
    """A parked object must be behind the head camera, not merely off the table.

    ORDER spawns every cube any pair uses so one launch can realise all of them, and parks the
    two a condition does not place. The old slots sat off the table but inside the camera's
    view, so every green/yellow episode showed the red and blue cubes on the floor beside the
    table -- objects no condition describes, in the observation the policy trains on.

    Asserted against the real camera pose rather than a magic number, because "off the table"
    and "out of frame" are different predicates and only the second one matters.
    """
    from arbiter.suites.spec import parking_slot

    cam = tuple(float(v) for v in HEAD_CAM["pos"])
    tgt = tuple(float(v) for v in HEAD_CAM["look_at"])
    d = math.dist(cam, tgt)
    fwd = tuple((t - c) / d for t, c in zip(tgt, cam))

    for i in range(12):
        p = parking_slot(i)
        v = tuple(a - b for a, b in zip(p, cam))
        dot = sum(a * b for a, b in zip(v, fwd))
        check(dot < -0.25,
              f"parking slot {i} at {tuple(round(c,2) for c in p)} is behind the head camera "
              f"(dot={dot:+.3f}, must be < -0.25)")


def test_topology_object_is_never_occluded() -> None:
    """The object must be visible over the barrier at EVERY swept height.

    TOPOLOGY puts the object on the far side and looks across the barrier at it, so a tall
    barrier can hide the thing the policy has to find. At the old start position the object's
    centre passed 3.4 mm below the sight line at h=0.13 -- and 0.12/0.13 are trained heights, so
    the unobservable view sat in the training set. A barrier the policy cannot see past is not a
    harder condition, it is an unobservable one, and it measures perception rather than routing.

    Asserted against the REAL camera pose from HEAD_CAM rather than a remembered number, so
    moving the camera or the endpoints cannot silently reintroduce it.
    """
    from arbiter.suites.spec import (
        barrier_heights,
        topology_endpoints,
        topology_max_visible_start_x,
    )

    cam = tuple(float(v) for v in HEAD_CAM["pos"])
    (start_x, _), _ = topology_endpoints()
    cube_half = 0.025
    margin = 0.02

    for h in barrier_heights():
        x_max = topology_max_visible_start_x(
            h, cam_x=cam[0], cam_z=cam[2], barrier_x=WORKSPACE.centre[0],
            table_top_z=TABLE_TOP_Z, object_half_h=cube_half, margin=margin,
        )
        check(start_x <= x_max,
              f"h={h:.2f}: object centre visible over the barrier "
              f"(start x {start_x:.3f} <= {x_max:.3f}, {margin * 100:.0f} cm margin)")

    # The barrier must still lie strictly between the two endpoints, or the axis is not about
    # routing past anything.
    (sx, _), (gx, _) = topology_endpoints()
    check(sx < WORKSPACE.centre[0] < gx,
          f"barrier at x={WORKSPACE.centre[0]} lies between start {sx} and goal {gx}")
    check(WORKSPACE.x_min <= sx and gx <= WORKSPACE.x_max,
          f"both endpoints inside the workspace ({sx}, {gx})")


def main() -> int:
    test_structure()
    test_v1_conventions_preserved()
    test_props_reference_real_assets()
    test_spawn_areas_inside_workspace()
    test_no_enumerated_sweep()
    test_camera_symmetric()
    test_camera_streams()
    test_reproducible()
    test_order_axis_extras()
    test_parking_slots_are_out_of_frame()
    test_topology_object_is_never_occluded()
    test_axis_metadata_matches()

    print()
    if FAILURES:
        print(f"[FAIL] {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("[PASS] all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
