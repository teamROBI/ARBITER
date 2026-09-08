#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the generated scene props hold the properties the benchmark depends on.

Generates into a temp dir and inspects the USD, so it never touches the data store. Run via
the USD-only launcher (no Kit boot):

    arbiter/sim/tools/usd_python.sh arbiter/sim/tools/tests/test_scene_props.py

The load-bearing check is **static, not rigid**. A barrier authored with RigidBodyAPI would be
knocked over by the arm, and TOPOLOGY — whose entire measurement is whether the policy routes
over or around a barrier of height h — would quietly become a different task. That failure
would not raise anything; it would just produce wrong numbers. Hence a test.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

try:
    from pxr import Usd, UsdGeom, UsdPhysics
except ImportError:
    print("[ERROR] pxr not found. Run via arbiter/sim/tools/usd_python.sh", flush=True)
    sys.exit(1)

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arbiter.sim.tools.create_scene import (  # noqa: E402
    BARRIER_DEPTH_M,
    BARRIER_WIDTH_M,
    TABLE_MARGIN_M,
    TABLE_THICKNESS_M,
    barrier_heights,
    generate,
    table_centre_xy,
    table_extents,
    table_scale,
)
from arbiter.collect.constants import ROUTE_AROUND_MARGIN_M  # noqa: E402
from arbiter.suites.spec import (  # noqa: E402
    BLOCKER_HEIGHT_M,
    BLOCKER_LEGACY_HEIGHT_M,
    DETOUR_OFFSETS,
    LANE_AXES,
    TOPOLOGY,
    TOPOLOGY_H_MAX,
    TOPOLOGY_TRAIN_H,
    WORKSPACE,
    barrier_half_width,
    blocker_span,
    bypass_y,
    expected_homotopy,
    forced_bypass_side,
)

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  {'ok   ' if cond else 'FAIL '} {label}")
    if not cond:
        FAILURES.append(label)


def _api_counts(usd_path: Path) -> tuple[int, int, int]:
    """(rigid, mass, collider) prim counts.

    The stage must stay referenced while the prims are touched — a prim handle whose stage has
    been collected raises "Accessed invalid expired prim", so the traversal happens here
    rather than returning prims to the caller.
    """
    stage = Usd.Stage.Open(str(usd_path))
    prims = list(stage.Traverse())
    rigid = sum(1 for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI))
    mass = sum(1 for p in prims if p.HasAPI(UsdPhysics.MassAPI))
    collide = sum(1 for p in prims if p.HasAPI(UsdPhysics.CollisionAPI))
    return rigid, mass, collide


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        generate(out)
        usds = sorted(out.glob("*.usd"))

        print("\n[inventory]")
        # barriers + table + fixture + the DETOUR blocker, then the floor MARKINGS: one start
        # zone, plus one lane per axis that declares a painted corridor.
        n_markings = 1 + len(LANE_AXES)
        # +5: table, fixture, the 0.40 m blocker, its visual-only ghost twin, and the legacy
        # 0.16 m variant kept only to reproduce the grids measured against it.
        expected = len(barrier_heights()) + 5 + n_markings
        check(len(usds) == expected,
              f"{expected} props authored: one per swept barrier height, plus table, fixture, "
              f"blocker, ghost blocker, legacy blocker "
              f"and {n_markings} floor marking(s) (got {len(usds)})")
        check(all(u.with_suffix(".meta.json").exists() for u in usds),
              "every USD has a .meta.json sidecar")
        names = {u.stem for u in usds}
        check("arb_table" in names, "table present")
        check("arb_fixture_wall" in names, "fixture wall present")
        check(sum(1 for n in names if n.startswith("arb_barrier_h")) == len(barrier_heights()),
              f"{len(barrier_heights())} barrier heights present, matching the axis sweep")

        print("\n[static, not rigid — the load-bearing check]")
        # Two classes, and the distinction is load-bearing in both directions.
        #
        # Obstacles (table, barrier, fixture) need exactly ONE collider and no rigid body: a
        # RigidBodyAPI here would let the arm knock the barrier over and turn TOPOLOGY into a
        # different task.
        #
        # Floor markings (start zone, lane) must have NO collider at all. They sit exactly where
        # objects spawn and where the gripper descends to grasp, so a 3 mm marking with a
        # collider becomes a ledge the cube rests on and the fingers catch. That was a real bug:
        # the marking was first authored through the obstacle path and inherited its collider.
        MARKINGS = {"arb_zone_start"} | {f"arb_lane_{a}" for a in LANE_AXES}
        # The ghost blocker belongs with the markings, not the obstacles: it is a visual-only
        # twin and a CollisionAPI on it would silently turn the perception control into a second
        # copy of the solid condition -- i.e. the control would agree with the thing it is
        # supposed to be distinguished from, and agreement would look like a result.
        NO_COLLIDE = MARKINGS | {"arb_blocker_ghost"}
        bad_obstacle, bad_marking = [], []
        for u in usds:
            rigid, mass, collide = _api_counts(u)
            tag = f"{u.stem}(rigid={rigid},mass={mass},collide={collide})"
            if u.stem in NO_COLLIDE:
                if rigid or mass or collide:
                    bad_marking.append(tag)
            elif rigid or mass or collide != 1:
                bad_obstacle.append(tag)
        n_obs = len(usds) - len(NO_COLLIDE)
        check(not bad_obstacle,
              f"all {n_obs} obstacles: no RigidBodyAPI, no MassAPI, exactly one collider"
              + (f" — offenders: {bad_obstacle[:5]}" if bad_obstacle else ""))
        check(not bad_marking,
              f"all {len(NO_COLLIDE)} collisionless props (floor markings + ghost blocker): no collider, no rigid body"
              + (f" — offenders: {bad_marking[:5]}" if bad_marking else ""))

        print("\n[DETOUR blocker — the seal geometry]")
        # The blocker exists to make ONE bypass lane impassable so the complement is forced by
        # physics rather than by the `h*` convention. Two ways that fails silently, both of which
        # the first implementation had:
        #
        #  - A GAP between barrier and blocker. Centred on the detour lane, a 0.14 m blocker at
        #    y=-0.22 spans [-0.29,-0.15] while the barrier at offset 0 ends at -0.10, leaving 5 cm
        #    of open table between the two props. The arm can aim at that, take neither detour,
        #    and score a success while the axis measures nothing.
        #  - An OVERLAP with the forced lane, which blocks both sides and makes the condition
        #    unachievable -- 180 episodes of a struggling arm is what the gate exists to prevent.
        check("arb_blocker" in names, "blocker present")
        # The criterion is the ARM's carry height, not the barrier's. 0.16 was picked as
        # "taller than any swept barrier" and measured 36/36 episodes flying straight over it --
        # only the cube, carried ~4 cm below the gripper, grazed the top, and two cleared it
        # outright and scored a lift-over as a success. Guard the real requirement.
        check(BLOCKER_HEIGHT_M > 0.35,
              f"blocker ({BLOCKER_HEIGHT_M} m) sits above the measured end-effector apex range "
              f"0.222-0.349 m, so the lane is sealed rather than grazed")
        check(BLOCKER_HEIGHT_M > TOPOLOGY_H_MAX,
              f"blocker also clears every swept barrier ({TOPOLOGY_H_MAX} m)")
        hw = barrier_half_width()
        gaps, overlaps, unreachable = [], [], []
        for off in DETOUR_OFFSETS:
            b_lo, b_hi = off - hw, off + hw
            lo, hi = blocker_span(off, around_margin=ROUTE_AROUND_MARGIN_M)
            fs = forced_bypass_side(off)
            fy = bypass_y(off, fs, around_margin=ROUTE_AROUND_MARGIN_M)
            if min(abs(lo - b_hi), abs(b_lo - hi)) > 1e-9:
                gaps.append(f"offset {off:+.2f}: barrier ends {b_hi:+.3f}/{b_lo:+.3f}, "
                            f"blocker spans [{lo:+.3f},{hi:+.3f}]")
            if lo <= fy <= hi:
                overlaps.append(f"offset {off:+.2f}: forced lane {fy:+.3f} inside blocker")
            if not (WORKSPACE.y_min <= fy <= WORKSPACE.y_max):
                unreachable.append(f"offset {off:+.2f}: forced lane {fy:+.3f} outside workspace")
        check(not gaps, f"barrier and blocker are flush at every DETOUR offset"
                        + (f" — gaps: {gaps}" if gaps else ""))
        check(not overlaps, "the forced detour lane is clear of the blocker"
                            + (f" — {overlaps}" if overlaps else ""))
        check(not unreachable, "every forced detour lane is inside the workspace"
                               + (f" — {unreachable}" if unreachable else ""))
        # +/-0.06 is excluded from DETOUR_OFFSETS deliberately: its forced lane lands at
        # y=+/-0.280, exactly ON the workspace boundary, so the arm would be collecting 180
        # episodes at its lateral reach limit while carrying a cube.
        check(all(abs(o) <= 0.03 + 1e-9 for o in DETOUR_OFFSETS),
              "DETOUR offsets stay within +/-0.03, where the forced lane keeps >=3 cm of "
              "workspace margin")

        check("arb_blocker_ghost" in names, "ghost blocker present")
        check("arb_blocker_legacy" in names, "legacy blocker present (reproduction only)")
        # Kept only so the v6/v7 DETOUR grids stay reproducible. Asserting it is BELOW the apex
        # range records *why* it is legacy: it cannot seal the lane, which is the whole reason
        # the default moved.
        check(BLOCKER_LEGACY_HEIGHT_M < 0.222,
              f"legacy blocker ({BLOCKER_LEGACY_HEIGHT_M} m) sits BELOW the measured apex range, "
              f"i.e. it does not seal the lane — reproduction only")
        solid = out / "arb_blocker.meta.json"
        ghost = out / "arb_blocker_ghost.meta.json"
        if solid.exists() and ghost.exists():
            a, b = json.loads(solid.read_text()), json.loads(ghost.read_text())
            same = all(abs(float(a[k][i]) - float(b[k][i])) < 1e-9
                       for k in ("size_m",) if k in a and k in b
                       for i in range(len(a[k])))
            check(same, "ghost blocker is geometrically identical to the solid one, so the "
                        "perception control differs in collision ONLY")

        print("\n[stage conventions]")
        for u in usds[:4] + usds[-1:]:
            stage = Usd.Stage.Open(str(u))
            check(UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y,
                  f"{u.stem}: Y-up, matching the rest of the asset library")
            check(UsdGeom.GetStageMetersPerUnit(stage) == 1.0, f"{u.stem}: metres per unit 1.0")
            check(stage.GetDefaultPrim().IsValid(), f"{u.stem}: has a default prim")

        print("\n[meta sidecars]")
        for u in usds:
            meta = json.loads(u.with_suffix(".meta.json").read_text())
            for k in ("bbox_size_m", "world_size_m", "static", "up_axis", "role"):
                if k not in meta:
                    check(False, f"{u.stem}: meta missing '{k}'")
                    break
            else:
                # world_z must equal authored Y: that index is what
                # sim_common.read_spawn_z_offset treats as the height.
                bb, ws = meta["bbox_size_m"], meta["world_size_m"]
                ok = (abs(ws[2] - bb[1]) < 1e-9 and abs(ws[0] - bb[0]) < 1e-9
                      and abs(ws[1] - bb[2]) < 1e-9 and meta["static"] is True)
                check(ok, f"{u.stem}: world_size_m is the Y-up bbox rotated 90 deg about X")

        print("\n[barrier geometry tracks the TOPOLOGY axis]")
        heights = barrier_heights()
        axis_heights = sorted({round(float(p["barrier_h"]), 6) for p in TOPOLOGY.grid()})
        check(set(axis_heights).issubset(set(heights)),
              f"every height the axis sweeps has an asset "
              f"({len(axis_heights)} swept, {len(heights)} authored)")
        check(all(any(abs(h - t) < 1e-6 for h in heights) for t in TOPOLOGY_TRAIN_H),
              "every trained height has an asset")

        for u in usds:
            if not u.stem.startswith("arb_barrier_h"):
                continue
            meta = json.loads(u.with_suffix(".meta.json").read_text())
            h = float(meta["barrier_h_m"])
            ws = meta["world_size_m"]
            # world = (depth_x, lateral_y, height_z). The lateral span MUST be the long axis:
            # a barrier whose 30 cm ran along x was a wall parallel to the approach that the
            # arm could sidestep, and TOPOLOGY would have measured nothing. This assertion
            # previously encoded that exact error, which is why it passed.
            ok = (abs(ws[2] - h) < 1e-9
                  and abs(ws[0] - BARRIER_DEPTH_M) < 1e-9
                  and abs(ws[1] - BARRIER_WIDTH_M) < 1e-9)
            check(ok, f"{u.stem}: world = (depth {BARRIER_DEPTH_M}, lateral "
                      f"{BARRIER_WIDTH_M}, height {h}) -- got {[round(v, 3) for v in ws]}")
            check(ws[1] > ws[0], f"{u.stem}: barrier is wider laterally than deep, i.e. it "
                                 f"stands ACROSS the approach")

        trained_flags = {}
        for u in usds:
            if u.stem.startswith("arb_barrier_h"):
                m = json.loads(u.with_suffix(".meta.json").read_text())
                trained_flags[round(float(m["barrier_h_m"]), 6)] = bool(m["trained"])
        check(all(trained_flags.get(round(t, 6)) for t in TOPOLOGY_TRAIN_H),
              "trained heights are flagged trained=true in their sidecar")
        check(sum(trained_flags.values()) == len(TOPOLOGY_TRAIN_H),
              f"exactly {len(TOPOLOGY_TRAIN_H)} heights flagged trained "
              f"(got {sum(trained_flags.values())})")

        print("\n[homotopy flip is spanned by real assets]")
        over = [h for h in heights if expected_homotopy(h) == "over"]
        around = [h for h in heights if expected_homotopy(h) == "around"]
        check(len(over) > 0 and len(around) > 0,
              f"assets exist on both sides of h* ({len(over)} over / {len(around)} around)")
        # A barrier asset must EXIST for trained heights on both sides of h*, since both
        # classes are now demonstrated. Previously every trained height was OVER, so the
        # around-class assets were only ever built for test conditions -- and a missing asset
        # for a trained height would surface as a collection crash, not a design error.
        trained_classes = {expected_homotopy(t) for t in TOPOLOGY_TRAIN_H}
        check(trained_classes == {"over", "around"},
              f"trained heights span both classes (got {sorted(trained_classes)})")
        missing = [t for t in TOPOLOGY_TRAIN_H
                   if round(t, 6) not in {round(h, 6) for h in heights}]
        check(not missing, f"every trained height has a barrier asset (missing {missing})")

        print("\n[table spans the workspace]")
        ts = table_scale()
        span_x = WORKSPACE.x_max - WORKSPACE.x_min
        span_y = WORKSPACE.y_max - WORKSPACE.y_min
        check(ts[0] > span_x and ts[2] > span_y,
              "tabletop is larger than the workspace it must cover")
        check(abs(ts[1] - TABLE_THICKNESS_M) < 1e-9, "table thickness is the authored Y extent")
        tmeta = json.loads((out / "arb_table.meta.json").read_text())
        check(abs(tmeta["top_z_offset_m"] - TABLE_THICKNESS_M * 0.5) < 1e-9,
              "table records its top-surface offset for placement")

        print("\n[left-right symmetry]")
        # The v1 confound: symmetric kinematics is not enough if the scene itself is not
        # symmetric. Primitives centred on y=0 are, by construction.
        check(abs(WORKSPACE.y_min + WORKSPACE.y_max) < 1e-9,
              "workspace is centred on y=0, so a mirrored scene is identical")
        check(abs(ts[2] - (span_y + 2 * TABLE_MARGIN_M)) < 1e-9,
              f"table y-span is workspace + 2x{TABLE_MARGIN_M} m margin (symmetric in y)")
        # x is deliberately NOT symmetric: the near edge sits behind the robot base at x=0 so
        # the arm has something to stand on. A symmetric margin left it floating 18 cm short.
        x0, x1, y0, y1 = table_extents()
        check(x0 <= 0.0, f"table near edge is behind the robot base (x0={x0:.3f} <= 0)")
        check(abs(x1 - (WORKSPACE.x_max + TABLE_MARGIN_M)) < 1e-9,
              f"table far edge is workspace + {TABLE_MARGIN_M} m margin")
        check(abs(ts[0] - (x1 - x0)) < 1e-9, "table x-span matches its declared extents")
        check(abs(y0 + y1) < 1e-9, f"table y extents stay symmetric ({y0:.3f}, {y1:.3f})")
        cx, cy = table_centre_xy()
        check(abs(cy) < 1e-9, "table centre is on y=0, so the scene mirrors cleanly")
        check(abs(cx - 0.5 * (x0 + x1)) < 1e-9, "recorded centre matches the extents")
        check(tmeta["world_centre_xy"] == [cx, cy],
              "sidecar records the centre, so layout and renderer place it identically")
        check(tmeta["world_extents_xy"] == [x0, x1, y0, y1], "sidecar records the extents")

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
