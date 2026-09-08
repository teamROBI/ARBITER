#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a scene layout per axis, from the axis definition.

**The model changed from v1, not just the values.** v1's layout enumerated a fixed list of 16
discrete ``object_spawn_areas``, and a condition *was* one of those ids. With a parameter
vector that model breaks down: POSITION alone has 667 grid points, and enumerating them as
named areas would be a 667-entry file whose contents duplicate the axis definition — a second
place for the geometry to live, and therefore a second place for it to drift.

So a v2 layout describes only what is genuinely static: the table, the robot, the cameras, the
object pool, and any props the axis needs. **Spawn positions come from the condition at
runtime**, computed from its parameter vector. The layout carries the workspace bounds so the
runtime can assert a position is inside them, but it does not list positions.

``object_spawn_areas`` is still emitted, holding the axis's *training* positions only. Two
reasons: existing runtime code reads that key, and the training positions are a genuinely
small, genuinely fixed set worth having on disk for inspection. Test positions are not listed
— they are swept.

Two v1 conventions are deliberately preserved (CLAUDE.md records both as intentional):

- ``session.scene.path`` stays ``""``. That forces
  ``runtime_layout.apply_portal_layout_override`` to resolve the scene portably from
  ``file_name`` against ``SCENE_RUNTIME_USD_DIR``. It once held a baked absolute path, which
  silently loaded the scene out of another project's tree.
- ``"schema": "acs_portal_layout.v2"`` and the ``$ROVI_PORTAL_LAYOUT_JSON`` env var keep their
  inherited names. Renaming them buys nothing and breaks the runtime.

Usage:

    python arbiter/suites/layout_gen.py --axis topology
    python arbiter/suites/layout_gen.py --all --out arbiter/suites/layouts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arbiter.suites.spec import (  # noqa: E402
    AXES,
    ORDER_OBJECTS,
    ORDER_TRAIN_SEQUENCES,
    object_keys_for,
    ORDER_OBJECT_POS as _ORDER_OBJECT_POS,
    ORDER_TARGET_POS as _ORDER_TARGET_POS,
    TOPOLOGY_TRAIN_H,
    WORKSPACE,
    Axis,
    axis as get_axis,
)

SCHEMA = "acs_portal_layout.v2"

#: Table top surface height in world z. The procedural table slab is 4 cm thick and its
#: sidecar records top_z_offset_m, so the runtime places the slab centre at TABLE_TOP_Z minus
#: that offset rather than recomputing it here.
TABLE_TOP_Z = 0.75

#: Franka base sits at the near edge of the table, facing +x across the workspace. Provisional
#: until the Phase 1 achievability gate confirms every test point is reachable.
ROBOT_POS = [0.0, 0.0, TABLE_TOP_Z]
ROBOT_RPY = [0.0, 0.0, 0.0]

#: Head camera, on the FAR side of the table looking back toward the robot.
#:
#: It was originally placed behind the arm at (-0.25, 0, +0.65) looking forward, which rendered
#: as a full-frame close-up of the robot's shoulder: the arm sits directly between that
#: viewpoint and the workspace, so the policy's only observation showed no table, no object and
#: no barrier. Measured by rendering it -- every programmatic check on the layout passed.
#:
#: The far-side placement is what robosuite/LIBERO call "agentview" and it is the standard for
#: a reason: the workspace is unobstructed and the arm appears behind it rather than in front.
#: Still centred on y=0, so the scene stays left-right symmetric -- v1's asymmetric camera was
#: an uncontrolled confound its symmetric-kinematics argument did not cover.
#: Head camera, matching LIBERO/robosuite's ``agentview`` geometry.
#:
#: Taken from robosuite's ``table_arena.xml``: agentview sits at ``pos 0.5 0 1.35`` with the
#: table top at z=0.8, i.e. **0.55 m above the table, 0.50 m beyond the table centre, looking
#: back toward the robot at 45 degrees below horizontal**, on the arena centreline. Decoding its
#: quaternion (0.653, 0.271, 0.271, 0.653) gives a forward direction of (-0.708, 0, -0.706) --
#: exactly 45 degrees down, yaw 180.
#:
#: Adopted rather than tuned by eye so the observation is comparable to the standard VLA
#: benchmark. Two earlier attempts here were both wrong and both passed every programmatic
#: check: behind the arm (rendered a close-up of the shoulder), then 1.0 m above the table
#: (a near-bird's-eye view where the workspace was a small patch of frame).
#:
#: Still on y=0. DIRECTION sweeps theta both ways and treats +45 and -45 as equidistant, so an
#: asymmetric camera would make two equal-distance conditions unequally visible.
HEAD_CAM_HEIGHT_ABOVE_TABLE = 0.55
HEAD_CAM_BEYOND_WORKSPACE = 0.50

HEAD_CAM = {
    "name": "cam_head",
    "pos": [WORKSPACE.centre[0] + HEAD_CAM_BEYOND_WORKSPACE, 0.0,
            TABLE_TOP_Z + HEAD_CAM_HEIGHT_ABOVE_TABLE],
    "look_at": [WORKSPACE.centre[0], 0.0, TABLE_TOP_Z + 0.05],
}

#: MuJoCo's ``fovy`` is the VERTICAL field of view, so LIBERO's 45 degrees is 45 degrees
#: vertically. Translated to our 672x376 frame that is a 14.15 mm focal length against
#: IsaacLab's 20.955 mm horizontal aperture, giving a 73-degree horizontal FOV.
#:
#: A first attempt used 24 mm, reasoning that LIBERO's square 128x128 makes its horizontal FOV
#: also 45 degrees and that matching *that* preserved framing tightness. Rendered, it was far
#: too tight: the barrier filled the frame and the arm was entirely outside it. LIBERO's
#: agentview does show the arm, so the vertical reading is both the literal specification and
#: the one that reproduces the reference view.
HEAD_CAM_FOCAL_MM = 14.15

#: Wrist camera, LIBERO's ``robot0_eye_in_hand`` equivalent. Attached to ``panda_hand`` so it
#: travels with the gripper, looking along the approach axis toward the fingertips.
#:
#: LIBERO ships this alongside agentview and v1 already *recorded* wrist streams at 240x424 --
#: the supplementary says so -- then never used them. Having it makes the observation space
#: match the reference benchmark rather than being a subset of it, and it gives the policy a
#: close-up during approach, which matters most exactly where the fixed agentview is weakest:
#: an object partly hidden behind a barrier.
#:
#: Optional per axis via ``cameras=``, since a second stream costs render time and dataset
#: size, and only some axes need the close-up.
WRIST_CAM = {
    "name": "cam_wrist",
    "parent_body": "panda_hand",
    # Behind the hand origin along -z (the approach axis), tilted to look at the fingertips.
    "offset_pos": [0.0, 0.0, -0.04],
    "width": 424,
    "height": 240,
    "focal_mm": 12.0,
}

#: Which camera streams an axis records. **Every axis records both**, and that uniformity is a
#: requirement rather than a preference: the benchmark trains one policy on all axes, which is
#: impossible if the observation space differs between them, and per-axis radii are only
#: comparable when measured through the same observation.
#:
#: The wrist camera used to be enabled on {topology, approach} alone -- the axes where the
#: fixed view is most occluded. That produced a merged dataset declaring `cam_wrist` while only
#: 210 of 630 episodes carried it, which a trainer would either reject or, worse, read as
#: garbage for two thirds of the data.
#:
#: Dropping a stream at training time needs no re-collection, so recording both everywhere also
#: makes head-only a free ablation. The reverse is not true.
WRIST_CAM_AXES = set(AXES)

#: Graspable objects. Procedural cubes from create_assets.py; colour is incidental and
#: balanced across splits, exactly as it was in v1.
OBJECT_POOL = ["arb_cube_red", "arb_cube_blue", "arb_cube_green"]

# ORDER's objects come from the spec (axes.ORDER_OBJECTS), imported above. A local list here
# shadowed that import and silently pinned the layout to the old two-cube pair while the spec
# had already moved to three.

#: APPROACH's bar. Long axis 10 cm, past the 8 cm jaw opening, so only the 3.5 cm width is
#: graspable and the bar's yaw dictates the required jaw azimuth.
APPROACH_OBJECTS = ["arb_bar_red", "arb_bar_blue"]
# Re-exported from the spec, which owns all benchmark geometry. Kept as names here because
# four tools already import them from this module.
ORDER_OBJECT_POS = _ORDER_OBJECT_POS
ORDER_TARGET_POS = _ORDER_TARGET_POS


def _pool_entry(asset_name: str) -> dict:
    return {"asset_name": asset_name, "library": "assets", "scale": [1.0, 1.0, 1.0]}


def _training_spawn_areas(ax: Axis) -> list[dict]:
    """The axis's training positions, as named areas. Test positions are swept, not listed."""
    areas: list[dict] = []
    seen: set[tuple[float, float]] = set()
    for params in ax.grid():
        if not ax.in_train(params):
            continue
        x = float(params.get("x", WORKSPACE.centre[0]))
        y = float(params.get("y", WORKSPACE.centre[1]))
        if (x, y) in seen:
            continue
        seen.add((x, y))
        areas.append({
            "id": f"train_area_{len(areas) + 1}",
            "pos": [round(x, 4), round(y, 4), TABLE_TOP_Z],
        })
    return areas


def _props(ax: Axis) -> list[dict]:
    """Static props the axis needs.

    Barriers and fixtures are separate procedural props referenced by name, never baked into
    the scene USD — the asset generator and the scene builder both assume the scene is one
    mesh, and a barrier merged into it could not be parameterized per episode anyway.

    A barrier is listed per *trained* height only. The swept test heights are resolved at
    runtime from the condition's ``barrier_h``, which is why the generator emits an asset for
    every swept height (see ``create_scene.barrier_heights``) but the layout names only
    the training ones.
    """
    if ax.name == "topology":
        cx, cy = WORKSPACE.centre
        return [
            {
                "asset_name": f"arb_barrier_h{int(round(h * 1000)):04d}",
                "library": "props",
                "role": "barrier",
                "barrier_h_m": h,
                "pos": [round(cx, 4), round(cy, 4), TABLE_TOP_Z],
                "swept_param": "barrier_h",
            }
            for h in sorted(TOPOLOGY_TRAIN_H)
        ]
    if ax.name == "approach":
        cx, cy = WORKSPACE.centre
        return [{
            "asset_name": "arb_fixture_wall",
            "library": "props",
            "role": "fixture",
            "pos": [round(cx, 4), round(cy, 4), TABLE_TOP_Z],
            "swept_param": "phi_deg",
        }]
    return []


def _cameras(ax: Axis) -> list[dict]:
    """Camera streams for one axis. Both, for every axis -- see WRIST_CAM_AXES."""
    cams = [dict(HEAD_CAM, width=672, height=376, focal_mm=HEAD_CAM_FOCAL_MM)]
    if ax.name in WRIST_CAM_AXES:
        cams.append(dict(WRIST_CAM))
    return cams


def build_layout(ax: Axis) -> dict:
    """One layout dict for one axis."""
    # APPROACH needs the bar: a cube's square footprint makes every jaw azimuth grasp it
    # equally, which is what made the axis's swept quantity invisible. object_keys_for is the
    # single definition; this pool must agree with it or the scene lacks the prop the
    # collector asks for.
    # Derived from object_keys_for, not from a parallel constant. The comment above has said
    # since it was written that object_keys_for is the single definition, but the code kept its
    # own list -- and that drifted the moment ORDER's pair became a condition parameter:
    # object_keys_for returned four cubes while ORDER_OBJECTS still named two, so the layout
    # would omit the very cubes the collector asks for. Union, so a pool may still carry a
    # spare object the axis does not name (a distractor), but never miss one it does.
    required = object_keys_for(ax.name)
    extra = (ORDER_OBJECTS if ax.name == "order"
             else APPROACH_OBJECTS if ax.name == "approach"
             else OBJECT_POOL)
    pool = list(dict.fromkeys([*required, *extra]))
    session = {
        "camera": {
            "distance_m": 1.8,
            "orbit_center": [0.45, 0.0, TABLE_TOP_Z],
            "pitch_deg": 24.0,
            "preset": "isometric",
            "yaw_deg": -30.0,
        },
        "object_pool": [_pool_entry(a) for a in pool],
        "object_spawn_areas": _training_spawn_areas(ax),
        "props": _props(ax),
        "robot": {
            "name": "franka_panda",
            "pos": ROBOT_POS,
            "rpy": ROBOT_RPY,
            "heading": 0.0,
        },
        "scene": {
            "auto_pos_z": TABLE_TOP_Z,
            "file_name": "arb_table.usd",
            "name": "arb_table",
            # Deliberately empty: forces portable resolution against SCENE_RUNTIME_USD_DIR.
            # Never bake an absolute path here.
            "path": "",
            "pos": [0.0, 0.0, TABLE_TOP_Z],
            "rot": [0.7071, 0.7071, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "custom_cams": _cameras(ax),
        # v2 additions: the runtime computes spawn positions from a condition's parameter
        # vector, so it needs the bounds to validate against and the axis to interpret.
        "workspace": {
            "x_min": WORKSPACE.x_min, "x_max": WORKSPACE.x_max,
            "y_min": WORKSPACE.y_min, "y_max": WORKSPACE.y_max,
            "table_top_z": TABLE_TOP_Z,
        },
        "axis": {
            "name": ax.name,
            "kind": ax.kind,
            "unit": ax.unit,
            "sweep_param": ax.sweep_param,
            "mechanism_metric": ax.mechanism_metric,
        },
    }
    if ax.name == "order":
        session["order"] = {
            # The pair is a CONDITION parameter, not a scene property: one launch realises
            # every pair. Recorded here so a layout still documents the axis's real design --
            # two pairs, one of which is trained in both orders to supply the contrast that
            # makes order-following learnable at all.
            "objects_in_slot_order": list(ORDER_OBJECTS),
            "train_sequences": ["".join(str(i) for i in t) for t in ORDER_TRAIN_SEQUENCES],
            "objects": object_keys_for("order"),
            "object_pos": ORDER_OBJECT_POS,
            "target_pos": ORDER_TARGET_POS,
        }
    return {
        "name": f"arbiter_{ax.name}",
        # No wall-clock stamp: a generated layout must be byte-identical across runs or it
        # shows up as a spurious diff and stops being reviewable.
        "saved_at": 0.0,
        "schema": SCHEMA,
        "generated_by": "arbiter/suites/layout_gen.py",
        "session": session,
    }


def write_layout(ax: Axis, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"arbiter_{ax.name}.json"
    path.write_text(json.dumps(build_layout(ax), indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", choices=sorted(AXES), help="generate one axis")
    ap.add_argument("--all", action="store_true", help="generate every axis")
    ap.add_argument("--out", default=str(Path(__file__).parent / "layouts"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.axis and not args.all:
        ap.error("pass --axis NAME or --all")

    names = sorted(AXES) if args.all else [args.axis]
    out = Path(args.out)
    for name in names:
        ax = get_axis(name)
        layout = build_layout(ax)
        s = layout["session"]
        tag = "[DRY]" if args.dry_run else "[OK] "
        if not args.dry_run:
            write_layout(ax, out)
        print(f"{tag} {name:10s} areas={len(s['object_spawn_areas']):3d} "
              f"props={len(s['props']):2d} pool={len(s['object_pool'])} "
              f"-> arbiter_{name}.json")
    if not args.dry_run:
        print(f"\n[OK]  wrote {len(names)} layout(s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
