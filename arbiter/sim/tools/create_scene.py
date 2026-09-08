#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate ARBITER's procedural scene props: table, barriers, fixture walls.

v1's tabletop was a Polycam scan reconstructed by the R2-M2 project — a single mesh with one
baseColor texture, not rebuildable here. v2 builds the scene from primitives instead, which
buys three things the measurement actually needs:

1. **Exactly known workspace geometry.** The support-distance metric compares expert paths
   against a known table plane and known barrier extents. A scanned mesh gives neither.
2. **Guaranteed left-right symmetry.** v1 claimed symmetric kinematics ruled out hardware
   asymmetry, but the scanned table, its texture, and the camera placement were all
   potentially asymmetric. Primitives are symmetric by construction.
3. **Reproducible from code**, with no dependency on another project.

**Static, not rigid.** Every prop here gets ``CollisionAPI`` and a physics material but
deliberately *no* ``RigidBodyAPI`` and no ``MassAPI``. That is the one thing not to get wrong:
a barrier authored as a rigid body would be knocked over by the arm, and TOPOLOGY — where the
whole measurement is whether the policy routes over or around a barrier of height h — would
silently degrade into a different task. Rigid-body props also need CCD; static ones do not.

**Y-up authoring, matching the rest of the asset library.** ``create_assets.py`` authors
Y-up and ``sim_common.place_object_at_area`` applies a 90-degree X rotation at spawn, so
authored Y becomes world Z. Props follow the same convention so one rule covers the whole
library, and ``read_spawn_z_offset`` (which reads ``bbox_size_m[1]`` as the Z half-height)
keeps working on them. The sidecar also records ``world_size_m`` — extents *after* that
rotation — because the route primitive needs a barrier's world height to choose its waypoint,
and deriving it at every call site is how conventions drift.

Barrier heights are generated across the TOPOLOGY sweep declared in ``arbiter/suites/spec.py``,
so the assets and the axis definition cannot disagree about which heights exist.

No Isaac Sim bootstrap: this authors USD only. Run it through the USD-only launcher, which
resolves pxr out of Isaac's bundled libs without booting Kit:

    arbiter/sim/tools/usd_python.sh arbiter/sim/tools/create_scene.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    print(
        "[ERROR] pxr not found. Run via: arbiter/sim/tools/usd_python.sh "
        "arbiter/sim/tools/create_scene.py",
        flush=True,
    )
    sys.exit(1)

try:
    from pxr import PhysxSchema
    _HAS_PHYSX = True
except ImportError:
    _HAS_PHYSX = False

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arbiter.suites.spec import (  # noqa: E402
    BARRIER_DEPTH_M,
    BARRIER_WIDTH_M,
    BLOCKER_DEPTH_M,
    BLOCKER_HEIGHT_M,
    BLOCKER_LEGACY_HEIGHT_M,
    BLOCKER_WIDTH_M,
    TOPOLOGY_TRAIN_H,
    WORKSPACE,
    LANE_AXES,
    barrier_heights,
    transport_lane,
)

OUTPUT_DIR = _REPO_ROOT / "data" / "usd" / "props"

# ── physics ──────────────────────────────────────────────────────────────────
# Higher friction than the graspable objects: the table must not let a placed object creep,
# and the barrier must not act as a slide. Restitution 0 so nothing bounces off a wall.
STATIC_FRICTION = 1.0
DYNAMIC_FRICTION = 1.0
RESTITUTION = 0.0

# ── geometry ─────────────────────────────────────────────────────────────────
# Table spans the workspace plus a margin so the arm never reaches past the edge. Provisional,
# like the WORKSPACE extents themselves — the Phase 1 achievability gate settles both.
TABLE_MARGIN_M = 0.10
TABLE_THICKNESS_M = 0.04
# Mid-tone, low-saturation. Deliberately darker than the white Franka so the arm reads against
# it -- at the old exposure the table blew out to near-white and the robot vanished into it.
TABLE_COLOR = (0.42, 0.43, 0.46)

# Barrier plan geometry lives in axes.py (the spec); only its colour is a rendering choice.
# Authoring order is Y-up (x, y_up, z) and _world_size maps that to (x, z, y_up), so the
# lateral span goes in the authored Z slot. Authoring it in X made the wall run PARALLEL to
# the approach instead of across it, and TOPOLOGY would have measured nothing.
#: Start-zone marker: where the object always spawns, on the axes that keep it fixed. Neutral
#: and desaturated ON PURPOSE -- the coloured target pads say "put it here", so the start mark
#: must not be mistaken for one. Slightly larger than the 5 cm cube so it frames the object
#: rather than hiding under it.
#: Transport corridor. Neutral so it is never mistaken for a target -- the coloured pads own
#: that meaning -- but MUCH darker than the table, not subtly darker.
#:
#: Measured: the table's 0.42 albedo renders at ~184/255, an effective gain of ~1.7, so the
#: first attempt at (0.31, 0.33, 0.37) came out within 2 grey levels of the table and was
#: invisible. This is the third time a marking in this scene has had to be darkened after a
#: render showed it wasn't there -- the pads, the table itself, now the lane. Pick contrast
#: against the RENDERED value, not the albedo.
LANE_COLOR = (0.20, 0.22, 0.26)
#: 3 mm, not 2: the lane sits 0.5*thickness above the table, and 1 mm invites z-fighting.
LANE_THICKNESS_M = 0.003

START_ZONE_COLOR = (0.22, 0.25, 0.30)
START_ZONE_SIZE_M = 0.08
START_ZONE_THICKNESS_M = 0.003

BARRIER_COLOR = (0.75, 0.28, 0.20)         # visually distinct from table and from every object

FIXTURE_HEIGHT_M = 0.10
FIXTURE_WIDTH_M = 0.12
FIXTURE_DEPTH_M = 0.02
FIXTURE_COLOR = (0.30, 0.34, 0.42)
# Distinct from BARRIER_COLOR on purpose. The barrier is the obstacle whose HEIGHT is swept and
# which may be crossed; the blocker is impassable and marks the lane that is closed. If they
# looked alike the scene would not show which of the two lanes is open, and the axis asks the
# policy to read exactly that.
BLOCKER_COLOR = (0.16, 0.20, 0.55)         # deep blue; saturated, per the pad lesson


# ── helpers ──────────────────────────────────────────────────────────────────

def _add_static_physics_material(stage, root_path, collision_prim) -> None:
    """UsdPhysics material bound to a *static* collider (no rigid body involved)."""
    mat_scope_path = root_path.AppendChild("PhysicsMaterials")
    mat_path = mat_scope_path.AppendChild("PMat_0")

    UsdGeom.Scope.Define(stage, str(mat_scope_path))
    mat = UsdShade.Material.Define(stage, str(mat_path))
    mat_prim = mat.GetPrim()

    pmat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    pmat.CreateStaticFrictionAttr().Set(STATIC_FRICTION)
    pmat.CreateDynamicFrictionAttr().Set(DYNAMIC_FRICTION)
    pmat.CreateRestitutionAttr().Set(RESTITUTION)

    if _HAS_PHYSX:
        try:
            pxm = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
            pxm.CreateFrictionCombineModeAttr().Set("max")
            pxm.CreateRestitutionCombineModeAttr().Set("min")
        except Exception:
            mat_prim.CreateAttribute(
                "physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("max")
            mat_prim.CreateAttribute(
                "physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token).Set("min")

    try:
        UsdShade.MaterialBindingAPI.Apply(collision_prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    except Exception:
        rel = collision_prim.GetRelationship("physics:material:binding")
        if not rel:
            rel = collision_prim.CreateRelationship("physics:material:binding", custom=False)
        rel.SetTargets([mat_prim.GetPath()])


def _bind_visual(stage, prim, name: str, color: tuple, roughness: float = 0.6) -> None:
    vis_mat = UsdShade.Material.Define(stage, f"/{name}/VisMaterial")
    shader = UsdShade.Shader.Define(stage, f"/{name}/VisMaterial/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    vis_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(prim).Bind(vis_mat)


def _world_size(scale_yup: tuple[float, float, float]) -> list[float]:
    """Y-up authored extents -> world extents after the 90-degree X spawn rotation.

    +90 about X maps (x, y, z) -> (x, -z, y), so world_z is the authored Y. That is the same
    index ``sim_common.read_spawn_z_offset`` treats as the height, which is what keeps the two
    consistent.
    """
    sx, sy, sz = scale_yup
    return [sx, sz, sy]


def make_static_box_usd(
    output_dir: Path,
    name: str,
    color: tuple,
    scale_yup: tuple[float, float, float],
    *,
    collider: bool = True,
    extra_meta: dict | None = None,
    roughness: float = 0.6,
) -> Path:
    """Author one static collider box. No RigidBodyAPI — see the module docstring."""
    output_dir.mkdir(parents=True, exist_ok=True)
    usd_path = output_dir / f"{name}.usd"
    if usd_path.exists():
        usd_path.unlink()

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())

    cube = UsdGeom.Cube.Define(stage, f"/{name}/Mesh")
    cube.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3f(*scale_yup))

    _bind_visual(stage, cube.GetPrim(), name, color, roughness)

    # Static: collider only. Adding RigidBodyAPI here is the mistake that would quietly
    # turn TOPOLOGY into a knock-the-barrier-over task.
    # collider=False makes a pure visual: no collision, no rigid body. Required for a floor
    # marking, which sits exactly where objects spawn and where the gripper descends to grasp.
    # With a collider a 3 mm marking becomes a ledge the cube rests on and the fingers catch.
    if collider:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _add_static_physics_material(stage, root.GetPrim().GetPath(), cube.GetPrim())

    stage.GetRootLayer().Save()

    dims = list(scale_yup)
    meta = {
        "bbox_size_m": dims,
        "bbox_longest_edge_m": max(dims),
        "world_size_m": _world_size(scale_yup),
        "meters_per_unit": 1.0,
        "static": True,
        "up_axis": "Y",
    }
    if extra_meta:
        meta.update(extra_meta)
    usd_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return usd_path


# ── prop definitions ─────────────────────────────────────────────────────────

#: How far the tabletop extends *behind* the robot base, which sits at x=0.
TABLE_BACK_MARGIN_M = 0.06


def table_extents() -> tuple[float, float, float, float]:
    """(x_near, x_far, y_near, y_far) of the tabletop in world coordinates.

    The near edge sits behind the robot base, not in front of the workspace. Sizing the slab as
    "workspace plus a margin" centred on the workspace put its near edge at x=0.18 while the
    Franka base is at x=0 -- an 18 cm gap, so the robot floated in mid-air with no mount.
    Physics did not care (the base is fixed) but the rendered scene *is* the policy's
    observation, and a robot standing on nothing is an implausible image to hand a VLA.
    """
    return (
        -TABLE_BACK_MARGIN_M,
        WORKSPACE.x_max + TABLE_MARGIN_M,
        WORKSPACE.y_min - TABLE_MARGIN_M,
        WORKSPACE.y_max + TABLE_MARGIN_M,
    )


def table_centre_xy() -> tuple[float, float]:
    """World XY centre of the slab. Consumers read this rather than recomputing it."""
    x0, x1, y0, y1 = table_extents()
    return (round(0.5 * (x0 + x1), 6), round(0.5 * (y0 + y1), 6))


def table_scale() -> tuple[float, float, float]:
    """Y-up scale for the tabletop slab."""
    x0, x1, y0, y1 = table_extents()
    return (x1 - x0, TABLE_THICKNESS_M, y1 - y0)


def _fmt_mm(h: float) -> str:
    return f"{int(round(h * 1000)):04d}"


def generate(output_dir: Path, *, dry_run: bool = False) -> dict[str, list[str]]:
    """Write the table, one barrier per swept height, and the fixture wall."""
    written: dict[str, list[str]] = {"table": [], "barrier": [], "fixture": []}

    tscale = table_scale()
    if not dry_run:
        make_static_box_usd(
            output_dir, "arb_table", TABLE_COLOR, tscale,
            extra_meta={
                "role": "table",
                "top_z_offset_m": TABLE_THICKNESS_M * 0.5,
                # Recorded so the layout and the renderer place the slab identically without
                # either re-deriving the extents.
                "world_centre_xy": list(table_centre_xy()),
                "world_extents_xy": list(table_extents()),
            },
        )
    written["table"].append(
        f"arb_table  {[round(v, 3) for v in _world_size(tscale)]} m (world)")

    for h in barrier_heights():
        name = f"arb_barrier_h{_fmt_mm(h)}"
        # Authored (x=depth, y_up=height, z=lateral) -> world (depth, lateral, height).
        scale = (BARRIER_DEPTH_M, h, BARRIER_WIDTH_M)
        if not dry_run:
            make_static_box_usd(
                output_dir, name, BARRIER_COLOR, scale,
                extra_meta={
                    "role": "barrier",
                    "barrier_h_m": h,
                    "trained": any(abs(h - t) < 1e-6 for t in TOPOLOGY_TRAIN_H),
                },
            )
        written["barrier"].append(f"{name}  h={h * 100:.0f} cm")

    # The DETOUR blocker: one prop, placed per condition, that closes the bypass lane
    # `route()`'s tie-break would otherwise take. A COLLIDER, unlike the target pads -- the pads
    # are kinematic and collisionless so a cube lands on them instead of bouncing off, whereas
    # this prop's whole function is to make a lane impassable. Static, like the barrier: authored
    # as a rigid body it would be shoved aside and the axis would measure nothing.
    bscale = (BLOCKER_DEPTH_M, BLOCKER_HEIGHT_M, BLOCKER_WIDTH_M)
    if not dry_run:
        make_static_box_usd(
            output_dir, "arb_blocker", BLOCKER_COLOR, bscale,
            extra_meta={"role": "blocker", "blocker_h_m": BLOCKER_HEIGHT_M},
        )
    written.setdefault("blocker", []).append(
        f"arb_blocker  {[round(v, 3) for v in _world_size(bscale)]} m (world)")

    # A VISUAL-ONLY twin of the blocker: identical geometry and colour, no CollisionAPI.
    #
    # It exists to separate the two things the real blocker introduces at once -- a geometric
    # constraint, and an object the policy has never seen. Measured, those are not
    # interchangeable: GR00T N1.7 REL re-routes cleanly around the solid blocker (33/36 took the
    # forced side) while ACT collapses in its presence (5/6 never cross the barrier at all), so
    # "the blocker changed the behaviour" does not by itself say which of the two did it.
    #
    # With the ghost, the lane is passable but still LOOKS closed. A policy that re-routes anyway
    # is responding to what it sees; one that drives straight through is responding only to
    # contact, and the solid-blocker re-route was physics rather than perception.
    if not dry_run:
        make_static_box_usd(
            output_dir, "arb_blocker_ghost", BLOCKER_COLOR, bscale,
            collider=False,
            extra_meta={"role": "blocker_ghost", "blocker_h_m": BLOCKER_HEIGHT_M},
        )
    written.setdefault("blocker", []).append(
        f"arb_blocker_ghost  {[round(v, 3) for v in _world_size(bscale)]} m (world, no collider)")

    # The original 0.16 m prop, kept ONLY so the v6/v7 DETOUR grids stay reproducible. It does
    # not seal the lane: the arm carries at 0.222-0.349 m and flies straight over it. New work
    # uses the default (`arb_blocker`, now 0.40 m).
    lscale_blk = (BLOCKER_DEPTH_M, BLOCKER_LEGACY_HEIGHT_M, BLOCKER_WIDTH_M)
    if not dry_run:
        make_static_box_usd(
            output_dir, "arb_blocker_legacy", BLOCKER_COLOR, lscale_blk,
            extra_meta={"role": "blocker", "blocker_h_m": BLOCKER_LEGACY_HEIGHT_M,
                        "note": "does not seal the lane; reproduction only"},
        )
    written.setdefault("blocker", []).append(
        f"arb_blocker_legacy  {[round(v, 3) for v in _world_size(lscale_blk)]} m (world)")

    fscale = (FIXTURE_DEPTH_M, FIXTURE_HEIGHT_M, FIXTURE_WIDTH_M)
    if not dry_run:
        make_static_box_usd(
            output_dir, "arb_fixture_wall", FIXTURE_COLOR, fscale,
            extra_meta={"role": "fixture"},
        )
    written["fixture"].append(
        f"arb_fixture_wall  {[round(v, 3) for v in _world_size(fscale)]} m (world)")

    # Static, because the axes that use it hold the spawn position fixed -- that is exactly the
    # condition under which marking it is sound. An axis that sweeps the spawn (POSITION) gets
    # no marker, or the table would be showing the answer.
    zscale = (START_ZONE_SIZE_M, START_ZONE_THICKNESS_M, START_ZONE_SIZE_M)
    if not dry_run:
        make_static_box_usd(output_dir, "arb_zone_start", START_ZONE_COLOR, zscale,
                            collider=False, extra_meta={"role": "start_zone"})
    written.setdefault("zone", []).append(
        f"arb_zone_start  {[round(v, 3) for v in _world_size(zscale)]} m (world)")

    # One lane asset per axis that declares one, sized from the spec's own endpoints so the
    # painted corridor cannot drift from the transport it depicts.
    for ax_name in sorted(LANE_AXES):
        lane = transport_lane(ax_name)
        if lane is None:
            continue
        _mid, length, width = lane
        lscale = (length, LANE_THICKNESS_M, width)
        name = f"arb_lane_{ax_name}"
        if not dry_run:
            make_static_box_usd(output_dir, name, LANE_COLOR, lscale,
                                collider=False,
                                extra_meta={"role": "lane", "axis": ax_name})
        written.setdefault("zone", []).append(
            f"{name}  {[round(v, 3) for v in _world_size(lscale)]} m (world)")

    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUTPUT_DIR), help="output directory for the USDs")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be written without authoring anything")
    args = ap.parse_args()

    out = Path(args.out)
    written = generate(out, dry_run=args.dry_run)

    tag = "[DRY]" if args.dry_run else "[OK] "
    print(f"\n{tag} output dir: {out}")
    for role, items in written.items():
        print(f"\n{tag} {role} ({len(items)}):")
        for line in items:
            print(f"       {line}")
    total = sum(len(v) for v in written.values())
    print(f"\n{tag} {total} prop(s); physx schema: "
          f"{'yes' if _HAS_PHYSX else 'no (raw attrs)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
