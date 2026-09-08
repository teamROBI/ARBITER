#!/usr/bin/env python3
"""Generate the graspable assets: primitive USD blocks for grasp-direction evaluation.

Shapes (world-space after the 90° X-rotation applied at spawn):
  *_front  — wide flat slab  12×6×3 cm  (narrow 3 cm depth, easy front pinch)
  *_top    — tall narrow col  4×4×14 cm  (small cross-section, easy top descent)

Also writes GLB preview files next to the USDs, for thumbnailing and quick
visual inspection. The assets are fully procedural -- no scanned mesh is used.

No Isaac Sim bootstrap is required — this script uses only pxr and trimesh.
Run with the agent venv (which has pxr + trimesh):
    arbiter/sim/.venv/bin/python arbiter/sim/tools/create_assets.py
"""
import json
import sys
from pathlib import Path

# ── pxr imports ─────────────────────────────────────────────────────────────
try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    print("[ERROR] pxr not found. Activate the project venv and retry.", flush=True)
    sys.exit(1)

try:
    from pxr import PhysxSchema
    _HAS_PHYSX = True
except ImportError:
    _HAS_PHYSX = False

# ── output directories ───────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR  = _REPO_ROOT / "data" / "usd" / "assets"
GLB_DIR     = OUTPUT_DIR / "glbs"

# ── object definitions ───────────────────────────────────────────────────────
# scale = (X, Y_up, Z) in Y-up USD space, meters.
# After the 90° X-rotation at spawn: Y_up → world-Z, Z → world-(-Y).
#   front slab  → world 0.12 W × 0.06 H × 0.03 D  (3 cm thin for front pinch)
#   top column  → world 0.04 W × 0.14 H × 0.04 D  (4×4 cm cross-section for top grasp)
OBJECTS = [
    ("arb_red_front",  (1.0, 0.15, 0.15), (0.04, 0.14, 0.04)),
    ("arb_red_top",    (1.0, 0.15, 0.15), (0.14, 0.04, 0.04)),
    ("arb_blue_front", (0.15, 0.15, 1.0), (0.04, 0.14, 0.04)),
    ("arb_blue_top",   (0.15, 0.15, 1.0), (0.14, 0.04, 0.04)),
]

# Rainbow cubes: 5×5×5 cm, ROYGBIV
CUBES = [
    ("arb_cube_red",    (1.0,  0.08, 0.08), (0.05, 0.05, 0.05)),
    ("arb_cube_orange", (1.0,  0.5,  0.0),  (0.05, 0.05, 0.05)),
    ("arb_cube_yellow", (1.0,  0.95, 0.0),  (0.05, 0.05, 0.05)),
    ("arb_cube_green",  (0.05, 0.8,  0.1),  (0.05, 0.05, 0.05)),
    ("arb_cube_blue",   (0.05, 0.3,  1.0),  (0.05, 0.05, 0.05)),
    ("arb_cube_indigo", (0.29, 0.0,  0.51), (0.05, 0.05, 0.05)),
    ("arb_cube_violet", (0.56, 0.0,  1.0),  (0.05, 0.05, 0.05)),
]

# APPROACH bars. The long axis is 10 cm, which exceeds the Panda's 8 cm jaw opening, so the
# bar can ONLY be grasped across its 3.5 cm width -- which is the whole point: the required jaw
# azimuth becomes a visible property of the object instead of an unstated convention.
#
# Scale is in the asset's Y-up frame and YUP_TO_ZUP is a 90 degree rotation about X, so
# (sx, sy, sz) spawns as world (sx, sz, sy): the long axis lands along world X and sy is height.
BARS = [
    ("arb_bar_red",  (1.0,  0.08, 0.08), (0.10, 0.035, 0.035)),
    ("arb_bar_blue", (0.05, 0.3,  1.0),  (0.10, 0.035, 0.035)),
]

# Buttons: flat cylinders, axis=Y in Y-up USD → lays flat (axis=Z) after spawn rotation.
# (radius_m, height_m), color
BUTTONS = [
    ("arb_button_green",  (0.0,  0.75, 0.2),  (0.03, 0.025)),
    ("arb_button_purple", (0.55, 0.0,  0.8),  (0.03, 0.025)),
    ("arb_button_orange", (1.0,  0.45, 0.0),  (0.03, 0.025)),
]

# Target pads: flat markers that make a transport goal *visible*.
#
# EXTENT sweeps transport distance, and with no pad the distance appeared in neither the image
# nor the instruction -- so the policy could only ever produce its habitual distance, and the
# axis was measuring the width of POSITION_TOLERANCE_M rather than any generalization limit.
# Every axis whose success predicate is "placed" has the same hole; POSITION and APPROACH do
# not, because "lift" needs no goal location.
#
# Half-width is deliberately POSITION_TOLERANCE_M (0.03), so the pad's footprint IS the region
# the predicate accepts. A pad larger than the tolerance would promise success where the
# checker fails.
#
# Kinematic and collisionless, because a pad is a marker and not an obstacle: it has to be
# repositioned per condition (EXTENT alone has 26 distinct lengths, so baking it into the stage
# the way the barrier is baked would mean one simulator launch per condition), and it must not
# catch the cube being placed on it or perturb the settle.
PAD_HALF_M = 0.03
PAD_THICKNESS_M = 0.004
# Mid-dark tones, not tints. The first pass used pale tints (1.0, 0.55, 0.55) and the render
# showed why that fails: on a white table under the scene's key light a tint is nearly
# invisible, and a goal marker the policy cannot resolve is no better than the missing target
# it replaced. Colour-matched to each cube but clearly darker, so pad and block are never
# confused -- and the pad is flat while the block is 3D, which separates them further.
PADS = [
    ("arb_pad_red",    (0.72, 0.18, 0.18)),
    ("arb_pad_blue",   (0.16, 0.34, 0.74)),
    ("arb_pad_green",  (0.14, 0.52, 0.22)),
    ("arb_pad_yellow", (0.78, 0.66, 0.10)),
]

MASS_KG          = 0.1
STATIC_FRICTION  = 2.0
DYNAMIC_FRICTION = 2.0
RESTITUTION      = 0.0


# ── helpers ──────────────────────────────────────────────────────────────────

def _add_physics_material(stage: "Usd.Stage", root_path: "Sdf.Path", collision_prim: "Usd.Prim"):
    """Create a UsdPhysics material and bind it to the collision prim."""
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
            if hasattr(pxm, "CreateImprovePatchFrictionAttr"):
                pxm.CreateImprovePatchFrictionAttr().Set(True)
            else:
                mat_prim.CreateAttribute("physxMaterial:improvePatchFriction", Sdf.ValueTypeNames.Bool).Set(True)
        except Exception:
            mat_prim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("max")
            mat_prim.CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token).Set("min")
            mat_prim.CreateAttribute("physxMaterial:improvePatchFriction", Sdf.ValueTypeNames.Bool).Set(True)

    try:
        UsdShade.MaterialBindingAPI.Apply(collision_prim).Bind(
            mat,
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )
    except Exception:
        rel = collision_prim.GetRelationship("physics:material:binding")
        if not rel:
            rel = collision_prim.CreateRelationship("physics:material:binding", custom=False)
        rel.SetTargets([mat_prim.GetPath()])


def _add_ccd(root_prim: "Usd.Prim"):
    """Apply CCD and solver iteration settings (PhysX-specific, best-effort)."""
    if not _HAS_PHYSX:
        root_prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
        root_prim.CreateAttribute("physxRigidBody:enableSpeculativeCCD", Sdf.ValueTypeNames.Bool).Set(True)
        root_prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(14)
        root_prim.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
        return
    try:
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
        if hasattr(rb, "CreateEnableCCDAttr"):
            rb.CreateEnableCCDAttr(True)
        else:
            root_prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
        if hasattr(rb, "CreateEnableSpeculativeCCDAttr"):
            rb.CreateEnableSpeculativeCCDAttr(True)
        else:
            root_prim.CreateAttribute("physxRigidBody:enableSpeculativeCCD", Sdf.ValueTypeNames.Bool).Set(True)
        if hasattr(rb, "CreateSolverPositionIterationCountAttr"):
            rb.CreateSolverPositionIterationCountAttr(14)
        else:
            root_prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(14)
        if hasattr(rb, "CreateSolverVelocityIterationCountAttr"):
            rb.CreateSolverVelocityIterationCountAttr(4)
        else:
            root_prim.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    except Exception:
        root_prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
        root_prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(14)


def make_block_usd(output_dir: Path, name: str, color: tuple, scale: tuple):
    usd_path = output_dir / f"{name}.usd"

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Root xform — carries RigidBodyAPI and mass
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())

    # Unit cube scaled to desired dimensions
    cube = UsdGeom.Cube.Define(stage, f"/{name}/Mesh")
    cube.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3f(*scale))

    # Visual material
    vis_mat = UsdShade.Material.Define(stage, f"/{name}/VisMaterial")
    shader  = UsdShade.Shader.Define(stage, f"/{name}/VisMaterial/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.0)
    vis_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(vis_mat)

    # Physics
    rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())  # noqa: F841
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr().Set(MASS_KG)

    _add_ccd(root.GetPrim())
    _add_physics_material(stage, root.GetPrim().GetPath(), cube.GetPrim())

    stage.GetRootLayer().Save()

    # Meta sidecar — bbox is exactly the scale vector (unit cube × scale = real dims)
    dims = list(scale)
    meta = {
        "bbox_size_m": dims,
        "bbox_longest_edge_m": max(dims),
        "meters_per_unit": 1.0,
    }
    meta_path = usd_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[OK] {usd_path.name}  bbox={[round(d*100, 1) for d in dims]} cm", flush=True)


def make_pad_usd(output_dir: Path, name: str, color: tuple):
    """A flat, kinematic, collisionless target marker.

    Differs from :func:`make_block_usd` in exactly three ways, each load-bearing:
    no ``CollisionAPI`` (the cube must land on the pad, not bounce off it), ``kinematicEnabled``
    (so it stays where it is written and is not pushed by the arm or gravity), and no CCD or
    physics material (both are meaningless without a collider).

    It is still a rigid body rather than a plain Xform: that is what lets the collector move it
    with ``write_root_state_to_sim``, the same call ``place_object`` uses, instead of needing a
    separate USD-editing path.
    """
    usd_path = output_dir / f"{name}.usd"

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())

    # Y-up stage, so the thin dimension is Y; the spawn rotation lays it flat on the table.
    scale = (2.0 * PAD_HALF_M, PAD_THICKNESS_M, 2.0 * PAD_HALF_M)
    cube = UsdGeom.Cube.Define(stage, f"/{name}/Mesh")
    cube.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3f(*scale))

    vis_mat = UsdShade.Material.Define(stage, f"/{name}/VisMaterial")
    shader = UsdShade.Shader.Define(stage, f"/{name}/VisMaterial/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    vis_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(vis_mat)

    rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rb.CreateKinematicEnabledAttr().Set(True)
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr().Set(MASS_KG)

    stage.GetRootLayer().Save()

    dims = list(scale)
    meta_path = usd_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "bbox_size_m": dims,
        "bbox_longest_edge_m": max(dims),
        "meters_per_unit": 1.0,
        "pad_half_m": PAD_HALF_M,
    }, indent=2))
    print(f"[OK] {usd_path.name}  pad half={PAD_HALF_M * 100:.1f} cm", flush=True)


def make_button_usd(output_dir: Path, name: str, color: tuple, radius_height: tuple):
    radius, height = radius_height
    usd_path = output_dir / f"{name}.usd"

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())

    cyl = UsdGeom.Cylinder.Define(stage, f"/{name}/Mesh")
    cyl.GetRadiusAttr().Set(radius)
    cyl.GetHeightAttr().Set(height)
    cyl.GetAxisAttr().Set("Y")  # Y-up → lays flat after 90° X spawn rotation

    vis_mat = UsdShade.Material.Define(stage, f"/{name}/VisMaterial")
    shader  = UsdShade.Shader.Define(stage, f"/{name}/VisMaterial/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.1)
    vis_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(cyl.GetPrim()).Bind(vis_mat)

    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr().Set(MASS_KG)

    _add_ccd(root.GetPrim())
    _add_physics_material(stage, root.GetPrim().GetPath(), cyl.GetPrim())

    stage.GetRootLayer().Save()

    diameter = radius * 2
    dims = [diameter, height, diameter]
    meta = {
        "bbox_size_m": dims,
        "bbox_longest_edge_m": max(dims),
        "meters_per_unit": 1.0,
    }
    (usd_path.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"[OK] {usd_path.name}  r={radius*100:.1f}cm h={height*100:.1f}cm", flush=True)


def make_button_glb(glb_dir: Path, name: str, color: tuple, radius_height: tuple):
    try:
        import trimesh
        import numpy as np
    except ImportError:
        print(f"[SKIP] trimesh not found — skipping GLB for {name}", flush=True)
        return

    radius, height = radius_height
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
    # trimesh cylinder is Z-axis by default; rotate 90° around X so flat face is down in Y-up GLTF viewer
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    r, g, b = [int(c * 255) for c in color]
    cyl.visual = trimesh.visual.ColorVisuals(mesh=cyl, face_colors=np.array([[r, g, b, 255]] * len(cyl.faces), dtype=np.uint8))

    glb_path = glb_dir / f"{name}_normalized.glb"
    cyl.export(str(glb_path))
    print(f"[OK] {glb_path.name}", flush=True)


def make_block_glb(glb_dir: Path, name: str, color: tuple, scale: tuple):
    """Write a normalized GLB preview file for the ACS portal viewer.

    The portal looks for <name>_normalized.glb first (preferred over raw GLB).
    Scale is (X, Y_up, Z) in meters; the GLB uses Y-up (GLTF standard).
    """
    try:
        import trimesh
        import numpy as np
    except ImportError:
        print(f"[SKIP] trimesh not found — skipping GLB for {name}", flush=True)
        return

    w, h, d = scale  # X, Y(up), Z
    box = trimesh.creation.box(extents=[w, h, d])

    # Convert 0-1 float color to 0-255 uint8 RGBA
    r, g, b = [int(c * 255) for c in color]
    box.visual = trimesh.visual.ColorVisuals(mesh=box, face_colors=np.array([[r, g, b, 255]] * len(box.faces), dtype=np.uint8))

    glb_path = glb_dir / f"{name}_normalized.glb"
    box.export(str(glb_path))
    print(f"[OK] {glb_path.name}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GLB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"USD output : {OUTPUT_DIR}", flush=True)
    print(f"GLB output : {GLB_DIR}", flush=True)
    for name, color, scale in OBJECTS:
        make_block_usd(OUTPUT_DIR, name, color, scale)
        make_block_glb(GLB_DIR, name, color, scale)
    for name, color, scale in CUBES:
        make_block_usd(OUTPUT_DIR, name, color, scale)

    for name, color, scale in BARS:
        make_block_usd(OUTPUT_DIR, name, color, scale)
        make_block_glb(GLB_DIR, name, color, scale)
    for name, color, rh in BUTTONS:
        make_button_usd(OUTPUT_DIR, name, color, rh)
        make_button_glb(GLB_DIR, name, color, rh)
    for name, color in PADS:
        make_pad_usd(OUTPUT_DIR, name, color)
    print(f"\nDone — "
          f"{len(OBJECTS) + len(CUBES) + len(BUTTONS) + len(PADS)} assets written", flush=True)


if __name__ == "__main__":
    main()
