# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Build the v2 scene for one axis, from that axis's layout.

One definition, shared by the renderer and the achievability gate. When the renderer had its
own copy, three composition bugs survived every programmatic check and were only caught by
looking at an image: a barrier rotated 90 degrees, a camera pointed at the robot's shoulder,
and a robot floating 18 cm off the table. A second copy of this logic is a second place for
that to happen.

Must be imported only after ``AppLauncher`` has started -- everything here touches isaaclab.
"""

from __future__ import annotations

import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from arbiter.collect.grasp import mat_to_quat
from arbiter.collect.constants import ROUTE_AROUND_MARGIN_M
from arbiter.suites.spec import (  # noqa: E402
    normalise_condition_key,  # noqa: E402
    AXES,
    ORDER_OBJECT_POS,
    WORKSPACE,
    blocker_xy,
    pad_keys_for_condition,
    parking_slot,
    start_zone_xys,
    transport_lane,
    target_xys,
)
from arbiter.suites.layout_gen import TABLE_TOP_Z, build_layout

_REPO_ROOT = Path(__file__).resolve().parents[3]
SCENE_USD_DIR = _REPO_ROOT / "data" / "usd" / "props"
OBJ_USD_DIR = _REPO_ROOT / "data" / "usd" / "assets"

#: Props are authored Y-up and get a +90 deg X rotation at spawn, so authored Y becomes world Z.
YUP_TO_ZUP = (0.70710678, 0.70710678, 0.0, 0.0)

#: Home joint pose, solved by IK to a grasp point over the workspace and then baked. A varying
#: start pose is uncontrolled variation the axis definition does not account for, so the
#: initial state has to be deterministic.
HOME_JOINT_POS = [-0.0157, -0.2754, 0.0159, -2.5895, 0.0059, 2.3142, 0.7809]

#: Whether to spawn the APPROACH fixture wall. Off: the implemented wall is a fixed obstruction
#: rather than the per-condition fixture the design calls for, and it blocks a 45-90 degree band
#: that includes a *trained* azimuth. Turn on only once the wall is positioned from the
#: condition under test. See the note in build_scene.
APPROACH_FIXTURE_ENABLED = False

#: Sun orientation, as a quaternion (w, x, y, z). A pure rotation about Y, so the light
#: direction lies in the xz-plane and carries no lateral component -- see build_scene.
#: 55 degrees from vertical, i.e. 35 degrees above the horizon.
SUN_ROT = (0.887155, 0.0, 0.461472, 0.0)


#: Standard deviation of the per-episode home-pose jitter, in radians (~1.1 deg per joint).
#: Small enough that every condition stays reachable, large enough that the approach segment of
#: two episodes of the same condition genuinely differ.
HOME_JITTER_RAD = 0.02


def jittered_home(seed: int, scale: float = HOME_JITTER_RAD) -> list[float]:
    """A slightly perturbed home pose, deterministic in ``seed``.

    Every part of this benchmark is otherwise deterministic -- fixed spawn pose, zero initial
    velocity, a scripted expert -- so two demonstrations of one condition would be identical
    frame for frame. Recording five copies of one trajectory inflates the episode count without
    adding information and makes the demo-diversity statistic the protocol calls for report
    variation that is not there.

    The arm's *starting configuration* is the one quantity no axis sweeps: POSITION, EXTENT and
    FACTORIAL vary the object and target, DIRECTION and APPROACH vary angles about them,
    TOPOLOGY varies the barrier, ORDER varies the instruction. Perturbing the home pose
    therefore varies the trajectory without touching any condition's parameter vector, which
    spawn-pose jitter could not do -- for POSITION the spawn pose *is* the swept parameter.

    Seeded from the condition key and repeat index so a rerun reproduces the dataset exactly.
    """
    import random

    rng = random.Random(seed)
    return [j + rng.gauss(0.0, scale) for j in HOME_JOINT_POS]


def episode_seed(condition_key: str, repeat: int) -> int:
    """Stable seed for one (condition, repeat). ``hash()`` is salted per process, so not that.

    The key is normalised first, so the seed depends on the condition rather than on what the
    project happens to call its assets -- see `normalise_condition_key`.
    """
    import hashlib

    key = normalise_condition_key(condition_key)
    h = hashlib.sha256(f"{key}#{repeat}".encode()).digest()
    return int.from_bytes(h[:8], "big")

#: Clearance so a spawned object rests on the surface rather than interpenetrating it.
SPAWN_CLEARANCE_M = 0.002


def thickness(usd_stem: str, directory: Path) -> float:
    """World-frame Z extent from the sidecar.

    ``world_size_m`` exists only on the scene props. The graspable assets carry
    ``bbox_size_m``, whose index 1 is the authored Y -- which becomes world Z after the spawn
    rotation, and is the same index ``sim_common.read_spawn_z_offset`` treats as the height.
    """
    meta = json.loads((directory / f"{usd_stem}.meta.json").read_text())
    if "world_size_m" in meta:
        return float(meta["world_size_m"][2])
    return float(meta["bbox_size_m"][1])


def look_at_quat(eye, target) -> tuple[float, float, float, float]:
    """Quaternion orienting a camera at ``target``, in CameraCfg's "world" convention.

    Baked into the spawn config rather than applied at runtime: a Camera sensor is not
    initialised during scene setup, so ``set_world_poses_from_view`` there silently leaves the
    camera at the origin. Measured -- both cameras reported world pos (0,0,0) and rendered
    identical frames of the ground plane.
    """
    import math

    f = [t - e for t, e in zip(target, eye)]
    n = math.sqrt(sum(c * c for c in f)) or 1.0
    f = [c / n for c in f]
    up = [0.0, 0.0, 1.0]
    d = sum(u * c for u, c in zip(up, f))
    z = [u - d * c for u, c in zip(up, f)]
    nz = math.sqrt(sum(c * c for c in z))
    if nz < 1e-8:
        z, nz = [0.0, 1.0, 0.0], 1.0
    z = [c / nz for c in z]
    y = [z[1] * f[2] - z[2] * f[1], z[2] * f[0] - z[0] * f[2], z[0] * f[1] - z[1] * f[0]]
    return tuple(mat_to_quat([[f[0], y[0], z[0]], [f[1], y[1], z[1]], [f[2], y[2], z[2]]]))


def build_scene(
    axis_name: str,
    *,
    device: str = "cuda:0",
    dt: float = 1.0 / 120.0,
    object_keys: list[str] | None = None,
    barrier_h: float | None = None,
    barrier_offset: float = 0.0,
    with_blocker: bool = False,
    blocker_ghost: bool = False,
    blocker_legacy: bool = False,
    with_cameras: bool = False,
    bright: bool = False,
    with_props: bool = True,
):
    """(scene, sim, info) for one axis.

    ``object_keys`` spawns those graspable objects as rigid bodies; every episode parks them
    off-workspace and teleports the one it needs, which is how a condition's parameter vector
    becomes a scene without rebuilding it. ``barrier_h`` selects which barrier asset to load
    for TOPOLOGY -- swapping the prop per episode is not possible once the stage is built, so a
    run covers one height at a time.
    """
    layout = build_layout(AXES[axis_name])["session"]

    sim = SimulationContext(SimulationCfg(dt=dt, device=device))
    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)

    cfg.ground = AssetBaseCfg(prim_path="/World/GroundPlane",
                              spawn=sim_utils.GroundPlaneCfg())
    cfg.dome = AssetBaseCfg(
        prim_path="/World/DomeLight",
        # Dropped from 1800/700. At the old intensities the table's own colour -- a mid grey,
        # (0.62, 0.60, 0.56) -- rendered as near-white, so the white Franka had almost no
        # figure/ground separation against it and the first target pads were invisible until
        # they were re-saturated. The fix belongs in the exposure, not in the albedo: darkening
        # the table further would have compensated for a light that was simply too strong.
        spawn=sim_utils.DomeLightCfg(
            intensity=900.0 if bright else 400.0, color=(0.85, 0.86, 0.90)),
    )
    # The sun MUST lie in the xz-plane, so it has no lateral component and the scene is
    # left-right symmetric. That symmetry is load-bearing, not cosmetic: it is what lets a
    # held-out azimuth's mirror image count as evidence the capability exists, which is the
    # argument DIRECTION and FACTORIAL rest on. The previous rot=(0.87, 0.32, 0.32, 0.0) aimed
    # the light along (-0.579, +0.579, -0.574) -- 35 degrees off the plane -- so shadows fell to
    # one side and +theta was not visually equivalent to -theta.
    #
    # This keeps the original 35 degree elevation and only rotates the azimuth onto the plane,
    # pointing the light along (-0.819, 0, -0.574): it travels away from the head camera, so
    # shadows fall behind objects and the arm does not shadow the object it is reaching for.
    # Asserted by arbiter/sim/env/tests/test_scene_symmetry.py.
    cfg.sun = AssetBaseCfg(
        prim_path="/World/Sun",
        spawn=sim_utils.DistantLightCfg(intensity=700.0, angle=1.5),
        init_state=AssetBaseCfg.InitialStateCfg(rot=SUN_ROT),
    )

    t_table = thickness("arb_table", SCENE_USD_DIR)
    tmeta = json.loads((SCENE_USD_DIR / "arb_table.meta.json").read_text())
    tcx, tcy = tmeta["world_centre_xy"]
    cfg.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=str(SCENE_USD_DIR / "arb_table.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(tcx, tcy, TABLE_TOP_Z - 0.5 * t_table), rot=YUP_TO_ZUP),
    )

    cfg.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.robot.init_state = cfg.robot.init_state.replace(pos=(0.0, 0.0, TABLE_TOP_Z))

    # Graspable objects, parked clear of the workspace. place_object() teleports the one an
    # episode needs; the rest stay parked so they cannot occlude or collide.
    keys = object_keys if object_keys is not None else [
        o["asset_name"] for o in layout["object_pool"]
    ]
    half_h: dict[str, float] = {}
    # An InteractiveScene is indexed by the *cfg attribute name*, not the prim path, so the
    # attribute name is what has to be recorded for later lookup.
    entity_name: dict[str, str] = {}
    for i, key in enumerate(keys):
        h = thickness(key, OBJ_USD_DIR)
        half_h[key] = 0.5 * h
        entity_name[key] = f"obj_{i}"
        setattr(cfg, f"obj_{i}", RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Obj_{key}",
            spawn=sim_utils.UsdFileCfg(usd_path=str(OBJ_USD_DIR / f"{key}.usd")),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=_parking(i), rot=YUP_TO_ZUP),
        ))

    # Target pads. Kinematic, collisionless markers that make a transport goal visible; see
    # PAD_AXES in axes.py for why only the "place" axes get them. Spawned parked and moved into
    # position per condition by place_pad, because the goal moves with the condition -- EXTENT
    # alone has 26 distinct lengths, so baking them in the way the barrier is baked would mean
    # one simulator launch per condition.
    pad_keys = _axis_pad_keys(axis_name) if with_props else []
    pad_entity: dict[str, str] = {}
    pad_half_h: dict[str, float] = {}
    for i, pkey in enumerate(pad_keys):
        pad_half_h[pkey] = 0.5 * thickness(pkey, OBJ_USD_DIR)
        pad_entity[pkey] = f"pad_{i}"
        setattr(cfg, f"pad_{i}", RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Pad_{pkey}",
            spawn=sim_utils.UsdFileCfg(usd_path=str(OBJ_USD_DIR / f"{pkey}.usd")),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=_parking(len(keys) + i), rot=YUP_TO_ZUP),
        ))

    # Transport corridor: a painted lane between fixed endpoints. Static and collisionless, and
    # spawned BEFORE the start zone so the zone reads on top of it where they overlap.
    if with_props and transport_lane(axis_name) is not None:
        (lx, ly), _len, _w = transport_lane(axis_name)
        t_lane = thickness(f"arb_lane_{axis_name}", SCENE_USD_DIR)
        setattr(cfg, "lane", AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Lane",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(SCENE_USD_DIR / f"arb_lane_{axis_name}.usd")),
            # rot=YUP_TO_ZUP is NOT optional. The scene props are authored Y-up and nothing
            # converts them automatically -- the barrier passes this same rotation. Omitted, the
            # authored height stays world Y and the authored width stays world Z, so a
            # 30x14 cm floor lane spawns as a 4 mm wide, 14 cm tall vertical fin. It renders as
            # a hairline, which reads as "the marking is too faint" rather than "the marking is
            # standing on edge" -- so it survived a colour change before being measured.
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(float(lx), float(ly), TABLE_TOP_Z + 0.5 * t_lane),
                rot=YUP_TO_ZUP),
        ))

    # Start zones: static markers at the spawn positions of axes that keep the spawn FIXED.
    # Static rather than kinematic like the pads, because nothing about them varies with the
    # condition -- that is precisely why marking them is sound. See START_ZONE_AXES.
    if with_props:
        t_zone = thickness("arb_zone_start", SCENE_USD_DIR)
        for zi, (zx, zy) in enumerate(start_zone_xys(axis_name)):
            setattr(cfg, f"zone_{zi}", AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/StartZone_{zi}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(SCENE_USD_DIR / "arb_zone_start.usd")),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(float(zx), float(zy), TABLE_TOP_Z + 0.5 * t_zone),
                    rot=YUP_TO_ZUP),
            ))

    props = layout["props"]
    barrier_top = None
    blocker_at = None

    # NOTE on the APPROACH fixture. The design calls for it to be *parameterized per
    # condition*: positioned so it blocks every azimuth except the one under test, making the
    # required grasp cued by affordance rather than by instruction. What is implemented is a
    # single wall at a fixed offset (+y, -x from the object), which is not that -- it is an
    # obstruction that happens to block one band.
    #
    # Measured consequence: the achievability gate failed APPROACH in a contiguous 45-90 degree
    # band, every failure a `grasp` stall ~4 cm into the descent -- including the TRAINED
    # phi=45, which by the gate's own definition makes the axis uncollectable. Instrumented to
    # be sure, because the same band is consistent with several other causes and this project
    # has already lost time to reasoning about geometry instead of measuring it:
    #
    #   - not cube geometry: phi=135 has the identical 7.07 cm diagonal jaw span and passes,
    #     while phi=90 has the same 5.00 cm span as the passing phi=0 and fails;
    #   - not a joint limit: at the stall every joint sits at 0.22-0.75 of its range;
    #   - not contact with the cube: the cube never moves from its spawn pose;
    #   - it is a hard obstruction. During the stall j2/j4/j6 -- the joints carrying vertical
    #     reach -- are frozen while j1/j3/j5/j7 drift steadily: the IK spinning in the null
    #     space of an end-effector it cannot lower.
    #
    # The wall is the only asymmetric thing in the scene, and it is on the +y side.
    #
    # This flag is what actually turns it off. It used to say in prose that the fixture was
    # "off by default", while the branch below still fired whenever the layout carried a
    # fixture prop -- which it always does, since layout_gen emits one as design intent. So the
    # wall kept spawning and kept failing a trained condition. APPROACH is tier 2 in the plan;
    # DIRECTION and TOPOLOGY carry the structural claim without it.
    if with_props and axis_name == "topology":
        h = barrier_h if barrier_h is not None else float(props[0]["barrier_h_m"])
        name = f"arb_barrier_h{int(round(h * 1000)):04d}"
        t_bar = thickness(name, SCENE_USD_DIR)
        cx, cy = WORKSPACE.centre
        # The offset MUST be honoured here. It was previously ignored, pinning the barrier at
        # y=cy while RouteTask derived its bypass from cy + offset -- so the task routed around
        # a barrier that was not where it thought. Measured: the only around-class conditions
        # that passed were offset=0, the one case where assumption and reality coincided; both
        # +/-0.06 put the bypass exactly on the real barrier's edge and the object hit it.
        # Same class of bug as barrier_h, and both come from a condition parameter that the
        # scene has to realise but silently did not.
        cfg.barrier = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Barrier",
            spawn=sim_utils.UsdFileCfg(usd_path=str(SCENE_USD_DIR / f"{name}.usd")),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(cx, cy + float(barrier_offset), TABLE_TOP_Z + 0.5 * t_bar),
                rot=YUP_TO_ZUP),
        )
        barrier_top = TABLE_TOP_Z + t_bar

        # DETOUR's blocker: closes the lane `route()`'s tie-break would otherwise take, so the
        # complement side is required by physics rather than by the `h*` convention. Placed
        # from the spec's own `blocker_xy`, which derives the lane from the same tie-break the
        # expert uses -- restating it here is how the barrier offset came to be honoured in one
        # place and ignored in another.
        #
        # Baked into the stage like the barrier, not placed per-episode like the pads: it is a
        # static collider, and a collider cannot be teleported per condition without the arm
        # tunnelling through it. So a run covers one (barrier_h, offset, blocker) at a time --
        # the same one-launch-per-configuration constraint TOPOLOGY already has.
        if with_blocker:
            # The ghost is the visual-only twin: same geometry and colour, no CollisionAPI. It
            # separates "re-routed because the lane is closed" from "re-routed because the lane
            # LOOKS closed", which the solid blocker confounds.
            # Legacy wins over ghost if both are asked for; they answer different questions
            # and combining them silently would produce a condition neither arm describes.
            if blocker_legacy:
                blk_name = "arb_blocker_legacy"
            elif blocker_ghost:
                blk_name = "arb_blocker_ghost"
            else:
                blk_name = "arb_blocker"
            t_blk = thickness(blk_name, SCENE_USD_DIR)
            bx, by = blocker_xy(float(barrier_offset),
                                around_margin=ROUTE_AROUND_MARGIN_M)
            cfg.blocker = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Blocker",
                spawn=sim_utils.UsdFileCfg(usd_path=str(SCENE_USD_DIR / f"{blk_name}.usd")),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(bx, by, TABLE_TOP_Z + 0.5 * t_blk),
                    rot=YUP_TO_ZUP),
            )
            blocker_at = (bx, by)
    elif (APPROACH_FIXTURE_ENABLED
          and with_props and props and props[0]["role"] == "fixture"):
        t_fx = thickness("arb_fixture_wall", SCENE_USD_DIR)
        cx, cy = WORKSPACE.centre
        cfg.fixture = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Fixture",
            spawn=sim_utils.UsdFileCfg(usd_path=str(SCENE_USD_DIR / "arb_fixture_wall.usd")),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(cx - 0.06, cy + 0.08, TABLE_TOP_Z + 0.5 * t_fx), rot=YUP_TO_ZUP),
        )

    if with_cameras:
        for cam in layout["custom_cams"]:
            _add_camera(cfg, cam)

    scene = InteractiveScene(cfg)
    sim.reset()
    return scene, sim, {
        # What the stage ACTUALLY holds, for assert_condition_realised. Reporting the
        # arguments back is the point: a parameter that never reached the stage cannot appear
        # here, so the guard catches exactly the bug that motivated it.
        "built": {"barrier_h": (barrier_h if axis_name == "topology" and with_props else None),
                  "barrier_offset": (float(barrier_offset)
                                     if axis_name == "topology" and with_props else None),
                  # 1.0/0.0 rather than True/False: assert_condition_realised compares floats,
                  # and a condition carries `blocker` as a 0/1 parameter.
                  "blocker": (1.0 if blocker_at is not None else 0.0)
                             if axis_name == "topology" and with_props else None},
        "layout": layout,
        "object_keys": keys,
        "object_half_height": half_h,
        "instance_name": entity_name,
        "pad_keys": pad_keys,
        "pad_half_height": pad_half_h,
        "pad_instance_name": pad_entity,
        "barrier_top_z": barrier_top,
        "blocker_xy": blocker_at,
        "table_top_z": TABLE_TOP_Z,
    }


#: Condition parameters that the SCENE must realise, per axis. Anything listed here has to be
#: reflected in the built stage; anything a task derives on its own (a transport target, a
#: grasp azimuth) does not appear.
SCENE_REALISED_PARAMS: dict[str, set[str]] = {
    "topology": {"barrier_h", "barrier_offset"},
}


def assert_condition_realised(axis_name: str, params: dict, built: dict) -> None:
    """Fail loudly if the stage does not match the condition it is supposed to represent.

    Two of the worst bugs in this phase were the same shape: a condition parameter the scene
    had to realise and silently did not. ``barrier_h`` was ignored, so the expert routed over
    an apex computed for a barrier that was not there and drove the object into the real one.
    ``barrier_offset`` was ignored, so the bypass landed on the real barrier's edge. Neither
    raised anything -- one produced failures that looked like a routing limitation, and the
    other would have produced *clean* data on an axis that had stopped testing what it claimed.

    Vigilance is not a control for that. This is: the caller passes what the stage was actually
    built with, and a mismatch is an error at build time rather than a number nobody questions.
    """
    required = SCENE_REALISED_PARAMS.get(axis_name, set())
    problems = []

    # The blocker is checked separately from `required` because it is OPTIONAL: the 12 collected
    # TOPOLOGY archives predate it and their conditions carry no `blocker` key at all, so
    # demanding one would fail every existing condition. What must never happen is a MISMATCH --
    # a condition asking for a sealed lane on a stage that has no blocker would let the expert
    # detour to the complement through empty space and record a demonstration of a configuration
    # that was never built. That is the exact failure mode `barrier_offset` had.
    if "blocker" in built:
        want_blk = bool(float(params.get("blocker", 0.0)))
        got_blk = bool(float(built.get("blocker") or 0.0))
        if want_blk != got_blk:
            problems.append(
                f"'blocker': condition says {'present' if want_blk else 'absent'}, "
                f"scene built {'present' if got_blk else 'absent'}")
    for name in sorted(required):
        if name not in params:
            problems.append(f"condition has no '{name}'")
            continue
        want = float(params[name])
        got = built.get(name)
        if got is None:
            problems.append(f"'{name}' is not realised in the scene at all")
        elif abs(float(got) - want) > 1e-6:
            problems.append(f"'{name}': condition says {want}, scene built {got}")
    if problems:
        raise ValueError(
            f"scene does not realise condition for axis '{axis_name}': "
            + "; ".join(problems)
            + ". A prop cannot change after the stage is built, so run one launch per "
              "distinct value rather than iterating conditions against a fixed scene."
        )


def _parking(i: int) -> tuple[float, float, float]:
    """Off-workspace parking slot. Defined by the spec -- see ``axes.parking_slot``."""
    return parking_slot(i)


def _add_camera(cfg, cam: dict) -> None:
    if cam["name"] == "cam_wrist":
        # Rides the hand, so its prim lives under the hand body and its offset is local.
        setattr(cfg, "cam_wrist", CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{cam['parent_body']}/WristCam",
            update_period=0.0,
            width=int(cam["width"]), height=int(cam["height"]),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=float(cam["focal_mm"]),
                                             clipping_range=(0.01, 10.0)),
            offset=CameraCfg.OffsetCfg(pos=tuple(cam["offset_pos"]), convention="ros"),
        ))
        return
    setattr(cfg, cam["name"], CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{cam['name']}",
        update_period=0.0,
        width=int(cam["width"]), height=int(cam["height"]),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=float(cam["focal_mm"]),
                                        clipping_range=(0.05, 20.0)),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(cam["pos"]),
            rot=look_at_quat(cam["pos"], cam["look_at"]),
            convention="world",
        ),
    ))


def _quat_mul(a, b):
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def spawn_rot(yaw_deg: float = 0.0):
    """Spawn orientation for a prop: the Y-up correction, then a yaw about world Z.

    APPROACH needs this. Its object is a bar whose long axis exceeds the jaw opening, so the
    bar can only be grasped across its width -- which makes the required jaw azimuth a
    *visible* property of the object rather than an unstated convention. Sweeping the bar's
    yaw is therefore the same sweep as the grasp azimuth, but one the policy can actually see.
    """
    import math

    if not yaw_deg:
        return YUP_TO_ZUP
    h = math.radians(float(yaw_deg)) * 0.5
    return _quat_mul((math.cos(h), 0.0, 0.0, math.sin(h)), YUP_TO_ZUP)


def _axis_pad_keys(axis_name: str) -> list[str]:
    """Every pad the SCENE must spawn for an axis (a superset of any one condition's)."""
    from arbiter.suites.spec import pad_keys_for
    return pad_keys_for(axis_name)


def place_pad(scene, sim, info: dict, key: str, xy) -> None:
    """Move one target pad to ``xy``, flat on the table.

    No settle loop and no return value, unlike :func:`place_object`: the pad is kinematic, so
    it does not fall and there is no resting height to measure. It is written once per episode
    before the object is placed.

    Silently does nothing if the axis has no such pad, so a caller can place pads
    unconditionally rather than every task duplicating the PAD_AXES test.
    """
    import torch

    inst = info.get("pad_instance_name", {}).get(key)
    if inst is None:
        return
    pad = scene[inst]
    z = info["table_top_z"] + info["pad_half_height"][key]
    root = pad.data.default_root_state.clone()
    root[:, 0:3] = torch.tensor([[float(xy[0]), float(xy[1]), z]], device=pad.device)
    root[:, 3:7] = torch.tensor([list(YUP_TO_ZUP)], device=pad.device)
    root[:, 7:] = 0.0
    pad.write_root_state_to_sim(root)
    pad.reset()


def park_objects(scene, sim, info: dict, keys) -> None:
    """Move objects off the table, to the parking slots they spawned in.

    Needed because ORDER's pair is a condition parameter: the scene spawns every cube any pair
    uses so one launch can realise all of them, and a condition places only two. Without
    parking, the green/yellow pair would still be sitting on the table during the next
    red/blue episode -- four cubes in frame, an observation no condition describes, and a
    contamination that looks like nothing at all in the recorded action.
    """
    import torch

    for key in keys:
        inst = info["instance_name"].get(key)
        if inst is None:
            continue
        obj = scene[inst]
        i = int(str(inst).rsplit("_", 1)[-1])
        root = obj.data.default_root_state.clone()
        root[:, 0:3] = torch.tensor([list(_parking(i))], device=obj.device)
        root[:, 3:7] = torch.tensor([list(YUP_TO_ZUP)], device=obj.device)
        root[:, 7:] = 0.0
        obj.write_root_state_to_sim(root)
        obj.reset()


def reset_condition_scene(scene, sim, info: dict, condition) -> tuple[list[float], list[str]]:
    """Put the stage into the state one condition describes. Returns (resting z, object keys).

    The single place that answers "which objects, where, and which goals are visible" for a
    condition. Five tools -- the collector, the gate, the evaluator, the probe and the
    renderer -- each had their own copy of an ``if axis_name == "order"`` branch, and this file
    already records what that costs: placement logic duplicated across tools is how TOPOLOGY's
    barrier offset came to be honoured in one place and ignored in another, producing clean
    data for a configuration that was never built.

    Order matters. Pads and parking come before the object is placed, so the settle happens in
    the scene the episode actually runs in.
    """
    from arbiter.suites.spec import object_start_xy, object_start_yaw, order_slot_objects

    if condition.axis == "order":
        used = order_slot_objects()
        starts = [tuple(xy) for xy in ORDER_OBJECT_POS]
        yaws = [0.0] * len(used)
    else:
        used = [condition.object_key]
        starts = [object_start_xy(condition)]
        yaws = [object_start_yaw(condition)]

    place_pads_for(scene, sim, info, condition)
    park_objects(scene, sim, info, [k for k in info["object_keys"] if k not in used])
    rest = [place_object(scene, sim, info, k, xy, yaw_deg=y)
            for k, xy, y in zip(used, starts, yaws)]
    return rest, used


def place_pads_for(scene, sim, info: dict, condition) -> None:
    """Place every pad a condition needs, from the spec's own target list.

    Pairs ``pad_keys_for`` with ``target_xys`` positionally, which is the contract both
    functions document: the pad the policy sees is placed at the coordinates the success
    predicate will check, because both come from ``transport_target``.
    """
    pads = pad_keys_for_condition(condition)
    targets = target_xys(condition)
    if len(pads) != len(targets):
        raise ValueError(
            f"{condition.axis}: {len(pads)} pads but {len(targets)} targets. The pad a policy "
            f"aims at would not be the target it is scored against."
        )
    for pkey, xy in zip(pads, targets):
        place_pad(scene, sim, info, pkey, xy)


def place_object(scene, sim, info: dict, key: str, xy, *, yaw_deg: float = 0.0,
                 settle_steps: int = 45) -> float:
    """Teleport one object onto the table at ``xy`` and settle. Returns its resting world z.

    Settling matters: an object dropped and read immediately reports a height it has not
    reached yet, and the lift check is relative to that baseline. v1 used 45 steps for the
    same reason.
    """
    import torch

    obj = scene[info["instance_name"][key]]
    z = info["table_top_z"] + info["object_half_height"][key] + SPAWN_CLEARANCE_M
    root = obj.data.default_root_state.clone()
    root[:, 0:3] = torch.tensor([[float(xy[0]), float(xy[1]), z]], device=obj.device)
    root[:, 3:7] = torch.tensor([list(spawn_rot(yaw_deg))], device=obj.device)
    root[:, 7:] = 0.0
    obj.write_root_state_to_sim(root)
    obj.reset()

    # Hold the arm at its current joint positions through the settle. write_joint_state_to_sim
    # sets state, not the position target, so without this the arm drifts from the home pose
    # back toward whatever target was last commanded while the object falls.
    robot = scene["robot"]
    robot.set_joint_position_target(robot.data.joint_pos.clone())

    for _ in range(settle_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    return float(obj.data.root_pos_w[0, 2].detach().cpu().item())
