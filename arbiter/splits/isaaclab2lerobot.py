# Copyright 2026 ROBI Contributors.
# Copyright 2025 ROBOTIS CO., LTD.
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import h5py
import json
import numpy as np
import argparse
import shutil
from tqdm import tqdm

import subprocess
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import sys as _sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from arbiter.suites.spec import (
    TRAINING_EXCLUSIONS,
    instruction_from_attrs,
    is_withheld_from_training,
    params_from_attrs,
)
from arbiter.splits.demo_selector import select_demo_names, validate_archive

ROBOT_TYPE = "ffw_sg2_rev1"

_ARM_JOINTS = [
    "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
    "arm_l_joint5", "arm_l_joint6", "arm_l_joint7", "gripper_l_joint1",
    "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
    "arm_r_joint5", "arm_r_joint6", "arm_r_joint7", "gripper_r_joint1",
    "head_joint1", "head_joint2", "lift_joint",
]
_MOBILE_JOINTS = _ARM_JOINTS + ["base_linear_x", "base_linear_y", "base_angular_z"]

_CAMERAS_HEAD_ONLY = {
    "cam_head": {"height": 376, "width": 672},
}
_CAMERAS_ALL = {
    "cam_head":        {"height": 376, "width": 672},
    "cam_wrist_left":  {"height": 240, "width": 424},
    "cam_wrist_right": {"height": 240, "width": 424},
}

_FRANKA_JOINTS = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7", "panda_finger_joint1",
]

_CAMERAS_FRANKA = {
    "cam_head": {"height": 376, "width": 672},
}

ROBOT_CONFIGS = {
    # v2: single-arm Franka. 8 dims = 7 arm joints + 1 gripper. This is the canonical
    # released dimensionality -- the spec has to instantiate on a Franka or an SO-100, so a
    # 14- or 19-dim vector with most entries parked is not a portable contract.
    "franka_panda": {
        "action_dim": 8,
        "state_dim": 8,
        "action_names": _FRANKA_JOINTS,
        "state_names": _FRANKA_JOINTS,
        "cameras": _CAMERAS_FRANKA,
    },
    "ffw_sg2_rev1": {
        "action_dim": 19,
        "state_dim": 19,
        "action_names": _ARM_JOINTS,
        "state_names": _ARM_JOINTS,
        "cameras": _CAMERAS_HEAD_ONLY,
    },
    "ffw_sg2_rev1_mobile": {
        "action_dim": 22,
        "state_dim": 19,  # joint_pos has no mobile base (velocity, not position)
        "action_names": _MOBILE_JOINTS,
        "state_names": _ARM_JOINTS,
        "cameras": _CAMERAS_ALL,
    },
}

def _detect_robot_config(dataset_file: str) -> str:
    """Auto-detect robot config from action shape of first episode."""
    with h5py.File(dataset_file, "r") as f:
        ep_keys = sorted(f["data"].keys())
        if ep_keys:
            actions = f["data"][ep_keys[0]]["actions"]
            dim = actions.shape[-1]
            for name, cfg in ROBOT_CONFIGS.items():
                if cfg["action_dim"] == dim:
                    return name
    return ROBOT_TYPE  # fallback


def _detect_cameras(dataset_file: str, robot_config_name: str) -> dict:
    """Camera set actually present in the archive, not the one the robot config assumes.

    v2 enables the wrist camera only on the axes whose holdout the wrist can see -- TOPOLOGY
    and APPROACH (`WRIST_CAM_AXES` in layout_gen). A fixed camera list per robot is therefore
    wrong in both directions: it silently drops `cam_wrist` on the two axes that have it, and
    would declare a video feature with no frames behind it on the five that do not.

    Resolution is read from the data as well, so a re-render at a different size does not
    require editing a table that nothing checks.
    """
    declared = ROBOT_CONFIGS[robot_config_name]["cameras"]
    with h5py.File(dataset_file, "r") as f:
        ep_keys = sorted(f["data"].keys())
        if not ep_keys:
            return dict(declared)
        obs = f["data"][ep_keys[0]]["obs"]
        found = {}
        for key in obs.keys():
            if not key.startswith("cam"):
                continue
            shape = obs[key].shape           # (T, H, W, C)
            if len(shape) != 4:
                continue
            found[key] = {"height": int(shape[1]), "width": int(shape[2])}
    if not found:
        return dict(declared)
    extra = sorted(set(found) - set(declared))
    missing = sorted(set(declared) - set(found))
    if extra:
        print(f"[INFO] cameras present beyond the robot config: {extra}")
    if missing:
        print(f"[INFO] robot config declares cameras this archive lacks, skipping: {missing}")
    return found

def _encode_video_frames_h264_avc1(img_dir: Path, video_path: Path, fps: int, overwrite: bool = True):
    """
    Replace LeRobot's default video encoder with libx264 + avc1 tag (MP4).
    This matches common 'real' datasets: codec_name=h264, codec_tag_string=avc1.
    """
    img_dir = Path(img_dir)
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    input_pattern = str(img_dir / "frame_%06d.png")

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-loglevel", "error",
        "-framerate", str(int(fps)),
        "-i", input_pattern,
        "-an",
        "-c:v", "libx264",
        "-tag:v", "avc1",
        "-pix_fmt", "yuv420p",
        "-crf", "28",
        "-preset", "veryfast",
        str(video_path),
    ]

    subprocess.run(cmd, check=True)


def force_lerobot_h264_avc1():
    """Monkeypatch LeRobot's encode_video_frames to force libx264+avc1."""
    import lerobot.datasets.video_utils as vu
    import lerobot.datasets.lerobot_dataset as ld
    vu.encode_video_frames = _encode_video_frames_h264_avc1
    ld.encode_video_frames = _encode_video_frames_h264_avc1

def get_env_features(fps: int, robot_config_name: str = ROBOT_TYPE, cameras: dict | None = None):
    config = ROBOT_CONFIGS[robot_config_name]
    cameras = config["cameras"] if cameras is None else cameras

    features = {
        "action": {
            "dtype": "float32",
            "shape": (config["action_dim"],),
            "names": config["action_names"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (config["state_dim"],),
            "names": config["state_names"],
        },
    }

    for cam_name, cam_cfg in cameras.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": [cam_cfg["height"], cam_cfg["width"], 3],
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.height": cam_cfg["height"],
                "video.width": cam_cfg["width"],
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    return features

def process_data(dataset: LeRobotDataset, task: str, demo_group: h5py.Group, demo_name: str,
                 robot_config_name: str = ROBOT_TYPE, cameras: dict | None = None) -> bool:
    """Process a single demonstration group from the HDF5 dataset and add it into the LeRobot dataset."""
    config = ROBOT_CONFIGS[robot_config_name]
    # The same detected set the features were declared from. Taking it from the robot config
    # here instead would either drop a stream the dataset declares or look for one the archive
    # does not have, and the two must agree frame for frame.
    camera_keys = list((config["cameras"] if cameras is None else cameras).keys())

    try:
        actions = np.array(demo_group['actions'], dtype=np.float32)
        joint_pos = np.array(demo_group['obs/joint_pos'], dtype=np.float32)

        camera_data = {}
        for cam_key in camera_keys:
            camera_data[cam_key] = np.array(demo_group[f'obs/{cam_key}'], dtype=np.uint8)

    except KeyError as e:
        print(f"Demo {demo_name} is not valid (missing key: {e}), skipping...")
        return False

    if actions.shape[0] < 10:
        print(f"Demo {demo_name} has insufficient frames ({actions.shape[0]}), skipping...")
        return False

    if actions.ndim == 1:
        actions = actions.reshape(-1, config["action_dim"])
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.reshape(-1, config["state_dim"])

    total_state_frames = actions.shape[0]

    for frame_index in tqdm(range(total_state_frames), desc=f"Processing demo {demo_name}"):
        frame = {
            "action": actions[frame_index],
            "observation.state": joint_pos[frame_index],
        }
        for cam_key in camera_keys:
            frame[f"observation.images.{cam_key}"] = camera_data[cam_key][frame_index]

        dataset.add_frame(frame=frame, task=task)

    return True

def convert_isaaclab_to_lerobot(
    repo_id: str, dataset_file: str, task_fallback: str,
    fps: int, push_to_hub: bool = False, overwrite: bool = False,
    root: str = "./datasets/lerobot/sim2real_data",
    demo_names: list[str] | None = None,
    demo_range: tuple[int, int] | None = None,
    split: str | None = None,
    expected_axis: str | None = None,
):
    """Convert an IsaacLab HDF5 dataset into LeRobot dataset format."""
    force_lerobot_h264_avc1()

    robot_config_name = _detect_robot_config(dataset_file)
    cameras = _detect_cameras(dataset_file, robot_config_name)
    print(f"[INFO] Robot config: {robot_config_name} (action={ROBOT_CONFIGS[robot_config_name]['action_dim']}-dim, state={ROBOT_CONFIGS[robot_config_name]['state_dim']}-dim)")
    print(f"[INFO] Cameras from archive: {sorted(cameras)}")

    now_episode_index = 0
    episode_conditions: list[dict] = []
    output_dir = Path(root) / repo_id

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite to delete it and re-convert."
            )
        print(f"[INFO] Overwriting existing dataset at {output_dir}")
        shutil.rmtree(output_dir)

    output_dir = str(output_dir)

    with h5py.File(dataset_file, "r") as f:
        if "fps" in f.attrs:
            stored_fps = int(f.attrs["fps"])
            if stored_fps != fps:
                print(f"[INFO] Using fps={stored_fps} from HDF5 (overriding default {fps})")
                fps = stored_fps

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        # The detected config, not a module constant: a Franka dataset labelled
        # ffw_sg2_rev1 would mislead every downstream consumer about its embodiment.
        robot_type=robot_config_name,
        features=get_env_features(fps, robot_config_name, cameras),
        root=output_dir,
    )

    print(f"[INFO] Processing HDF5 file: {dataset_file}")
    with h5py.File(dataset_file, "r") as f:
        if demo_names is not None:
            # Caller pre-filtered; process exactly these demos, no success check needed.
            selected = demo_names
            print(f"[INFO] Using {len(selected)} pre-selected demos")
        elif split is not None:
            # v2: the axis owns the split. validate_archive refuses an archive collected for
            # a different axis, because converting against the wrong one produces a dataset
            # whose split silently disagrees with the benchmark.
            axis_name = validate_archive(dataset_file, expected_axis=expected_axis)
            selected = select_demo_names(dataset_file, split)
            print(f"[INFO] axis '{axis_name}' split '{split}': "
                  f"{len(selected)} demos to convert")
        else:
            all_names = list(f["data"].keys())
            if demo_range is not None:
                start, end = demo_range
                all_names = all_names[start:end]
                print(f"[INFO] Demo slice [{start}:{end}] → {len(all_names)} demos")
            else:
                print(f"[INFO] Found {len(all_names)} demos")
            selected = [
                name for name in all_names
                if not ("success" in f["data"][name].attrs and not f["data"][name].attrs["success"])
            ]
            print(f"[INFO] {len(selected)} successful demos to convert")

        # Reported at the end: a conversion that silently fell back to recorded strings for most
        # episodes is the failure mode this block exists to make visible.
        n_derived = n_recorded = n_changed = 0
        recorded_warned = False

        # Withhold the conditions the spec says must not reach the training set. This is
        # separate from the axis's own train/test split: `in_train` answers "is this condition
        # part of the sweep holdout", this answers "does the policy get to see it", and once a
        # second non-swept holdout is layered on, those stop being the same question.
        #
        # For TOPOLOGY that is the around-class at offset -0.06, the only offset whose detour
        # goes RIGHT. Leaving it in makes the blocker holdout a 6.1 cm recombination of a shape
        # training already contains; withholding it puts the same holdout at 19 cm against every
        # axis. Filtered here rather than at collection because the episodes are perfectly good
        # -- they are simply not shown to the policy.
        if split == "train":
            keep, withheld = [], 0
            for name in selected:
                a = dict(f["data"][name].attrs)
                try:
                    ps = params_from_attrs(a)
                    ax = str(a.get("axis", axis_name))
                except Exception:
                    keep.append(name); continue
                if is_withheld_from_training(ax, ps):
                    withheld += 1
                else:
                    keep.append(name)
            if withheld:
                print(f"[INFO] withheld {withheld} episode(s) from training: "
                      f"{TRAINING_EXCLUSIONS.get(axis_name, '(see axes.TRAINING_EXCLUSIONS)')}")
                selected = keep

        for demo_name in tqdm(selected, desc="Processing demos"):
            demo_group = f["data"][demo_name]

            # Re-derive the instruction from axes.py rather than trusting the copy the
            # archive recorded, because the spec owns that function and the recorded string is
            # only a snapshot of it. Nothing about a demonstration depends on the wording --
            # the scripted experts route by geometry and never read it -- so re-deriving makes
            # a wording fix cost a re-conversion instead of re-collecting in the simulator.
            #
            # That is what repaired the camera-frame bearing bug: every DIRECTION instruction
            # was 90 degrees from what the head camera showed, and the 630 collected episodes
            # were still perfectly good.
            #
            # v2 writes "instruction" (Condition.to_attrs); v1 archives wrote
            # "language_instruction". Both are kept as the fallback for archives that predate
            # the parameter vector. Falling back to a generic task string would be silent and,
            # for ORDER, fatal: the instruction is the *only* thing separating its two
            # conditions, so losing it collapses the axis into duplicate episodes.
            attrs = dict(demo_group.attrs)
            try:
                lang = instruction_from_attrs(attrs)
                n_derived += 1
            except (KeyError, ValueError) as exc:
                lang = str(attrs.get("instruction",
                                     attrs.get("language_instruction", "")) or "")
                if not recorded_warned:
                    print(f"[WARN] cannot re-derive instruction from attrs ({exc}); "
                          f"falling back to the recorded string")
                    recorded_warned = True
                n_recorded += 1
            lang = str(lang).strip().lower()
            if lang and lang != str(attrs.get("instruction", "")).strip().lower():
                n_changed += 1
            task_str = lang or task_fallback

            valid = process_data(dataset, task_str, demo_group, demo_name,
                                 robot_config_name, cameras)

            if valid:
                # LeRobot's schema has no per-episode slot for a parameter vector, and the
                # sweep analysis needs one: without it a converted dataset cannot say which
                # condition an episode came from or where it sat on the axis.
                episode_conditions.append({
                    "episode_index": now_episode_index,
                    "source_demo": demo_name,
                    "task": task_str,
                    **{k: (v.item() if hasattr(v, "item") else
                           v.decode() if isinstance(v, bytes) else v)
                       for k, v in demo_group.attrs.items()},
                })
                now_episode_index += 1
                dataset.save_episode()
                print(f"Saved episode {now_episode_index} (task: '{task_str}')")

    conditions_path = Path(output_dir) / "meta" / "arb_conditions.jsonl"
    conditions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(conditions_path, "w") as cf:
        for row in episode_conditions:
            cf.write(json.dumps(row, default=str) + "\n")
    print(f"[INFO] wrote {len(episode_conditions)} condition records -> {conditions_path}")
    print(f"[INFO] instructions: {n_derived} re-derived from axes.py, "
          f"{n_recorded} taken from the archive, {n_changed} differ from what was recorded")
    if n_recorded and not n_derived:
        print("[WARN] no instruction was re-derived. The dataset's language is whatever the "
              "archive happened to record, so a spec fix will NOT be reflected here.")

    images_dir = Path(output_dir) / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IsaacLab dataset to LeRobot format")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to dataset HDF5 file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--push_to_hub", action="store_true", help="Push dataset to HuggingFace Hub")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing output dir before converting")
    parser.add_argument("--demo_names", nargs="+", default=None,
                        help="Explicit list of demo names to process (skips success filtering)")
    parser.add_argument("--demo_range", type=int, nargs=2, default=None, metavar=("START", "END"),
                        help="Process only demos[START:END] (0-based indices into sorted demo list)")
    parser.add_argument("--split", type=str, default=None,
                        choices=("train", "test", "all"),
                        help="Export one split. Membership is decided by the axis the archive "
                             "was collected for, not by a flag here, so the dataset cannot "
                             "disagree with the benchmark. 'all' is for the k-shot "
                             "intervention, which deliberately injects held-out conditions.")
    parser.add_argument("--expected-axis", type=str, default=None,
                        help="Refuse to convert unless the archive was collected for this "
                             "axis. Recommended in scripts: converting against the wrong "
                             "axis fails silently otherwise.")

    repo_root = Path(__file__).resolve().parents[2]
    # Datasets land under data/datasets/, on the /data1-backed tree. Nothing here may name a
    # sibling project: the output root is derived from this repo's own root and nowhere else.
    default_root = str(repo_root / "data" / "datasets" / "lerobot")

    parser.add_argument("--repo_id", type=str, default=None, help="Repo ID (auto-generated from path if omitted)")
    parser.add_argument("--root", type=str, default=default_root, help=f"Output root directory (default: {default_root})")

    args = parser.parse_args()

    stats_path = Path(args.dataset_file).parent / "run_statistics.json"
    run_stats = {}
    if stats_path.is_file():
        try:
            run_stats = json.loads(stats_path.read_text())
            print(f"[INFO] Read run_statistics.json from {stats_path}")
        except Exception:
            pass

    task_type = run_stats.get("task_type") or "pick"
    fps = run_stats.get("fps") or args.fps

    # Auto-generate repo_id from path: .../sim_record/<scene>/<task_type>/...
    if args.repo_id is None:
        parts = Path(args.dataset_file).parts
        sim_idx = next((i for i, p in enumerate(parts) if p == "sim_record"), None)
        scene_name = parts[sim_idx + 1] if sim_idx is not None else "unknown"
        repo_id = f"sim_ffw_sg2_{scene_name}_{task_type}"
        if "sim2real" not in Path(args.dataset_file).name:
            repo_id += "_raw_sim"
        if args.split is not None:
            repo_id += f"_{args.split}"
    else:
        repo_id = args.repo_id

    print(f"[INFO] repo_id:   {repo_id}")
    print(f"[INFO] task_type: {task_type}")
    print(f"[INFO] fps:       {fps}")

    convert_isaaclab_to_lerobot(
        repo_id=repo_id,
        dataset_file=args.dataset_file,
        task_fallback=task_type,
        fps=fps,
        push_to_hub=args.push_to_hub,
        overwrite=args.overwrite,
        root=args.root,
        demo_names=args.demo_names,
        demo_range=tuple(args.demo_range) if args.demo_range else None,
        split=args.split,
        expected_axis=args.expected_axis,
    )
