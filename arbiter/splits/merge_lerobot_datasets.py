#!/usr/bin/env python3
"""Merge multiple LeRobot dataset parts (produced by parallel conversion) into one dataset."""

import argparse
import json
import shutil
import pandas as pd
from pathlib import Path


def load_tasks(meta_dir: Path) -> dict[str, int]:
    """Returns {task_str: local_task_index}."""
    result = {}
    tasks_file = meta_dir / "tasks.jsonl"
    if tasks_file.exists():
        for line in tasks_file.read_text().strip().splitlines():
            if line.strip():
                t = json.loads(line)
                result[t["task"]] = t["task_index"]
    return result


def merge_lerobot_datasets(input_dirs: list[Path], output_dir: Path):
    input_dirs = [Path(d) for d in input_dirs]

    # Validate all parts exist
    for d in input_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Part dataset not found: {d}")

    # Collect all unique tasks → assign global task indices (preserving insertion order)
    all_tasks: dict[str, int] = {}
    for part_dir in input_dirs:
        for task_str in load_tasks(part_dir / "meta"):
            if task_str not in all_tasks:
                all_tasks[task_str] = len(all_tasks)

    # Read features/fps/etc. from first part's info.json
    info = json.loads((input_dirs[0] / "meta" / "info.json").read_text())
    camera_keys = [
        k.removeprefix("observation.images.")
        for k in info["features"]
        if k.startswith("observation.images.")
    ]

    # Create output directory structure
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for cam in camera_keys:
        (output_dir / "videos" / "chunk-000" / f"observation.images.{cam}").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    global_ep_idx = 0
    global_frame_offset = 0
    episode_meta = []
    episode_stats: list[dict] = []
    condition_meta: list[dict] = []

    for part_dir in input_dirs:
        local_tasks = load_tasks(part_dir / "meta")
        # local task_index → global task_index
        task_idx_map = {local_idx: all_tasks[task_str] for task_str, local_idx in local_tasks.items()}

        part_parquets = sorted((part_dir / "data" / "chunk-000").glob("episode_*.parquet"))
        if not part_parquets:
            print(f"[WARN] No episodes found in {part_dir}, skipping")
            continue

        def _by_ep(path: Path) -> dict:
            """Load a jsonl keyed by episode_index. Missing file is not fatal: a v1-era part
            may predate it, and the merge should say so rather than silently drop episodes."""
            if not path.is_file():
                print(f"[WARN] {part_dir.name}: no {path.name}; merged set will be incomplete")
                return {}
            out = {}
            for line in path.read_text().strip().splitlines():
                if not line:
                    continue
                row = json.loads(line)
                out[int(row["episode_index"])] = row
            return out

        part_stats = _by_ep(part_dir / "meta" / "episodes_stats.jsonl")
        part_conds = _by_ep(part_dir / "meta" / "arb_conditions.jsonl")

        for parquet_path in part_parquets:
            local_ep_idx = int(parquet_path.stem.split("_")[-1])

            df = pd.read_parquet(parquet_path)
            ep_length = len(df)

            # Remap indices
            df["episode_index"] = global_ep_idx
            df["index"] = df["index"] - df["index"].min() + global_frame_offset
            df["task_index"] = df["task_index"].map(task_idx_map)

            out_parquet = output_dir / "data" / "chunk-000" / f"episode_{global_ep_idx:06d}.parquet"
            df.to_parquet(out_parquet, index=False)

            # Copy video files
            for cam in camera_keys:
                src = part_dir / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{local_ep_idx:06d}.mp4"
                dst = output_dir / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{global_ep_idx:06d}.mp4"
                if src.exists():
                    shutil.copy2(src, dst)
                else:
                    print(f"[WARN] Missing video: {src}")

            task_indices = df["task_index"].unique().tolist()
            ep_tasks = [t for t, idx in all_tasks.items() if idx in task_indices]
            episode_meta.append({"episode_index": global_ep_idx, "tasks": ep_tasks, "length": ep_length})

            # LeRobot v2.1 requires episodes_stats.jsonl. Without it LeRobotDataset stops
            # treating the directory as a local dataset and goes to the Hub for version refs,
            # which 404s on a repo id that only exists on disk -- so a merged set that looked
            # complete could not be opened at all.
            st = part_stats.get(local_ep_idx)
            if st is not None:
                st = dict(st)
                st["episode_index"] = global_ep_idx
                episode_stats.append(st)

            # The condition sidecar is what carries each episode's parameter vector; LeRobot has
            # no slot for it and the sweep analysis needs it, so it has to be remapped too.
            cond = part_conds.get(local_ep_idx)
            if cond is not None:
                cond = dict(cond)
                cond["episode_index"] = global_ep_idx
                cond["source_dataset"] = part_dir.name
                condition_meta.append(cond)

            global_frame_offset += ep_length
            global_ep_idx += 1

        print(f"[INFO] Merged {len(part_parquets)} episodes from {part_dir.name}")

    # Write tasks.jsonl
    tasks_lines = [
        json.dumps({"task_index": idx, "task": task})
        for task, idx in sorted(all_tasks.items(), key=lambda x: x[1])
    ]
    (output_dir / "meta" / "tasks.jsonl").write_text("\n".join(tasks_lines) + "\n")

    # Write episodes.jsonl
    ep_lines = [json.dumps(ep) for ep in episode_meta]
    (output_dir / "meta" / "episodes.jsonl").write_text("\n".join(ep_lines) + "\n")

    # Write episodes_stats.jsonl -- required by LeRobot v2.1
    if episode_stats:
        (output_dir / "meta" / "episodes_stats.jsonl").write_text(
            "\n".join(json.dumps(e) for e in episode_stats) + "\n")
        if len(episode_stats) != global_ep_idx:
            print(f"[WARN] {len(episode_stats)} episode stats for {global_ep_idx} episodes")
    else:
        print("[WARN] no episodes_stats.jsonl written; LeRobot v2.1 will not open this dataset")

    # Write arb_conditions.jsonl -- the parameter vector per episode
    if condition_meta:
        (output_dir / "meta" / "arb_conditions.jsonl").write_text(
            "\n".join(json.dumps(c, default=str) for c in condition_meta) + "\n")
        print(f"[INFO] carried {len(condition_meta)} condition records into the merged set")

    # Write info.json
    info["total_episodes"] = global_ep_idx
    info["total_frames"] = global_frame_offset
    info["total_tasks"] = len(all_tasks)
    info["total_videos"] = global_ep_idx * len(camera_keys)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{global_ep_idx}"}
    (output_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    print(f"[INFO] Done: {global_ep_idx} episodes, {global_frame_offset} frames → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge parallel LeRobot dataset parts into one")
    parser.add_argument("--input_dirs", nargs="+", required=True, help="Part dataset directories (in order)")
    parser.add_argument("--output_dir", required=True, help="Output merged dataset directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        print(f"[INFO] Removing existing output: {output_dir}")
        shutil.rmtree(output_dir)

    merge_lerobot_datasets([Path(d) for d in args.input_dirs], output_dir)
