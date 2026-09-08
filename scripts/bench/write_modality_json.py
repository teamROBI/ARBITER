#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Emit the `meta/modality.json` GR00T needs, derived from each dataset's own `info.json`.

A LeRobot dataset stores `observation.state` and `action` as flat vectors. GR00T needs to know
which slice is which joint, and which LeRobot video column backs which modality key. That
mapping lives in `meta/modality.json`, which the LeRobot converter does not write.

Derived rather than hand-written, for the usual reason: the slice boundaries and the camera
resolutions are already recorded in `info.json`, and a hand-written copy of them is a second
source of truth that goes wrong silently. If the action ever stops being 8-dim, or a camera is
renamed, this notices instead of emitting a mapping that indexes into the wrong columns —
which would train the model on joint 3 while calling it joint 5.

Usage:
    python scripts/bench/write_modality_json.py                     # every v2 dataset
    python scripts/bench/write_modality_json.py --root <dir> --check # verify, do not write
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Which LeRobot video column feeds which GR00T modality key. GR00T's vision stack expects the
#: primary view to be called `image` and the wrist view `wrist_image`; we record them as
#: cam_head and cam_wrist.
CAMERA_KEYS = {
    "observation.images.cam_head": "image",
    "observation.images.cam_wrist": "wrist_image",
}

DEFAULT_GLOBS = [
    "data/datasets/lerobot/*/",
    "data/datasets/lerobot_merged/*/",
]


def build(info: dict) -> dict:
    """modality.json for one dataset, from its info.json."""
    feats = info["features"]

    def named_slices(field: str) -> dict:
        f = feats[field]
        dim = f["shape"][0]
        names = f.get("names")
        if not names:
            raise ValueError(f"{field} has no `names`; cannot map slices to joints")
        if len(names) != dim:
            raise ValueError(f"{field}: {len(names)} names for {dim} dims")
        # One dim per joint, in the recorded order. Contiguous single-element slices rather
        # than a guessed grouping: LIBERO's gripper spans two state dims and one action dim,
        # and copying that shape here would silently shift every joint after it.
        return {n: {"start": i, "end": i + 1} for i, n in enumerate(names)}

    video = {}
    for col, key in CAMERA_KEYS.items():
        if col in feats:
            video[key] = {"original_key": col}
    if not video:
        raise ValueError(f"no known camera columns; dataset has {sorted(feats)}")

    return {
        "state": named_slices("observation.state"),
        "action": named_slices("action"),
        "video": video,
        # GR00T reads the task string through task_index, as its own LIBERO sample does.
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=None,
                    help="dataset directory; repeatable. Defaults to every v2 dataset.")
    ap.add_argument("--check", action="store_true",
                    help="verify an existing modality.json matches the dataset, do not write")
    args = ap.parse_args()

    roots: list[Path] = []
    if args.root:
        roots = [Path(r) for r in args.root]
    else:
        for g in DEFAULT_GLOBS:
            roots += [Path(p) for p in sorted(glob.glob(str(REPO_ROOT / g)))]
    if not roots:
        print("[ERROR] no datasets found")
        return 1

    bad = 0
    for root in roots:
        if not root.is_absolute():
            root = REPO_ROOT / root
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            print(f"[WARN] {root.name}: no meta/info.json, skipping")
            continue
        try:
            want = build(json.loads(info_path.read_text()))
        except (KeyError, ValueError) as e:
            print(f"[FAIL] {root.name}: {e}")
            bad += 1
            continue

        out = root / "meta" / "modality.json"
        if args.check:
            have = json.loads(out.read_text()) if out.is_file() else None
            if have == want:
                print(f"[ok]   {root.name}")
            else:
                print(f"[FAIL] {root.name}: modality.json "
                      f"{'missing' if have is None else 'does not match info.json'}")
                bad += 1
        else:
            out.write_text(json.dumps(want, indent=4) + "\n")
            n_state = len(want["state"])
            print(f"[ok]   {root.name}: {n_state}-dim state/action, "
                  f"video {sorted(want['video'])}")

    if bad:
        print(f"\n[FAIL] {bad} dataset(s)")
        return 1
    print(f"\n[PASS] {len(roots)} dataset(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
