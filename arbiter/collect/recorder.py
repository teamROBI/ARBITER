# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Write scripted-expert episodes to HDF5 in the schema the converter reads back.

One archive per axis, one group per episode, and **the condition's parameter vector on every
episode's attrs** — that last part is the whole reason v2 can express a continuous sweep at
all, and it is what ``arbiter/splits/demo_selector.py`` reads to decide a split. The selector
asks the axis rather than carrying its own predicates, so a dataset cannot disagree with the
benchmark about what "held out" means.

Two properties this file exists to guarantee:

**Only successful episodes advance the queue.** A failed attempt is still written (with
``success=False``) so it can be inspected, but it does not count toward the demonstrations owed
for its condition. Without that, a condition the expert finds hard silently ends up
under-represented and the collected distribution stops matching the declared one — which for a
coverage benchmark corrupts the measurement rather than merely wasting time.

**The recorded action is the commanded joint target, not the achieved state.** Those differ by
the tracking lag, and a policy trained on achieved states learns to predict where the arm
already is rather than where to send it. The distinction is the same one that made
``close_gripper`` and ``move_to`` fail earlier in this project.

Images are gzip-compressed in the archive. At 672x376x3 a single head-camera frame is 758 KB
raw, so a 250-step episode would be ~190 MB and a full axis hundreds of gigabytes. The archive
is an intermediate anyway — LeRobot conversion re-encodes to H.264 — so paying a little CPU
here to keep it on disk is worth it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


class EpisodeRecorder:
    """Accumulate one episode's tensors, then append it to an axis archive.

    Deliberately not streaming: an episode is only known to be a keeper once its success
    predicate has run, and appending frame-by-frame would mean either rewriting the group on
    failure or leaving partial episodes in the file for a reader to filter.
    """

    def __init__(self, image_keys: list[str]) -> None:
        self.image_keys = list(image_keys)
        self.reset()

    def reset(self) -> None:
        self.actions: list[np.ndarray] = []
        self.joint_pos: list[np.ndarray] = []
        self.images: dict[str, list[np.ndarray]] = {k: [] for k in self.image_keys}
        self._discard_from = 0

    # ── capture ──────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray, joint_pos: np.ndarray,
             images: dict[str, np.ndarray] | None = None) -> None:
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.joint_pos.append(np.asarray(joint_pos, dtype=np.float32))
        if images:
            for k, img in images.items():
                if k in self.images:
                    self.images[k].append(np.asarray(img, dtype=np.uint8))

    def mark_discard_point(self) -> None:
        """Drop everything captured so far.

        The expert opens each episode with a discard hold while RTX loads textures; those
        frames show a half-rendered scene and must not reach the dataset. ``TaskContext.hold``
        signals it through ``clear_recording_cache``.
        """
        self._discard_from = len(self.actions)

    @property
    def n_frames(self) -> int:
        return max(0, len(self.actions) - self._discard_from)

    def _slice(self, seq: list) -> list:
        return seq[self._discard_from:]

    # ── write ────────────────────────────────────────────────────────────────

    def write(self, path: Path, index: int, attrs: dict[str, Any], *, success: bool,
              compression: str | None = "gzip", level: int = 4) -> str:
        """Append this episode to ``path`` and return the group name.

        Opened in append mode per episode rather than held open for the run: a collection run
        is hours long, and a process killed with the file handle open loses everything since
        the last flush. Reopening costs milliseconds against a multi-second episode.
        """
        if self.n_frames == 0:
            raise ValueError("no frames recorded; nothing to write")

        path.parent.mkdir(parents=True, exist_ok=True)
        name = f"episode_{index}"
        with h5py.File(str(path), "a") as f:
            data = f.require_group("data")
            if name in data:
                del data[name]
            g = data.create_group(name)

            g.create_dataset("actions", data=np.stack(self._slice(self.actions)))
            obs = g.create_group("obs")
            obs.create_dataset("joint_pos", data=np.stack(self._slice(self.joint_pos)))
            for k, frames in self.images.items():
                fr = self._slice(frames)
                if not fr:
                    continue
                obs.create_dataset(
                    k, data=np.stack(fr),
                    compression=compression, compression_opts=level if compression else None,
                    chunks=(1,) + np.asarray(fr[0]).shape,
                )

            for key, value in attrs.items():
                g.attrs[key] = value
            g.attrs["success"] = bool(success)
            g.attrs["n_frames"] = int(self.n_frames)
        return name


def next_episode_index(path: Path) -> int:
    """Next free index in an archive, so a resumed run appends rather than overwrites."""
    if not path.is_file():
        return 0
    with h5py.File(str(path), "r") as f:
        if "data" not in f:
            return 0
        idx = [int(n.rsplit("_", 1)[1]) for n in f["data"].keys() if n.startswith("episode_")]
        return max(idx) + 1 if idx else 0


def archive_summary(path: Path) -> dict:
    """Counts and frame totals, for checking a run without loading the images."""
    if not path.is_file():
        return {"episodes": 0, "successful": 0, "frames": 0}
    with h5py.File(str(path), "r") as f:
        if "data" not in f:
            return {"episodes": 0, "successful": 0, "frames": 0}
        eps = list(f["data"].keys())
        ok = sum(1 for n in eps if bool(f["data"][n].attrs.get("success", True)))
        frames = sum(int(f["data"][n].attrs.get("n_frames", 0)) for n in eps)
        return {"episodes": len(eps), "successful": ok, "frames": frames,
                "size_mb": round(path.stat().st_size / 1e6, 1)}
