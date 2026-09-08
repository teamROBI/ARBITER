#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""The recorder must write archives the split machinery can read back.

Needs h5py, which lives in the sim venv rather than system python:

    venvs/sim/bin/python arbiter/collect/tests/test_recorder.py

``test_demo_selector`` proves the *schema* round-trips by writing archives by hand. This proves
the thing that actually collects data writes that same schema — a distinction that matters,
because a hand-written fixture agreeing with a hand-written reader says nothing about the
producer sitting between them.

The specific failures guarded here are all silent ones. A recorder that dropped the parameter
vector, or paired actions with the wrong frame count, or kept the discarded warm-up frames,
would produce an archive that opens cleanly, reports plausible episode counts, and trains a
policy on subtly wrong data.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import h5py
    import numpy as np
except ImportError:
    print("[ERROR] needs h5py. Run: venvs/sim/bin/python "
          "arbiter/collect/tests/test_recorder.py", flush=True)
    sys.exit(1)

from arbiter.collect.recorder import (  # noqa: E402
    EpisodeRecorder,
    archive_summary,
    next_episode_index,
)
from arbiter.suites.spec import (  # noqa: E402
    AXES,
    enumerate_conditions,
    params_from_attrs,
)

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  {'ok   ' if cond else 'FAIL '} {label}")
    if not cond:
        FAILURES.append(label)


def fake_episode(rec: EpisodeRecorder, n: int, *, discard_first: int = 0,
                 img_shape=(4, 6, 3)) -> None:
    """Drive the recorder the way the collector does, warm-up frames included."""
    for i in range(discard_first):
        rec.step(action=[float(i)] * 8, joint_pos=[0.0] * 8,
                 images={k: np.zeros(img_shape, np.uint8) for k in rec.image_keys})
    if discard_first:
        rec.mark_discard_point()
    for i in range(n):
        rec.step(
            action=[float(i)] * 7 + [0.04],
            joint_pos=[float(i) * 0.1] * 7 + [0.04],
            images={k: np.full(img_shape, i % 255, np.uint8) for k in rec.image_keys},
        )


def test_roundtrip() -> None:
    print("\nrecorder -> HDF5 -> back")
    cond = enumerate_conditions(AXES["position"], ["arb_cube_red"], splits=("train",))[0]

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "position.hdf5"
        rec = EpisodeRecorder(image_keys=["cam_head"])
        fake_episode(rec, 12)
        rec.write(path, 0, cond.to_attrs(), success=True)

        with h5py.File(str(path), "r") as f:
            g = f["data"]["episode_0"]
            check(g["actions"].shape == (12, 8), f"actions are (T,8): {g['actions'].shape}")
            check(g["obs"]["joint_pos"].shape == (12, 8),
                  f"state is (T,8) — the 8-dim single-arm convention the plan pins")
            check(g["obs"]["cam_head"].shape == (12, 4, 6, 3),
                  f"images are (T,H,W,3): {g['obs']['cam_head'].shape}")
            check(bool(g.attrs["success"]) is True, "success flag written")
            check(int(g.attrs["n_frames"]) == 12, "n_frames matches the arrays")

            # The whole point: the parameter vector survives.
            recovered = params_from_attrs(dict(g.attrs))
            check(set(recovered) == set(cond.params),
                  f"parameter names survive: {sorted(recovered)}")
            same = all(abs(float(recovered[k]) - float(v)) < 1e-6
                       for k, v in cond.params.items()
                       if isinstance(v, (int, float)))
            check(same, "numeric parameter values survive the HDF5 attr round trip")
            check(str(g.attrs["axis"]) == "position", "axis recorded")
            check(abs(float(g.attrs["sweep_coord"]) - cond.sweep_coord) < 1e-6,
                  "sweep coordinate recorded")


def test_discard() -> None:
    print("\nwarm-up frames are dropped, not recorded")
    cond = enumerate_conditions(AXES["position"], ["arb_cube_red"], splits=("train",))[0]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "a.hdf5"
        rec = EpisodeRecorder(image_keys=["cam_head"])
        fake_episode(rec, 10, discard_first=7)
        check(rec.n_frames == 10, f"n_frames excludes the discarded prefix: {rec.n_frames}")
        rec.write(path, 0, cond.to_attrs(), success=True)
        with h5py.File(str(path), "r") as f:
            g = f["data"]["episode_0"]
            check(g["actions"].shape[0] == 10, "discarded frames absent from the archive")
            # The discarded frames used action[0]=0..6; the kept ones restart at 0 with a
            # 0.04 gripper. If the prefix leaked in, the gripper column would show 0.0 first.
            # float32 does not hold 0.04 exactly, so compare with a tolerance -- the same
            # coercion trap the split round-trip test warns about.
            check(abs(float(g["actions"][0, 7]) - 0.04) < 1e-6,
                  "first kept frame is a real frame, not a warm-up one")


def test_resume_and_summary() -> None:
    print("\nappending, resuming and summarising")
    conds = enumerate_conditions(AXES["position"], ["arb_cube_red"], splits=("train",))[:3]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "b.hdf5"
        check(next_episode_index(path) == 0, "fresh archive starts at 0")

        rec = EpisodeRecorder(image_keys=[])
        for i, c in enumerate(conds):
            rec.reset()
            fake_episode(rec, 5 + i)
            rec.write(path, i, c.to_attrs(), success=(i != 1))

        check(next_episode_index(path) == 3, "resume index is one past the last episode")
        s = archive_summary(path)
        check(s["episodes"] == 3, f"episode count: {s['episodes']}")
        check(s["successful"] == 2, f"failed episodes are kept but marked: {s['successful']}")
        check(s["frames"] == 5 + 6 + 7, f"frame total: {s['frames']}")

        # A second process appending must not overwrite.
        rec.reset()
        fake_episode(rec, 4)
        rec.write(path, next_episode_index(path), conds[0].to_attrs(), success=True)
        check(archive_summary(path)["episodes"] == 4, "append does not clobber earlier episodes")


def test_selector_accepts_recorder_output() -> None:
    print("\nthe split machinery reads what the recorder writes")
    try:
        from arbiter.splits.demo_selector import summarize
    except ImportError as e:
        check(False, f"could not import demo_selector: {e}")
        return

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "position.hdf5"
        rec = EpisodeRecorder(image_keys=[])
        both = (enumerate_conditions(AXES["position"], ["arb_cube_red"], splits=("train",))[:4]
                + enumerate_conditions(AXES["position"], ["arb_cube_red"],
                                       splits=("test",))[:3])
        for i, c in enumerate(both):
            rec.reset()
            fake_episode(rec, 6)
            rec.write(path, i, c.to_attrs(), success=True)

        summary = summarize(path)
        check(summary["train"] + summary["test"] + summary["failed"] == 7,
              f"selector sees every episode: {summary}")
        check(summary["train"] == 4 and summary["test"] == 3,
              f"selector splits recorder output correctly: "
              f"train={summary['train']} test={summary['test']}")


def test_empty_refused() -> None:
    print("\nan empty episode is refused rather than written")
    cond = enumerate_conditions(AXES["position"], ["arb_cube_red"], splits=("train",))[0]
    with tempfile.TemporaryDirectory() as td:
        rec = EpisodeRecorder(image_keys=[])
        try:
            rec.write(Path(td) / "c.hdf5", 0, cond.to_attrs(), success=True)
            check(False, "writing zero frames should raise")
        except ValueError:
            check(True, "writing zero frames raises rather than creating an empty group")


def main() -> int:
    test_roundtrip()
    test_discard()
    test_resume_and_summary()
    test_selector_accepts_recorder_output()
    test_empty_refused()

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
