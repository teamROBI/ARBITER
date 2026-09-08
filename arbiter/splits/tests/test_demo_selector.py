#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Round-trip a condition through a real HDF5 archive and back to a split decision.

Needs h5py, which lives in the sim venv rather than system python:

    venvs/sim/bin/python arbiter/splits/tests/test_demo_selector.py

This is the verification the plan calls for: a parameterized condition must survive
layout -> collection -> HDF5 attrs -> split predicate without drifting. The drift this guards
against is silent and total — if the parameter vector does not survive, the converter puts
held-out episodes into the training set and the measurement is destroyed with nothing raising.

The subtle one is type coercion. HDF5 returns numpy scalars and bytes where the axis predicates
were written against Python floats and str, and ``float(np.float32(0.42)) != 0.42`` exactly. A
predicate using ``list.index()`` or strict equality behaves differently on the way back out
than it did at collection time, which is exactly the kind of bug that shows up as "a few
conditions ended up in the wrong split".
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
          "arbiter/splits/tests/test_demo_selector.py", flush=True)
    sys.exit(1)

from arbiter.splits.demo_selector import (  # noqa: E402
    SelectionError,
    episode_split,
    read_episode_params,
    select_demo_names,
    summarize,
    validate_archive,
)
from arbiter.suites.spec import AXES, enumerate_conditions  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  {'ok   ' if cond else 'FAIL '} {label}")
    if not cond:
        FAILURES.append(label)


def write_archive(path: Path, conditions, *, success=None) -> None:
    """Write an archive the way collection will: one group per episode, attrs from to_attrs."""
    with h5py.File(str(path), "w") as f:
        data = f.create_group("data")
        for i, cond in enumerate(conditions):
            g = data.create_group(f"episode_{i}")
            # Minimal arrays so the group resembles a real episode.
            g.create_dataset("actions", data=np.zeros((12, 8), dtype=np.float32))
            g.create_group("obs").create_dataset(
                "joint_pos", data=np.zeros((12, 8), dtype=np.float32))
            for k, v in cond.to_attrs().items():
                g.attrs[k] = v
            if success is not None:
                g.attrs["success"] = bool(success[i])


# ──────────────────────────────────────────────────────────────────────────────

def test_roundtrip_all_axes() -> None:
    print("\n[parameter vector survives the HDF5 round trip]")
    with tempfile.TemporaryDirectory() as td:
        for name, ax in AXES.items():
            conds = enumerate_conditions(ax, ["cube"])[:24]
            p = Path(td) / f"{name}.hdf5"
            write_archive(p, conds)

            mismatched = []
            with h5py.File(str(p), "r") as f:
                for i, cond in enumerate(conds):
                    got = read_episode_params(f["data"][f"episode_{i}"].attrs)
                    if got != cond.params:
                        mismatched.append((cond.key(), got, cond.params))
            check(not mismatched,
                  f"{name}: {len(conds)} conditions round-trip exactly"
                  + (f" — first mismatch {mismatched[0]}" if mismatched else ""))


def test_types_normalised() -> None:
    """HDF5 gives back numpy scalars and bytes; the axis predicates expect float and str."""
    print("\n[types are normalised on the way back out]")
    with tempfile.TemporaryDirectory() as td:
        conds = enumerate_conditions(AXES["order"], ["cube"])   # has a str parameter
        p = Path(td) / "order.hdf5"
        write_archive(p, conds)
        with h5py.File(str(p), "r") as f:
            got = read_episode_params(f["data"]["episode_0"].attrs)
        check(all(isinstance(v, (int, float, str)) for v in got.values()),
              f"no numpy scalars or bytes leak through (got {got})")
        check(isinstance(got.get("order"), str), "a string parameter comes back as str")

    with tempfile.TemporaryDirectory() as td:
        conds = enumerate_conditions(AXES["direction"], ["cube"])
        p = Path(td) / "dir.hdf5"
        write_archive(p, conds)
        with h5py.File(str(p), "r") as f:
            got = read_episode_params(f["data"]["episode_0"].attrs)
        check(isinstance(got.get("theta_deg"), float), "a float parameter comes back as float")


def test_split_agrees_with_axis() -> None:
    """The whole point: the split read back must equal the split the axis assigned."""
    print("\n[split read back == split the axis assigned]")
    with tempfile.TemporaryDirectory() as td:
        for name, ax in AXES.items():
            conds = enumerate_conditions(ax, ["cube"])
            p = Path(td) / f"{name}.hdf5"
            write_archive(p, conds)
            wrong = []
            with h5py.File(str(p), "r") as f:
                for i, cond in enumerate(conds):
                    got = episode_split(f["data"][f"episode_{i}"].attrs)
                    if got != cond.split:
                        wrong.append((cond.key(), got, cond.split))
            check(not wrong,
                  f"{name}: all {len(conds)} splits agree"
                  + (f" — first {wrong[0]}" if wrong else ""))


def test_select_demo_names() -> None:
    print("\n[selection by split]")
    ax = AXES["direction"]
    conds = enumerate_conditions(ax, ["cube"])
    n_train = sum(1 for c in conds if c.split == "train")
    n_test = len(conds) - n_train

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "dir.hdf5"
        write_archive(p, conds)
        check(len(select_demo_names(p, "train")) == n_train,
              f"train selection returns {n_train} episodes")
        check(len(select_demo_names(p, "test")) == n_test,
              f"test selection returns {n_test} episodes")
        check(len(select_demo_names(p, "all")) == len(conds),
              "'all' returns both splits, for the k-shot injection intervention")
        check(select_demo_names(p, "train") == sorted(select_demo_names(p, "train")),
              "selection is sorted, so conversion order is reproducible")
        try:
            select_demo_names(p, "nope")  # type: ignore[arg-type]
            check(False, "an unknown split is rejected")
        except ValueError:
            check(True, "an unknown split is rejected")


def test_failed_episodes_excluded() -> None:
    print("\n[failed episodes are excluded but retained in the archive]")
    ax = AXES["direction"]
    conds = enumerate_conditions(ax, ["cube"], splits=("train",))
    flags = [i % 2 == 0 for i in range(len(conds))]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "dir.hdf5"
        write_archive(p, conds, success=flags)
        kept = select_demo_names(p, "train")
        check(len(kept) == sum(flags), f"only successful episodes selected ({len(kept)})")
        check(len(select_demo_names(p, "train", require_success=False)) == len(conds),
              "require_success=False keeps the failures, for debugging")
        with h5py.File(str(p), "r") as f:
            check(len(f["data"]) == len(conds), "failures remain in the archive")


def test_validate_archive() -> None:
    print("\n[archive validation hard-errors rather than warning]")
    ax = AXES["topology"]
    conds = enumerate_conditions(ax, ["cube"])[:4]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "topo.hdf5"
        write_archive(p, conds)
        check(validate_archive(p) == "topology", "returns the archive's axis")
        check(validate_archive(p, "topology") == "topology", "accepts a matching expected axis")
        try:
            validate_archive(p, "direction")
            check(False, "refuses to convert an archive against the wrong axis")
        except SelectionError as e:
            check("Refusing" in str(e), f"refuses the wrong axis ({str(e)[:60]}...)")

        # A v1-style archive with no axis attribute must be rejected, not silently mishandled.
        v1 = Path(td) / "v1.hdf5"
        with h5py.File(str(v1), "w") as f:
            g = f.create_group("data").create_group("demo_0")
            g.attrs["spawn_area_id"] = "spawn_area_5"
            g.attrs["target_object_key"] = "arb_red_top"
        try:
            validate_archive(v1)
            check(False, "a v1 archive is rejected with a clear message")
        except SelectionError as e:
            check("v1 archive" in str(e), f"a v1 archive is rejected clearly ({str(e)[:70]}...)")

        empty = Path(td) / "empty.hdf5"
        with h5py.File(str(empty), "w") as f:
            f.create_group("data")
        try:
            validate_archive(empty)
            check(False, "an empty archive is rejected")
        except SelectionError:
            check(True, "an empty archive is rejected")

        mixed = Path(td) / "mixed.hdf5"
        write_archive(mixed, enumerate_conditions(AXES["direction"], ["cube"])[:2]
                      + enumerate_conditions(AXES["topology"], ["cube"])[:2])
        try:
            validate_archive(mixed)
            check(False, "an archive mixing two axes is rejected")
        except SelectionError as e:
            check("mixed axes" in str(e), "an archive mixing two axes is rejected")

        try:
            validate_archive(Path(td) / "nope.hdf5")
            check(False, "a missing archive is rejected")
        except SelectionError:
            check(True, "a missing archive is rejected")


def test_summarize() -> None:
    print("\n[summary catches a stale-axis collection]")
    ax = AXES["direction"]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "dir.hdf5"
        conds = enumerate_conditions(ax, ["cube"])
        write_archive(p, conds)
        s = summarize(p)
        check(s["axis"] == "direction", "reports the axis")
        check(s["train"] > 0 and s["test"] > 0, f"counts both splits ({s['train']}/{s['test']})")
        check(s["sweep_coord_train_max"] == 0.0,
              "training episodes sit at sweep coordinate 0 — a nonzero max here would mean "
              "collection ran against a stale axis definition")
        check(s["sweep_coord_test_max"] > 0.0,
              f"test episodes span a real range (max {s['sweep_coord_test_max']:.1f})")
        check(s["n_conditions"] == len({c.key().split('@')[0] for c in conds}),
              f"distinct conditions counted ({s['n_conditions']})")


def main() -> int:
    test_roundtrip_all_axes()
    test_types_normalised()
    test_split_agrees_with_axis()
    test_select_demo_names()
    test_failed_episodes_excluded()
    test_validate_archive()
    test_summarize()

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
