# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Select demonstrations from an HDF5 archive by asking the axis which split they belong to.

This replaces v1's ``subset_filters.py`` for v2 data, and the change is structural rather than
cosmetic. v1 defined its own split predicates over ``(spawn_area_idx, object_key)``:

    "a_tl_br": lambda area, key: area in {1, 2, 3, 4, 13, 14, 15, 16}

Two things were wrong with that. It could only express discrete named areas, so a continuous
sweep was inexpressible. And it was a *second* definition of the split, separate from the one
the evaluation used — nothing structurally prevented the collected training set and the
evaluated ID condition from disagreeing about what "held out" meant.

Here there is one definition. Each episode records the axis it was collected for and its
parameter vector (see :meth:`arbiter.suites.spec.Condition.to_attrs`), and membership is decided
by calling that axis's own ``in_train``. A run cannot drift from the axis definition because it
does not carry a copy of it.

Layering: ``arbiter/suites/spec.py`` is stdlib-only and this module is the one that imports h5py.
The dependency runs selector -> axes, never the reverse — v1 had ``arb_suites.py`` importing
from ``subset_filters``, which dragged h5py into the evaluation path for no reason (CLAUDE.md
records that as a thing to avoid).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import h5py

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arbiter.suites.spec import AXES, Params, axis as get_axis  # noqa: E402

Split = Literal["train", "test", "all"]


class SelectionError(RuntimeError):
    """Raised when an archive cannot be interpreted against an axis definition."""


def read_episode_axis(attrs) -> str:
    """The axis an episode was collected for, from its HDF5 group attrs."""
    raw = attrs.get("axis", "")
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def read_episode_params(attrs) -> Params:
    """Recover the parameter vector written by ``Condition.to_attrs``.

    HDF5 hands back numpy scalars and bytes rather than Python floats and str, so values are
    normalised here. Leaving numpy scalars in place would make the axis predicates' ``isclose``
    and ``index`` calls behave subtly differently from how they behave at collection time.
    """
    params: Params = {}
    for key in attrs:
        name = key.decode() if isinstance(key, bytes) else str(key)
        if not name.startswith("param_"):
            continue
        v = attrs[key]
        if isinstance(v, bytes):
            params[name[6:]] = v.decode()
        elif hasattr(v, "item"):
            params[name[6:]] = v.item()
        else:
            params[name[6:]] = v
    return params


def episode_split(attrs) -> Split:
    """Which split an episode belongs to, decided by its own axis."""
    axis_name = read_episode_axis(attrs)
    if axis_name not in AXES:
        raise SelectionError(
            f"episode records axis '{axis_name}', which is not defined. "
            f"Known axes: {', '.join(sorted(AXES))}"
        )
    return "train" if get_axis(axis_name).in_train(read_episode_params(attrs)) else "test"


def _is_success(attrs) -> bool:
    """Absent means success — only an explicit False excludes an episode.

    Matches the v1 convention. Collection writes ``success=False`` for episodes that failed
    validation and keeps them in the archive for debugging, so they must be filtered here
    rather than assumed gone.
    """
    if "success" not in attrs:
        return True
    return bool(attrs["success"])


def validate_archive(dataset_file: str | Path, expected_axis: str | None = None) -> str:
    """Check an archive is interpretable, and return the axis it was collected for.

    Hard-errors rather than warning, because the failure it guards against is converting an
    archive against the wrong axis — which produces a dataset whose split silently disagrees
    with the benchmark. v1 had the same guard for the same reason.
    """
    path = Path(dataset_file)
    if not path.is_file():
        raise SelectionError(f"archive not found: {path}")

    with h5py.File(str(path), "r") as f:
        if "data" not in f:
            raise SelectionError(f"{path}: no top-level 'data' group")
        names = sorted(f["data"].keys())
        if not names:
            raise SelectionError(f"{path}: 'data' group is empty")

        found = {read_episode_axis(f["data"][n].attrs) for n in names}
        found.discard("")
        if not found:
            raise SelectionError(
                f"{path}: no episode records an 'axis' attribute. This looks like a v1 archive; "
                f"v2 requires the parameter-vector schema (see arbiter/suites/spec.py)."
            )
        if len(found) > 1:
            raise SelectionError(f"{path}: mixed axes in one archive: {sorted(found)}")

        axis_name = found.pop()
        if axis_name not in AXES:
            raise SelectionError(f"{path}: unknown axis '{axis_name}'")
        if expected_axis is not None and axis_name != expected_axis:
            raise SelectionError(
                f"{path} was collected for axis '{axis_name}', not '{expected_axis}'. "
                f"Refusing to convert the wrong axis."
            )
    return axis_name


def select_demo_names(
    dataset_file: str | Path,
    split: Split = "train",
    *,
    require_success: bool = True,
) -> list[str]:
    """Successful episode names in one split, sorted.

    ``split="all"`` keeps both, which is what a coverage-scaling run wants when it deliberately
    injects held-out conditions into training (the k-shot intervention).
    """
    if split not in ("train", "test", "all"):
        raise ValueError(f"split must be train/test/all, got {split!r}")

    selected: list[str] = []
    with h5py.File(str(dataset_file), "r") as f:
        for name in sorted(f["data"].keys()):
            attrs = f["data"][name].attrs
            if require_success and not _is_success(attrs):
                continue
            if split != "all" and episode_split(attrs) != split:
                continue
            selected.append(name)
    return selected


def summarize(dataset_file: str | Path) -> dict:
    """Counts per split, plus the sweep-coordinate range actually present.

    The sweep range matters as a sanity check on a collected archive: a training set is
    supposed to sit at coordinate 0, so a nonzero maximum among train episodes means something
    was collected against a stale axis definition.
    """
    axis_name = validate_archive(dataset_file)
    out = {
        "axis": axis_name,
        "train": 0,
        "test": 0,
        "failed": 0,
        "sweep_coord_train_max": 0.0,
        "sweep_coord_test_max": 0.0,
        "conditions": set(),
    }
    with h5py.File(str(dataset_file), "r") as f:
        for name in sorted(f["data"].keys()):
            attrs = f["data"][name].attrs
            if not _is_success(attrs):
                out["failed"] += 1
                continue
            split = episode_split(attrs)
            out[split] += 1
            coord = float(attrs.get("sweep_coord", 0.0))
            key = f"sweep_coord_{split}_max"
            out[key] = max(out[key], coord)
            params = read_episode_params(attrs)
            out["conditions"].add(tuple(sorted(params.items())))
    out["n_conditions"] = len(out["conditions"])
    del out["conditions"]
    return out
