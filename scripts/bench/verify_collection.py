#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Check a collected v2 training set before anything is trained on it.

Needs h5py, which lives in the sim venv:

    venvs/sim/bin/python scripts/bench/verify_collection.py

Every check here targets a failure that leaves a *plausible* dataset behind. A collection run
that silently skipped conditions, or wrote the same trajectory fifteen times, or lost the
parameter vector, still produces an archive that opens cleanly and reports a sensible episode
count. The training run would then succeed and the measurement would be wrong.

What it asserts:

**Every declared training condition was collected, at equal depth.** The queue advances only on
success, so a condition the expert finds hard would otherwise end up with fewer demonstrations
than its neighbours -- and for a benchmark about coverage, thinning out exactly where the task
is hard biases the very thing being measured. Unequal depth is reported as a warning during a
run in progress (the queue round-robins, so counts differ until it finishes) and as a failure
once the axis is complete.

**Repeats of one condition are genuinely different.** The simulator is deterministic, so
without per-episode home-pose jitter every repeat would be byte-identical: fifteen copies of one
trajectory, an inflated episode count, and a demo-diversity statistic reporting variation that
does not exist.

**Seeds are unique.** A reused seed reproduces a trajectory exactly, which is the same failure
arriving by a different route.

**The parameter vector survived.** Without it the split cannot be recovered downstream and
held-out episodes can end up in the training set.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import h5py
    import numpy as np
except ImportError:
    print("[ERROR] needs h5py. Run: venvs/sim/bin/python "
          "scripts/bench/verify_collection.py", flush=True)
    sys.exit(1)

from arbiter.suites.spec import AXES, enumerate_conditions, object_keys_for, params_from_attrs  # noqa: E402

FAILURES: list[str] = []
WARNINGS: list[str] = []


def check(cond: bool, label: str, *, warn_only: bool = False) -> None:
    tag = "ok   " if cond else ("warn " if warn_only else "FAIL ")
    print(f"  {tag} {label}")
    if not cond:
        (WARNINGS if warn_only else FAILURES).append(label)


def declared_train(axis: str) -> int:
    # From the spec, not a hardcoded pair: ORDER now carries its pair as a condition parameter
    # and object_keys_for returns all four cubes, so a literal list here would undercount.
    return len(enumerate_conditions(AXES[axis], object_keys_for(axis), splits=("train",)))


def condition_signature(attrs) -> tuple:
    """Identify a condition from its recorded attrs.

    Compared on the parameter vector rather than a key string: HDF5 returns numpy scalars and
    bytes, so the reconstructed key would not match the collector's spelling character for
    character even when the condition is the same.
    """
    params = params_from_attrs(dict(attrs))
    return (tuple(sorted((k, str(v)) for k, v in params.items())),
            str(attrs.get("object_key", "")))



def check_no_jams(path, group, name, report):
    """Flag episodes where the arm was loaded but the motion stalled.

    The signature of a transport collision: the commanded joint target runs well ahead of the
    achieved joint state (the arm is being resisted) while the arm barely advances. Both of the
    defects found by watching rollouts show up here -- TOPOLOGY's around route ground the cube
    against the barrier corner for ~300 control steps, and neither the success predicate nor the
    achievability gate noticed, because both only ever asked whether the object arrived.

    Checked per episode rather than per axis: a jam that happens in one of fifteen repeats is
    still a demonstration teaching the policy to push through an obstacle.
    """
    import numpy as np

    a = group["actions"][:]
    q = group["obs"]["joint_pos"][:]
    if len(a) < 30:
        return
    err = np.abs(a[:, :7] - q[:, :7]).max(axis=1)
    vel = np.concatenate([[0.0], np.abs(np.diff(q[:, :7], axis=0)).max(axis=1)])
    # Loaded and creeping. Thresholds from the measured cases: a clean transport sits near
    # 0.05 rad of tracking error, the ground one held 0.36 rad for ten seconds.
    bad = (err > 0.15) & (vel < 6e-3)
    runs, start = [], None
    for i, v in enumerate(bad):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= 45:            # 1.5 s at 30 Hz
                runs.append((start, i))
            start = None
    if start is not None and len(bad) - start >= 45:
        runs.append((start, len(bad)))
    for s0, e0 in runs:
        report(False, f"{name}: JAM frames {s0}-{e0} ({(e0 - s0) / 30:.1f}s, "
                      f"peak tracking error {err[s0:e0].max():.3f} rad)")
    if not runs:
        report(True, f"{name}: no jams")

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/collect")
    ap.add_argument("--in-progress", action="store_true",
                    help="a run is still going; report unequal demo counts as a warning")
    ap.add_argument("--expect-demos", type=int, default=0,
                    help="required demonstrations per condition; 0 infers from the data")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    archives = sorted(glob.glob(str(root / "*.hdf5")))
    if not archives:
        print(f"[ERROR] no archives under {root}")
        return 1

    axes: dict[str, dict] = {}
    jams: list[str] = []
    for path in archives:
        with h5py.File(path, "r") as f:
            if "data" not in f:
                continue
            for name in f["data"]:
                check_no_jams(path, f["data"][name], f"{Path(path).stem}/{name}",
                              lambda ok, msg: None if ok else jams.append(msg))
                attrs = f["data"][name].attrs
                ax = str(attrs["axis"])
                d = axes.setdefault(ax, {"n": 0, "ok": 0, "frames": 0, "conds": {},
                                         "splits": {}, "seeds": set(), "no_params": 0})
                d["n"] += 1
                d["frames"] += int(attrs.get("n_frames", 0))
                d["splits"][str(attrs["split"])] = d["splits"].get(str(attrs["split"]), 0) + 1
                if "episode_seed" in attrs:
                    d["seeds"].add(int(attrs["episode_seed"]))
                if not params_from_attrs(dict(attrs)):
                    d["no_params"] += 1
                if bool(attrs.get("success", True)):
                    d["ok"] += 1
                    sig = condition_signature(attrs)
                    d["conds"][sig] = d["conds"].get(sig, 0) + 1

    print(f"\n{'axis':11s} {'eps':>5s} {'ok':>5s} {'train':>6s} {'test':>5s} "
          f"{'conds':>6s} {'declared':>9s} {'demos':>9s} {'frames':>8s}")
    total_eps = total_frames = 0
    for ax in sorted(axes):
        d = axes[ax]
        counts = sorted(d["conds"].values())
        rng = f"{counts[0]}-{counts[-1]}" if counts else "-"
        total_eps += d["n"]
        total_frames += d["frames"]
        print(f"{ax:11s} {d['n']:5d} {d['ok']:5d} {d['splits'].get('train', 0):6d} "
              f"{d['splits'].get('test', 0):5d} {len(d['conds']):6d} "
              f"{declared_train(ax):9d} {rng:>9s} {d['frames']:8d}")
    print(f"{'TOTAL':11s} {total_eps:5d} {'':5s} {'':6s} {'':5s} {'':6s} {'':9s} "
          f"{'':9s} {total_frames:8d}")

    print("\nevery declared training condition was collected")
    for ax in sorted(axes):
        got, want = len(axes[ax]["conds"]), declared_train(ax)
        # TOPOLOGY is collected one (height, offset) per launch, so a partial set of archives
        # legitimately covers a subset of the axis.
        warn = args.in_progress or ax == "topology"
        check(got == want, f"{ax}: {got} of {want} declared train conditions", warn_only=warn)

    print("\nequal demonstration depth per condition")
    for ax in sorted(axes):
        counts = sorted(axes[ax]["conds"].values())
        if not counts:
            continue
        equal = counts[0] == counts[-1]
        msg = f"{ax}: {counts[0]}-{counts[-1]} demos per condition"
        if args.expect_demos and equal:
            equal = counts[0] == args.expect_demos
            msg += f" (expected {args.expect_demos})"
        check(equal, msg, warn_only=args.in_progress)

    print("\nevery episode succeeded")
    for ax in sorted(axes):
        d = axes[ax]
        check(d["ok"] == d["n"], f"{ax}: {d['ok']}/{d['n']}")

    print("\nthe parameter vector survived on every episode")
    for ax in sorted(axes):
        d = axes[ax]
        check(d["no_params"] == 0,
              f"{ax}: {d['no_params']} episodes missing param_* attrs")

    print("\nseeds are unique (a reused seed reproduces a trajectory exactly)")
    for ax in sorted(axes):
        d = axes[ax]
        if d["seeds"]:
            check(len(d["seeds"]) == d["n"],
                  f"{ax}: {len(d['seeds'])} distinct seeds for {d['n']} episodes")

    print("\nrepeats of one condition are genuinely different trajectories")
    for path in archives:
        with h5py.File(path, "r") as f:
            if "data" not in f:
                continue
            groups: dict[tuple, list[str]] = {}
            for name in f["data"]:
                groups.setdefault(condition_signature(f["data"][name].attrs), []).append(name)
            worst = None
            for names in groups.values():
                if len(names) < 2:
                    continue
                base = f["data"][names[0]]["actions"][:]
                for other in names[1:]:
                    arr = f["data"][other]["actions"][:]
                    m = min(len(base), len(arr))
                    diff = float(np.abs(base[:m] - arr[:m]).max())
                    worst = diff if worst is None else min(worst, diff)
            if worst is not None:
                check(worst > 1e-6,
                      f"{Path(path).stem}: smallest difference across repeats "
                      f"= {worst:.5f} rad")

    print()
    if WARNINGS and not FAILURES:
        print(f"[PASS with {len(WARNINGS)} warning(s)]")
        for w in WARNINGS:
            print("  ~", w)
        return 0
    if FAILURES:
        print(f"[FAIL] {len(FAILURES)} check(s):")
        for x in FAILURES:
            print("  -", x)
        for w in WARNINGS:
            print("  ~", w)
        return 1
    print(f"[PASS] {total_eps} episodes, {total_frames} frames verified")
    # Transport jams: the arm loaded and creeping. Reported per episode because a single
    # ground repeat is still a demonstration that teaches pushing through an obstacle.
    print("\nno transport jams (arm resisted while barely advancing)")
    if jams:
        for j in jams[:20]:
            check(False, j)
        if len(jams) > 20:
            print(f"  ... and {len(jams) - 20} more")
    else:
        check(True, f"no jams in {sum(d['n'] for d in axes.values())} episodes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
