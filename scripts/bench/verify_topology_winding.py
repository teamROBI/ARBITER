#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Check that TOPOLOGY's expert really takes the homotopy class the condition calls for.

The plan lists this as a gating verification, and it has already caught a real failure: the
around-route once travelled *above* the barrier for several runs while reporting success. That
is a silent semantic collapse -- nothing raises, the success predicate is satisfied, the data
looks clean, and the axis has simply stopped distinguishing its two classes. A success rate
cannot detect it. Only the path can.

Reads the probe trajectories, so it needs no simulator. For every TOPOLOGY condition it finds
where the end-effector crosses the barrier's plane and asks how it got past:

    over   -- it crossed above the barrier top
    around -- it crossed beyond the barrier's lateral edge
    THROUGH -- neither, which is geometrically impossible and means the path is wrong

and compares that against `expected_homotopy(h)`, which is what the barrier asset, the axis
definition and `RouteTask` all derive from.

Usage:
    python scripts/bench/verify_topology_winding.py
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def load_probe(path) -> dict:  # noqa: E402
    """Read a probe/rollout JSON. Inlined from TANGO's metrics.sweep, whose remaining
    contents (discrete Frechet, support distance, radius fits) are deliberately not part of
    ARBITER -- support distance is the axis this project does not use."""
    import json
    with open(path) as f:
        return json.load(f)
from arbiter.suites.spec import (  # noqa: E402
    WORKSPACE,
    barrier_half_width,
    expected_homotopy,
    topology_endpoints,
)

#: The table top is z = 0 in the arm's base frame, which is the frame the probe records, so a
#: barrier of height h has its top at exactly z = h.
CLEARANCE_EPS_M = 0.005

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  {'ok   ' if cond else 'FAIL '} {label}")
    if not cond:
        FAILURES.append(label)


def classify(path: list[list[float]], barrier_h: float, barrier_y: float) -> tuple[str, dict]:
    """How the end-effector got past the barrier plane.

    Only the crossings matter, not the whole path: the arm is above the table at the start and
    end of every episode regardless of class, so a summary over all waypoints would call
    everything "over".
    """
    cx, _ = WORKSPACE.centre
    half_w = barrier_half_width()
    top = barrier_h

    crossings = []
    for a, b in zip(path, path[1:]):
        if (a[0] - cx) * (b[0] - cx) <= 0 and a[0] != b[0]:
            t = (cx - a[0]) / (b[0] - a[0])
            y = a[1] + t * (b[1] - a[1])
            z = a[2] + t * (b[2] - a[2])
            crossings.append((y, z))
    if not crossings:
        return "NO-CROSSING", {}

    # The FIRST crossing is the transport; every episode then places the object and retracts
    # back across the plane high and centred. Ranking crossings by clearance instead picks that
    # retract whenever its height above a low barrier exceeds the transport's lateral margin --
    # which reported 11 of 36 conditions as "over" purely because the barrier was short, while
    # the transport crossing was plainly around it (h=0.08: y beyond the edge by 0.051 m at
    # 0.029 m *below* the top).
    y, z = crossings[0]
    over = z > top + CLEARANCE_EPS_M
    around = abs(y - barrier_y) > half_w + CLEARANCE_EPS_M
    detail = {"y": y, "z": z, "z_above_top": z - top,
              "lateral_beyond_edge": abs(y - barrier_y) - half_w,
              "n_crossings": len(crossings)}
    # Passing the plane *below* the barrier top is only possible around its edge, so that is
    # the decisive test; height alone is not, since an around-route may still be high when the
    # barrier is very short.
    if around:
        return "around", detail
    if over:
        return "over", detail
    return "THROUGH", detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/output/analysis/trajectories")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    files = sorted(glob.glob(str(root / "topology_*.json")))
    if not files:
        print(f"[ERROR] no topology probe files under {root}")
        return 1

    (sx, _), (gx, _) = topology_endpoints()
    print(f"barrier plane x={WORKSPACE.centre[0]:.2f}, half-width {barrier_half_width():.2f} m, "
          f"transport {sx:.2f} -> {gx:.2f}\n")

    rows = []
    for f in files:
        for r in load_probe(f)["records"]:
            if not r.get("success"):
                continue
            h = float(r["params"]["barrier_h"])
            off = float(r["params"].get("barrier_offset", 0.0))
            got, d = classify(r["path"], h, WORKSPACE.centre[1] + off)
            rows.append((h, off, r["split"], expected_homotopy(h), got, d))

    rows.sort(key=lambda x: (x[0], x[1]))
    print(f"{'h':>5s} {'off':>6s} {'split':>6s} {'expected':>9s} {'actual':>9s} "
          f"{'z-top':>8s} {'lat-edge':>9s}")
    for h, off, split, want, got, d in rows:
        mark = " " if want == got else "  <-- MISMATCH"
        za = f"{d.get('z_above_top', float('nan')):+.4f}" if d else "     -"
        la = f"{d.get('lateral_beyond_edge', float('nan')):+.4f}" if d else "     -"
        print(f"{h:5.2f} {off:+6.2f} {split:>6s} {want:>9s} {got:>9s} {za:>8s} {la:>9s}{mark}")

    print()
    n_through = sum(1 for r in rows if r[4] == "THROUGH")
    n_none = sum(1 for r in rows if r[4] == "NO-CROSSING")
    n_match = sum(1 for r in rows if r[3] == r[4])
    check(n_none == 0, f"every path crosses the barrier plane ({n_none} did not)")
    check(n_through == 0,
          f"no path passes THROUGH the barrier ({n_through} did) -- "
          f"geometrically impossible, so a path that reports it is wrong")
    check(n_match == len(rows),
          f"realised homotopy matches the condition: {n_match}/{len(rows)}")

    by_class: dict[str, int] = {}
    for r in rows:
        by_class[r[4]] = by_class.get(r[4], 0) + 1
    print(f"\n  realised classes: {dict(sorted(by_class.items()))}")
    check(len(by_class) >= 2,
          f"both homotopy classes are actually demonstrated: {sorted(by_class)}")

    print()
    if FAILURES:
        print(f"[FAIL] {len(FAILURES)} check(s):")
        for x in FAILURES:
            print("  -", x)
        return 1
    print(f"[PASS] {len(rows)} TOPOLOGY conditions take the class they claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
