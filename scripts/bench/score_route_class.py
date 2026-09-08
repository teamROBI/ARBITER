#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Compute TOPOLOGY's declared mechanism metric, `topology_correct_rate`, from eval rollouts.

The axis has always declared this metric and never computed it. Only the success predicate ran,
and that checks *position*: whether the object reached the target. It cannot see which side of
the barrier the arm went, so "100% at h=0.09" is ambiguous between two very different claims --
the policy selected the AROUND class the expert demonstrates there, or it went over and the
predicate could not tell.

That ambiguity matters more than it sounds, because `h*` is a **convention about the expert**,
not a physical constraint. It was set to 0.08 to balance the sweep, and nothing stops a policy
from lifting over a 0.13 m barrier. So the axis can report a perfect score while never
exercising the discontinuity it exists to test.

Reuses `classify()` from verify_topology_winding.py -- the same geometry that validates the
expert's own demonstrations, so expert and policy are judged by one rule.

    venvs/sim/bin/python scripts/bench/score_route_class.py --dir data/output/eval/groot_v6_rel
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "scripts" / "bench"))
from verify_topology_winding import classify  # noqa: E402

from arbiter.suites.spec import WORKSPACE, expected_homotopy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/output/eval/groot_v6_rel")
    args = ap.parse_args()

    files = sorted(glob.glob(str(REPO_ROOT / args.dir / "topology_*.json")))
    if not files:
        print(f"[ERROR] no topology eval files under {args.dir}")
        return 1

    print(f"  {'h':>5s} {'expected':>9s} {'n':>4s} {'SR':>7s} {'route-correct':>14s}  observed routes")
    n_pathless = 0
    rows = []
    for f in files:
        d = json.load(open(f))
        recs = d["records"]
        if not recs:
            continue
        h = float(recs[0]["params"]["barrier_h"])
        off = float(recs[0]["params"].get("barrier_offset", 0.0))
        want = expected_homotopy(h)
        seen = collections.Counter()
        correct = ok = 0
        for r in recs:
            ok += int(r.get("success", False))
            path = r.get("eef_path")
            if not path:
                n_pathless += 1
                continue
            got, _ = classify(path, h, WORKSPACE.centre[1] + off)
            seen[got] += 1
            correct += int(got == want)
        n = len(recs)
        graded = sum(seen.values())
        rate = f"{100 * correct / graded:.1f}%" if graded else "n/a"
        print(f"  {h:5.2f} {want:>9s} {n:4d} {100 * ok / n:6.1f}% {rate:>14s}  {dict(seen)}")
        rows.append((h, want, ok, n, correct, graded))

    if n_pathless:
        print(f"\n[WARN] {n_pathless} rollouts carried no eef_path and could not be graded.")
        print("       They predate path recording; re-run those evaluations to grade them.")

    graded_rows = [r for r in rows if r[5]]
    if graded_rows:
        tot_ok = sum(r[2] for r in graded_rows)
        tot_n = sum(r[3] for r in graded_rows)
        tot_c = sum(r[4] for r in graded_rows)
        tot_g = sum(r[5] for r in graded_rows)
        print(f"\n  overall: SR {100 * tot_ok / tot_n:.1f}%   "
              f"route-correct {100 * tot_c / tot_g:.1f}%")
        if tot_g and tot_c / tot_g < 0.5 <= tot_ok / tot_n:
            print("\n  [FINDING] The policy reaches the target while using the WRONG route class.")
            print("            The axis is scoring transport, not homotopy selection: going over")
            print("            is physically available at every swept height, so h* constrains")
            print("            only the expert's choice and never the policy's.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
