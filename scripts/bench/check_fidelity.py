#!/usr/bin/env python3
# Copyright 2026 ARBITER Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Decide whether ARBITER's copied scene is the scene TANGO's checkpoints were trained on.

Two reproductions, both measured off TANGO's stored eval JSONs rather than quoted from prose:

1. **Control** -- no blocker, the six around-class heights (0.08-0.13), 6 episodes each:
   36/36 success, every rollout route class ``around``, every one on the ``-y`` lane. This is the
   habitual behaviour, and it is the most sensitive single check available: it exercises the
   barrier prop, the camera, the expert's tie-break geometry and the policy's route selection at
   once, and it has no slack -- TANGO's number is 100%.

2. **Sweep** -- across ``h*``: 100% route-correct at every height, with the class flipping from
   ``over`` to ``around`` exactly at ``h* = 0.08``. This checks the barrier *heights* are what
   the stage thinks they are; a systematic offset would move the flip.

A miss means a prop dimension, camera parameter, lighting intensity or spawn rotation drifted in
the copy, and every subsequent number would measure the drift rather than the policy.

Deliberately strict. TANGO's own numbers are 100% on both, so a tolerance here would be
tolerance for an unexplained difference -- and the whole purpose of the gate is that an
unexplained difference stops the project rather than propagating into it.

Usage:
    python scripts/bench/check_fidelity.py --control DIR --sweep DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts" / "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)

from verify_topology_winding import classify  # noqa: E402

from arbiter.suites import spec as S  # noqa: E402

#: TANGO's v7 control: the six around-class heights at 6 episodes each.
EXPECT_CONTROL_N = 36
EXPECT_CONTROL_SIDE = "-y"     # the expert's tie-break lane at offset 0
EXPECT_CONTROL_ROUTE = "around"


def _records(dirpath: Path) -> list[dict]:
    out = []
    for f in sorted(dirpath.rglob("*.json")):
        try:
            j = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.extend(j.get("records") or [])
    return out


def _scored(recs: list[dict]) -> list[dict]:
    rows = []
    for r in recs:
        path = r.get("eef_path") or []
        h = float(r["params"]["barrier_h"])
        off = float(r["params"].get("barrier_offset", 0.0))
        if len(path) < 2:
            rows.append({"h": h, "route": "NO-PATH", "side": None,
                         "success": bool(r.get("success"))})
            continue
        route, detail = classify(path, barrier_h=h, barrier_y=off)
        side = ("-y" if detail["y"] < off else "+y") if route == "around" else None
        rows.append({"h": h, "route": route, "side": side,
                     "success": bool(r.get("success"))})
    return rows


def check_control(dirpath: Path) -> list[str]:
    recs = _records(dirpath)
    if not recs:
        # An empty run is a FAIL, never a pass. TANGO's gate once printed
        # "VERDICT PASS: fully achievable" having tested zero conditions.
        return [f"control: no records under {dirpath} -- an empty run is a failure, not a pass"]

    rows = _scored(recs)
    fails = []
    n = len(rows)
    if n != EXPECT_CONTROL_N:
        fails.append(f"control: {n} rollouts, expected {EXPECT_CONTROL_N} "
                     f"(6 around-class heights x 6 episodes)")

    n_ok = sum(r["success"] for r in rows)
    if n_ok != n:
        fails.append(f"control: {n_ok}/{n} success, expected all -- TANGO's v7 control is 36/36")

    routes = Counter(r["route"] for r in rows)
    if set(routes) != {EXPECT_CONTROL_ROUTE}:
        fails.append(f"control: route classes {dict(routes)}, expected all "
                     f"{EXPECT_CONTROL_ROUTE!r}")

    sides = Counter(r["side"] for r in rows if r["side"])
    if set(sides) != {EXPECT_CONTROL_SIDE}:
        fails.append(f"control: lanes {dict(sides)}, expected all {EXPECT_CONTROL_SIDE!r} "
                     f"(the expert's tie-break lane at offset 0)")

    heights = sorted({r["h"] for r in rows})
    expected_h = sorted(h for h in S.barrier_heights() if S.expected_homotopy(h) == "around")
    if heights != expected_h:
        fails.append(f"control: heights {heights}, expected {expected_h}")

    print(f"  control : n={n} success={n_ok} routes={dict(routes)} lanes={dict(sides)}")
    return fails


def check_sweep(dirpath: Path) -> list[str]:
    recs = _records(dirpath)
    if not recs:
        return [f"sweep: no records under {dirpath} -- an empty run is a failure, not a pass"]

    rows = _scored(recs)
    fails = []
    by_h: dict[float, Counter] = {}
    for r in rows:
        by_h.setdefault(r["h"], Counter())[r["route"]] += 1

    for h in sorted(by_h):
        want = S.expected_homotopy(h)
        got = by_h[h]
        n = sum(got.values())
        correct = got.get(want, 0)
        flag = "ok " if correct == n else "BAD"
        print(f"  sweep   : h={h:.2f} expect={want:<6s} {correct}/{n} correct  {dict(got)}  {flag}")
        if correct != n:
            fails.append(f"sweep: h={h} expected all {want!r}, got {dict(got)}")

    spans = {S.expected_homotopy(h) for h in by_h}
    if spans != {"over", "around"}:
        fails.append(f"sweep: heights span only {spans}; the sweep must cross h*="
                     f"{S.TOPOLOGY_H_STAR} or it checks no discontinuity")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True)
    ap.add_argument("--sweep", required=True)
    args = ap.parse_args()

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else REPO_ROOT / q

    fails = check_control(resolve(args.control)) + check_sweep(resolve(args.sweep))

    print()
    if fails:
        print(f"[GATE FAIL] {len(fails)} check(s) did not reproduce TANGO's v7 results:")
        for f in fails:
            print(f"  - {f}")
        print()
        print("The copied scene is not the scene the checkpoint was trained on. Do not run the")
        print("arbitration cells against it: every number would measure the drift. Render the")
        print("scene and compare against TANGO's -- programmatic checks are blind to")
        print("composition, and a rotated prop or a mis-aimed camera passes every assertion.")
        return 1

    print("[GATE PASS] the copied scene reproduces TANGO's v7 control and height sweep.")
    print("            The borrowed checkpoints are valid on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
