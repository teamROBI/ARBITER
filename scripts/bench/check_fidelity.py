#!/usr/bin/env python3
# Copyright 2026 ARBITER Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Decide whether ARBITER's copied scene is the scene TANGO's checkpoints were trained on.

## Why this is not an exact replication, and what that costs

The first version of this gate asserted upstream's numbers exactly -- 36/36 success, no
tolerance -- on the reasoning that a tolerance is tolerance for an unexplained difference. That
was wrong, and the first run showed why: it scored 4/6 at one height, which read as scene drift
and was not.

`episode_seed` hashes the condition key, and `Condition.key()` ends in ``@<object_key>``. ARBITER
names its assets ``arb_*`` where upstream names them ``tango_*``, so **every initial condition is
a different draw** -- measured at up to 0.053 rad (3.06 deg) per joint of home-pose jitter. Worse,
it cannot be fixed by normalising the prefix (which ARBITER now does, for its own
reproducibility): the normalised seed matches neither original, because upstream seeded off a
string this repo deliberately does not contain. Reproducing upstream's exact rollouts would mean
seeding off the sibling project's asset prefix, which is precisely the coupling this repo exists
without. Upstream's stored eval records do not carry their seeds either.

So identical rollouts are unavailable, and the gate has to separate two things the first version
conflated.

## The split

**Geometric, asserted exactly.** Which side of the barrier the arm passed, and which homotopy
class it used, are properties of the *scene*: barrier height and width, prop placement, the
expert's tie-break geometry. A 3-degree difference in start pose does not move a route from
around-left to around-right, nor move where the class flips. A drifted prop dimension or a
mis-set barrier height does. These are the drift-sensitive signals, and they carry no tolerance:

  - every rollout that crosses the barrier plane uses the class the height demands;
  - every ``around`` crossing is on the ``-y`` lane, the expert's tie-break side at offset 0;
  - the class flips from ``over`` to ``around`` exactly at ``h* = 0.08``.

**Motor, reported against a floor.** Whether the object actually reached the target is
jitter-sensitive, and h* is where it is most sensitive because the required route changes there.
Upstream scored 36/36 across 36 different seeds, so a matching scene should stay high -- but a
few lost episodes on a fresh draw are not evidence of drift. The floor below is a judgement call,
stated as one rather than dressed up as a derivation.

A geometric failure stops the project. A motor shortfall says look at the footage.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts" / "bench")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from verify_topology_winding import classify  # noqa: E402

from arbiter.suites import spec as S  # noqa: E402

#: Upstream's control grid: the six around-class heights at 6 episodes each.
EXPECT_CONTROL_N = 36
#: The expert's tie-break lane at barrier offset 0.
EXPECT_LANE = "-y"

#: Success-rate floor, as a fraction. A judgement call, not a derivation: upstream is 100% over
#: 36 seeds, and a scene that matches should not lose more than a handful of episodes to a
#: different draw of home-pose jitter. Set deliberately loose, because the *geometric* checks are
#: what this gate leans on; a motor shortfall above this floor is reported, not fatal.
SUCCESS_FLOOR = 0.80
#: Crossing-rate floor. A rollout that never reaches the barrier plane has made no route choice,
#: so a scene that produced many of them would leave the geometric checks with little to say.
CROSSING_FLOOR = 0.85


def _records(dirpath: Path) -> list[dict]:
    out: list[dict] = []
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
        h = float(r["params"]["barrier_h"])
        off = float(r["params"].get("barrier_offset", 0.0))
        path = r.get("eef_path") or []
        if len(path) < 2:
            rows.append({"h": h, "route": "NO-PATH", "side": None,
                         "success": bool(r.get("success"))})
            continue
        route, detail = classify(path, barrier_h=h, barrier_y=off)
        side = ("-y" if detail["y"] < off else "+y") if route == "around" else None
        rows.append({"h": h, "route": route, "side": side,
                     "success": bool(r.get("success"))})
    return rows


def _report(label: str, rows: list[dict]) -> tuple[list[str], list[str]]:
    """Return (fatal, advisory) messages for one grid."""
    fatal: list[str] = []
    advisory: list[str] = []

    by_h: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        by_h[r["h"]].append(r)

    n_total = len(rows)
    n_ok = sum(r["success"] for r in rows)
    n_crossed = sum(1 for r in rows if r["route"] in ("around", "over"))

    print(f"  {label}: n={n_total} success={n_ok} crossed={n_crossed}")
    for h in sorted(by_h):
        rs = by_h[h]
        want = S.expected_homotopy(h)
        routes = Counter(r["route"] for r in rs)
        lanes = Counter(r["side"] for r in rs if r["side"])
        crossed = [r for r in rs if r["route"] in ("around", "over")]
        wrong_class = [r for r in crossed if r["route"] != want]
        wrong_lane = [r for r in crossed
                      if r["route"] == "around" and r["side"] != EXPECT_LANE]
        mark = "ok " if not (wrong_class or wrong_lane) else "BAD"
        print(f"    h={h:.2f} expect={want:<6s} success={sum(r['success'] for r in rs)}/{len(rs)}"
              f"  routes={dict(routes)} lanes={dict(lanes)}  {mark}")

        # --- geometric: no tolerance ---
        if wrong_class:
            fatal.append(f"{label} h={h}: {len(wrong_class)} crossing(s) used the wrong class "
                         f"(expected {want!r}, saw {dict(Counter(r['route'] for r in wrong_class))}). "
                         f"Route class is a property of the scene, not of the start pose.")
        if wrong_lane:
            fatal.append(f"{label} h={h}: {len(wrong_lane)} around-crossing(s) on the wrong lane "
                         f"(expected {EXPECT_LANE!r}). The tie-break lane is geometric.")

    # --- motor: floors ---
    if n_total:
        sr = n_ok / n_total
        cr = n_crossed / n_total
        if sr < SUCCESS_FLOOR:
            fatal.append(f"{label}: success {n_ok}/{n_total} = {sr:.0%}, below the "
                         f"{SUCCESS_FLOOR:.0%} floor. Upstream is 100% here; a gap this large is "
                         f"more than a different jitter draw.")
        elif n_ok != n_total:
            advisory.append(f"{label}: success {n_ok}/{n_total} = {sr:.0%} against upstream's "
                            f"100%. Within the floor and expected from a fresh draw of "
                            f"home-pose jitter, but worth a look at the failing rollouts.")
        if cr < CROSSING_FLOOR:
            fatal.append(f"{label}: only {n_crossed}/{n_total} = {cr:.0%} of rollouts reached "
                         f"the barrier plane, below the {CROSSING_FLOOR:.0%} floor. With that "
                         f"many non-crossings the geometric checks have little to judge.")
    return fatal, advisory


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True)
    ap.add_argument("--sweep", required=True)
    args = ap.parse_args()

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else REPO_ROOT / q

    fatal: list[str] = []
    advisory: list[str] = []

    control = _records(resolve(args.control))
    if not control:
        # An empty run is a failure, never a pass. Upstream's gate once printed
        # "VERDICT PASS: fully achievable" having tested zero conditions.
        fatal.append(f"control: no records under {args.control} -- an empty run is a failure")
    else:
        rows = _scored(control)
        if len(rows) != EXPECT_CONTROL_N:
            advisory.append(f"control: {len(rows)} rollouts, upstream's grid is "
                            f"{EXPECT_CONTROL_N} (6 heights x 6 episodes)")
        heights = sorted({r["h"] for r in rows})
        expected_h = sorted(h for h in S.barrier_heights()
                            if S.expected_homotopy(h) == "around")
        if heights != expected_h:
            advisory.append(f"control: heights {heights}, upstream's are {expected_h}")
        f, a = _report("control", rows)
        fatal += f
        advisory += a

    sweep = _records(resolve(args.sweep))
    if not sweep:
        fatal.append(f"sweep: no records under {args.sweep} -- an empty run is a failure")
    else:
        rows = _scored(sweep)
        spans = {S.expected_homotopy(r["h"]) for r in rows}
        if spans != {"over", "around"}:
            fatal.append(f"sweep: heights span only {spans}; it must cross "
                         f"h*={S.TOPOLOGY_H_STAR} or it checks no discontinuity")
        f, a = _report("sweep", rows)
        fatal += f
        advisory += a

    print()
    for a in advisory:
        print(f"[NOTE] {a}")
    if advisory:
        print()

    if fatal:
        print(f"[GATE FAIL] {len(fatal)} geometric/floor check(s) failed:")
        for f in fatal:
            print(f"  - {f}")
        print()
        print("A geometric failure means the copied scene is not the scene the checkpoint was")
        print("trained on: a prop dimension, a barrier height, a camera parameter. Render the")
        print("scene and compare -- programmatic checks are blind to composition, and a rotated")
        print("prop or a mis-aimed camera passes every assertion.")
        return 1

    print("[GATE PASS] route class and lane match upstream at every height, the class flips at")
    print("            h*, and success clears the floor. The borrowed checkpoints are valid on")
    print("            this scene. Note this is a distributional match, not an identical-rollout")
    print("            replication -- see the module docstring for why that is unavailable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
