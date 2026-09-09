#!/usr/bin/env python3
# Copyright 2026 ARBITER Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Score arbitration cells: did the requested channel win?

Success rate cannot answer that. A rollout can reach the target having taken the lane nobody
asked for, and on the conflict cells reaching the target is not even the right criterion --
`CONFLICT-VT` asks whether the policy *declined* an infeasible instruction. So compliance is
scored on the behaviour itself: which lateral lane the transport crossed on, or which homotopy
class it used.

Both come from `verify_topology_winding.classify`, the same geometry that validates the expert's
own demonstrations, so expert and policy are judged by one rule. `classify` returns the signed
crossing ``y``, and the side is its sign relative to the barrier centre.

Two things this refuses to do, both of them ways a number here could mislead:

1. **It will not report a side score for a rollout that went over the barrier.** The blocker
   seals a lateral lane and cannot seal the vertical one, so a policy that lifts over the middle
   touches neither prop and has not chosen a lane at all. Those rollouts are counted as
   ``no-lane`` and excluded from the side denominator rather than silently scored as failures --
   ACT does exactly this at every height, and reading it as "took the wrong lane" would invent a
   decision the policy never made.

2. **It will not report a single arm as a result.** Compliance on one arm cannot separate
   obedience from habit, which is the failure mode under study. Where both arms of a paired cell
   are present it reports the difference; where only one is, it says so.

Usage:
    venvs/spec/bin/python scripts/bench/score_arbitration.py --dir data/output/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "scripts" / "bench"))

from verify_topology_winding import classify  # noqa: E402

from arbiter.suites import cells as C  # noqa: E402
from arbiter.suites import spec as S  # noqa: E402


def side_of(crossing_y: float, barrier_offset: float) -> str:
    """Which lateral lane a crossing is on, in the spec's own vocabulary."""
    return "-y" if crossing_y < barrier_offset else "+y"


def score_record(rec: dict, barrier_offset: float) -> dict:
    """Route class and lane for one rollout, or an explanation of why neither is available."""
    path = rec.get("eef_path") or []
    if len(path) < 2:
        return {"route": "NO-PATH", "side": None,
                "note": "no eef_path recorded; the rollout cannot be scored on behaviour"}
    h = float(rec["params"]["barrier_h"])
    route, detail = classify(path, barrier_h=h, barrier_y=barrier_offset)
    side = side_of(detail["y"], barrier_offset) if route == "around" else None
    return {"route": route, "side": side, "detail": detail}


def load(dirpath: Path) -> list[dict]:
    runs = []
    for f in sorted(dirpath.rglob("*.json")):
        try:
            j = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "records" not in j:
            continue
        j["_path"] = f
        runs.append(j)
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory of eval JSONs (searched recursively)")
    ap.add_argument("--offset", type=float, default=None,
                    help="barrier offset, if the runs predate the provenance block")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    runs = load(root)
    if not runs:
        # An empty scan is not a pass. TANGO's gate once printed VERDICT PASS having tested
        # nothing, because the success condition was trivially true at zero conditions.
        print(f"[FAIL] no eval JSONs with records under {root}", file=sys.stderr)
        return 1

    # (cell, arm, omega) -> tallies
    cells_seen: dict[tuple, dict] = defaultdict(
        lambda: {"n": 0, "compliant": 0, "no_lane": 0, "routes": defaultdict(int),
                 "sides": defaultdict(int), "success": 0, "channel": None, "expect": None}
    )
    plain: list[dict] = []

    for j in runs:
        prov = j.get("provenance") or {}
        cell_info = prov.get("cell")
        offset = args.offset
        if offset is None:
            recs = j.get("records") or [{}]
            offset = float(recs[0].get("params", {}).get("barrier_offset", 0.0))

        if not cell_info:
            plain.append(j)
            continue

        key = (cell_info["name"], cell_info["arm"], prov.get("cag_omega", 1.0))
        t = cells_seen[key]
        t["channel"] = prov.get("language_channel")
        t["expect"] = (cell_info["expect_kind"], cell_info["expect_value"])
        requires_around = cell_info.get("requires_around", False)

        for rec in j["records"]:
            sc = score_record(rec, offset)
            t["n"] += 1
            t["routes"][sc["route"]] += 1
            t["success"] += 1 if rec.get("success") else 0
            if sc["side"]:
                t["sides"][sc["side"]] += 1

            kind, want = t["expect"]
            if kind == "route_class":
                t["compliant"] += 1 if sc["route"] == want else 0
            else:
                # A side expectation is only meaningful if a lane was actually taken.
                if requires_around and sc["route"] != "around":
                    t["no_lane"] += 1
                    continue
                t["compliant"] += 1 if sc["side"] == want else 0

    if plain:
        print(f"[NOTE] {len(plain)} run(s) carry no cell provenance and were skipped; "
              f"they predate --cell or were plain axis runs.\n")

    print(f"{'cell':13s} {'arm':13s} {'w':>4s} {'chan':9s} "
          f"{'compliance':>12s} {'no-lane':>8s} {'SR':>8s}  routes")
    print("-" * 104)
    for (name, arm, omega), t in sorted(cells_seen.items()):
        denom = t["n"] - t["no_lane"]
        comp = f"{t['compliant']}/{denom}" if denom else "n/a"
        pct = f" ({t['compliant']/denom:5.1%})" if denom else ""
        routes = dict(sorted(t["routes"].items()))
        print(f"{name:13s} {arm:13s} {omega:4g} {str(t['channel']):9s} "
              f"{comp+pct:>12s} {t['no_lane']:8d} {t['success']}/{t['n']:<6d}  {routes}")

    # Paired contrast: the only form in which a naming cell is a result.
    print()
    for cell_name in C.CELLS:
        cell = C.cell(cell_name)
        naming = [a.name for a in cell.arms if a.route_word is not None]
        if len(naming) < 2:
            continue
        omegas = {k[2] for k in cells_seen if k[0] == cell_name}
        for omega in sorted(omegas):
            have = {a: cells_seen.get((cell_name, a, omega)) for a in naming}
            if not all(have.values()):
                missing = [a for a, v in have.items() if not v]
                print(f"[PARTIAL] {cell_name} (w={omega:g}): arm(s) {missing} not present, so "
                      f"no paired effect. Compliance on one arm cannot separate obedience from "
                      f"habit.")
                continue
            # The paired effect must hold the BEHAVIOUR fixed and vary the instruction:
            #
            #     P(lane = X | instruction says X) - P(lane = X | instruction says not-X)
            #
            # Subtracting the two arms' *compliance* rates instead is wrong, and wrong in the
            # worst direction: each arm is scored against a different target, so a policy that
            # ignores language completely scores 100% on one arm and 0% on the other and the
            # difference prints as +-100% -- a perfect effect, from a total null. Fixing one
            # outcome and asking how its frequency moves with the instruction gives 0% there,
            # which is the truth.
            lo, hi = naming[0], naming[1]
            target = C.cell(cell_name).arm(hi).expect_value
            fracs = {}
            for a in (hi, lo):
                t = have[a]
                d = t["n"] - t["no_lane"]
                fracs[a] = (t["sides"].get(target, 0) / d) if d else float("nan")
            effect = fracs[hi] - fracs[lo]
            print(f"[PAIRED] {cell_name} (w={omega:g}): "
                  f"P(lane={target} | asked for it) = {fracs[hi]:.1%}, "
                  f"P(lane={target} | asked for the other) = {fracs[lo]:.1%}")
            print(f"          causal effect of the instruction on lane choice = {effect:+.1%}"
                  f"{'   <-- language does not steer' if abs(effect) < 0.05 else ''}")
            for a in (hi, lo):
                t = have[a]
                print(f"          {a:<12s} compliance {t['compliant']}/"
                      f"{t['n'] - t['no_lane']}  lanes {dict(t['sides'])}")

    if not cells_seen:
        print("[FAIL] no cell-tagged runs found; nothing to score", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
