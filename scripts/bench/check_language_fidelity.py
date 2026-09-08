#!/usr/bin/env python3
# Copyright 2026 ARBITER Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Does ARBITER's spec reproduce a dataset's own language, byte for byte?

The companion to `check_fidelity.py`, and the half that needs no GPU. The visual gate asks
whether the copied *scene* is the scene a checkpoint was trained on; this asks whether the copied
*spec* still derives the language that checkpoint was trained with.

It matters because ARBITER's instruction machinery is used two ways at once. At conversion time
it writes the dataset's language; at evaluation time `cells.py` builds spoken phrases from the
same functions and asserts they are in the training vocabulary. If the derivation drifted, the
second use would silently speak sentences the first never produced -- and the symptom would be a
null result that looks like "language does not steer the route".

It is also the check that catches an asset-naming mismatch. `_obj_phrase` strips the project
prefix, so a dataset collected under a different prefix falls through the curated dictionary into
the generic path: ``tango_cube_red`` renders as *"red tango cube"*, which no annotator would
write and which nothing at inference reproduces. That happens silently, and every re-derived
instruction in the dataset would carry it.

Exit 0 only if every episode matches at both levels.

Usage:
    python scripts/bench/check_language_fidelity.py \\
        --dataset data/datasets/lerobot_merged/arbiter_v8_subtask
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arbiter.suites.spec import (  # noqa: E402
    instruction_from_attrs,
    sub_instructions_from_attrs,
)

#: The condition sidecar. LeRobot has no per-episode parameter-vector slot, so the axis and its
#: parameters are carried alongside; without it the language cannot be re-derived at all.
SIDECAR = "arb_conditions.jsonl"
LEGACY_SIDECAR = "tango_conditions.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="a merged LeRobot dataset directory")
    ap.add_argument("--max-report", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.dataset)
    if not root.is_absolute():
        root = REPO_ROOT / root
    meta = root / "meta"

    sidecar = meta / SIDECAR
    if not sidecar.is_file():
        legacy = meta / LEGACY_SIDECAR
        if legacy.is_file():
            print(f"[FAIL] found {LEGACY_SIDECAR} but not {SIDECAR}.\n"
                  f"       This dataset was collected under another project's asset prefix. Its "
                  f"object keys will fall through the curated phrase dictionary and render as "
                  f"e.g. 'red tango cube'. Rename the sidecar and rewrite the prefixed values "
                  f"before using it -- the task strings must be left alone, since they define "
                  f"the dataset's task-index partition.", file=sys.stderr)
            return 1
        print(f"[FAIL] no condition sidecar at {sidecar}", file=sys.stderr)
        return 1

    rows = [json.loads(l) for l in sidecar.read_text().splitlines() if l.strip()]
    if not rows:
        # An empty scan is a failure, not a pass.
        print(f"[FAIL] {sidecar} is empty -- nothing was checked", file=sys.stderr)
        return 1

    episodes_path = meta / "episodes.jsonl"
    eps: dict[int, dict] = {}
    if episodes_path.is_file():
        for line in episodes_path.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                eps[e["episode_index"]] = e

    task_ok = 0
    sub_ok = 0
    sub_checked = 0
    problems: list[str] = []
    objects: Counter = Counter()

    for r in rows:
        objects[r.get("object_key")] += 1

        try:
            derived = instruction_from_attrs(r)
        except Exception as e:  # the spec raises rather than guessing, which is the point
            problems.append(f"task  axis={r.get('axis')} ep={r.get('episode_index')}: {e}")
            continue
        if derived == r.get("task"):
            task_ok += 1
        else:
            problems.append(f"task  axis={r.get('axis')} ep={r.get('episode_index')}: "
                            f"derived {derived!r} != recorded {r.get('task')!r}")

        recorded = [s["text"] for s in (eps.get(r.get("episode_index"), {}).get("sub_tasks") or [])]
        if not recorded:
            continue
        sub_checked += 1
        try:
            derived_sub = sub_instructions_from_attrs(r)
        except Exception as e:
            problems.append(f"phase axis={r.get('axis')} ep={r.get('episode_index')}: {e}")
            continue
        if derived_sub == recorded:
            sub_ok += 1
        else:
            problems.append(f"phase axis={r.get('axis')} ep={r.get('episode_index')}: "
                            f"derived {derived_sub} != recorded {recorded}")

    n = len(rows)
    print(f"  dataset     : {root.name}")
    print(f"  objects     : {dict(objects)}")
    print(f"  task-level  : {task_ok}/{n} re-derive identically")
    if sub_checked:
        print(f"  phase-level : {sub_ok}/{sub_checked} re-derive identically")
    else:
        print("  phase-level : no sub_tasks recorded (task-channel dataset)")

    print()
    if problems:
        print(f"[FAIL] {len(problems)} mismatch(es); first {args.max_report}:")
        for p in problems[:args.max_report]:
            print(f"  - {p}")
        print()
        print("The spec no longer derives the language this dataset was built with. A checkpoint")
        print("trained on it cannot be spoken to using cells.py phrasing: the sentences would be")
        print("out of vocabulary, and the null that follows would look like a finding.")
        return 1

    print(f"[LANG PASS] all {n} episodes re-derive identically at every recorded level.")
    print("            The spec and this dataset agree; cells.py phrasing is in-vocabulary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
