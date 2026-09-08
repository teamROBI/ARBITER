#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Write per-phase instructions into a converted dataset's ``meta/episodes.jsonl``.

This is Phase B's data change. The measured deficit is that the text pathway carries no
trajectory information: every training episode had ONE constant string, so GR00T's action-DiT
text cross-attention (4 of its 16 blocks) only ever saw a task label. GR00T already ships the
channel -- ``sub_tasks`` in ``episodes.jsonl``, read by ``create_language_from_meta`` with a
half-action-horizon lead -- so this needs a re-conversion and a retrain, not a re-collection.

Phase text comes from ``axes.sub_instructions_from_attrs`` and frame spans from
``axes.sub_task_spans``, applied to the gripper command recorded in the source archive. Neither
is restated here: the spec owns the wording, so a wording fix costs a re-run of this script
rather than 645 episodes of simulator time -- the same property that let the camera-frame bearing
bug be repaired without recollecting DIRECTION.

Two things it refuses to do quietly, both of which would produce a plausible-looking dataset
that trains the wrong thing:

- **Leave a frame uncovered.** GR00T assigns ``""`` to any frame no sub_task covers, and an
  empty instruction measured 0/6 even with no obstacle present. Coverage is asserted per episode.
- **Mismatch phases to spans.** If the vocabulary and the segmentation disagree, every phase text
  lands on the wrong motion and the retrain looks like the intervention failing.

    python scripts/bench/emit_sub_tasks.py --root data/datasets/lerobot_merged/arbiter_v7 --dry-run
    python scripts/bench/emit_sub_tasks.py --root data/datasets/lerobot_merged/arbiter_v7
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
    sub_instructions_from_attrs,
    sub_task_spans,
)

#: Gripper channel in the recorded 8-dim action (7 arm joints + 1 gripper). Pinned by the
#: action/state convention, which v1 managed to contradict across three documents.
GRIPPER_DIM = 7


def gripper_closed_trace(actions) -> list[bool]:
    """Per-frame "is the commanded gripper closed", from the recorded ACTION not the state.

    The command, deliberately: the rate-limited primitives step a command toward a target and
    recording achieved state instead is a documented failure mode here. A midpoint threshold is
    enough because the command is driven to both rails and the trace is not noisy.
    """
    g = [float(a[GRIPPER_DIM]) for a in actions]
    lo, hi = min(g), max(g)
    if hi - lo < 1e-6:
        raise ValueError("gripper command never moves; cannot segment phases")
    mid = 0.5 * (lo + hi)
    return [v < mid for v in g]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="converted LeRobot dataset directory")
    ap.add_argument("--collect", default="data/collect",
                    help="archive directory the dataset was converted from")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written without touching anything")
    args = ap.parse_args()

    import h5py  # imported late so --help works without the sim venv

    root = Path(args.root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    eps_path = root / "meta" / "episodes.jsonl"
    cond_path = root / "meta" / "arb_conditions.jsonl"
    for p in (eps_path, cond_path):
        if not p.exists():
            print(f"[ERROR] missing {p}")
            return 1

    episodes = [json.loads(l) for l in open(eps_path) if l.strip()]
    conds = {c["episode_index"]: c for c in
             (json.loads(l) for l in open(cond_path) if l.strip())}
    print(f"[INFO] {len(episodes)} episodes, {len(conds)} condition records")

    collect = Path(args.collect)
    if not collect.is_absolute():
        collect = REPO_ROOT / collect

    handles: dict[str, object] = {}
    n_ok = 0
    per_axis: Counter = Counter()
    phase_counts: Counter = Counter()
    problems: list[str] = []

    try:
        for ep in episodes:
            idx = ep["episode_index"]
            c = conds.get(idx)
            if c is None:
                problems.append(f"episode {idx}: no condition record")
                continue
            # arbiter_topology_h0120_o+0000 -> topology_h0120_o+0000.hdf5
            stem = str(c["source_dataset"]).replace("arbiter_", "")
            arch = collect / f"{stem}.hdf5"
            if not arch.exists():
                problems.append(f"episode {idx}: no archive {arch.name}")
                continue
            if stem not in handles:
                handles[stem] = h5py.File(arch, "r")
            grp = handles[stem]["data"][c["source_demo"]]

            attrs = dict(grp.attrs)
            try:
                texts = sub_instructions_from_attrs(attrs)
                closed = gripper_closed_trace(grp["actions"])
                spans = sub_task_spans(closed, len(texts))
            except (KeyError, ValueError) as exc:
                problems.append(f"episode {idx} ({stem}/{c['source_demo']}): {exc}")
                continue

            # Coverage is the invariant that matters; assert it per episode rather than trusting
            # the helper, because an uncovered frame trains on "" and looks like nothing.
            if spans[0][0] != 0 or spans[-1][1] != len(closed):
                problems.append(f"episode {idx}: spans do not tile [0,{len(closed)})")
                continue
            if any(spans[i][1] != spans[i + 1][0] for i in range(len(spans) - 1)):
                problems.append(f"episode {idx}: gap between spans")
                continue

            ep["sub_tasks"] = [{"start": int(a), "end": int(b), "text": t}
                               for (a, b), t in zip(spans, texts)]
            n_ok += 1
            per_axis[str(attrs.get("axis"))] += 1
            phase_counts[len(texts)] += 1
    finally:
        for h in handles.values():
            h.close()

    print(f"[INFO] built sub_tasks for {n_ok}/{len(episodes)} episodes")
    print(f"[INFO] per axis: {dict(sorted(per_axis.items()))}")
    print(f"[INFO] phases per episode: {dict(sorted(phase_counts.items()))}")
    if problems:
        print(f"[ERROR] {len(problems)} episode(s) failed:")
        for p in problems[:10]:
            print(f"          {p}")
        print("[ERROR] refusing to write a partially annotated dataset -- a mix of annotated "
              "and bare episodes trains the text pathway inconsistently and the retrain would "
              "not be interpretable.")
        return 1

    if args.dry_run:
        ex = episodes[0]
        print("\n[DRY RUN] nothing written. First episode would get:")
        for st in ex.get("sub_tasks", []):
            print(f"          [{st['start']:4d},{st['end']:4d})  {st['text']}")
        return 0

    with open(eps_path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    print(f"[OK] wrote {eps_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
