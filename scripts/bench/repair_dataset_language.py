#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Re-derive every converted episode's instruction from ``axes.py``, in place.

Two language bugs shipped into the LeRobot datasets, and both are wording-only:

* DIRECTION's bearing words were in **table** axes, not the head camera's. theta=0 was called
  "right" while it travels toward the bottom of the frame, so every DIRECTION and FACTORIAL
  instruction was rotated 90 degrees away from the policy's own observation.
* ``_obj_phrase`` was a bare underscore-to-space, so ``arb_cube_red`` became
  "arb cube red" -- with the colour, the only thing distinguishing ORDER's two objects,
  buried at the end of a three-token noun phrase.

Nothing about a *demonstration* depends on the wording: the scripted experts route by geometry
and never read the instruction. Re-collecting would be 630 episodes of simulator time and
re-converting ~90 minutes of video re-encode, when the only wrong bytes are strings in
``meta/``. The per-frame parquet stores an integer ``task_index``, so it does not need touching
**provided the rename does not regroup episodes**.

That proviso is the whole risk, and it is checked rather than assumed: the old and new
instructions must induce the *same partition* over episodes. If a wording change merged two
task groups or split one, the stored indices would silently point at the wrong strings, so this
refuses to patch and tells you to re-convert that dataset instead.

    python scripts/bench/repair_dataset_language.py --dry-run
    python scripts/bench/repair_dataset_language.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arbiter.suites.spec import instruction_from_attrs  # noqa: E402

META_FILES = ("tasks.jsonl", "episodes.jsonl", "arb_conditions.jsonl")


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def check(ds: Path) -> tuple[dict[int, str], dict[str, str]] | str:
    """Return (per-episode new instruction, old->new rename) or a reason it cannot be patched."""
    meta = ds / "meta"
    cond_p = meta / "arb_conditions.jsonl"
    if not cond_p.exists():
        return "no meta/arb_conditions.jsonl, so the parameter vector is unavailable"

    new: dict[int, str] = {}
    for rec in _read_jsonl(cond_p):
        attrs = {k: v for k, v in rec.items() if k not in ("episode_index", "source_demo", "task")}
        try:
            new[int(rec["episode_index"])] = instruction_from_attrs(attrs).strip().lower()
        except (KeyError, ValueError) as exc:
            return f"cannot re-derive episode {rec.get('episode_index')}: {exc}"

    old: dict[int, str] = {}
    for rec in _read_jsonl(meta / "episodes.jsonl"):
        tasks = rec.get("tasks") or []
        if len(tasks) != 1:
            return f"episode {rec['episode_index']} has {len(tasks)} tasks, expected exactly 1"
        old[int(rec["episode_index"])] = str(tasks[0]).strip().lower()

    if set(old) != set(new):
        return f"{len(old)} episodes in episodes.jsonl vs {len(new)} in arb_conditions.jsonl"

    # The partition test. Same grouping <=> old->new is a well-defined bijection on the
    # distinct strings: every old string maps to exactly one new string, and no two old
    # strings collide onto the same new one.
    fwd: dict[str, set[str]] = {}
    for ep, o in old.items():
        fwd.setdefault(o, set()).add(new[ep])
    split = {o: v for o, v in fwd.items() if len(v) > 1}
    if split:
        return ("a wording change SPLIT a task group, so stored task_index values would be "
                f"ambiguous: { {k: sorted(v) for k, v in split.items()} }")
    rename = {o: next(iter(v)) for o, v in fwd.items()}
    merged: dict[str, list[str]] = {}
    for o, n in rename.items():
        merged.setdefault(n, []).append(o)
    dup = {n: sorted(os) for n, os in merged.items() if len(os) > 1}
    if dup:
        return f"a wording change MERGED task groups, which changes the task table: {dup}"
    return new, rename


def patch(ds: Path, new: dict[int, str], rename: dict[str, str], backup: bool) -> None:
    meta = ds / "meta"
    if backup:
        bk = meta / "language_backup"
        bk.mkdir(exist_ok=True)
        for f in META_FILES:
            if (meta / f).exists() and not (bk / f).exists():
                shutil.copy2(meta / f, bk / f)

    tasks = _read_jsonl(meta / "tasks.jsonl")
    for row in tasks:
        row["task"] = rename.get(str(row["task"]).strip().lower(), row["task"])
    _write_jsonl(meta / "tasks.jsonl", tasks)

    eps = _read_jsonl(meta / "episodes.jsonl")
    for row in eps:
        row["tasks"] = [new[int(row["episode_index"])]]
    _write_jsonl(meta / "episodes.jsonl", eps)

    conds = _read_jsonl(meta / "arb_conditions.jsonl")
    for row in conds:
        s = new[int(row["episode_index"])]
        row["task"] = s
        if "instruction" in row:
            row["instruction"] = s
    _write_jsonl(meta / "arb_conditions.jsonl", conds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/datasets/lerobot")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    dss = sorted(d for d in root.iterdir() if (d / "meta" / "episodes.jsonl").exists())
    if not dss:
        print(f"[ERROR] no LeRobot datasets under {root}")
        return 1

    ok = bad = unchanged = 0
    for ds in dss:
        res = check(ds)
        if isinstance(res, str):
            print(f"[REFUSE] {ds.name}: {res}")
            bad += 1
            continue
        new, rename = res
        changed = {o: n for o, n in rename.items() if o != n}
        if not changed:
            print(f"[OK    ] {ds.name}: already current ({len(new)} eps)")
            unchanged += 1
            continue
        print(f"[PATCH ] {ds.name}: {len(new)} eps, {len(changed)}/{len(rename)} strings")
        for o, n in sorted(changed.items()):
            print(f"           {o!r}\n        -> {n!r}")
        if not args.dry_run:
            patch(ds, new, rename, backup=not args.no_backup)
        ok += 1

    print(f"\n[INFO] {ok} patched, {unchanged} already current, {bad} refused"
          + ("  (dry run, nothing written)" if args.dry_run else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
