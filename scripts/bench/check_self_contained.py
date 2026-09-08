#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Fail if ARBITER resolves into a sibling project or a machine-specific path.

Self-containment is a hard constraint: nothing may resolve into ``TANGO``, ``R2-M2``,
``ACS_ROBI`` or any other checkout, at runtime or build time. ARBITER's sim stack was *copied*
from TANGO rather than imported, precisely so the two can diverge; a reference that creeps back
in would silently re-couple them. TANGO learned this the hard way: v1 inherited nine files with
absolute ``/home/jokim/projects/...`` paths that worked on exactly one machine, and venvs
symlinked into ACS_ROBI so ``isaaclab`` imported out of another project's tree.

Unlike TANGO's version, this scans the **worktree**, not ``git ls-files``. TANGO's checker
missed 17 untracked files -- including its entire ACT baseline and most of scripts/bench -- so
its green result was partly vacuous. A check that cannot see the code is worse than no check.

Mentioning a sibling project in prose is fine and often necessary -- the provenance section has
to say where the repo came from. Depending on one is not. Those are hard to separate
automatically, so every surviving occurrence is listed in ``ALLOWED`` with a reason. A new
occurrence anywhere else fails the check, which forces the question to be answered rather than
accumulating silently the way v1's did.

Usage:
    python scripts/bench/check_self_contained.py           # fail only on new references
    python scripts/bench/check_self_contained.py --strict  # also fail on the known baseline
    python scripts/bench/check_self_contained.py --list    # show accepted occurrences too
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: (pattern, human name). Kept separate so the report can say which rule fired.
PATTERNS = [
    (re.compile(r"\bTANGO\b"), "sibling project TANGO"),
    (re.compile(r"R2-M2"), "sibling project R2-M2"),
    (re.compile(r"ACS_ROBI"), "sibling project ACS_ROBI"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+/projects/"), "machine-specific absolute path"),
]

#: Occurrences that are known and accepted, as ``path -> reason``. Prose only: anything here
#: must be a comment, a docstring or documentation, never a value the code resolves.
ALLOWED: dict[str, str] = {
    ".gitignore": "ignore rules",
    "CLAUDE.md": "provenance and history, read by future sessions",
    "README.md": "provenance section",
    "scripts/bench/check_self_contained.py": "this checker names the patterns it searches for",
    "arbiter/collect/constants.py":
        "docstring recording that these constants were copied from TANGO, not imported",
    "arbiter/policy/gr00t/modality_config.py":
        "docstrings recording why the embodiment is 8-dim and why the fork is not patched",
    "arbiter/sim/tools/create_scene.py":
        "docstring explaining why the scene is generated rather than scanned",
    "scripts/bench/verify_topology_winding.py":
        "docstring recording that load_probe was inlined instead of importing TANGO's metrics",
    "scripts/setup/patch_act_state_dim.sh":
        "comments recording the 8-dim-vs-14-dim decision inherited from TANGO",
    "arbiter/suites/cells.py":
        "cites the TANGO measurements each cell exists to follow up -- the ghost-blocker 21/24, "
        "the 20/20 route-correct positive control, the 17-of-18 drive into a sealed lane. The "
        "numbers are the cells' justification and belong beside them",
    "scripts/bench/run_finetune.sh":
        "comment recording that TANGO's retention of every checkpoint is why SAVE_LIMIT "
        "defaults to 3 here",
    "scripts/bench/train_dropout.sh":
        "cites TANGO's empty-instruction 0/6 result as the reason dropout training is needed "
        "before a guidance sweep means anything",
    "scripts/bench/check_language_fidelity.py":
        "explains that a dataset carrying another project's asset prefix renders phrases "
        "through the generic path",
    "scripts/bench/fidelity_gate.sh":
        "the gate exists to reproduce TANGO's v7 results on the copied scene, so it has to name "
        "them; it reads no TANGO path at runtime",
    "scripts/bench/check_fidelity.py":
        "carries the expected v7 control and sweep figures the gate asserts against",
    "scripts/bench/score_arbitration.py":
        "cites TANGO's measured blocker breakdown as the reason no-lane rollouts are excluded "
        "from the side denominator",
    "arbiter/sim/pyproject.toml":
        "comment recording which of TANGO's sim dependencies were deliberately dropped (the "
        "DDS teleop stack) and why, so they are not re-added by reflex",
    "arbiter/sim/setup_sim_env.sh":
        "comment recording that the CycloneDDS from-source build was dropped",
    "arbiter/policy/cag.py":
        "the module warning cites TANGO's empty-instruction 0/6 result, which is the reason the "
        "unconditional branch must be validated before a guidance sweep means anything",
    "arbiter/policy/tests/test_cag.py":
        "asserts that the refusal message carries that same evidence",
    "arbiter/suites/view.py":
        "warns that the camera reproduces the view TANGO's checkpoints were trained on, which "
        "is why the FOV is not a free parameter",
    "arbiter/suites/tests/test_view.py":
        "records the measured 0.222-0.349 m end-effector apex range from TANGO that sets the "
        "blocker height, and why the framing criterion conflicts with it",
    "arbiter/suites/tests/test_cells.py":
        "cites TANGO's empty- and scrambled-instruction results as the reason the grounded-"
        "vocabulary test exists",
}

#: Files carrying a known, accepted cross-project or machine-specific reference that is *not*
#: prose. Empty by construction: ARBITER was built by copying, so it starts clean. An entry here
#: is a debt, and it should be paid rather than grow.
V1_LEGACY: dict[str, str] = {}

#: Live references that are not accepted, with what to do about them. Also empty at the start.
KNOWN_LIVE: dict[str, str] = {}

SKIP_DIRS = {".git", "third_party", "__pycache__", "venvs", "data", "output"}


SCAN_SUFFIXES = {".py", ".sh", ".toml", ".json", ".jsonl", ".md", ".yaml", ".yml", ".cfg", ".txt"}


def source_files() -> list[Path]:
    """Every source file in the worktree, tracked or not.

    Deliberately not ``git ls-files``: untracked files are exactly where a fresh dependency
    hides, and they are still code that runs.
    """
    files = []
    for p in sorted(REPO_ROOT.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(REPO_ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if p.suffix not in SCAN_SUFFIXES:
            continue
        files.append(rel)
    return files


def untracked_count(files: list[Path]) -> int:
    """How many scanned files git does not track -- the blind spot this scan closes."""
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return -1
    tracked = {Path(x) for x in out.splitlines()}
    return sum(1 for f in files if f not in tracked)


def scan(show_all: bool, strict: bool) -> int:
    findings: list[tuple[str, int, str, str]] = []
    for rel in source_files():
        path = REPO_ROOT / rel
        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for pattern, name in PATTERNS:
                if pattern.search(line):
                    # A repo-root absolute path is machine-specific too, but it is a different
                    # and lesser problem than resolving into someone else's checkout.
                    findings.append((str(rel), i, name, line.strip()[:110]))
                    break

    allowed, live, legacy, unclassified = [], [], [], []
    for f in findings:
        rel = f[0]
        if rel in ALLOWED:
            allowed.append(f)
        elif rel in KNOWN_LIVE:
            live.append(f)
        elif rel in V1_LEGACY:
            legacy.append(f)
        else:
            unclassified.append(f)

    if show_all and allowed:
        print(f"Allowed ({len(allowed)} occurrences in {len({a[0] for a in allowed})} files):")
        for rel, ln, name, txt in allowed:
            print(f"  {rel}:{ln}  [{name}]")
        print()

    if live:
        print(f"LIVE cross-project references ({len(live)}):")
        for rel, ln, name, txt in live:
            print(f"  {rel}:{ln}  [{name}]")
            print(f"      {txt}")
        for rel, why in KNOWN_LIVE.items():
            print(f"  -> {rel}: {why}")
        print()

    if unclassified:
        print(f"UNCLASSIFIED references ({len(unclassified)}) -- new since the last audit:")
        for rel, ln, name, txt in unclassified:
            print(f"  {rel}:{ln}  [{name}]")
            print(f"      {txt}")
        print()
        print("Either remove the dependency, or add the file to ALLOWED with a reason if the")
        print("occurrence is prose.")

    if legacy:
        print(f"legacy, pending removal ({len(legacy)} occurrences in "
              f"{len({f[0] for f in legacy})} files):")
        for rel, why in V1_LEGACY.items():
            n = sum(1 for f in legacy if f[0] == rel)
            if n:
                print(f"  {rel} ({n}): {why}")
        print()

    scanned = source_files()
    n_files = len(scanned)
    n_untracked = untracked_count(scanned)
    # The point of the check is to stop *new* references appearing, the way v1 accumulated
    # nine of them unnoticed. The known baseline is reported every run but only fails under
    # --strict, so CI stays green on it and turns red the moment something new shows up.
    fatal = len(unclassified) + (len(live) + len(legacy) if strict else 0)
    if fatal == 0:
        print(f"[PASS] {n_files} source files ({n_untracked} untracked, scanned anyway): "
              f"nothing unclassified. ({len(live)} live, {len(legacy)} legacy known.)")
        return 0
    print(f"[FAIL] {len(unclassified)} unclassified, {len(live)} live, {len(legacy)} legacy "
          f"(scanned {n_files} source files, {n_untracked} of them untracked)")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="also print accepted occurrences")
    ap.add_argument("--strict", action="store_true",
                    help="also fail on the known live and legacy baseline, not just on "
                         "newly-appeared references")
    args = ap.parse_args()
    return scan(args.list, args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
