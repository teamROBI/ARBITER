#!/usr/bin/env python3
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Every prop spawned into the scene must carry the Y-up -> Z-up spawn rotation.

The asset library is authored Y-up and **nothing converts it automatically**: each spawn passes
``rot=YUP_TO_ZUP`` by hand. Omit it and the authored height stays world Y while the authored
width stays world Z, so a flat 30x14 cm floor lane spawns as a 4 mm wide, 14 cm tall vertical
fin.

That failure is nasty because of how it *looks*. It renders as a hairline, which reads as "this
marking is too faint" rather than "this marking is standing on edge" -- so it survived two
colour changes before anyone measured the geometry. It reached collected data: a stray vertical
hairline sat in every episode of the five axes that carry a start marking.

Checked by parsing the source rather than by building a scene, so it runs in the stdlib with no
Isaac, no GPU and no simulator launch -- which is what makes it cheap enough to run every time.
The existing prop test checks the *assets*; this checks how they are *placed*.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCENE = REPO_ROOT / "arbiter" / "sim" / "env" / "scene.py"

FAILURES: list[str] = []
SPAWNING_CFGS = {"AssetBaseCfg", "RigidObjectCfg"}
#: Prims that are deliberately not asset-file spawns and so need no conversion.
EXEMPT_PREFIXES = ("cam_", "Camera", "light", "Light", "dome", "sun", "ground", "terrain")


def check(cond: bool, label: str) -> None:
    print(f"  {'ok   ' if cond else 'FAIL '} {label}")
    if not cond:
        FAILURES.append(label)


def _init_state_call(node: ast.Call) -> ast.Call | None:
    for kw in node.keywords:
        if kw.arg == "init_state" and isinstance(kw.value, ast.Call):
            return kw.value
    return None


def _has_yup_rot(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "rot":
            return isinstance(kw.value, ast.Name) and kw.value.id == "YUP_TO_ZUP"
    return False


def _spawns_usd_file(node: ast.Call) -> bool:
    """Only asset-file spawns need the conversion; procedural shapes are authored Z-up."""
    for kw in node.keywords:
        if kw.arg == "spawn" and isinstance(kw.value, ast.Call):
            fn = kw.value.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", "")
            return name == "UsdFileCfg"
    return False


def main() -> int:
    src = SCENE.read_text()
    tree = ast.parse(src)

    spawns = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "attr", None) or getattr(fn, "id", "")
        if name in SPAWNING_CFGS and _spawns_usd_file(node):
            spawns.append(node)

    print(f"\n[spawn rotation — {SCENE.name}]")
    check(len(spawns) >= 5,
          f"found {len(spawns)} USD-file prop spawns to check (expected at least 5)")

    for node in spawns:
        prim = "?"
        for kw in node.keywords:
            if kw.arg == "prim_path":
                prim = ast.unparse(kw.value)[:60]
        if any(e in prim for e in EXEMPT_PREFIXES):
            continue
        init = _init_state_call(node)
        check(init is not None, f"line {node.lineno}: {prim} declares an init_state")
        if init is None:
            continue
        check(_has_yup_rot(init),
              f"line {node.lineno}: {prim} passes rot=YUP_TO_ZUP")

    print("\n[the constant itself]")
    yup = [n for n in ast.walk(tree)
           if isinstance(n, ast.Assign)
           and any(isinstance(t, ast.Name) and t.id == "YUP_TO_ZUP" for t in n.targets)]
    check(len(yup) == 1, "YUP_TO_ZUP is defined exactly once")
    if yup:
        vals = [v.value for v in yup[0].value.elts]  # type: ignore[attr-defined]
        w = vals[0]
        check(abs(w - 0.70710678) < 1e-6,
              f"YUP_TO_ZUP is a 90-degree rotation about X (w={w})")

    if FAILURES:
        print(f"\n[FAIL] {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\n[PASS] all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
