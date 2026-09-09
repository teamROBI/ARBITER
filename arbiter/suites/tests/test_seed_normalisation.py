# Copyright 2026 ARBITER Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Condition keys must identify a condition, not a project's asset-naming convention.

Written after a real failure. ARBITER's USD assets are named ``arb_*`` where the upstream
project's are ``tango_*``, and `Condition.key()` ends in ``@<object_key>``. Because
`episode_seed` hashes that string, the rename silently redrew every initial condition: all six
seeds at one barrier height changed, and the jittered home pose moved by up to 0.053 rad
(3.06 deg) per joint.

The consequence was worse than the cause. The first copy-fidelity run scored 4/6 where upstream's
stored result is 6/6, which reads as "the copied scene drifted" -- the exact conclusion the gate
exists to draw -- when in fact the scene may be identical and only the starting poses differed.
A fidelity gate that cannot separate those two is not a fidelity gate.

So the prefix is normalised out of the seed, and these tests hold that in place.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arbiter.suites import spec as S  # noqa: E402


def _seed(condition_key: str, repeat: int) -> int:
    """Mirror of `scene.episode_seed`, which cannot be imported without Isaac Lab.

    Duplicated deliberately and narrowly: the point of the test is that the *normalisation*
    the seed depends on behaves, and `test_seed_formula_matches_scene_py` below checks this
    mirror still matches the real implementation by reading its source.
    """
    key = S.normalise_condition_key(condition_key)
    return int.from_bytes(hashlib.sha256(f"{key}#{repeat}".encode()).digest()[:8], "big")


UPSTREAM = "topology[barrier_h=0.08,barrier_offset=0]@tango_cube_red"
OURS = "topology[barrier_h=0.08,barrier_offset=0]@arb_cube_red"


def test_a_rename_does_not_redraw_initial_conditions():
    """The whole point: same condition, different asset prefix, same seeds."""
    for repeat in range(12):
        assert _seed(UPSTREAM, repeat) == _seed(OURS, repeat), (
            f"repeat {repeat} draws a different initial condition under a renamed asset; a "
            f"replication would not be reproducing the run it claims to"
        )


def test_normalisation_strips_only_known_project_prefixes():
    assert S.normalise_condition_key(OURS).endswith("@cube_red")
    assert S.normalise_condition_key(UPSTREAM).endswith("@cube_red")
    # An unprefixed object name must survive intact. Stripping the first token of anything
    # would turn cube_red into red, which is a different (and nonexistent) asset.
    assert S.normalise_condition_key("order[x=1]@cube_red").endswith("@cube_red")
    assert S.normalise_condition_key("x[y=1]@bar_red").endswith("@bar_red")


def test_only_the_object_part_is_touched():
    """Parameters carry the physical identity and must pass through byte-identical."""
    key = "topology[barrier_h=0.08,barrier_offset=-0.06]@arb_cube_red"
    out = S.normalise_condition_key(key)
    assert out.startswith("topology[barrier_h=0.08,barrier_offset=-0.06]@")
    # A prefix appearing inside the parameter block must not be rewritten.
    odd = "axis[note=arb_cube_red]@arb_cube_red"
    assert S.normalise_condition_key(odd) == "axis[note=arb_cube_red]@cube_red"


def test_a_key_without_an_object_is_returned_unchanged():
    assert S.normalise_condition_key("plainkey") == "plainkey"
    assert S.normalise_condition_key("") == ""


def test_distinct_conditions_still_get_distinct_seeds():
    """Normalisation must not collapse conditions that genuinely differ."""
    seeds = {
        _seed(f"topology[barrier_h={h},barrier_offset=0]@arb_cube_red", r)
        for h in (0.08, 0.09, 0.10)
        for r in range(4)
    }
    assert len(seeds) == 12, "normalisation collapsed distinct (condition, repeat) pairs"

    # Different objects at the same condition must also stay distinct.
    a = _seed("order[seq=012]@arb_cube_red", 0)
    b = _seed("order[seq=012]@arb_cube_blue", 0)
    assert a != b


def test_seed_formula_matches_scene_py():
    """Guard the mirror above against the real implementation drifting away from it.

    Read from source rather than imported: `scene.py` pulls in Isaac Lab at module level, and
    this suite must keep running in an environment that has no simulator.
    """
    src = (REPO_ROOT / "arbiter" / "sim" / "env" / "scene.py").read_text()
    body = src[src.index("def episode_seed("):]
    body = body[:body.index("\ndef ", 1)] if "\ndef " in body[1:] else body
    assert "normalise_condition_key(condition_key)" in body, (
        "episode_seed no longer normalises the key; a rename would silently redraw every "
        "initial condition again"
    )
    assert 'sha256' in body and 'h[:8]' in body, (
        "the seed formula changed; update the mirror in this test to match"
    )
