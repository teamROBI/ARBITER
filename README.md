# ARBITER

**Pixels or Words?** — when a policy can demonstrably execute several behaviors, which channel
gets to choose, and can we make language the one that does without giving up the visual
competence that currently does the choosing?

## The question

A GR00T N1.7 checkpoint re-routes around an obstacle **43/48** times when a blocker makes the
detour visible, and **0/42** times when a fluent instruction names the same route. The language
channel is live — a mid-episode switch moves the endpoint 36%, and switching a sub-task ordering
instruction changes which block is picked up 6/6 — it simply carries task identity and nothing
about the path.

That is not a question about reach. It is a question about **arbitration**.

## The four-cell table

Each cell fixes capability (both behaviors are demonstrated, at that condition) and varies which
channel is *right*:

| Cell | Scene | Instruction | Correct authority |
|---|---|---|---|
| `L-AUTH` | both lanes open | names a side | **language** |
| `V-AUTH` | barrier height swept across `h*` | route-silent | **vision** |
| `CONFLICT-VF` | ghost blocker — visible, no collider | names the apparently-sealed lane | **language** |
| `CONFLICT-VT` | solid blocker genuinely seals a lane | names the sealed lane | **vision** (refuse) |

`CONFLICT-VF` is the cell existing benchmarks cannot build: vision is *false*, so a policy that
obeys pixels is wrong. A single global language-authority scale cannot be correct on all four
rows at once — it moves them all the same direction. That is the method thesis.

## Provenance

The Isaac Sim testbed — procedural scene, the `route` primitive, the achievability gate, the
route classifier, the LeRobot conversion path, the GR00T modality config — was **copied** from
TANGO (`tango` @ `v2`), not imported. The two projects ask different questions and must diverge:
TANGO measures how far competence extends past the demonstrations (a radius); ARBITER holds
capability fixed and asks which channel picks. Support distance, generalization radius and
coverage cost are deliberately *not* part of this repo.

Nothing here may resolve into a sibling checkout. Enforced by:

```bash
python scripts/bench/check_self_contained.py --strict
```

Unlike TANGO's version, it scans the worktree rather than `git ls-files` — untracked files are
exactly where a fresh dependency hides.

## Layout

`data/` and `venvs/` are symlinks onto `/data1`; nothing heavy lives on `/home`.

```
arbiter/suites/cells.py   THE SPEC. stdlib-only, single source of truth
arbiter/sim/              scene construction, Isaac motion backend, collection/gate/eval tools
arbiter/collect/          the route primitive, experts, success predicates
arbiter/splits/           HDF5 -> LeRobot conversion and merge
arbiter/policy/gr00t/     modality config and env switches
scripts/bench/            drivers, scoring, verification, hygiene
```

## Tests

The spec is stdlib-only by design, so most tests need nothing but pytest:

```bash
PYTHONPATH=. venvs/spec/bin/python -m pytest \
  arbiter/suites/tests arbiter/collect/tests arbiter/splits/tests \
  arbiter/sim/tools/tests/test_spawn_rotation.py arbiter/sim/env/tests -q
```

`arbiter/sim/tools/tests/test_scene_props.py` needs USD and runs via
`arbiter/sim/tools/usd_python.sh`.

## Status

Phase 0 (pilot / go-no-go) in progress. See the plan for the sequence; the load-bearing early
check is **copy fidelity**: reproducing TANGO's control at 18/18 with LEFT ×18 and its barrier
sweep at 20/20 route-correct, which is what proves the copied scene is the scene the borrowed
checkpoint was trained on.
