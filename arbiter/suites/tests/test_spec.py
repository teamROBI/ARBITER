# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Invariants for the v2 condition schema.

Runs with plain python — no Isaac, no h5py, no pytest required:

    python arbiter/suites/tests/test_axes.py

The point of these is that the schema's guarantees are checked, not documented. Two of them
are v1 regressions we must not repeat: train/test membership must be derived from one place
(v1 could let the eval split and the data split drift), and the parameter vector must survive
the HDF5 round trip (v1 only stored a discrete area id, which is what made continuous sweeps
impossible).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arbiter.suites.spec import (  # noqa: E402
    APPROACH,
    APPROACH_TRAIN_YAW,
    approach_phi,
    AXES,
    DIRECTION,
    DIRECTION_TRAIN_THETA,
    METRIC_AXES,
    POSITION,
    POSITION_TRAIN_CELLS,
    COMPOSITION_AXES,
    STRUCTURAL_AXES,
    TOPOLOGY,
    TOPOLOGY_H_STAR,
    TOPOLOGY_TRAIN_H,
    WORKSPACE,
    Condition,
    axis,
    enumerate_conditions,
    expected_homotopy,
    params_from_attrs,
    sample_sweep,
)

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}")
        FAILURES.append(label)


# ──────────────────────────────────────────────────────────────────────────────

def test_registry() -> None:
    print("\n[registry]")
    check(len(AXES) == 7, "7 axes registered")
    check(len(METRIC_AXES) == 2, f"2 metric axes (got {METRIC_AXES})")
    check(len(STRUCTURAL_AXES) == 4, f"4 structural axes (got {STRUCTURAL_AXES})")
    # FACTORIAL is its own class. It asks whether coverage *multiplies* across axes, which is a
    # different claim from "invent a shape" and falsifies in the opposite direction: FACTORIAL
    # succeeding weakens the paradigm argument, a structural axis succeeding strengthens the
    # policy's case. Typing them alike conflated the two.
    check(len(COMPOSITION_AXES) == 1, f"1 composition axis (got {COMPOSITION_AXES})")
    check(len(METRIC_AXES) + len(STRUCTURAL_AXES) + len(COMPOSITION_AXES) == len(AXES),
          "every axis is classified exactly once")
    check(all(a.name == n for n, a in AXES.items()), "registry keys match axis names")
    check(all(a.mechanism_metric for a in AXES.values()), "every axis names a mechanism metric")
    try:
        axis("nope")
        check(False, "unknown axis raises KeyError")
    except KeyError:
        check(True, "unknown axis raises KeyError")


def test_grids_nonempty_and_split() -> None:
    print("\n[grids]")
    for name, a in AXES.items():
        grid = list(a.grid())
        n_train = sum(1 for p in grid if a.in_train(p))
        n_test = len(grid) - n_train
        check(len(grid) > 0, f"{name}: grid non-empty ({len(grid)})")
        check(n_train > 0, f"{name}: has training conditions ({n_train})")
        check(n_test > 0, f"{name}: has test conditions ({n_test})")


def test_sweep_coord_zero_on_train() -> None:
    """A trained parameter vector is at distance zero from the training domain, by definition."""
    print("\n[sweep_coord: zero on train]")
    for name, a in AXES.items():
        train = [p for p in a.grid() if a.in_train(p)]
        worst = max(abs(a.sweep_coord(p)) for p in train)
        check(worst < 1e-6, f"{name}: sweep_coord==0 for all {len(train)} train points "
                            f"(worst {worst:.2e})")


def test_sweep_coord_positive_off_train() -> None:
    print("\n[sweep_coord: positive off train]")
    for name, a in AXES.items():
        test = [p for p in a.grid() if not a.in_train(p)]
        bad = [p for p in test if a.sweep_coord(p) <= 0.0]
        check(not bad, f"{name}: sweep_coord>0 for all {len(test)} test points "
                       f"({len(bad)} violations)")


def test_position_geometry() -> None:
    """POSITION is the positive control, so its sweep must actually span a usable range."""
    print("\n[position geometry]")
    grid = list(POSITION.grid())
    coords = [POSITION.sweep_coord(p) for p in grid if not POSITION.in_train(p)]
    reach = max(coords)
    check(len(POSITION_TRAIN_CELLS) == 9, "9 training cells")
    check(all(POSITION.in_train({"x": x, "y": y}) for x, y in POSITION_TRAIN_CELLS),
          "every training cell is on the test grid and recognised as train")
    check(reach > 15.0, f"sweep reaches >15 cm of displacement (got {reach:.1f} cm)")
    check(all(WORKSPACE.contains(float(p["x"]), float(p["y"])) for p in grid),
          "every grid point lies inside the workspace")


def test_direction_geometry() -> None:
    """Centre-start in both splits, and constant radius, are the two v1 Reverse repairs."""
    print("\n[direction geometry]")
    grid = list(DIRECTION.grid())
    radii = {float(p["radius"]) for p in grid}
    check(len(radii) == 1, f"constant transport radius across the sweep (got {radii})")
    far = max(DIRECTION.sweep_coord(p) for p in grid)
    check(far > 90.0, f"sweep reaches >90 deg from the trained sector (got {far:.1f})")
    check(max(DIRECTION_TRAIN_THETA) - min(DIRECTION_TRAIN_THETA) == 90.0,
          "training sector spans 90 deg")
    # Angular distance must wrap: theta=352.5 is 7.5 deg from the trained theta=0, not 352.5.
    check(math.isclose(DIRECTION.sweep_coord({"theta_deg": 352.5, "radius": 0.18}), 7.5,
                       abs_tol=1e-6),
          "angular sweep_coord wraps around 360")


def test_approach_jaw_periodicity() -> None:
    """A parallel-jaw azimuth is 180-periodic. Getting this wrong INVERTS the metric.

    Regression test for a measured bug: with a 360-periodic distance, phi=180 reported a sweep
    coordinate of 135 degrees -- the farthest test point on the axis -- while being physically
    the trained phi=0 grasp. A policy would have succeeded at the supposedly-hardest points,
    the fitted radius would have come out enormous, and a structural axis would have looked
    metric. That is the paper's central claim failing backwards from a sign convention.
    """
    print("\n[approach: jaw azimuth is 180-periodic]")
    grid = list(APPROACH.grid())
    yaws = sorted(float(p["yaw_deg"]) for p in grid)

    check(max(yaws) < 180.0,
          f"sweep is half-open [0,180): 180 duplicates 0 (max yaw {max(yaws)})")
    check(all(APPROACH.sweep_coord({"yaw_deg": t}) < 1e-9 for t in APPROACH_TRAIN_YAW),
          "trained yaws sit at coord 0")

    # The bar's yaw is what the policy sees; phi is what the expert must command. They are one
    # bijection apart, and every condition must carry both consistently -- if they ever drift,
    # the expert grasps at an angle the observation does not call for and the axis measures
    # nothing.
    for p_ in grid:
        want = approach_phi(float(p_["yaw_deg"]))
        check(abs(float(p_["phi_deg"]) - want) < 1e-9,
              f"yaw={p_['yaw_deg']}: phi={p_['phi_deg']} is the grasp the bar's yaw calls for ({want})")
    check(len({round(float(p_["phi_deg"]), 4) for p_ in grid}) == len(grid),
          "each condition needs a distinct jaw azimuth, or conditions are duplicates")

    # A bar at yaw and at yaw+180 is the same bar, so the metric must not tell them apart.
    for yaw in (0.0, 22.5, 47.0, 91.0, 179.0):
        a = APPROACH.sweep_coord({"yaw_deg": yaw})
        b = APPROACH.sweep_coord({"yaw_deg": yaw + 180.0})
        check(abs(a - b) < 1e-9,
              f"yaw={yaw} and yaw={yaw + 180} give the same coord ({a:.3f} vs {b:.3f})")

    for t in APPROACH_TRAIN_YAW:
        check(APPROACH.in_train({"yaw_deg": t + 180.0}),
              f"yaw={t + 180} is the trained yaw={t} bar, so it is in-train")

    worst = max(APPROACH.sweep_coord(p_) for p_ in grid)
    check(worst <= 90.0 + 1e-9,
          f"no coord exceeds 90 deg, the max possible jaw separation (got {worst})")

    # Enough distinct bins to fit a radius against.
    n_distinct = len({round(APPROACH.sweep_coord(p), 4) for p in grid})
    check(n_distinct >= 6, f"at least 6 distinct sweep coordinates (got {n_distinct})")

    # 180 was also the one azimuth the reachability probe could not reach.
    check(180.0 not in yaws, "yaw=180 (duplicates 0) is not in the sweep")
    phis_ = sorted(float(p_["phi_deg"]) for p_ in grid)
    check(all(0.0 <= v < 180.0 for v in phis_),
          f"every derived phi stays in [0,180): {min(phis_)} .. {max(phis_)}")


def test_topology_flip() -> None:
    """The discontinuity the axis exists to probe."""
    print("\n[topology flip]")
    # BOTH classes must be trained. They were not: every trained height used to be below h*,
    # so AROUND was held out entirely and the policy never received gradient showing that
    # lateral circumvention is available at all. 0% on the around conditions was then certain
    # for any policy, which measures the split rather than the policy -- the same confound this
    # project's plan levels at v1's Inversion suite.
    trained = {expected_homotopy(h) for h in TOPOLOGY_TRAIN_H}
    check(trained == {"over", "around"},
          f"both homotopy classes are demonstrated in training (got {sorted(trained)})")
    n_over = sum(expected_homotopy(h) == "over" for h in TOPOLOGY_TRAIN_H)
    check(n_over == len(TOPOLOGY_TRAIN_H) - n_over,
          f"trained heights split evenly between classes ({n_over} over, "
          f"{len(TOPOLOGY_TRAIN_H) - n_over} around)")
    # No trained height may also be a test height, or the holdout leaks outright.
    test_h = {round(float(p["barrier_h"]), 6)
              for p in TOPOLOGY.grid() if not TOPOLOGY.in_train(p)}
    overlap = sorted(test_h & {round(h, 6) for h in TOPOLOGY_TRAIN_H})
    check(not overlap, f"no height is both trained and tested (overlap {overlap})")
    check(expected_homotopy(TOPOLOGY_H_STAR - 0.001) == "over", "just below h*: over")
    check(expected_homotopy(TOPOLOGY_H_STAR + 0.001) == "around", "just above h*: around")
    grid = list(TOPOLOGY.grid())
    around_tests = [p for p in grid
                    if not TOPOLOGY.in_train(p)
                    and expected_homotopy(float(p["barrier_h"])) == "around"]
    over_tests = [p for p in grid
                  if not TOPOLOGY.in_train(p)
                  and expected_homotopy(float(p["barrier_h"])) == "over"]
    check(len(around_tests) > 0,
          f"sweep includes held-out AROUND conditions ({len(around_tests)})")
    check(len(over_tests) > 0,
          f"sweep includes held-out OVER conditions ({len(over_tests)})")
    # h* itself must be tested, never trained: it is the height where the class flips, so it is
    # the one condition that cannot be answered by copying the nearest demonstration.
    check(any(math.isclose(float(p["barrier_h"]), TOPOLOGY_H_STAR, abs_tol=1e-6)
              for p in grid if not TOPOLOGY.in_train(p)),
          f"h* = {TOPOLOGY_H_STAR} is a test height, not a trained one")


def test_factorial_covers_every_level() -> None:
    """The whole point: each factor level is trained, but half the combinations are not."""
    print("\n[factorial]")
    a = axis("factorial")
    grid = list(a.grid())
    train = [p for p in grid if a.in_train(p)]
    test = [p for p in grid if not a.in_train(p)]
    check(len(train) > 0 and len(test) > 0, f"{len(train)} trained / {len(test)} held-out cells")
    check({int(p["pos_idx"]) for p in train} == {int(p["pos_idx"]) for p in grid},
          "every position level appears in training")
    check({float(p["theta_deg"]) for p in train} == {float(p["theta_deg"]) for p in grid},
          "every direction level appears in training")
    check(all(not a.in_train(p) for p in test), "held-out cells are not trained")
    overlap = {(int(p["pos_idx"]), float(p["theta_deg"])) for p in train} & {
        (int(p["pos_idx"]), float(p["theta_deg"])) for p in test}
    check(not overlap, "no cell is both trained and held out")


def test_attrs_roundtrip() -> None:
    """Parameter vector survives HDF5 attrs. v1 stored only a discrete area id."""
    print("\n[attrs round-trip]")
    for name, a in AXES.items():
        conds = enumerate_conditions(a, ["red_block"])
        ok = True
        for c in conds[:50]:
            recovered = params_from_attrs(c.to_attrs())
            if recovered != c.params:
                ok = False
                print(f"        {name}: {recovered} != {c.params}")
                break
        check(ok, f"{name}: params round-trip through to_attrs/params_from_attrs")

    c = enumerate_conditions(DIRECTION, ["red_block"])[0]
    attrs = c.to_attrs()
    for k in ("axis", "object_key", "split", "sweep_coord", "instruction"):
        check(k in attrs, f"attrs carry '{k}'")
    check(all(isinstance(v, (int, float, str)) for v in attrs.values()),
          "all attr values are HDF5-safe scalars")


def test_enumerate_and_keys() -> None:
    print("\n[enumerate]")
    objs = ["red_block", "blue_block"]
    conds = enumerate_conditions(POSITION, objs)
    check(len(conds) == len(list(POSITION.grid())) * len(objs),
          "one condition per (grid point x object)")
    check(len({c.key() for c in conds}) == len(conds), "condition keys are unique")
    train_only = enumerate_conditions(POSITION, objs, splits=("train",))
    check(all(c.split == "train" for c in train_only), "split filter is honoured")
    check(all(c.sweep_coord == 0.0 for c in train_only), "train conditions report coord 0")
    check(all(c.instruction for c in conds), "every condition carries an instruction")


def test_sample_sweep() -> None:
    """Evaluation must sample the sweep range, cover every bin, and admit what it dropped."""
    print("\n[sample_sweep]")
    objs = ["red_block", "blue_block", "green_block"]

    s = sample_sweep(POSITION, objs, n_bins=10, per_bin=6, seed=0)
    print(f"        position: {s.summary()}")
    check(s.n_train == len(POSITION_TRAIN_CELLS) * len(objs), "all train conditions kept")
    check(s.n_test_sampled <= 10 * 6, f"test sample bounded by n_bins*per_bin "
                                      f"({s.n_test_sampled})")
    check(s.dropped > 0 and s.n_test_total > s.n_test_sampled,
          f"dropped count is reported ({s.dropped} of {s.n_test_total})")
    check(all(c > 0 for c in s.bin_counts),
          f"no empty sweep bin (counts {s.bin_counts})")
    check(len(s.bin_edges) == 11, "bin edges bracket the range")
    test_coords = [c.sweep_coord for c in enumerate_conditions(POSITION, objs, splits=("test",))]
    check(abs(s.bin_edges[0] - min(test_coords)) < 1e-6,
          "bins are anchored at the minimum OBSERVED test coord, not at 0")
    check(abs(s.bin_edges[-1] - max(test_coords)) < 1e-6, "bins reach the maximum test coord")

    # Determinism: the eval set must be reproducible from the seed alone.
    again = sample_sweep(POSITION, objs, n_bins=10, per_bin=6, seed=0)
    check([c.key() for c in s.conditions] == [c.key() for c in again.conditions],
          "same seed gives an identical evaluation set")
    other = sample_sweep(POSITION, objs, n_bins=10, per_bin=6, seed=1)
    check([c.key() for c in s.conditions] != [c.key() for c in other.conditions],
          "a different seed gives a different sample")

    for name, a in AXES.items():
        ss = sample_sweep(a, objs, n_bins=6, per_bin=4, seed=0)
        empty = sum(1 for c in ss.bin_counts if c == 0)
        check(ss.n_test_sampled > 0, f"{name}: sampled some test conditions "
                                     f"({ss.n_test_sampled}/{ss.n_test_total})")
        check(empty == 0, f"{name}: every populated bin sampled (empty={empty})")
        check(all(c.split == "train" for c in ss.conditions[:ss.n_train]),
              f"{name}: train conditions come first")


def test_no_heavy_imports() -> None:
    """The layering fix: axes.py must not drag h5py/numpy/isaac into the eval path."""
    print("\n[layering]")
    for mod in ("h5py", "numpy", "torch", "isaaclab"):
        check(mod not in sys.modules, f"importing axes did not pull in {mod}")


def test_frozen_condition() -> None:
    print("\n[immutability]")
    c = enumerate_conditions(DIRECTION, ["red_block"])[0]
    try:
        c.split = "train"       # type: ignore[misc]
        check(False, "Condition is frozen")
    except Exception:
        check(True, "Condition is frozen")
    check(isinstance(c, Condition), "enumerate returns Condition instances")


def test_obj_phrases_are_curated() -> None:
    """Every asset an axis really uses must have a hand-written instruction phrase.

    _obj_phrase falls back to a de-prefixed, colour-first rewrite so synthetic names in these
    tests still work, but a real asset must not rely on it: the phrase goes verbatim into
    meta/tasks.jsonl at train time, and the colour word is the only thing separating ORDER's two
    conditions. This is the check that "arb cube red" ever reached the dataset at all.
    """
    from arbiter.suites.spec import _OBJ_PHRASE, object_keys_for

    used = {o for ax in AXES for o in object_keys_for(ax)}
    missing = sorted(used - set(_OBJ_PHRASE))
    assert not missing, f"assets used by an axis but not in _OBJ_PHRASE: {missing}"
    for asset in sorted(used):
        phrase = _OBJ_PHRASE[asset]
        assert "arb" not in phrase, f"{asset}: project prefix leaked into {phrase!r}"
        assert "_" not in phrase, f"{asset}: underscore left in {phrase!r}"
    print(f"  ok: {len(used)} assets, all curated: "
          + ", ".join(f"{a}->{_OBJ_PHRASE[a]!r}" for a in sorted(used)))


def test_direction_words_are_camera_frame() -> None:
    """The bearing word must describe the head camera's view, not the table's axes.

    The camera sits at (1.0, 0, 1.3) looking at (0.5, 0, 0.8), so its forward is ~(-1, 0, -1)/r2
    and, with world up, image-right is +y and image-up is -x. theta is measured from +x, so
    theta=0 travels toward the *bottom* of the frame and theta=90 toward the right.

    The table previously said theta=0 was "right", rotating every DIRECTION instruction 90
    degrees away from what the policy could see. Asserted rather than commented because the
    mistake is invisible in any number the benchmark reports.
    """
    from arbiter.suites.spec import _compass_label

    for theta, want in ((0.0, "near"), (90.0, "right"), (180.0, "far"), (270.0, "left"),
                        (45.0, "near-right"), (135.0, "far-right"),
                        (225.0, "far-left"), (315.0, "near-left")):
        got = _compass_label(theta)
        assert got == want, f"theta={theta}: camera frame says {want!r}, spec says {got!r}"
    print("  ok: 8 bearings match the head camera's frame")


def test_instructions_match_what_the_scene_shows() -> None:
    """The language and the scene must agree about what the goal is.

    Two rules, both violated before this test existed:

    * **A pad axis must name the target.** Five axes spawn a saturated target pad and three of
      them never mentioned it -- DIRECTION and FACTORIAL said "move it to the near side" and
      EXTENT said "move it forward", while the thing to aim at sat in plain view. EXTENT was the
      worst case: its language named neither the pad nor the distance it sweeps, so the
      instruction carried no information about the condition at all.
    * **A lift axis must NOT name a target,** because there is no goal location to name.
      POSITION and APPROACH lift, and promising a target the scene does not contain would be a
      straightforward lie to the policy.
    """
    from arbiter.suites.spec import object_keys_for, pad_keys_for

    for name, ax in sorted(AXES.items()):
        conds = enumerate_conditions(ax, object_keys_for(name))
        texts = {c.instruction for c in conds}
        has_pad = bool(pad_keys_for(name))
        names_target = all("target" in t for t in texts)
        if has_pad:
            check(names_target, f"{name}: every instruction names the target (pad axis)")
        else:
            check(not any("target" in t for t in texts),
                  f"{name}: no instruction promises a target (lift axis)")

    # An axis whose language varies must vary it per swept value, not arbitrarily: DIRECTION has
    # 8 bearing words, FACTORIAL 4, ORDER 6 orderings. A drop here means conditions collapsed
    # into one instruction and became indistinguishable to the policy.
    expect = {"direction": 8, "factorial": 4, "order": 6}
    for name, n in sorted(expect.items()):
        texts = {c.instruction for c in enumerate_conditions(AXES[name], object_keys_for(name))}
        check(len(texts) == n,
              f"{name}: {n} distinct instructions expected, got {len(texts)}")

    # And the axes whose swept quantity must be READ FROM THE SCENE keep language constant:
    # TOPOLOGY's barrier height and POSITION's spawn are visible, and naming them would hand
    # over the very thing the axis asks the policy to perceive.
    for name in ("topology", "position", "approach", "extent"):
        texts = {c.instruction for c in enumerate_conditions(AXES[name], object_keys_for(name))}
        check(len(texts) == 1,
              f"{name}: language stays constant across conditions (got {len(texts)})")


def main() -> int:
    test_registry()
    test_grids_nonempty_and_split()
    test_sweep_coord_zero_on_train()
    test_sweep_coord_positive_off_train()
    test_position_geometry()
    test_direction_geometry()
    test_approach_jaw_periodicity()
    test_topology_flip()
    test_factorial_covers_every_level()
    test_attrs_roundtrip()
    test_obj_phrases_are_curated()
    test_direction_words_are_camera_frame()
    test_instructions_match_what_the_scene_shows()
    test_enumerate_and_keys()
    test_sample_sweep()
    test_no_heavy_imports()
    test_frozen_condition()

    print()
    if FAILURES:
        print(f"[FAIL] {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("[PASS] all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
