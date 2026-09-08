# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""ARBITER geometry and condition spec: conditions as parameter vectors.

v1 represented a condition as a *discrete named area* plus an object key, and derived
train/test membership from a predicate over ``(area_idx, object_key)``. That cannot express
``theta = 137 deg`` or ``barrier_h = 6.2 cm``, so it cannot express a continuous sweep — and a
continuous sweep is the whole measurement in v2. Here a condition carries a **parameter
vector** instead; a named area is just a coarse quantization of one, so this schema subsumes
the v1 one rather than complicating it.

Two distances matter and must not be confused:

``sweep_coord``
    Distance from the training domain **in parameter space**, in the axis's native unit (cm,
    degrees, ordinal steps). Free to compute, available at construction time, and it defines
    the x-axis of every SR curve.

support distance
    Distance from the training set **in trajectory space** — Frechet between the scripted
    expert's reference path for this condition and the nearest training path. This is the
    principled metric the radius is fitted against. It lives in the metrics module, not here,
    because it needs expert rollouts.

The two agree in ordering for well-behaved axes; where they disagree, trajectory space wins.

**This module has no heavy dependencies on purpose.** ``arbiter/splits/subset_filters.py``
imports h5py at module scope, and CLAUDE.md records that the eval path must never pull it in.
v1 had ``arb_suites.py`` importing ``SUBSETS`` *from* ``subset_filters``, which inverted the
dependency the wrong way. Axis and split definitions live here (stdlib only); the h5py-backed
demo selector imports from this module.

Geometry note: the numbers in ``WORKSPACE`` and each axis's training domain are *provisional*.
They are sized for a Franka Panda at a table edge but must be validated in Phase 1 against
actual reachability and against the scripted-expert achievability gate — the gate requires
~100% expert success at every test point, and it is what turns these guesses into settled
values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator, Literal

ParamValue = float | int | str
Params = dict[str, ParamValue]

#: Three classes, not two. "metric" and "structural" are the asymmetry the paper predicts --
#: interpolating a distance versus producing a shape never demonstrated. FACTORIAL asks a third
#: question: whether coverage *multiplies* across axes. Typing it "structural" conflated
#: "invent a shape" with "compose trained factors", which are different claims with different
#: falsifications -- FACTORIAL succeeding weakens the paradigm argument, while a structural
#: axis succeeding strengthens the policy's case.
AxisKind = Literal["metric", "structural", "composition"]
Split = Literal["train", "test"]


# ──────────────────────────────────────────────────────────────────────────────
# Workspace
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Workspace:
    """Tabletop region the benchmark operates in, robot base frame, metres.

    The *test* extent, deliberately larger than any axis's training domain: a sweep has to
    reach parameter values well outside training to find the radius, so training domains are
    sited centrally and the sweep runs out to the edges.
    """

    x_min: float = 0.28
    x_max: float = 0.72
    y_min: float = -0.28
    y_max: float = 0.28
    z_table: float = 0.0

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    @property
    def centre(self) -> tuple[float, float]:
        return (0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max))


WORKSPACE = Workspace()


# ──────────────────────────────────────────────────────────────────────────────
# Condition
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Condition:
    """One evaluable condition: an axis, a parameter vector, and an object.

    ``split`` is *derived* from the axis's training domain, never stored independently — that
    was a v1 failure mode where the eval split and the data split could drift apart.
    """

    axis: str
    params: Params
    object_key: str
    instruction: str
    split: Split
    sweep_coord: float

    def key(self) -> str:
        """Stable identifier, used for filenames and for --conditions selection."""
        ps = ",".join(f"{k}={_fmt(v)}" for k, v in sorted(self.params.items()))
        return f"{self.axis}[{ps}]@{self.object_key}"

    def to_attrs(self) -> dict[str, ParamValue]:
        """Flat, HDF5-attr-safe dict. Round-trips via :func:`params_from_attrs`."""
        out: dict[str, ParamValue] = {
            "axis": self.axis,
            "object_key": self.object_key,
            "split": self.split,
            "sweep_coord": float(self.sweep_coord),
            "instruction": self.instruction,
        }
        for k, v in self.params.items():
            out[f"param_{k}"] = v
        return out


def params_from_attrs(attrs: dict[str, ParamValue]) -> Params:
    """Recover the parameter vector from HDF5 episode attrs written by :meth:`to_attrs`."""
    return {k[len("param_"):]: v for k, v in attrs.items() if str(k).startswith("param_")}


def _fmt(v: ParamValue) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


# ──────────────────────────────────────────────────────────────────────────────
# Axis
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Axis:
    """One held-out descriptor and the sweep that probes it.

    ``grid`` yields every parameter vector, train and test together; ``in_train`` decides
    membership; ``sweep_coord`` reports parameter-space distance from the training domain.
    Keeping all three on one object is what makes it impossible for the collected training
    set and the evaluated split to disagree.
    """

    name: str
    kind: AxisKind
    unit: str
    sweep_param: str
    grid: Callable[[], Iterator[Params]]
    in_train: Callable[[Params], bool]
    sweep_coord: Callable[[Params], float]
    instruction: Callable[[Params, str], str]
    mechanism_metric: str
    notes: str = ""


def enumerate_conditions(
    axis: Axis,
    object_keys: list[str],
    splits: tuple[Split, ...] = ("train", "test"),
) -> list[Condition]:
    """All conditions for one axis, bucketed into train/test by the axis's own domain."""
    conds: list[Condition] = []
    for params in axis.grid():
        split: Split = "train" if axis.in_train(params) else "test"
        if split not in splits:
            continue
        # Rounded here, once, rather than at each consumer. Sweep coordinates come out of
        # float subtraction (|0.09 - 0.07| * 100 = 1.9999999999999998), and an unrounded value
        # epsilon below an integer boundary gets truncated into the wrong bin by
        # sample_sweep -- which then reports empty bins as coverage gaps that do not exist.
        # The underlying quantities are physically quantised (1 cm, 11.25 deg), so rounding is
        # information-preserving.
        coord = 0.0 if split == "train" else round(float(axis.sweep_coord(params)), 6)
        # One condition per (params, object) EXCEPT where the params already name the objects.
        objs = object_keys[:1] if axis.name in PARAM_OBJECT_AXES else object_keys
        for obj in objs:
            conds.append(
                Condition(
                    axis=axis.name,
                    params=dict(params),
                    object_key=obj,
                    instruction=axis.instruction(params, obj),
                    split=split,
                    sweep_coord=coord,
                )
            )
    return conds


@dataclass(frozen=True)
class SweepSample:
    """A sampled evaluation set plus an explicit account of what was left out.

    A grid defines the *space*; evaluation samples from it. POSITION alone has 658 test grid
    points, so evaluating every one at 20 episodes across several objects is tens of thousands
    of rollouts — and pointless, because fitting a radius needs the sweep *range* covered, not
    every point in it.

    ``dropped`` and ``bin_counts`` exist so the bound is stated rather than hidden. A sampled
    sweep that silently discards most of its grid reads as full coverage when it is not.
    """

    conditions: list[Condition]
    n_train: int
    n_test_sampled: int
    n_test_total: int
    bin_counts: list[int]
    bin_edges: list[float]

    @property
    def dropped(self) -> int:
        return self.n_test_total - self.n_test_sampled

    def summary(self) -> str:
        empty = sum(1 for c in self.bin_counts if c == 0)
        return (
            f"{len(self.conditions)} conditions: {self.n_train} train (all kept), "
            f"{self.n_test_sampled}/{self.n_test_total} test sampled "
            f"({self.dropped} dropped) across {len(self.bin_counts)} sweep bins"
            + (f"; {empty} bin(s) EMPTY" if empty else "")
        )


def sample_sweep(
    ax: Axis,
    object_keys: list[str],
    *,
    n_bins: int = 10,
    per_bin: int = 6,
    seed: int = 0,
) -> SweepSample:
    """Stratify test conditions by ``sweep_coord`` and sample evenly across the range.

    Every *train* condition is kept — they are the ID control that anchors the radius fit, and
    there are few of them. Test conditions are binned into ``n_bins`` equal-width bins over
    ``[0, max sweep_coord]`` and sampled deterministically, so an evaluation set is
    reproducible from ``seed`` alone.

    An empty bin is reported, not silently tolerated: a gap in the sweep range is exactly what
    would make a radius fit unreliable.
    """
    import random

    rng = random.Random(seed)
    train = enumerate_conditions(ax, object_keys, splits=("train",))
    test = enumerate_conditions(ax, object_keys, splits=("test",))

    if not test:
        return SweepSample(list(train), len(train), 0, 0, [], [])

    # Bin over the *observed* test range, not [0, max]. Training values sit at coord 0 and
    # test values start one grid step away, so a bin anchored at 0 is empty by construction —
    # which is what a naive equal-width binning reports as a coverage gap that isn't one.
    distinct = sorted({round(c.sweep_coord, 6) for c in test})
    lo, hi = distinct[0], distinct[-1]

    # A low-cardinality axis (ORDER has one test value, FACTORIAL two) cannot be equal-width
    # binned into n_bins without leaving most of them empty. Stratify by distinct value.
    n = min(n_bins, len(distinct))

    if n <= 1 or hi <= lo:
        take = sorted(rng.sample(test, min(per_bin * max(n, 1), len(test))),
                      key=lambda c: c.key())
        return SweepSample(list(train) + take, len(train), len(take), len(test),
                           [len(take)], [lo, hi])

    if len(distinct) <= n_bins:
        # One stratum per distinct value: no empty bin is possible.
        groups: dict[float, list[Condition]] = {d: [] for d in distinct}
        for c in test:
            groups[round(c.sweep_coord, 6)].append(c)
        buckets = [groups[d] for d in distinct]
        edges = list(distinct)
    else:
        width = (hi - lo) / n
        buckets = [[] for _ in range(n)]
        for c in test:
            # The epsilon guards the same truncation hazard as the rounding in
            # enumerate_conditions, for any caller that builds Conditions by hand.
            i = min(int((round(c.sweep_coord, 6) - lo) / width + 1e-9), n - 1)
            buckets[i].append(c)
        edges = [round(lo + i * width, 6) for i in range(n + 1)]

    picked: list[Condition] = []
    counts: list[int] = []
    for b in buckets:
        take = sorted(rng.sample(b, min(per_bin, len(b))), key=lambda c: c.key()) if b else []
        picked.extend(take)
        counts.append(len(take))

    return SweepSample(
        conditions=list(train) + picked,
        n_train=len(train),
        n_test_sampled=len(picked),
        n_test_total=len(test),
        bin_counts=counts,
        bin_edges=edges,
    )


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

#: Asset name -> the words a person would use. Must match meta/tasks.jsonl verbatim at train
#: time, so it is defined once here and re-derived by the converter rather than transcribed.
#:
#: The asset names carry a project prefix and adjective-last order ("arb_cube_red"), which a
#: plain underscore-to-space gives as "arb cube red" -- a phrase no annotator would write and
#: no pretrained language encoder has useful structure for. The colour is the only thing
#: distinguishing ORDER's two objects, so burying it at the end of a three-token noun phrase
#: put load on exactly the word the axis depends on.
_OBJ_PHRASE = {
    "arb_cube_red": "red block",
    "arb_cube_blue": "blue block",
    "arb_cube_green": "green block",
    "arb_cube_yellow": "yellow block",
    "arb_bar_red": "red bar",
}

#: Axes where the condition's *parameters* name the objects, so a condition must NOT be
#: multiplied across the object list. ORDER carries its pair as a parameter, and
#: object_keys_for("order") returns every cube any pair uses so one launch can realise them
#: all -- zipping the two would emit each condition once per cube and quadruple the axis.
PARAM_OBJECT_AXES = frozenset({"order"})


_COLOURS = ("red", "blue", "green", "yellow", "black", "white")


def _obj_phrase(asset_name: str) -> str:
    """'arb_cube_red' -> 'red block'; unmapped names get readable English, not a mangled one.

    Curated phrases win. The fallback drops a project prefix and moves a trailing colour to the
    front, so an asset added without a map entry reads as "red cube" rather than
    "arb cube red". It is a fallback and not the mechanism: ``test_obj_phrases_are_curated``
    asserts every asset any axis actually uses has an entry here, which is the check that
    matters, while keeping synthetic names in the unit tests working.
    """
    name = str(asset_name)
    if name in _OBJ_PHRASE:
        return _OBJ_PHRASE[name]
    parts = [t for t in name.split("_") if t and t != "arb"]
    if len(parts) > 1 and parts[-1] in _COLOURS:
        parts = [parts[-1]] + parts[:-1]
    return " ".join(parts)


def _frange(lo: float, hi: float, step: float) -> Iterator[float]:
    """Inclusive float range, rounded to kill accumulation error in dict keys."""
    n = int(round((hi - lo) / step))
    for i in range(n + 1):
        yield round(lo + i * step, 6)


def _ang_dist(a: float, b: float) -> float:
    """Smallest absolute angular separation in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _min_ang_dist(theta: float, refs: list[float]) -> float:
    return min(_ang_dist(theta, r) for r in refs)


def _jaw_dist(a: float, b: float) -> float:
    """Angular separation between two parallel-jaw azimuths: 180-periodic, not 360.

    A jaw closing along +x is the same grasp as one closing along -x, so azimuths differing by
    180 degrees are *identical*, not maximally apart. Using the 360-periodic distance here is
    not a rounding issue -- it inverts the metric at the top of the range. Measured: phi=180
    was reporting a sweep coordinate of 135 degrees, the farthest test point on the axis, while
    being physically the trained phi=0 grasp. A policy would have succeeded there, the radius
    fit would have come out enormous, and a structural axis would have looked metric.
    """
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def _min_jaw_dist(phi: float, refs: list[float]) -> float:
    return min(_jaw_dist(phi, r) for r in refs)


def _min_euclid(x: float, y: float, refs: list[tuple[float, float]]) -> float:
    return min(math.hypot(x - rx, y - ry) for rx, ry in refs)


# ──────────────────────────────────────────────────────────────────────────────
# POSITION — metric, positive control
# ──────────────────────────────────────────────────────────────────────────────
# Training lattice sits centrally so the test grid can run out to ~19 cm of displacement.
# 12 cm spacing; 3 x 3 = 9 cells.

POSITION_TRAIN_X = [0.38, 0.50, 0.62]
POSITION_TRAIN_Y = [-0.12, 0.0, 0.12]
POSITION_TRAIN_CELLS = [(x, y) for x in POSITION_TRAIN_X for y in POSITION_TRAIN_Y]
POSITION_TEST_STEP = 0.02


def _position_grid() -> Iterator[Params]:
    for x in _frange(WORKSPACE.x_min, WORKSPACE.x_max, POSITION_TEST_STEP):
        for y in _frange(WORKSPACE.y_min, WORKSPACE.y_max, POSITION_TEST_STEP):
            yield {"x": x, "y": y}


def _position_in_train(p: Params) -> bool:
    return any(
        math.isclose(float(p["x"]), cx, abs_tol=1e-6)
        and math.isclose(float(p["y"]), cy, abs_tol=1e-6)
        for cx, cy in POSITION_TRAIN_CELLS
    )


POSITION = Axis(
    name="position",
    kind="metric",
    unit="cm",
    sweep_param="displacement",
    grid=_position_grid,
    in_train=_position_in_train,
    sweep_coord=lambda p: 100.0
    * _min_euclid(float(p["x"]), float(p["y"]), POSITION_TRAIN_CELLS),
    instruction=lambda p, o: f"lift the {_obj_phrase(o)} off the table",
    mechanism_metric="terminal_endpoint_error_cm",
    notes="Positive control. Evaluate this axis FIRST: an instrument that cannot detect the "
          "success it is calibrated for is not measuring anything.",
)


# ──────────────────────────────────────────────────────────────────────────────
# EXTENT — metric
# ──────────────────────────────────────────────────────────────────────────────

EXTENT_TRAIN_LENGTHS = [0.10, 0.14, 0.18, 0.20]
EXTENT_TEST_LENGTHS = list(_frange(0.10, 0.35, 0.01))

# Forward transport only. An earlier version swept three azimuths as well, which was a design
# error: varying direction inside the EXTENT axis confounds it with DIRECTION, and the whole
# point of a per-axis radius is that each axis moves one factor. It also put targets off the
# table -- from the workspace centre, only 0.22 m fits along +x, well short of the 0.35 m
# sweep.
EXTENT_AZIMUTHS = [0.0]

# Sited at the near edge rather than the centre, so 0.35 m of forward travel stays on the
# table (0.72 - 0.32 = 0.40 m of room).
#: EXTENT's spawn. Moved off (0.32, 0.0) because that point coincided EXACTLY with TOPOLOGY's
#: start after TOPOLOGY's own start moved to x=0.32 to clear the barrier occlusion. Both axes
#: then transported along +x from the same point, and TOPOLOGY -- the largest axis at 180
#: episodes -- transports 0.30 m, which is one of EXTENT's *held-out* lengths.
#:
#: Measured consequence: for every EXTENT test length from 0.26 m up, the nearest training
#: trajectory came from TOPOLOGY, and the support distance collapsed from 15.0 cm (against
#: EXTENT alone) to 5.0 cm (against all axes). EXTENT had no extrapolation region left, and its
#: 100% test score was a success inside the radius rather than beyond it.
#:
#: Chosen by grid search over spawn and azimuth, maximising the minimum endpoint separation from
#: every trained transport in the other six axes while keeping all 26 lengths >= 3 cm inside the
#: workspace: (0.31, -0.23) at azimuth 0 gives 13.0 cm, against 0.0 cm before.
EXTENT_START_X = 0.31
EXTENT_START_Y = -0.23


def _extent_grid() -> Iterator[Params]:
    for length in EXTENT_TEST_LENGTHS:
        for az in EXTENT_AZIMUTHS:
            yield {
                "length": length, "azimuth": az,
                "x": EXTENT_START_X, "y": EXTENT_START_Y,
            }


def _extent_in_train(p: Params) -> bool:
    return any(math.isclose(float(p["length"]), t, abs_tol=1e-6) for t in EXTENT_TRAIN_LENGTHS)


EXTENT = Axis(
    name="extent",
    kind="metric",
    unit="cm",
    sweep_param="length",
    grid=_extent_grid,
    in_train=_extent_in_train,
    sweep_coord=lambda p: 100.0
    * min(abs(float(p["length"]) - t) for t in EXTENT_TRAIN_LENGTHS),
    # Deliberately NOT the _compass_label wording DIRECTION uses, even though EXTENT
    # transports along azimuth 0 and so travels in DIRECTION's "near" direction.
    #
    # EXTENT's swept quantity is *distance*, and nothing the policy can observe carries it:
    # there is no target marker prop in any layout, and the instruction names no length. Both
    # axes would then issue the identical string "move it to the near side" for targets 10 to
    # 35 cm out -- one instruction, contradictory supervision.
    #
    # Sharing the word does not create that ambiguity, it exposes it: EXTENT at length 0.18 IS
    # DIRECTION at theta 0, radius 0.18, the same condition twice. Until the axis gets a
    # visible target (or a length in the language), the honest reading of its radius is the
    # width of POSITION_TOLERANCE_M around the habitual trained distance, not a generalization
    # limit. Keeping the wording distinct stops the defect from also corrupting DIRECTION.
    # Names the TARGET, not a bare "forward". EXTENT was the only axis whose language
    # mentioned neither its swept quantity (the distance) nor the marker showing it, so the
    # instruction carried no information about the condition at all. No bearing word: EXTENT
    # transports along one fixed azimuth, so a direction would be constant and say nothing.
    instruction=lambda p, o: f"pick up the {_obj_phrase(o)} and put it on the target",
    mechanism_metric="transport_completion_fraction",
    notes="Second metric axis, so 'metric radii are large' rests on more than one point. Cap "
          "the long end where the scripted expert stops hitting 100% — reach limits confound "
          "it otherwise.",
)


# ──────────────────────────────────────────────────────────────────────────────
# DIRECTION — structural. The repaired v1 Reverse.
# ──────────────────────────────────────────────────────────────────────────────
# Object starts at the workspace centre in BOTH train and test, which removes v1 Reverse's
# terminal-state confound (there, the centre only ever appeared as an episode's final state).
# All targets sit on one radius, so reach difficulty is constant across the sweep and no
# time-reversal claim is needed — direction is just a continuous parameter.

DIRECTION_RADIUS = 0.18
DIRECTION_TRAIN_THETA = [0.0, 22.5, 45.0, 67.5, 90.0]
DIRECTION_TEST_STEP = 7.5

#: Bearing words in the **head camera's** frame, which is the only frame the policy can see.
#:
#: theta is measured in the table frame from +x, so a displacement is
#: (cos theta, sin theta). The head camera sits at (1.0, 0, 1.3) looking at (0.5, 0, 0.8), so
#: its forward is ~(-0.707, 0, -0.707) and, with world up, image-right is **+y** while image-up
#: is **-x**. That makes theta=0 point at the *bottom* of the frame, not the right.
#:
#: This table previously read (0, "right"), i.e. raw table axes with +x called "right". Every
#: DIRECTION instruction was therefore rotated 90 degrees from what the camera showed, and the
#: axis was scoring a policy on language that did not describe its own observation. Caught by
#: watching a rollout: the "right" condition moved the cube down the frame.
#:
#: "far"/"near" rather than "top"/"bottom" deliberately: this is an oblique view of a tabletop,
#: and "move it to the top" invites the reading "lift it", which is a different axis entirely.
#: Words are unchanged; how they are USED changed. They used to be dropped into "move it to
#: the near side", which names a direction but never the thing being aimed at -- while a
#: saturated target pad sat in plain view. Every place axis now names the target and uses the
#: bearing to say where it is.
_COMPASS = [
    (0.0, "near"), (45.0, "near-right"), (90.0, "right"), (135.0, "far-right"),
    (180.0, "far"), (225.0, "far-left"), (270.0, "left"), (315.0, "near-left"),
]


def _compass_label(theta: float) -> str:
    return min(_COMPASS, key=lambda c: _ang_dist(theta, c[0]))[1]


def _direction_grid() -> Iterator[Params]:
    for theta in _frange(0.0, 360.0 - DIRECTION_TEST_STEP, DIRECTION_TEST_STEP):
        yield {"theta_deg": theta, "radius": DIRECTION_RADIUS}


def _direction_in_train(p: Params) -> bool:
    return any(
        math.isclose(float(p["theta_deg"]), t, abs_tol=1e-6) for t in DIRECTION_TRAIN_THETA
    )


DIRECTION = Axis(
    name="direction",
    kind="structural",
    unit="deg",
    sweep_param="theta_deg",
    grid=_direction_grid,
    in_train=_direction_in_train,
    sweep_coord=lambda p: _min_ang_dist(float(p["theta_deg"]), DIRECTION_TRAIN_THETA),
    instruction=lambda p, o: (
        f"pick up the {_obj_phrase(o)} and put it on the target to the "
        f"{_compass_label(float(p['theta_deg']))} side"
    ),
    mechanism_metric="signed_angular_error_deg",
    notes="Instruction names the compass direction, and the direction vocabulary is shared "
          "across train and test, so the task stays well-posed off-support.",
)


# ──────────────────────────────────────────────────────────────────────────────
# TOPOLOGY — structural, flagship
# ──────────────────────────────────────────────────────────────────────────────
# Start, goal and object fixed; a barrier of height h sits between them. Below h* the expert
# routes OVER, above it AROUND. Training contains only over-paths. The observation varies
# smoothly while the required trajectory changes discontinuously, endpoints held fixed — so
# no reachability, IK, or perceptual-novelty explanation is available for a failure.

# Where the required path class flips from over to around. Lowered from 0.12 so the sweep is
# balanced under the camera's line-of-sight limit below: with h* = 0.12 and h_max = 0.14 only
# three heights would have demanded the AROUND class, too few for the flagship axis.
TOPOLOGY_H_STAR = 0.08
# Both homotopy classes are trained, at heights the test set does not use.
#
# Previously all four trained heights were below h*, so every demonstration was OVER and the
# AROUND class was held out entirely. That made 0% on the around conditions certain regardless
# of capability -- the policy had never been shown that lateral circumvention is a thing a
# trajectory can do -- and it is the same "never received gradient" confound as ORDER's.
#
# 0.02/0.05 demonstrate OVER, 0.12/0.13 demonstrate AROUND, and the test heights are the eight
# in between: 0.03/0.04/0.06/0.07 (over, unseen height) and 0.08/0.09/0.10/0.11 (around, unseen
# height). Still 4 trained heights x 3 offsets, so the collection budget is unchanged.
#
# The axis now asks the better question. Not "can a policy invent a homotopy class from
# nothing", whose answer is no by construction, but "given both classes demonstrated, can it
# select the right one at an unseen barrier height" -- hardest exactly at h* = 0.08, where the
# sweep coordinate is also largest, since 0.08 is 3 cm from the nearest trained height either
# way.
TOPOLOGY_TRAIN_H = [0.02, 0.05, 0.12, 0.13]
TOPOLOGY_TEST_STEP = 0.01
# Capped by camera line of sight, measured not guessed. The object starts 12 cm behind the
# barrier, so the head camera must see over the barrier top to observe it at t=0. From LIBERO's
# agentview geometry (0.55 m above the table, 0.50 m beyond the workspace centre) the sight
# line clears a barrier top of 0.897 m, i.e. h = 0.147. A barrier the policy cannot see past is
# not a harder condition, it is an unobservable one -- and that measures perception, not
# routing.
#
# To lift this cap the object would have to start on the *camera* side of the barrier, so the
# arm routes out, grasps, and routes back. That is a richer test and worth doing, but it
# changes the task definition rather than a constant.
# Capped at 0.13, not 0.14, by MEASUREMENT rather than by the camera. The exhaustive gate
# passed every barrier height up to 0.13 (36/36 across 3 offsets) and failed all three offsets
# at 0.14: the around-route's lateral traverse is blocked when the arm has to reach past a wall
# that tall at full extension. h=0.14 was the only unachievable configuration in the whole
# sweep, so it comes out rather than being carried as a known-bad condition.
#
# Still 12 heights with h* = 0.08 giving 6 over / 6 around, so the class split stays balanced.
TOPOLOGY_H_MAX = 0.13
TOPOLOGY_OFFSETS = [-0.06, 0.0, 0.06]


def _topology_grid() -> Iterator[Params]:
    for h in _frange(0.02, TOPOLOGY_H_MAX, TOPOLOGY_TEST_STEP):
        for off in TOPOLOGY_OFFSETS:
            yield {"barrier_h": h, "barrier_offset": off}


def _topology_in_train(p: Params) -> bool:
    # Deliberately height-only. The detour-direction holdout is NOT expressed here, and the
    # attempt to do so is instructive: folding it in makes h=0.12 simultaneously trained (at
    # offsets 0 and +0.06) and tested (at -0.06), so two test conditions get `sweep_coord = 0`
    # and the axis reports test points at zero distance from support. Two tests caught it.
    #
    # The cause is that TOPOLOGY's sweep coordinate is **barrier height**, while the detour
    # holdout varies the required *direction* at a fixed height. One coordinate cannot describe
    # both, and a single radius fitted across them describes neither -- the same defect EXTENT
    # had when interpolation and extrapolation points were pooled.
    #
    # So the exclusion lives at the dataset layer instead: see TRAINING_EXCLUSIONS.
    return any(math.isclose(float(p["barrier_h"]), t, abs_tol=1e-6) for t in TOPOLOGY_TRAIN_H)


#: Conditions whose collected episodes are withheld from the TRAINING SET, without changing the
#: axis's own train/test split. `(axis, predicate)`; the converter honours it and the spec owns it.
#:
#: Distinct from `in_train`, which answers "is this condition part of the axis's sweep holdout".
#: This answers "does the policy get to see these demonstrations", and the two are not the same
#: question the moment a second, non-swept holdout is layered on top.
#:
#: Why any of this: `route()`'s tie-break detours toward whichever barrier edge the end-effector
#: is nearer, which at offset -0.06 is +y (RIGHT) while offsets 0 and +0.06 go -y (LEFT). So
#: training demonstrated BOTH detour directions -- 60 left, 30 right -- and the blocker holdout
#: (right detour at offset 0) was only a **6.1 cm** recombination of a shape training contained.
#: Its nearest trained neighbour was the offset -0.06 right detour, bypass lane 4 mm away.
#:
#: Withholding those 30 episodes, measured against the existing probes before spending anything:
#:
#:     as built                     6.1 - 6.9 cm
#:     right detours withheld       21.8 - 22.5 cm  (d_own)
#:     right detours withheld       18.9 - 19.0 cm  (d_all, against all seven axes)
#:
#: 19 cm is 3.4x the largest measured radius, so this is the far-from-support trajectory cell the
#: suite otherwise lacks. Cross-axis leakage costs only 13% -- the nearest neighbour across every
#: axis becomes ORDER's multi-block path, not DIRECTION's rightward transport as feared.
#:
#: Withholding beats re-collecting with the tie-break pinned, and not only on cost: pinning to
#: LEFT would force offset -0.06 to bypass at y = -0.28, exactly the workspace boundary, which
#: `bypass_is_reachable` rejects. Withholding needs no simulator time -- the episodes exist and
#: are simply not converted.
#:
#: OVER-class heights at this offset stay in training: an over-route never travels laterally, so
#: it cannot demonstrate a detour direction.
TRAINING_EXCLUSIONS: dict[str, str] = {
    "topology": "around-class conditions at barrier_offset = -0.06",
}


def is_withheld_from_training(axis_name: str, params: Params) -> bool:
    """Should this condition's episodes be kept out of the training set? See TRAINING_EXCLUSIONS."""
    if axis_name != "topology":
        return False
    if expected_homotopy(float(params["barrier_h"])) != "around":
        return False
    return math.isclose(float(params.get("barrier_offset", 0.0)), -0.06, abs_tol=1e-6)


#: Barrier plan geometry, in the spec because both the asset generator and the scripted
#: expert need it and a restated copy is a copy that drifts. RouteTask previously hardcoded
#: 0.15 with a comment claiming it matched the generator -- which is the failure mode this
#: file exists to prevent.
#:
#: Narrowed from 0.30: in a 56 cm workspace that forced the around-route out to y = 0.21,
#: within 7 cm of the workspace edge, and the tallest barrier then failed because the arm had
#: to reach past it at full extension. 0.20 keeps "around" a real 16 cm detour while staying
#: comfortably inside the reachable envelope.
BARRIER_WIDTH_M = 0.20                  # lateral span (world Y)
BARRIER_DEPTH_M = 0.02                  # depth (world X): thin, so "over" is about height


def barrier_half_width() -> float:
    return 0.5 * BARRIER_WIDTH_M


#: TOPOLOGY's fixed endpoints, as offsets in X from the workspace centre. The object starts on
#: one side of the barrier and must end on the other, and *both* the scene placement and the
#: expert's goal derive from these -- they used to be written out separately in the gate, the
#: renderer and ``RouteTask``, which meant a barrier could silently be placed somewhere other
#: than between the two points the expert was routing around.
# Start moved out from -0.12 to -0.18 so the object is never hidden behind the barrier.
#
# The object sits on the FAR side and the head camera looks across the barrier at it, so the
# sight line has to clear the barrier top. At the old -0.12 (x=0.38) the object's centre passed
# 3.4 mm BELOW that line at h=0.13 -- occluded, and 0.12/0.13 are now *trained* heights, so the
# unobservable view was in the training set rather than only the test set.
#
# Fixed by moving the object, not the camera: the head camera reproduces LIBERO/robosuite
# `agentview` geometry and is what makes radii comparable to the field's standard, so it is not
# free to move. Starting on the camera side was the other option and is worse -- it changes the
# task from "carry past a barrier" to "reach out, grasp, come back", i.e. a different axis.
#
# At x=0.32 the sight line clears a 0.13 barrier by 34 mm. See
# `topology_max_visible_start_x` for the derivation, and
# `test_topology_object_is_never_occluded` for the assertion against the real camera pose.
TOPOLOGY_START_DX = -0.18
TOPOLOGY_GOAL_DX = 0.12


def topology_max_visible_start_x(
    barrier_h: float,
    *,
    cam_x: float,
    cam_z: float,
    barrier_x: float,
    table_top_z: float,
    object_half_h: float,
    margin: float = 0.0,
) -> float:
    """Largest start x whose object centre is still visible over a barrier of height h.

    Pure planar geometry in the camera's x-z plane (both camera and object sit on y=0). The
    sight line from the camera to the object crosses the barrier plane at a height that falls
    as the object moves *toward* the camera, because the line gets steeper. So visibility sets
    an upper bound on the start x, and the object has to start far enough away.

    Camera parameters are arguments rather than imports: this module is stdlib-only by design
    and the camera lives in ``layout_gen``, which imports from here.
    """
    span = cam_z - (table_top_z + object_half_h)
    if span <= 0:
        raise ValueError("camera must sit above the object")
    head = cam_z - (table_top_z + barrier_h + margin)
    if head <= 0:
        raise ValueError(f"barrier of height {barrier_h} reaches the camera height")
    return cam_x - (cam_x - barrier_x) * span / head


# ── DETOUR: the bypass side, and the blocker that forces it ──────────────────────────────────
#
# TOPOLOGY holds out the barrier HEIGHT and lets the height decide over-vs-around. What it
# cannot hold out is the *side* of the lateral detour, because `route()` picks that from an
# emergent tie-break -- whichever barrier edge the end-effector happens to be nearer -- and the
# choice was never written down anywhere. Measured against the collected data that tie-break
# yields:
#
#     offset -0.06  ->  bypass +y   (30 episodes)
#     offset  0.00  ->  bypass -y   (30 episodes)
#     offset +0.06  ->  bypass -y   (30 episodes)
#
# So "training always detours one way" is FALSE: both sides are demonstrated, 30 vs 60
# episodes. That is worth stating precisely because it decides the whole design of the axis
# below. It means the recoverability guarantee is satisfied WITHIN the axis -- each side
# receives gradient -- so holding out a (offset, side) COMBINATION is a compositional holdout
# and not the "never received gradient" confound that made the old all-over TOPOLOGY and the
# old single-instruction ORDER unable to report anything.
#
# The held-out cell is therefore the complement of the tie-break: at each offset, force the
# detour to the side the expert does NOT take there. A blocker prop occupies the tie-break lane
# so the complement is required by physics rather than by convention -- which is the specific
# weakness of TOPOLOGY's `h*`, a number chosen to balance the sweep and which nothing stops a
# policy from ignoring by lifting over.


def bypass_side(barrier_offset: float) -> str:
    """Which side ``route()``'s tie-break detours to, at this barrier offset: ``'-y'``/``'+y'``.

    This mirrors the tie-break in :meth:`arbiter.collect.context.TaskContext.route` exactly, and
    exists so the side stops being an emergent property of that ``<=`` and becomes a stated
    part of the spec. The gate, the blocker placement and the evaluator all need to agree on
    it; three copies of a tie-break is how they come to disagree.
    """
    _, cy = WORKSPACE.centre
    start_y = cy                      # TOPOLOGY's start and goal both sit on the centre line
    hw = barrier_half_width()
    y_lo, y_hi = barrier_offset - hw, barrier_offset + hw
    return "-y" if abs(start_y - y_lo) <= abs(start_y - y_hi) else "+y"


def forced_bypass_side(barrier_offset: float) -> str:
    """The held-out side at this offset -- the complement of :func:`bypass_side`."""
    return "+y" if bypass_side(barrier_offset) == "-y" else "-y"


def bypass_y(barrier_offset: float, side: str, *, around_margin: float) -> float:
    """Lateral coordinate the detour must reach to clear the barrier on ``side``.

    ``around_margin`` is passed in rather than imported: this module is stdlib-only and the
    margin lives in ``arbiter.collect.constants``, which is free to import from here but not the
    other way round. It is measured to the OUTSIDE of the hand, not the end-effector origin --
    the reason it is 0.12 and not 0.06 is that the gripper fouled the barrier corner while the
    cube cleared, and the arm ground there for ~300 steps without the success predicate or the
    achievability gate noticing.
    """
    if side not in ("-y", "+y"):
        raise ValueError(f"side must be '-y' or '+y', got {side!r}")
    hw = barrier_half_width()
    if side == "-y":
        return barrier_offset - hw - around_margin
    return barrier_offset + hw + around_margin


def bypass_is_reachable(barrier_offset: float, side: str, *, around_margin: float) -> bool:
    """Does the detour waypoint for ``side`` fall inside the validated workspace?

    This is the gate on the whole axis. The forced side is the far side of the barrier, so its
    waypoint sits further out than anything TOPOLOGY currently demonstrates, and at the wider
    offsets it lands ON the workspace boundary. Collecting 180 episodes of an arm straining at
    its reach limit is exactly the failure the achievability gate exists to prevent.
    """
    y = bypass_y(barrier_offset, side, around_margin=around_margin)
    return WORKSPACE.y_min <= y <= WORKSPACE.y_max


#: The blocker that forces the complement. A collider, unlike the target pads -- the pads are
#: deliberately `kinematicEnabled` with no `CollisionAPI` so a cube lands on them rather than
#: bouncing off, whereas this prop's entire function is to make one lane impassable.
#:
#: Sized to span the bypass lane it occupies, and tall enough that lifting over it is not a
#: cheap third homotopy class: TOPOLOGY's own cap is `TOPOLOGY_H_MAX = 0.13`, set by
#: measurement (h=0.14 failed at all three offsets in the exhaustive gate, the only
#: unachievable configuration in the whole sweep), so a blocker at 0.16 is above anything the
#: arm was ever shown to clear while carrying the cube.
#: Barrier offsets the DETOUR axis may use. NOT `TOPOLOGY_OFFSETS`, and the exclusion is a
#: measurement rather than a preference: at +/-0.06 the forced detour lane lands at y=+/-0.280,
#: which is EXACTLY the workspace boundary -- zero margin, with the arm at its lateral reach
#: limit while carrying a cube. Within +/-0.03 the forced lane keeps at least 3 cm of margin.
#:
#: The plan's pre-condition for this axis was "gate before collecting -- the widest detour
#: reaches the workspace edge, and 180 episodes of a struggling arm is the failure this gate
#: exists to prevent." This constant is that gate's answer.
DETOUR_OFFSETS = [-0.03, 0.0, 0.03]

#: Blocker height. The criterion is the **arm's carry height**, not the barrier's.
#:
#: This was 0.16, chosen as "taller than any swept barrier" (`TOPOLOGY_H_MAX = 0.13`) on the
#: reasoning that lifting over would then be unavailable. Measured, that reasoning was wrong: over
#: 36 rollouts the end-effector apex runs **0.222 - 0.349 m**, so 36/36 flew over a 0.16 m
#: blocker and only the cube -- carried ~4 cm below the gripper -- grazed the top. Two episodes
#: cleared it outright and scored a "success" that was a lift-over rather than a detour.
#:
#: 0.40 sits above every measured apex, and a gated run confirmed the expert still makes the
#: forced detour (100% at h=0.09 and 0.12) while the policy scores 0/18 with no lift-overs.
#:
#: Raising it costs no visibility, unlike the barrier: the barrier straddles the camera's sight
#: line (camera and object both on y=0), which is what capped it at 0.13, whereas the blocker is
#: displaced laterally with its nearest edge at y=-0.10, so the sight line clears it by 10 cm at
#: any height.
BLOCKER_HEIGHT_M = 0.40

#: The original 0.16 m prop, kept only to reproduce the runs measured against it (the v6/v7
#: DETOUR grids). It does NOT seal the lane -- see above. New work should not use it.
BLOCKER_LEGACY_HEIGHT_M = 0.16

#: Deprecated alias. `BLOCKER_HEIGHT_M` is now the tall value, so these are the same number;
#: retained so scripts written against the interim name keep working.
BLOCKER_TALL_HEIGHT_M = BLOCKER_HEIGHT_M
#: Lateral span (world Y). Wide enough that the blocker reaches from the barrier's edge to past
#: the workspace boundary at every usable offset, because the blocker's inner edge is placed
#: flush against the barrier's -- see `blocker_xy`.
#:
#: It is NOT centred on the detour lane, which was the first attempt and was wrong: a 0.14 m
#: pad centred on the lane at y=-0.22 spans [-0.29,-0.15], while the barrier at offset 0 ends at
#: -0.10. That leaves a **5 cm gap** between the two obstacles, and 5 cm is a corridor the arm
#: can aim at -- so the policy could slip between barrier and blocker, take neither detour, and
#: the axis would report a success while testing nothing. Exactly the class of bug that gave
#: TOPOLOGY clean-looking data for a configuration that was never built.
#:
#: 0.22 covers the widest case: at offset +/-0.03 the near barrier edge sits 0.07 from centre
#: and the workspace ends at 0.28, so 0.21 m of span is needed. The excess hangs past the
#: workspace boundary, which is harmless -- nothing has to reach there.
BLOCKER_WIDTH_M = 0.22
BLOCKER_DEPTH_M = 0.02                   # depth (world X), same thin profile as the barrier


def blocker_xy(barrier_offset: float, *, around_margin: float) -> tuple[float, float]:
    """Where the blocker sits, so that it seals the lane ``route()`` would otherwise take.

    Its inner edge is flush with the barrier's edge on the tie-break side, and it extends
    outward from there. Barrier and blocker then form ONE continuous obstruction with a single
    gap, on the forced side -- which is the whole point: the detour has to be the complement,
    not a shortcut between two separate props.

    ``around_margin`` is accepted for signature symmetry with :func:`bypass_y` and to keep every
    caller passing the same geometry source; the placement itself is set by the barrier edge and
    the blocker's own width, not by the margin.
    """
    cx, _ = WORKSPACE.centre
    hw = barrier_half_width()
    side = bypass_side(barrier_offset)
    half_blk = 0.5 * BLOCKER_WIDTH_M
    if side == "-y":
        return (cx, barrier_offset - hw - half_blk)
    return (cx, barrier_offset + hw + half_blk)


def blocker_span(barrier_offset: float, *, around_margin: float) -> tuple[float, float]:
    """``(y_inner, y_outer)`` of the blocker, for the gate's no-gap and no-overlap checks."""
    _, by = blocker_xy(barrier_offset, around_margin=around_margin)
    half_blk = 0.5 * BLOCKER_WIDTH_M
    return (by - half_blk, by + half_blk)


def topology_endpoints() -> tuple[tuple[float, float], tuple[float, float]]:
    """``(start_xy, goal_xy)`` for the TOPOLOGY axis, in table coordinates."""
    cx, cy = WORKSPACE.centre
    return ((cx + TOPOLOGY_START_DX, cy), (cx + TOPOLOGY_GOAL_DX, cy))


def transport_target(condition) -> tuple[float, float] | None:
    """Where a transport-type axis expects the object to end up, in table coordinates.

    Owned by the spec because three things must agree on it: the scripted expert that drives
    the object there, the success predicate that checks it arrived, and the policy evaluator
    that scores a rollout. Written out separately in the evaluator it would drift from the
    expert, and the symptom would be an axis where the policy is graded against a target the
    demonstrations never used -- a success rate that is wrong while looking plausible.

    Returns None for axes with no transport target (the lift-type axes).
    """
    axis_name = condition.axis
    if axis_name in ("position", "approach"):
        return None
    if axis_name == "topology":
        return topology_endpoints()[1]

    p = condition.params
    cx, cy = WORKSPACE.centre
    # DIRECTION always starts at the centre; the others may carry their own origin.
    if "x" in p and "y" in p and axis_name != "direction":
        cx, cy = float(p["x"]), float(p["y"])

    if "theta_deg" in p:
        radius = float(p.get("radius", p.get("length", 0.18)))
        theta = math.radians(float(p["theta_deg"]))
    elif "length" in p:
        radius = float(p["length"])
        theta = math.radians(float(p.get("azimuth", 0.0)))
    else:
        return None
    return (cx + radius * math.cos(theta), cy + radius * math.sin(theta))



# ──────────────────────────────────────────────────────────────────────────────
# target pads — making a transport goal visible
# ──────────────────────────────────────────────────────────────────────────────

#: ORDER's fixed object and target positions.
#:
#: These lived in ``layout_gen`` while ``transport_target`` lived here, so the spec owned the
#: goal for six axes and a layout module owned it for the seventh -- exactly the split this file
#: exists to prevent. ``layout_gen`` re-exports them, so the four tools that import them from
#: there are unaffected.
ORDER_OBJECT_POS: list[list[float]] = [[0.36, -0.10], [0.36, 0.0], [0.36, 0.10]]
ORDER_TARGET_POS: list[list[float]] = [[0.46, -0.10], [0.46, 0.0], [0.46, 0.10]]

#: Axes whose success predicate is "the object came to rest at a target". Those goals are a
#: location, so the location has to be observable -- see ``PAD_HALF_M`` in
#: ``create_assets.py``. POSITION and APPROACH lift and have no goal location, so they
#: get no pad and their observation is unchanged.
PAD_AXES = frozenset({"extent", "direction", "factorial", "topology", "order"})

#: Each graspable object's matching pad. Colour-matched so a two-object condition says which
#: goal belongs to which block without relying on the instruction.
PAD_FOR_OBJECT: dict[str, str] = {
    "arb_cube_red": "arb_pad_red",
    "arb_cube_blue": "arb_pad_blue",
    "arb_cube_green": "arb_pad_green",
    "arb_cube_yellow": "arb_pad_yellow",
}


def pad_keys_for(axis_name: str) -> list[str]:
    """Pad assets an axis needs, aligned with ``object_keys_for(axis_name)``."""
    if axis_name not in PAD_AXES:
        return []
    return [PAD_FOR_OBJECT[o] for o in object_keys_for(axis_name)]


def pad_keys_for_condition(condition) -> list[str]:
    """Pads this specific condition places, aligned with :func:`target_xys`.

    Differs from :func:`pad_keys_for` for ORDER only: that returns every pad the *scene* spawns
    (four, one per cube any pair uses), while a single condition places the two belonging to
    its own pair.
    """
    return pad_keys_for(condition.axis)


def target_xys(condition) -> list[tuple[float, float]]:
    """Every goal position for a condition, aligned with ``object_keys_for(condition.axis)``.

    One entry for the single-object transport axes, two for ORDER, and empty for the lift axes.
    Wraps :func:`transport_target` rather than restating it, so the pad a policy sees and the
    target the predicate checks cannot end up in different places -- which would be worse than
    no pad at all, since the policy would be trained to aim at the wrong spot.
    """
    if condition.axis == "order":
        return [(float(x), float(y)) for x, y in ORDER_TARGET_POS]
    t = transport_target(condition)
    return [] if t is None else [t]



#: Where an object goes when a condition does not use it.
#:
#: **Behind the head camera**, which is the only place guaranteed out of frame. The slots used to
#: sit at (-1.5 - 0.3i, -1.5, 0.05), off the table but squarely inside the camera's view: every
#: ORDER episode of one pair showed the other pair's two cubes resting on the floor beside the
#: table, plainly resolvable, and no condition describes them. ORDER is the only axis that spawns
#: objects it does not place, so it was the only axis affected -- and it is also the axis where a
#: stray coloured cube is most likely to be read as a cue.
#:
#: The camera sits at (1.0, 0, 1.3) looking at (0.5, 0, 0.8), so its forward is -(1, 0, 1)/sqrt2
#: and a point is behind it exactly when ``px + pz > 2.3``. At pz = 0.05 that means px > 2.25,
#: which is why a slot at x = 2.2 -- the obvious first guess -- is still in frame for i = 0.
#: These start at 3.0, leaving 0.75 m of margin.
PARKING_X0 = 3.0
PARKING_DX = 0.3
PARKING_Y = -1.2
PARKING_Z = 0.05


def parking_slot(i: int) -> tuple[float, float, float]:
    """Off-workspace parking position for spawn index ``i``, clear of the camera frustum."""
    return (PARKING_X0 + PARKING_DX * float(i), PARKING_Y, PARKING_Z)


def object_keys_for(axis_name: str) -> list[str]:
    """Graspable objects an axis uses. One definition, because four callers need to agree.

    ORDER needs a fixed pair. APPROACH needs the bar rather than a cube: a cube has a square
    footprint, so every jaw azimuth grasps it equally well and the axis's swept quantity is
    invisible. Everything else uses one cube.
    """
    if axis_name == "order":
        return list(ORDER_OBJECTS)
    if axis_name == "approach":
        return ["arb_bar_red"]
    return ["arb_cube_red"]


def object_start_yaw(condition) -> float:
    """Yaw the object spawns at, in degrees. Only APPROACH sweeps it."""
    return float(condition.params.get("yaw_deg", 0.0))


def object_start_xy(condition) -> tuple[float, float]:
    """Where the target object spawns for a condition, from its parameter vector.

    One definition, because the achievability gate, the filmstrip renderer and the collector
    must place the object identically -- otherwise the gate certifies a configuration that
    collection never actually runs.
    """
    cx, cy = WORKSPACE.centre
    axis_name = condition.axis
    if axis_name == "topology":
        return topology_endpoints()[0]
    if axis_name == "position":
        return (float(condition.params["x"]), float(condition.params["y"]))
    if axis_name in ("extent", "factorial"):
        return (float(condition.params.get("x", cx)), float(condition.params.get("y", cy)))
    return (cx, cy)                    # direction, approach, order start at the centre


def barrier_heights() -> list[float]:
    """Every barrier height the TOPOLOGY sweep can ask for, trained and swept.

    Lives here rather than in the asset generator so that the generator, the layout generator,
    and the axis all read one definition. When it lived in the pxr-gated generator module, any
    pure-logic consumer had to import pxr to ask a geometry question.
    """
    n = int(round((TOPOLOGY_H_MAX - 0.02) / TOPOLOGY_TEST_STEP))
    sweep = [round(0.02 + i * TOPOLOGY_TEST_STEP, 6) for i in range(n + 1)]
    return sorted({*sweep, *(round(h, 6) for h in TOPOLOGY_TRAIN_H)})


def expected_homotopy(barrier_h: float) -> str:
    """Which path class the expert demonstrates at this barrier height.

    The discontinuity the axis is built around: it flips at ``TOPOLOGY_H_STAR`` while the
    observation changes smoothly.
    """
    return "over" if float(barrier_h) < TOPOLOGY_H_STAR else "around"


TOPOLOGY = Axis(
    name="topology",
    kind="structural",
    unit="cm",
    sweep_param="barrier_h",
    grid=_topology_grid,
    in_train=_topology_in_train,
    sweep_coord=lambda p: 100.0
    * min(abs(float(p["barrier_h"]) - t) for t in TOPOLOGY_TRAIN_H),
    instruction=lambda p, o: (
        f"move the {_obj_phrase(o)} past the barrier and put it on the target"
    ),
    mechanism_metric="topology_correct_rate",
    notes="Flagship. Build first: it exercises the barrier prop and the route expert, the "
          "hardest new machinery. Report where SR collapses relative to where topology flips.",
)


# ──────────────────────────────────────────────────────────────────────────────
# ORDER — structural. Instantiates (obs seen, trajectory novel).
# ──────────────────────────────────────────────────────────────────────────────
# Two objects, two targets, fixed positions; pixel-identical at t=0. Only the instruction —
# and therefore only the required trajectory — differs. Needs the instruction-sensitivity
# control, or a failure is ambiguous with "never reads the instruction", a different paper.

#: Three objects in three fixed slots, and all 3! = 6 orderings of them.
#:
#: The axis used to be a single pair with one held-out reversal, which left it resting on
#: exactly ONE test condition -- too thin to carry a structural claim. Three objects give six
#: orderings, three of which are held out.
#:
#: TRAIN is the three cyclic rotations. That choice is what satisfies the recoverability
#: guarantee *by construction*: across (0,1,2), (1,2,0) and (2,0,1) every object appears in
#: every position of the sequence exactly once. So "blue goes second" is demonstrated somewhere,
#: and no fixed positional habit -- "always move the left one first", "always red first" --
#: survives contact with the training set. The earlier one-pair design had the opposite
#: property: a single instruction, so the policy learned "always red first" and scored 0/20 on
#: the reversal, guaranteed, for any policy.
#:
#: TEST is the three odd permutations, each one transposition from a trained ordering. It also
#: removes the need for a separate contrast pair: training three orderings of the SAME objects
#: is itself the evidence that the ordering words carry information.
ORDER_OBJECTS: tuple[str, str, str] = (
    "arb_cube_red", "arb_cube_blue", "arb_cube_green",
)
ORDER_TRAIN_SEQUENCES: list[tuple[int, ...]] = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]


def _permutations3() -> list[tuple[int, ...]]:
    import itertools
    return list(itertools.permutations(range(len(ORDER_OBJECTS))))


def _kendall(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Number of pairs the two orderings disagree about -- the adjacent-transposition distance.

    The natural "distance from support" for a sequence: how many swaps separate a held-out
    ordering from the nearest demonstrated one. With three objects every test ordering happens
    to sit at distance 1; the metric is written generally so adding a fourth object grades the
    axis without touching anything else.
    """
    import itertools
    pa = {v: i for i, v in enumerate(a)}
    pb = {v: i for i, v in enumerate(b)}
    return sum(1 for x, y in itertools.combinations(range(len(a)), 2)
               if (pa[x] < pa[y]) != (pb[x] < pb[y]))


def _seq_key(seq) -> str:
    return "".join(str(i) for i in seq)


def _order_grid() -> Iterator[Params]:
    for seq in _permutations3():
        yield {"seq": _seq_key(seq)}


def _seq_of(p: Params) -> tuple[int, ...]:
    return tuple(int(c) for c in str(p["seq"]))


def _order_in_train(p: Params) -> bool:
    return _seq_of(p) in ORDER_TRAIN_SEQUENCES


def _order_sweep(p: Params) -> float:
    return float(min(_kendall(_seq_of(p), t) for t in ORDER_TRAIN_SEQUENCES))


def order_slot_objects() -> list[str]:
    """The three cubes in **spawn-slot** order, which never depends on the sequence.

    Slot order is fixed. If the slot a cube spawned in tracked the order it had to be moved in,
    a positional rule would solve the axis without reading the instruction at all.
    """
    return list(ORDER_OBJECTS)


def order_move_sequence(condition) -> list[int]:
    """Indices into :func:`order_slot_objects`, in the order the task must move them."""
    return list(_seq_of(condition.params))


def _order_instruction(p: Params, _o: str) -> str:
    names = [_obj_phrase(ORDER_OBJECTS[i]) for i in _seq_of(p)]
    # Later objects named by colour alone, matching the shorter natural phrasing.
    tail = ", then the " + ", then the ".join(n.split()[0] for n in names[1:])
    return f"put the {names[0]} on its target{tail}"


ORDER = Axis(
    name="order",
    kind="structural",
    unit="ordinal",
    sweep_param="seq",
    grid=_order_grid,
    in_train=_order_in_train,
    sweep_coord=_order_sweep,
    # Named by colour, not by position. "the first block" never said *which* block: "first"
    # was a fixed spawn slot the policy had to infer, so an ORDER failure could not separate
    # "cannot compose a novel sub-task sequence" -- the thing this axis exists to measure --
    # from "cannot resolve an ambiguous referring expression". ORDER is the (observation seen,
    # trajectory novel) cell nothing in v1 covered, so that ambiguity sat directly on the
    # paper's novel claim. The blocks are already visually distinct.
    #
    # The mapping is fixed by OrderTask, which drives objects[0] then objects[1] for "AB" and
    # the reverse for "BA", against object_keys = [red, blue]. So AB is red-then-blue. Asserted
    # in arbiter/collect/tests/test_tasks.py: the wording and the execution must not drift.
    instruction=_order_instruction,
    mechanism_metric="order_correct_rate",
    notes="Check overlap with arXiv 2607.29687 (compositional generalization in sequential "
          "robot tasks) before committing to this axis. Ordinal, so enumerate exhaustively "
          "rather than via sample_sweep.",
)


# ──────────────────────────────────────────────────────────────────────────────
# APPROACH — structural, extension. The repaired v1 Approach.
# ──────────────────────────────────────────────────────────────────────────────
# A pose-invariant object (cube or cylinder) removes v1's confound, where approach was
# perfectly entangled with object pose — front-graspable and top-graspable were two poses of
# the same bar, and v1's own failure analysis showed the policy keying on pose. Here fixture
# walls make only certain azimuths feasible, so the required approach is cued by affordance
# rather than by instruction.

#: Trained bar yaws. phi = yaw + 90 is a bijection mod 180, so sweeping the yaw
#: is the same sweep as the grasp azimuth -- but expressed in the observable.
APPROACH_TRAIN_YAW = [0.0, 45.0]

# Half-open [0, 180): 180 is the same physical grasp as 0, so including it added a duplicate
# condition -- and it was the single azimuth the reachability probe found unreachable, because
# panda_joint7 runs out of range as the wrist rotates (measured position error climbing
# monotonically: 0.0002 m at phi=0 to 0.0226 m at phi=180).
#
# 11.25-degree steps rather than 22.5: with two trained azimuths 45 degrees apart and
# 180-periodic distance, the sweep only spans 0-67.5 degrees, so a coarse step leaves too few
# distinct bins to fit a radius against.
APPROACH_TEST_YAW = [round(11.25 * i, 4) for i in range(16)]


#: The bar's long axis is 10 cm, past the Panda's 8 cm jaw opening, so it can only be grasped
#: across its 3.5 cm width. The required jaw azimuth is therefore fixed by the bar's yaw, which
#: means the *observation* determines it -- the whole point of using a bar.
#:
#: The offset is 0, and that was measured rather than reasoned. `grasp_frame`'s first column is
#: documented as "the azimuth heading" and its test as "jaw axis is the azimuth heading", which
#: reads like the finger-separation axis; it is in fact the direction the fingers are *aligned
#: along*. Assuming the former put every grasp 90 degrees out and the gate returned 0/16, every
#: condition stalling 2.5 cm short with the jaw straddling the 10 cm length. Direct test at
#: yaw=0: phi=0 lifts the bar and closes the jaw to 0.0162 m, half its 3.5 cm width; phi=90
#: stalls. A cube could not have caught this -- a square footprint grasps equally at any
#: azimuth, which is exactly why the old APPROACH passed while measuring nothing.
APPROACH_JAW_OFFSET_DEG = 0.0


def approach_phi(yaw_deg: float) -> float:
    """Required jaw azimuth for a bar spawned at ``yaw_deg``. 180-periodic, like the jaw."""
    return (float(yaw_deg) + APPROACH_JAW_OFFSET_DEG) % 180.0


def _approach_grid() -> Iterator[Params]:
    # Swept as the bar's yaw, which is what a policy can see. phi is carried alongside because
    # the scripted expert needs it, derived here so the two cannot disagree.
    for yaw in APPROACH_TEST_YAW:
        yield {"yaw_deg": yaw, "phi_deg": approach_phi(yaw)}


APPROACH = Axis(
    name="approach",
    kind="structural",
    unit="deg",
    sweep_param="yaw_deg",
    grid=_approach_grid,
    in_train=lambda p: _min_jaw_dist(float(p["yaw_deg"]), APPROACH_TRAIN_YAW) < 1e-6,
    sweep_coord=lambda p: _min_jaw_dist(float(p["yaw_deg"]), APPROACH_TRAIN_YAW),
    instruction=lambda p, o: f"lift the {_obj_phrase(o)} off the table",
    mechanism_metric="wrist_azimuth_error_deg",
    notes="Rebuilt. The axis previously swept the grasp azimuth of a CUBE at a fixed pose with "
          "a fixed instruction, so all 16 conditions were observationally identical -- one "
          "distinct instruction and one distinct object position across the whole axis. The "
          "required azimuth existed only in the expert's choice, invisible to the policy, so a "
          "held-out azimuth was unproducible in principle and a failure would have measured "
          "the ambiguity rather than trajectory generalization. Now the object is a bar whose "
          "long axis exceeds the jaw opening: its yaw dictates the grasp, and the yaw is "
          "visible. A per-condition fixture was the other candidate fix; this needs no fixture.",
)


# ──────────────────────────────────────────────────────────────────────────────
# FACTORIAL — the multiplicative-cost test
# ──────────────────────────────────────────────────────────────────────────────
# Every factor level appears in training; half the combinations do not. This is the cell v1
# structurally could not have, because there every holdout routed recoverability through the
# contralateral arm. Checkerboard keeps each held-out cell adjacent to trained cells on both
# factors, so support distance stays small and roughly matched across held-out cells — the
# comparison is against trained cells at similar d, not against an easier condition.

# Positions constrained so every (position x theta) cell puts its target on the table. With
# r=0.18 and theta in {0,45,90,135} that means x in [0.407, 0.540] and y <= 0.100; an earlier
# choice of (0.38/0.62, +-0.12) put 8 of 16 cells off the table, which would have silently
# turned half the FACTORIAL cells into unreachable-target failures -- and FACTORIAL is the one
# result the multiplicative-cost argument rests on.
FACTORIAL_POSITIONS = [(0.42, -0.16), (0.42, -0.02), (0.52, -0.16), (0.52, -0.02)]
FACTORIAL_THETA = [0.0, 45.0, 90.0, 135.0]


def _factorial_grid() -> Iterator[Params]:
    for pi in range(len(FACTORIAL_POSITIONS)):
        for ti in range(len(FACTORIAL_THETA)):
            x, y = FACTORIAL_POSITIONS[pi]
            yield {
                "pos_idx": pi, "theta_deg": FACTORIAL_THETA[ti],
                "x": x, "y": y, "radius": DIRECTION_RADIUS,
            }


def _factorial_in_train(p: Params) -> bool:
    """Checkerboard: (pos_idx + theta_idx) even is trained."""
    ti = FACTORIAL_THETA.index(float(p["theta_deg"]))
    return (int(p["pos_idx"]) + ti) % 2 == 0


FACTORIAL = Axis(
    name="factorial",
    kind="composition",
    unit="cell",
    sweep_param="cell",
    grid=_factorial_grid,
    in_train=_factorial_in_train,
    sweep_coord=lambda p: 0.0 if _factorial_in_train(p) else 1.0,
    instruction=lambda p, o: (
        f"pick up the {_obj_phrase(o)} and put it on the target to the "
        f"{_compass_label(float(p['theta_deg']))} side"
    ),
    mechanism_metric="held_cell_vs_trained_cell_sr",
    notes="Decides whether coverage cost is additive or multiplicative — the single result the "
          "whole argument rests on. NOT a sweep: enumerate exhaustively rather than via "
          "sample_sweep, because every one of the few held-out cells carries signal.",
)


# ──────────────────────────────────────────────────────────────────────────────
# registry
# ──────────────────────────────────────────────────────────────────────────────

AXES: dict[str, Axis] = {
    a.name: a
    for a in (POSITION, EXTENT, DIRECTION, TOPOLOGY, ORDER, APPROACH, FACTORIAL)
}

METRIC_AXES = [a.name for a in AXES.values() if a.kind == "metric"]
STRUCTURAL_AXES = [a.name for a in AXES.values() if a.kind == "structural"]
COMPOSITION_AXES = [a.name for a in AXES.values() if a.kind == "composition"]


def axis(name: str) -> Axis:
    if name not in AXES:
        raise KeyError(f"Unknown axis '{name}'. Available: {', '.join(sorted(AXES))}")
    return AXES[name]



#: Axes whose object always spawns in the same place, so that place can be marked ON the scene
#: instead of being left implicit.
#:
#: The rule is per-axis and follows from what each axis sweeps. POSITION sweeps the spawn
#: position itself across 667 cells -- marking it would paint the answer on the table. EXTENT,
#: DIRECTION and FACTORIAL sweep the *target*, so their goal stays a movable pad. TOPOLOGY,
#: ORDER and APPROACH hold both ends fixed and vary something else entirely (barrier height,
#: ordering, bar yaw), so their geometry is scene furniture and belongs in the scene.
#:
#: An earlier objection to per-axis scene design was too broad: a distinctive appearance does
#: let a policy identify the axis, but identifying the axis does not help with the held-out
#: condition *within* it. The confound only exists if a marking reveals the swept quantity, and
#: none of these do.
START_ZONE_AXES = frozenset({"extent", "direction", "topology", "order", "approach"})


def start_zone_xys(axis_name: str) -> list[tuple[float, float]]:
    """Fixed spawn positions to mark for an axis; empty when the spawn is swept."""
    if axis_name not in START_ZONE_AXES:
        return []
    if axis_name == "order":
        return [(float(x), float(y)) for x, y in ORDER_OBJECT_POS]
    conds = enumerate_conditions(AXES[axis_name], object_keys_for(axis_name))
    xys = {tuple(round(float(v), 6) for v in object_start_xy(c)) for c in conds}
    if len(xys) != 1:
        raise ValueError(
            f"{axis_name} is in START_ZONE_AXES but has {len(xys)} distinct spawn positions; "
            f"a marked start must not move with the condition."
        )
    return [tuple(next(iter(xys)))]




#: Axes that get a painted transport CORRIDOR on the table: both endpoints fixed, so the lane
#: encodes only fixed geometry.
#:
#: TOPOLOGY only, and the restriction is the whole point. A lane is safe exactly when it does
#: not encode the axis's swept quantity. TOPOLOGY sweeps barrier height, so a lane between its
#: fixed endpoints says "carry from here to there" and reveals nothing -- and because the
#: barrier fully spans the lane, the picture also shows the direct path being blocked, which is
#: the task. DIRECTION and EXTENT are the opposite case: a bearing rose or a distance ruler
#: would paint the swept azimuth or the swept length onto the table and hand the policy the
#: answer, which is why those two keep a movable pad showing only the current condition's goal.
#: ORDER's endpoints are fixed too, but its colour-matched pads already say the same thing and
#: three lanes would only add clutter.
LANE_AXES = frozenset({"topology"})

#: Narrower than the barrier (half-width 0.10) so the barrier visibly spans the corridor, and
#: wider than the 5 cm cube so the cube sits inside it.
LANE_WIDTH_M = 0.14


def transport_lane(axis_name: str) -> tuple[tuple[float, float], float, float] | None:
    """``((mid_x, mid_y), length_x, width_y)`` for the painted corridor, or None."""
    if axis_name not in LANE_AXES:
        return None
    (sx, sy), (gx, gy) = topology_endpoints()
    if abs(sy - gy) > 1e-9:
        raise ValueError("lane assumes the endpoints share a y; rotate the prim otherwise")
    return ((0.5 * (sx + gx), sy), abs(gx - sx), LANE_WIDTH_M)


def support_regime(axis_name: str, params: Params) -> str:
    """Whether a condition sits INSIDE the hull of trained values, or beyond it.

    "interpolation" | "extrapolation" | "n/a" for the ordinal and categorical axes.

    Distance from support says how FAR a condition is; this says on which SIDE. They are
    different claims and the benchmark was conflating them: EXTENT's trained lengths are
    [0.10, 0.14, 0.18, 0.20], so test lengths 0.11-0.19 are interpolation between demonstrations
    while 0.21-0.35 are extrapolation past all of them, and one radius fitted over both describes
    neither.

    It also sharpens what TOPOLOGY now shows. Trained heights sit at both ends (0.02/0.05 and
    0.12/0.13), so EVERY test height is interpolation in the swept parameter -- yet the required
    path class still flips at h*. That is the asymmetry the paper predicts, in its cleanest
    form: the observation varies smoothly and interpolably while the required trajectory does
    not.
    """
    def _regime(v: float, trained: list[float]) -> str:
        return "interpolation" if min(trained) <= v <= max(trained) else "extrapolation"

    if axis_name == "extent":
        return _regime(float(params["length"]), list(EXTENT_TRAIN_LENGTHS))
    if axis_name == "topology":
        return _regime(float(params["barrier_h"]), list(TOPOLOGY_TRAIN_H))
    if axis_name == "direction":
        return _regime(float(params["theta_deg"]) % 360.0,
                       [t % 360.0 for t in DIRECTION_TRAIN_THETA])
    if axis_name == "approach":
        return _regime(float(params["yaw_deg"]) % 180.0,
                       [y % 180.0 for y in APPROACH_TRAIN_YAW])
    if axis_name == "position":
        xs = [c[0] for c in POSITION_TRAIN_CELLS]
        ys = [c[1] for c in POSITION_TRAIN_CELLS]
        inside = (min(xs) <= float(params["x"]) <= max(xs)
                  and min(ys) <= float(params["y"]) <= max(ys))
        return "interpolation" if inside else "extrapolation"
    return "n/a"


#: Phase-level instructions, for binding language to motion rather than to task identity.
#:
#: The whole measured deficit is that the text pathway carries no trajectory information: every
#: training episode had ONE constant string, so the action DiT's text cross-attention only ever
#: saw a task label. Per-phase text is the training-time fix, and GR00T already ships the channel
#: -- `sub_tasks` in `episodes.jsonl`, read by `create_language_from_meta` with a half-horizon
#: lead.
#:
#: The design constraint that makes or breaks it: **the route word has to be compositional.**
#: TOPOLOGY only ever demonstrates the LEFT detour (see TRAINING_EXCLUSIONS), so if "right"
#: appears nowhere else in training it is out of vocabulary for the action head and no amount of
#: phase labelling helps -- the same gradient-starvation confound that made the old all-over
#: TOPOLOGY and single-instruction ORDER unable to report anything.
#:
#: Checked rather than assumed: DIRECTION and FACTORIAL train **three** of the eight compass
#: words, not all of them -- `near`, `near-right`, `right`. That is enough, because **"right" is
#: one of them**, and "left" is grounded separately by TOPOLOGY's own around-left demos. So the
#: wording here deliberately reuses `_compass_label` across axes: at test time "around the right
#: side of the barrier" composes a bearing word the model has grounded (from DIRECTION) with a
#: barrier context it has seen (from TOPOLOGY's left detours), instead of asking for a token it
#: has never received gradient on.
#:
#: The corollary is a real constraint on this vocabulary: a phase word that appears on ONE axis
#: only cannot be composed, so new sub-instruction wording has to be checked against what the
#: training set actually grounds, not against what reads naturally.
#:
#: Phases come from the recorded gripper command, not from the expert's own labels: the archives
#: store `actions` and `obs/joint_pos` only, and `ctx.phases` is never written. Validated across
#: all 645 collected episodes -- 435 show two gripper transitions (pick / transport / release),
#: 45 show six (ORDER's three blocks), and 165 show one (POSITION and APPROACH, the lift axes,
#: which never release). Grasp lands at 76 +/- 4 steps.
#: Phase counts are fixed by the gripper segmentation, not by taste. One transition (the lift
#: axes, which never release) gives two intervals; two transitions (the place axes) give three;
#: six (ORDER's three blocks) give seven, which is why ORDER folds release into carry. Every
#: frame must land inside some span: GR00T assigns `""` to any frame no sub_task covers, and an
#: empty instruction measured **0/6 even with no obstacle present** -- far outside the input
#: distribution, since every collected episode carried a real sentence.
_PHASE_TEXT_LIFT = ("reach for the {obj}", "lift the {obj} off the table")


#: The observation key the policy reads language from, for a given channel.
#:
#: Lives here because **two processes must agree on it and they cannot import each other**: the
#: trainer's modality config imports `gr00t`, which the sim venv does not have, and the evaluator
#: runs in the sim venv. GR00T resolves language at inference as
#: `observation["language"][modality_keys[0]]`, so a mismatch is not a warning -- the model
#: either raises or silently receives no instruction, and a Phase B run would then read as "the
#: intervention did nothing" when the plumbing never connected.
#:
#: "sub_task" is passed through literally because GR00T's loader special-cases it
#: (`LANG_KEYS = ["task", "sub_task"]`); "task" resolves to the dataset's annotation key.
LANG_OBS_KEY = {
    "task": "annotation.human.action.task_description",
    "sub_task": "sub_task",
}


def lang_obs_key(channel: str) -> str:
    """Observation key for a language channel; raises on anything GR00T would not recognise."""
    try:
        return LANG_OBS_KEY[channel]
    except KeyError:
        raise ValueError(
            f"unknown language channel {channel!r}; GR00T recognises "
            f"{sorted(LANG_OBS_KEY)} only"
        ) from None


def sub_task_spans(gripper_closed: list[bool], n_phases: int) -> list[tuple[int, int]]:
    """Frame spans for ``n_phases`` phase texts, from the recorded gripper command.

    ``gripper_closed[i]`` is whether the commanded gripper is closed at frame ``i``. Its
    transitions are the phase boundaries: close ends a reach, open ends a carry.

    Two properties this guarantees, both load-bearing:

    - **Complete coverage.** The returned spans tile ``[0, len(gripper_closed))`` with no gaps.
      GR00T assigns ``""`` to any frame no sub_task covers, and an empty instruction measured
      **0/6 even with no obstacle** -- it is far outside the input distribution, since every
      collected episode carried a real sentence. A gap here would train the text pathway on
      nothing for those frames.
    - **Exact count.** Returns exactly ``n_phases`` spans or raises. A silent mismatch would
      shift every phase text onto the wrong motion -- the failure would look like the
      intervention not working, rather than like a bug.

    Trailing frames beyond the last transition extend the final span, which is why the last
    phase text of each axis is written to remain true through the settle.
    """
    n = len(gripper_closed)
    if n == 0:
        raise ValueError("empty gripper trace")
    edges = [i for i in range(1, n) if gripper_closed[i] != gripper_closed[i - 1]]
    bounds = [0] + edges + [n]
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    # Fold trailing spans into the last phase so coverage stays complete and the count matches.
    if len(spans) > n_phases:
        head = spans[: n_phases - 1]
        spans = head + [(spans[n_phases - 1][0], n)]
    if len(spans) != n_phases:
        raise ValueError(
            f"gripper trace yields {len(spans)} span(s) for {n_phases} phase text(s); "
            f"transitions at {edges}. The phase vocabulary and the segmentation disagree -- "
            f"fix the vocabulary rather than padding, or the texts land on the wrong motion."
        )
    return spans


def sub_instructions_from_attrs(attrs: dict[str, ParamValue]) -> list[str]:
    """Phase-level instructions for one episode, in execution order.

    Returned as text only; the frame ranges come from the recorded gripper command at conversion
    time, so this function stays stdlib-only and independent of h5py.

    Raises rather than guessing, for the same reason `instruction_from_attrs` does: a silently
    generic phase string would train the text pathway on a label that carries nothing, which is
    precisely the condition being corrected.
    """
    ax_name = attrs.get("axis")
    if ax_name is None:
        raise KeyError("episode attrs carry no 'axis'; cannot derive sub-instructions")
    obj = attrs.get("object_key")
    if obj is None:
        raise KeyError(f"episode attrs carry no 'object_key' (axis {ax_name})")
    ax = str(ax_name)
    phrase = _obj_phrase(str(obj))
    params = params_from_attrs(attrs)

    if ax in ("position", "approach"):
        # Lift axes: no target, so no transport phase and nothing to say about a route.
        return [t.format(obj=phrase) for t in _PHASE_TEXT_LIFT]

    if ax == "order":
        # TWO phases per block, not three, and the count is forced by the segmentation rather
        # than chosen: three blocks give six gripper transitions, hence seven frame intervals
        # (reach1, carry1, reach2, carry2, reach3, carry3, tail). A release phase of its own
        # would need nine texts for seven intervals. Folding release into the carry it ends
        # matches intervals exactly, and the tail after the last release extends the final carry
        # so **every frame is covered** -- see `sub_task_spans`, and note that GR00T assigns
        # `""` to any frame no sub_task covers, which measured 0/6 even with no obstacle.
        out: list[str] = []
        for slot in _seq_of(params):
            ph = _obj_phrase(ORDER_OBJECTS[slot])
            out += [f"reach for the {ph}", f"carry the {ph} to its target and let go"]
        return out

    if ax == "topology":
        h = float(params["barrier_h"])
        if expected_homotopy(h) == "over":
            mid = f"lift the {phrase} over the barrier"
        else:
            side = bypass_side(float(params.get("barrier_offset", 0.0)))
            # Same vocabulary DIRECTION grounds: -y is "left", +y is "right" in the camera frame.
            word = "left" if side == "-y" else "right"
            mid = f"carry the {phrase} around the {word} side of the barrier"
        return [f"reach for the {phrase}", mid,
                f"put the {phrase} down on the target and let go"]

    if ax in ("direction", "factorial"):
        word = _compass_label(float(params["theta_deg"]))
        return [f"reach for the {phrase}",
                f"carry the {phrase} to the {word} side",
                f"put the {phrase} down on the target and let go"]

    if ax == "extent":
        return [f"reach for the {phrase}",
                f"carry the {phrase} out to the target",
                f"put the {phrase} down on the target and let go"]

    raise ValueError(f"no phase vocabulary for axis {ax!r}")


def instruction_from_attrs(attrs: dict[str, ParamValue]) -> str:
    """Re-derive an episode's instruction from its recorded HDF5 attrs.

    The archive stores the instruction *string* it was collected under, but that string is a pure
    function of (axis, parameter vector, object) and the spec owns that function. Reading the
    recorded copy makes every wording change a re-collection, and lets the dataset's language
    drift away from what ``axes.py`` says the axis asks for.

    Nothing about a demonstration depends on the wording: the scripted experts route by geometry
    and never read the instruction. So re-deriving here is what makes an instruction fix cost a
    re-conversion instead of ~50 minutes of simulator time -- which is how the camera-frame
    bearing bug (see ``_COMPASS``) was repaired without recollecting DIRECTION or FACTORIAL.

    Raises rather than guessing: a silently generic task string would be fatal for ORDER, where
    the instruction is the only thing separating the two conditions.
    """
    ax_name = attrs.get("axis")
    if ax_name is None:
        raise KeyError("episode attrs carry no 'axis'; cannot re-derive the instruction")
    obj = attrs.get("object_key")
    if obj is None:
        raise KeyError(f"episode attrs carry no 'object_key' (axis {ax_name})")
    return axis(str(ax_name)).instruction(params_from_attrs(attrs), str(obj))
