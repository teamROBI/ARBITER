# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for scripted expert tasks, and the deterministic collection queue.

A task's job is to demonstrate one condition. The queue's job is to make sure every condition
gets exactly the demonstrations it is supposed to, in a reproducible order.

The queue design is carried over because it has one property worth keeping: **it only advances
on a confirmed success.** A failed attempt leaves the queue untouched and the same condition is
retried, so the dataset ends up with a fixed number of *successful* demonstrations per
condition rather than a fixed number of *attempts*. Without that, a condition the expert finds
hard silently ends up under-represented, and the training distribution stops matching the
declared one — which for a coverage benchmark would corrupt the measurement itself.

What changed from v1: the queue is over v2 :class:`~arbiter.suites.spec.Condition` objects
carrying parameter vectors, not ``(object_key, area_id)`` pairs. Splits come from the axis, so
a collection run cannot disagree with the axis definition about what is in training.
"""

from __future__ import annotations

from typing import Callable, Generator

from arbiter.suites.spec import Axis, Condition, enumerate_conditions


class Task:
    """One scripted demonstration behaviour.

    Subclass and implement :meth:`run` as a generator composing
    :class:`~arbiter.collect.context.TaskContext` primitives with ``yield from``. Raise
    ``TaskFailed(retriable=True)`` to reject an attempt without abandoning the condition.
    """

    #: Set False for tasks where a failed grasp means the condition is unreachable rather
    #: than the attempt being unlucky.
    allow_retry: bool = True

    def __init__(self, condition: Condition) -> None:
        self.condition = condition

    @property
    def instruction(self) -> str:
        """Language for this episode. Comes from the condition, so it matches the axis."""
        return self.condition.instruction

    def run(self, ctx) -> Generator:
        raise NotImplementedError


class ConditionQueue:
    """Deterministic queue over (condition x repeat), advancing only on confirmed success.

    Reproducible from the axis alone: no RNG decides *which* conditions get collected, only
    per-episode nuisance variation (spawn yaw) inside an episode. Two runs of the same axis
    therefore cover the same conditions in the same order.

    ``partition_idx`` / ``partition_total`` split the work across parallel collectors by
    striding, so each gets an interleaved slice rather than a contiguous block — a contiguous
    split would make one worker collect only far-from-support conditions, and a crash would
    then bias the dataset along the very axis being measured.
    """

    def __init__(
        self,
        axis: Axis,
        object_keys: list[str],
        *,
        demos_per_condition: int = 5,
        splits: tuple[str, ...] = ("train",),
        partition_idx: int = 0,
        partition_total: int = 1,
        predicate: Callable[[Condition], bool] | None = None,
    ) -> None:
        if partition_total < 1 or not (0 <= partition_idx < partition_total):
            raise ValueError(
                f"bad partition {partition_idx}/{partition_total}"
            )
        self.axis = axis
        self.demos_per_condition = max(1, int(demos_per_condition))

        conds = enumerate_conditions(axis, object_keys, splits=splits)  # type: ignore[arg-type]
        if predicate is not None:
            # TOPOLOGY bakes its barrier into the stage at build time, so one launch can only
            # collect the conditions matching the loaded height. Filtering here rather than
            # skipping at run time keeps ``total_episodes`` honest -- a progress counter that
            # includes conditions this process will never attempt reads as a stalled run.
            conds = [c for c in conds if predicate(c)]
            if not conds:
                raise ValueError(f"predicate excluded every condition of axis '{axis.name}'")
        conds.sort(key=lambda c: c.key())
        self._all = conds
        self._pending: list[Condition] = [
            c for i, c in enumerate(conds) if i % partition_total == partition_idx
        ]
        self._remaining: dict[str, int] = {
            c.key(): self.demos_per_condition for c in self._pending
        }
        self._order: list[Condition] = list(self._pending)
        self._cursor = 0

    # ── inspection ───────────────────────────────────────────────────────────

    @property
    def total_episodes(self) -> int:
        return len(self._order) * self.demos_per_condition

    @property
    def collected(self) -> int:
        return self.total_episodes - sum(self._remaining.values())

    @property
    def done(self) -> bool:
        return all(v == 0 for v in self._remaining.values())

    def summary(self) -> str:
        return (
            f"axis={self.axis.name} conditions={len(self._order)} "
            f"demos/condition={self.demos_per_condition} "
            f"collected={self.collected}/{self.total_episodes}"
        )

    # ── iteration ────────────────────────────────────────────────────────────

    def next_condition(self) -> Condition | None:
        """The condition to attempt next, or None when the queue is exhausted.

        Round-robins rather than draining one condition at a time, so an interrupted run still
        yields a dataset spread across the whole axis instead of a prefix of it.
        """
        if self.done:
            return None
        n = len(self._order)
        for offset in range(n):
            c = self._order[(self._cursor + offset) % n]
            if self._remaining[c.key()] > 0:
                self._cursor = (self._cursor + offset) % n
                return c
        return None

    def confirm_success(self, condition: Condition) -> None:
        """Record one successful demonstration. Only this advances the queue."""
        k = condition.key()
        if k not in self._remaining:
            raise KeyError(f"condition not in this queue partition: {k}")
        if self._remaining[k] > 0:
            self._remaining[k] -= 1
        self._cursor = (self._cursor + 1) % max(len(self._order), 1)

    def record_failure(self, condition: Condition) -> None:
        """A failed attempt. Deliberately does not decrement — it advances the cursor only,
        so a persistently-failing condition does not block the rest of the axis."""
        if condition.key() not in self._remaining:
            raise KeyError(f"condition not in this queue partition: {condition.key()}")
        self._cursor = (self._cursor + 1) % max(len(self._order), 1)
