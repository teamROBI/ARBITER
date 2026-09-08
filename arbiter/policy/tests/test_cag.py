# Copyright 2026 ARBITER Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Counterfactual Action Guidance.

No server and no GPU: the inner policy is a stub that records what it was asked, which is enough
to pin every property that matters. The two that matter most:

* ``omega = 1`` must be *exactly* the conditional policy, byte for byte and with a single forward
  pass. If it were merely numerically close, the ``omega = 1`` column of the sweep would not be
  comparable to the unguided baseline, and the whole tradeoff curve loses its origin.
* An unvalidated unconditional branch must **refuse to run**. GR00T has never seen an empty
  instruction unless it was trained with dropout, and a curve built against that is an artifact
  rather than a measurement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arbiter.policy import cag  # noqa: E402

LANG_KEY = "annotation.human.action.task_description"
JOINTS = [f"joint_{i}" for i in range(3)]


class StubPolicy:
    """Returns an action that depends on the instruction, so mixing is observable."""

    def __init__(self, horizon: int = 4):
        self.horizon = horizon
        self.seen: list[str] = []

    def get_action(self, observation: dict) -> dict:
        instr = observation["language"][LANG_KEY][0][0]
        self.seen.append(instr)
        # Conditional branch -> 1.0; null branch -> 0.0. Makes the arithmetic legible.
        v = 0.0 if instr == "" else 1.0
        return {k: np.full((self.horizon, 1), v, dtype=np.float32) for k in JOINTS}


def _obs(instruction: str = "carry the red block around the right side of the barrier") -> dict:
    return {
        "video": {"image": np.zeros((1, 1, 4, 4, 3), dtype=np.uint8)},
        "state": {k: np.zeros((1, 1, 1), dtype=np.float32) for k in JOINTS},
        "language": {LANG_KEY: [[instruction]]},
    }


def _cag(omega: float, **kw) -> tuple[cag.CagPolicy, StubPolicy]:
    inner = StubPolicy()
    kw.setdefault("null_is_in_distribution", True)
    return cag.CagPolicy(inner, omega, **kw), inner


# ---------------------------------------------------------------- the identity at omega = 1

def test_omega_one_is_exactly_the_conditional_policy_and_one_forward_pass():
    policy, inner = _cag(1.0)
    out = policy.get_action(_obs())
    for k in JOINTS:
        assert np.array_equal(out[k], np.ones((4, 1), dtype=np.float32))
    assert inner.seen == ["carry the red block around the right side of the barrier"], (
        "omega=1 must not evaluate the unconditional branch at all"
    )
    assert policy.n_forward == 1


def test_omega_one_needs_no_validated_null():
    """It cancels, so the guard must not obstruct the baseline column of the sweep."""
    inner = StubPolicy()
    p = cag.CagPolicy(inner, 1.0, null_is_in_distribution=False)
    assert p.get_action(_obs()) is not None


# ---------------------------------------------------------------- the arithmetic

def test_omega_zero_is_the_unconditional_policy():
    policy, inner = _cag(0.0)
    out = policy.get_action(_obs())
    for k in JOINTS:
        assert np.allclose(out[k], 0.0)
    assert inner.seen[1] == "", "the second call must carry the null instruction"


@pytest.mark.parametrize("omega,expected", [(0.5, 0.5), (1.5, 1.5), (2.0, 2.0), (5.0, 5.0)])
def test_guidance_extrapolates_linearly(omega, expected):
    """uncond=0, cond=1 -> mixed = 0 + omega*(1-0) = omega."""
    policy, _inner = _cag(omega)
    out = policy.get_action(_obs())
    for k in JOINTS:
        assert np.allclose(out[k], expected), f"omega={omega} gave {out[k].flat[0]}"


def test_mixing_preserves_shape_and_dtype():
    policy, _ = _cag(2.0)
    out = policy.get_action(_obs())
    for k in JOINTS:
        assert out[k].shape == (4, 1)
        assert out[k].dtype == np.float32


def test_mixing_applies_across_the_whole_chunk():
    """Guidance is applied to every step of the horizon, not just the first."""
    inner = StubPolicy(horizon=16)
    policy = cag.CagPolicy(inner, 3.0, null_is_in_distribution=True)
    out = policy.get_action(_obs())
    for k in JOINTS:
        assert out[k].shape == (16, 1)
        assert np.allclose(out[k], 3.0)


# ---------------------------------------------------------------- the refusal

def test_unvalidated_null_is_refused():
    with pytest.raises(ValueError, match="unconditional branch"):
        cag.CagPolicy(StubPolicy(), 2.0, null_is_in_distribution=False)


def test_refusal_message_names_the_remedy():
    try:
        cag.CagPolicy(StubPolicy(), 2.0, null_is_in_distribution=False)
    except ValueError as e:
        msg = str(e)
        assert "instruction-dropout" in msg or "instruction dropout" in msg
        assert "0/6" in msg, "the message should carry the evidence, not just an assertion"
    else:
        pytest.fail("expected a refusal")


def test_negative_omega_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        cag.CagPolicy(StubPolicy(), -1.0, null_is_in_distribution=True)


# ---------------------------------------------------------------- observation surgery

def test_replace_instruction_swaps_only_the_language():
    o = _obs()
    swapped = cag.replace_instruction(o, "")
    assert swapped["language"][LANG_KEY] == [[""]]
    assert o["language"][LANG_KEY] == [["carry the red block around the right side of the barrier"]], (
        "the original observation must not be mutated"
    )
    # Shallow by design: the video tensor is shared, not copied, so guidance does not double
    # memory traffic per chunk.
    assert swapped["video"] is o["video"]
    assert swapped["state"] is o["state"]


def test_missing_language_field_raises():
    with pytest.raises(KeyError, match="no 'language' field"):
        cag.replace_instruction({"video": {}, "state": {}}, "")


def test_ambiguous_language_key_raises():
    """Only the first modality key reaches the model, so two keys means the caller is guessing."""
    o = _obs()
    o["language"] = {LANG_KEY: [["a"]], "sub_task": [["b"]]}
    with pytest.raises(ValueError, match="2 keys"):
        cag.replace_instruction(o, "")


def test_empty_language_dict_raises():
    o = _obs()
    o["language"] = {}
    with pytest.raises(ValueError, match="non-empty dict"):
        cag.replace_instruction(o, "")


# ---------------------------------------------------------------- branch-mismatch guards

def test_key_mismatch_between_branches_raises():
    with pytest.raises(ValueError, match="action keys differ"):
        cag.mix_actions({"a": np.zeros(2)}, {"b": np.zeros(2)}, 2.0)


def test_shape_mismatch_between_branches_raises():
    with pytest.raises(ValueError, match="shape"):
        cag.mix_actions({"a": np.zeros((4, 1))}, {"a": np.zeros((8, 1))}, 2.0)


def test_empty_action_dict_raises():
    with pytest.raises(ValueError, match="empty"):
        cag.mix_actions({}, {}, 2.0)


# ---------------------------------------------------------------- provenance

def test_describe_records_the_doubled_cost_and_the_null():
    policy, _ = _cag(2.0)
    policy.get_action(_obs())
    policy.get_action(_obs())
    d = policy.describe()
    assert d["policy"] == "cag"
    assert d["omega"] == 2.0
    assert d["null_instruction"] == ""
    assert d["null_is_in_distribution"] is True
    assert d["forward_passes"] == 4, "two chunks at two passes each"


def test_on_call_hook_sees_all_three_action_dicts():
    """So a rollout can log how far guidance moved the action, not just the result."""
    captured = []
    inner = StubPolicy()
    policy = cag.CagPolicy(
        inner, 2.0, null_is_in_distribution=True,
        on_call=lambda c, u, m: captured.append((c, u, m)),
    )
    policy.get_action(_obs())
    assert len(captured) == 1
    c, u, m = captured[0]
    assert np.allclose(c[JOINTS[0]], 1.0)
    assert np.allclose(u[JOINTS[0]], 0.0)
    assert np.allclose(m[JOINTS[0]], 2.0)
