# Copyright 2026 ARBITER Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Counterfactual Action Guidance (CAG) as an inference-time wrapper.

The baseline ARBITER has to beat, from *When Vision Overrides Language* (arXiv 2602.17659).
Classifier-free guidance over the text channel:

    a = a_uncond + omega * (a_cond - a_uncond)

`omega = 1` recovers the conditional policy exactly; `omega = 0` recovers the unconditional one;
`omega > 1` extrapolates away from the unconditioned prediction, amplifying whatever the
instruction contributed. Training-free, which is what makes it a serious competitor: it needs no
extra demonstrations and no architectural change.

**Why this is the method argument, not just a baseline.** ``omega`` is a *single global scale*
applied to every dimension of behaviour at once. Raising it to win ARBITER's language-authority
cells necessarily pulls the same lever in the cells where vision is the correct authority. One
monotone gain cannot move two rows up and two rows down, so no value of ``omega`` is correct on
the whole four-cell table. Measuring that curve is the point of running this.

.. warning::
   **The unconditional branch must be in-distribution or the whole sweep is meaningless.**

   GR00T trained on ARBITER-style data has never seen an empty instruction: every episode carries
   a real sentence. TANGO measured an empty string at **0/6 with no obstacle in the scene at
   all** -- so ``pi(a | o, "")`` from such a checkpoint is not a language-unconditioned policy, it
   is an out-of-distribution one, and ``a_cond - a_uncond`` is then a difference against noise.
   A tradeoff curve built on it would be an artifact.

   The fix is a checkpoint fine-tuned with instruction dropout, so that the null instruction is
   in-distribution and one checkpoint can serve both branches. `require_validated_null` exists to
   make that a deliberate assertion rather than an accident.

Two forward passes per chunk, so rollouts run at roughly half speed.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import numpy as np

#: The null instruction. Empty string is what an instruction-dropout fine-tune is trained to
#: treat as "no language", matching the dropout token used during training.
EMPTY_INSTRUCTION = ""


class _Policy(Protocol):
    def get_action(self, observation: dict) -> dict: ...


def replace_instruction(observation: dict, instruction: str) -> dict:
    """Shallow-copy ``observation`` with its language field replaced.

    Shallow on purpose: the video tensors are large and are not modified, so copying them per
    chunk would double memory traffic for no reason. Only the ``language`` sub-dict is rebuilt.
    """
    if "language" not in observation:
        raise KeyError(
            "observation carries no 'language' field, so CAG cannot form an unconditional "
            f"branch. Keys present: {sorted(observation)}"
        )
    lang = observation["language"]
    if not isinstance(lang, dict) or not lang:
        raise ValueError(
            f"observation['language'] must be a non-empty dict keyed by the modality key, got "
            f"{type(lang).__name__}"
        )
    if len(lang) != 1:
        # GR00T resolves language as observation["language"][modality_keys[0]]; more than one
        # key means the caller is guessing which one reaches the model.
        raise ValueError(
            f"observation['language'] has {len(lang)} keys ({sorted(lang)}); exactly one is "
            f"expected because only the first modality key reaches the model"
        )
    key = next(iter(lang))
    out = dict(observation)
    out["language"] = {key: [[instruction]]}
    return out


def mix_actions(cond: dict, uncond: dict, omega: float) -> dict:
    """``uncond + omega * (cond - uncond)``, per action key.

    Mixing happens on the *chunk*: GR00T returns ``ACTION_HORIZON`` steps at once, and guidance
    applies to all of them. That is also the mechanism behind the prediction this project makes
    about CAG on temporally extended factors -- guidance is stateless, so it re-argues the route
    from scratch at every chunk boundary rather than holding a commitment across them.
    """
    if not cond:
        raise ValueError("conditional action dict is empty")
    if set(cond) != set(uncond):
        raise ValueError(
            f"action keys differ between branches: conditional-only {sorted(set(cond) - set(uncond))}, "
            f"unconditional-only {sorted(set(uncond) - set(cond))}"
        )
    out: dict[str, Any] = {}
    for k, c in cond.items():
        u = uncond[k]
        c_arr = np.asarray(c, dtype=np.float64)
        u_arr = np.asarray(u, dtype=np.float64)
        if c_arr.shape != u_arr.shape:
            raise ValueError(
                f"action {k!r} has shape {c_arr.shape} conditional vs {u_arr.shape} "
                f"unconditional; the two branches must be the same policy on the same "
                f"observation, differing only in the instruction"
            )
        mixed = u_arr + omega * (c_arr - u_arr)
        out[k] = mixed.astype(np.asarray(c).dtype, copy=False)
    return out


def require_validated_null(
    *, null_is_in_distribution: bool, omega: float, null_instruction: str
) -> None:
    """Refuse to run a guidance sweep against an unvalidated unconditional branch.

    ``omega == 1`` is exempt: it reduces to the conditional policy exactly, so the unconditional
    branch cancels and never influences the output. Every other value depends on it.
    """
    if abs(omega - 1.0) < 1e-12:
        return
    if not null_is_in_distribution:
        raise ValueError(
            f"CAG at omega={omega} needs an unconditional branch the policy understands, and "
            f"this run has not asserted one.\n"
            f"The null instruction is {null_instruction!r}. A checkpoint trained without "
            f"instruction dropout has never seen it -- TANGO measured an empty instruction at "
            f"0/6 with no obstacle present at all -- so a_cond - a_uncond would be a difference "
            f"against out-of-distribution noise and the resulting tradeoff curve would be an "
            f"artifact.\n"
            f"Either serve an instruction-dropout checkpoint and pass the flag asserting so, or "
            f"run at omega=1 (which is just the conditional policy)."
        )


class CagPolicy:
    """Wraps a policy so each `get_action` becomes a guided mix of two forward passes.

    The inner policy is called twice on the same observation -- once with the episode's
    instruction, once with the null -- so both branches come from one checkpoint and one server.
    That is the intended configuration: with instruction dropout, a single fine-tune makes both
    branches in-distribution, which is simpler and better-controlled than pairing the policy with
    a separately-trained vision-only model.
    """

    def __init__(
        self,
        inner: _Policy,
        omega: float,
        *,
        null_instruction: str = EMPTY_INSTRUCTION,
        null_is_in_distribution: bool = False,
        on_call: Callable[[dict, dict, dict], None] | None = None,
    ):
        if omega < 0.0:
            raise ValueError(f"omega must be non-negative, got {omega}")
        require_validated_null(
            null_is_in_distribution=null_is_in_distribution,
            omega=omega,
            null_instruction=null_instruction,
        )
        self.inner = inner
        self.omega = float(omega)
        self.null_instruction = null_instruction
        self.null_is_in_distribution = null_is_in_distribution
        self.on_call = on_call
        #: Counted so a rollout record can state the doubled inference cost rather than implying
        #: the run was as cheap as an unguided one.
        self.n_forward = 0

    def get_action(self, observation: dict) -> dict:
        cond = self.inner.get_action(observation)
        self.n_forward += 1
        if abs(self.omega - 1.0) < 1e-12:
            # Exactly the conditional policy. Skipping the second pass is not an optimisation
            # detail -- it keeps the omega=1 column of the sweep identical to the unguided
            # baseline instead of merely numerically close to it.
            return cond
        uncond = self.inner.get_action(
            replace_instruction(observation, self.null_instruction)
        )
        self.n_forward += 1
        mixed = mix_actions(cond, uncond, self.omega)
        if self.on_call is not None:
            self.on_call(cond, uncond, mixed)
        return mixed

    def describe(self) -> dict:
        """Provenance for the rollout record, so a curve cannot be read out of context."""
        return {
            "policy": "cag",
            "omega": self.omega,
            "null_instruction": self.null_instruction,
            "null_is_in_distribution": self.null_is_in_distribution,
            "forward_passes": self.n_forward,
        }
