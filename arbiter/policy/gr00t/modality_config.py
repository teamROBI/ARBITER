# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""GR00T modality config for ARBITER. Passed to the trainer via ``--modality_config_path``.

GR00T's own ``libero_sim`` embodiment is the closest shipped match — Franka Panda in sim, and
TANGO deliberately copied LIBERO's ``agentview`` camera geometry — but it cannot be reused,
because its action space is a 7-dim end-effector pose delta
(``x y z roll pitch yaw gripper``). TANGO's action is **8-dim joint position targets**:
7 arm joints plus the gripper.

That convention is pinned by the plan and is not a detail to trade away for a shipped config.
v1's action dimensionality disagreed with itself across three documents, and re-deriving EEF
deltas from the recorded joint targets would reintroduce exactly the commanded-versus-achieved
ambiguity the collector was built to avoid: the recorded action is what ``movep`` sent, and an
EEF delta recovered from it is a different quantity.

So TANGO registers under ``EmbodimentTag.NEW_EMBODIMENT``, which is GR00T's supported path for
a robot it does not ship.

    python -m gr00t.experiment.launch_finetune \\
        --embodiment_tag new_embodiment \\
        --modality_config_path arbiter/policy/gr00t/modality_config.py \\
        ...

The keys here must match the slice names in each dataset's ``meta/modality.json``, which
``scripts/bench/write_modality_json.py`` emits from the dataset's own ``info.json`` so the two
cannot drift.
"""

from __future__ import annotations

import os

from gr00t.configs.data.data_config import ModalityConfig
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS, register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType

#: 7 arm joints + gripper. Order matches `_FRANKA_JOINTS` in `arbiter/splits/isaaclab2lerobot.py`
#: and the `names` recorded in every dataset's info.json.
ARBITER_JOINTS = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7", "panda_finger_joint1",
]

#: Which language channel the model reads: "task" (default) or "sub_task", from ARBITER_LANG_KEY.
#:
#: This is the Phase B switch. GR00T's loader treats a modality key that is literally in its
#: `LANG_KEYS = ["task", "sub_task"]` as coming from `episodes.jsonl` rather than through
#: `modality.json`; anything else must be an `annotation.*` key. And **only the first key reaches
#: the model** -- extra keys are paraphrase augmentation at training time, dropped at inference
#: -- so this is a choice between the two channels, not a way to have both.
#:
#: "task" is the per-episode string every dataset carries, and it is exactly what the measured
#: deficit is about: one constant sentence per episode, so the action DiT's four text
#: cross-attention blocks only ever saw a task label and the channel carries nothing about the
#: path. "sub_task" is per-phase text with a half-horizon lead, written by
#: `scripts/bench/emit_sub_tasks.py` from `axes.sub_instructions_from_attrs`.
#:
#: Kept switchable rather than replaced so the v6/v7 runs stay reproducible -- they are the
#: "before" half of the comparison, and a silently different language channel would make the
#: retrain uninterpretable.
LANG_KEY = os.environ.get("ARBITER_LANG_KEY", "task").strip().lower()
if LANG_KEY not in ("task", "sub_task"):
    raise ValueError(
        f"ARBITER_LANG_KEY must be 'task' or 'sub_task', got {LANG_KEY!r}. "
        f"Those are the only two channels GR00T's loader recognises (LANG_KEYS)."
    )
#: The dataset's own annotation key is what "task" resolves to; "sub_task" is passed through
#: literally because the loader special-cases it.
LANG_MODALITY_KEY = ("annotation.human.action.task_description" if LANG_KEY == "task"
                     else "sub_task")

#: Instruction-dropout probability, from ARBITER_LANG_DROPOUT. 0.0 (off) unless set.
#:
#: This is what makes a CAG sweep interpretable. Counterfactual Action Guidance needs
#: ``pi(a | o, null)``, and a normally-trained checkpoint has never seen a null instruction --
#: every episode carries a real sentence. TANGO measured an empty string at **0/6 with no
#: obstacle in the scene at all**, so that policy is out-of-distribution rather than
#: unconditioned, and ``a_cond - a_uncond`` would be a difference against noise.
#:
#: Training with dropout p makes "" in-distribution, so a *single* checkpoint serves both
#: branches of the guidance mix. That is simpler and better-controlled than pairing the policy
#: with a separately-trained vision-only model, because the two branches then differ in exactly
#: one input and share every weight.
#:
#: p = 0.5 is the usual classifier-free-guidance choice and is what Phase 0.5 uses. The dropout
#: also doubles as the no-language baseline: at inference, feeding "" gives what vision and
#: proprioception alone achieve.
LANG_DROPOUT = float(os.environ.get("ARBITER_LANG_DROPOUT", "0.0"))
if not 0.0 <= LANG_DROPOUT < 1.0:
    raise ValueError(
        f"ARBITER_LANG_DROPOUT must be in [0, 1), got {LANG_DROPOUT}. At 1.0 the language "
        f"channel never carries anything and the run is a no-language ablation, which should "
        f"be stated as such rather than reached through a dropout rate."
    )

#: Action chunk length. 16 steps at TANGO's 30 Hz control rate is ~0.53 s of future actions,
#: which is the same horizon `libero_sim` uses at 20 Hz over a shorter span. Worth revisiting:
#: TOPOLOGY's around-route detour lasts longer than half a second, so a chunk that cannot span
#: the commitment point may be unable to express the routing decision at all.
ACTION_HORIZON = 16

#: "abs" (default) or "rel", from ARBITER_ACTION_REP.
#:
#: TANGO's premise is about trajectory generalization, and the two parameterizations are not
#: equally able to express it. An absolute joint target is specific to where the object is; a
#: delta encodes a motion that is the same wherever the object sits. So a radius measured under
#: ABS may be partly a property of the action space rather than of the policy -- which is a
#: result worth having either way, and a confound the paper has to report.
#:
#: GR00T does the conversion itself: with `rep=RELATIVE` it subtracts the reference state at
#: load and adds it back at inference, so the evaluator still receives absolute targets and
#: needs no change. Transforming the dataset instead would double-apply against this.
ACTION_REP = os.environ.get("ARBITER_ACTION_REP", "abs").strip().lower()
if ACTION_REP not in ("abs", "rel"):
    raise ValueError(f"ARBITER_ACTION_REP must be 'abs' or 'rel', got {ACTION_REP!r}")


def _action_configs() -> list[ActionConfig] | None:
    """One ActionConfig per action modality key; GR00T zips them positionally.

    The arm joints go relative and the gripper stays absolute. A gripper command is a target
    width, not a displacement, and v1 drew the same line -- its gate records that the arms were
    relative while the grippers were not.

    `state_key` is left unset, so GR00T falls back to the matching key name. That works only
    because the state modality keys ARE the joint names, which is why the per-joint layout is
    kept rather than regrouped: it makes ABS and REL differ in exactly one field.
    """
    if ACTION_REP == "abs":
        return None
    return [
        ActionConfig(
            rep=(ActionRepresentation.RELATIVE if j != "panda_finger_joint1"
                 else ActionRepresentation.ABSOLUTE),
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        )
        for j in ARBITER_JOINTS
    ]


ARBITER_MODALITY_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        # Both streams on every axis. The wrist view is why APPROACH is learnable at all --
        # the bar's yaw is what dictates the grasp, and the head camera sees it obliquely.
        modality_keys=["image", "wrist_image"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=list(ARBITER_JOINTS),
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=list(ARBITER_JOINTS),
        action_configs=_action_configs(),
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[LANG_MODALITY_KEY],
    ),
}

# Registering twice raises, and the trainer imports this module once per rank worker, so a
# re-import must be a no-op rather than a crash.
if EmbodimentTag.NEW_EMBODIMENT.value not in MODALITY_CONFIGS:
    register_modality_config(ARBITER_MODALITY_CONFIG, EmbodimentTag.NEW_EMBODIMENT)

    # RELATIVE in the modality config is only half the switch: the processor gates every
    # relative conversion on the model's `use_relative_action` as well, so setting one without
    # the other silently trains on absolute targets while claiming to be relative.
    #
    # It is not exposed through FinetuneConfig, and load_dict's model branch replaces the whole
    # config rather than patching a field. Setting the dataclass default here works because the
    # trainer imports this module before it constructs the model config -- and it keeps the
    # change in TANGO's own file rather than in the shared Isaac-GR00T fork.
    #
    # CAVEAT, verified rather than assumed: the vendored trainer already sets it unconditionally
    # at `gr00t/experiment/launch_finetune.py:92` --
    #
    #     config.model.use_relative_action = True
    #
    # with no branch on action representation. So under a fine-tune launched through that path
    # the flag is True for **ABS runs as well**, and the block below is not what turns it on.
    # It is harmless there only because every relative conversion in the processor is additionally
    # gated on `action_configs is not None`, which an ABS modality config leaves unset -- so the
    # branches never fire. That is a coincidence of the current processor, not a guarantee: if
    # those gates ever change, ABS runs start training on relative targets silently, and the
    # symptom would be an action-space comparison where both arms are the same arm.
    # Left as-is deliberately -- the fork is shared, so it is not TANGO's to patch.
    # Training seed. `config.data.seed` defaults to 42 and `FinetuneConfig` never sets it, so
    # every run so far has been the same seed -- which is why no per-axis v6/v7 delta can be read
    # as a difference (POSITION's -13.3 pt is 1.6 sigma with both ID controls at 100%).
    #
    # Patched the same way `use_relative_action` is, and for the same reason it works: the trainer
    # calls `load_modality_config` (launch_finetune.py:57) BEFORE `get_default_config()` at line
    # 59, so a dataclass default set here lands in the constructed config. Keeps the change in
    # TANGO's own file rather than in the shared Isaac-GR00T fork.
    if os.environ.get("ARBITER_SEED"):
        _seed = int(os.environ["ARBITER_SEED"])
        from gr00t.configs.data.data_config import DataConfig
        DataConfig.seed = _seed
        print(f"[ARBITER] training seed = {_seed} (default is 42)")

    if ACTION_REP == "rel":
        from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
        Gr00tN1d7Config.use_relative_action = True
        print("[ARBITER] use_relative_action=True (arm joints relative, gripper absolute)")

    if LANG_DROPOUT > 0.0:
        # Wrap the episode loader's language construction rather than transforming the dataset.
        # Two reasons it belongs here:
        #
        #  1. The dataset on disk stays a single artifact. A dropout-augmented *copy* would
        #     double 24G of LeRobot data and, worse, bake one dropout draw into the data so
        #     every epoch saw the same episodes unconditioned -- which is not dropout, it is a
        #     fixed partition.
        #  2. `create_language_from_meta` is called once per episode load, so the draw is
        #     per-episode-per-load: across epochs the same episode is sometimes conditioned and
        #     sometimes not, which is the classifier-free-guidance recipe. Dropping per *frame*
        #     would instead teach the model that an instruction can vanish mid-trajectory, a
        #     distribution nothing at inference reproduces.
        #
        # The null token is "" because that is already what GR00T assigns to any frame no
        # sub_task covers, so it is the same token the loader can emit on its own.
        import random as _random

        from gr00t.data.dataset import lerobot_episode_loader as _loader

        _EpisodeLoader = _loader.LeRobotEpisodeLoader
        _orig_create_language = _EpisodeLoader.create_language_from_meta

        def _create_language_with_dropout(self, episode_meta, nframes, lang_key):
            langs = _orig_create_language(self, episode_meta, nframes, lang_key)
            if _random.random() < LANG_DROPOUT:
                return [""] * len(langs)
            return langs

        _create_language_with_dropout.__doc__ = (
            "create_language_from_meta with per-episode instruction dropout at "
            f"p={LANG_DROPOUT}, so the null instruction is in-distribution and one checkpoint "
            "can serve both branches of a CAG mix. See arbiter/policy/cag.py."
        )
        _EpisodeLoader.create_language_from_meta = _create_language_with_dropout
        print(f"[ARBITER] instruction dropout p={LANG_DROPOUT} (per episode load); "
              f"null token is the empty string")
    print(f"[ARBITER] registered modality config under "
          f"'{EmbodimentTag.NEW_EMBODIMENT.value}': "
          f"{len(ARBITER_JOINTS)}-dim joint action/state, "
          f"{ACTION_HORIZON}-step chunk, cameras {ARBITER_MODALITY_CONFIG['video'].modality_keys}")
