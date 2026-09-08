#!/usr/bin/env bash
# Phase 0.5 -- fine-tune with instruction dropout, so a CAG sweep is interpretable.
#
# Counterfactual Action Guidance needs pi(a | o, null). A normally-trained GR00T has never seen a
# null instruction -- every episode carries a real sentence -- and TANGO measured an empty string
# at 0/6 *with no obstacle in the scene at all*. So that policy is out-of-distribution rather
# than unconditioned, and a_cond - a_uncond would be a difference against noise. Every point of
# an omega sweep built on it would be an artifact.
#
# Training with dropout p makes "" in-distribution, so ONE checkpoint serves both branches of the
# mix. That is better controlled than pairing the policy with a separately-trained vision-only
# model: the two branches then share every weight and differ in exactly one input.
#
# It also yields the no-language baseline for free -- feed "" at inference and you have what
# vision and proprioception alone achieve.
#
# WHY THE sub_task CHANNEL AND THIS DATASET. Three of the four arbitration cells are only
# interpretable on sub_task, because topology's task-level instruction is route-silent and never
# names a side. So the guided policy has to read the channel that grounds a side. Training on the
# same sub_task dataset as the no-dropout checkpoint keeps the pair comparable: same data, same
# channel, same action space -- only dropout differs.
#
# Usage:
#   bash scripts/bench/train_dropout.sh
#   LANG_DROPOUT=0.3 STEPS=4000 bash scripts/bench/train_dropout.sh
#
# Needs all four GPUs for several hours (TANGO's equivalent runs ~29 min per 400 steps).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# 0.5 is the usual classifier-free-guidance choice: it spends half the gradient on each branch,
# so neither the conditional nor the unconditional policy is a rounding error.
export LANG_DROPOUT="${LANG_DROPOUT:-0.5}"
export LANG_KEY="${LANG_KEY:-sub_task}"
export ACTION_REP="${ACTION_REP:-rel}"
export DATASET="${DATASET:-${REPO_ROOT}/data/datasets/lerobot_merged/arbiter_v8_subtask}"
export STEPS="${STEPS:-4000}"
export SAVE_EVERY="${SAVE_EVERY:-400}"
export SAVE_LIMIT="${SAVE_LIMIT:-3}"
export NAME="${NAME:-n1d7_${ACTION_REP}_${LANG_KEY}_drop$(printf '%.0f' "$(echo "${LANG_DROPOUT} * 100" | bc)")}"

[[ -d "${DATASET}" ]] || { echo "[TRAIN] dataset not found: ${DATASET}" >&2; exit 1; }
[[ -f "${DATASET}/meta/modality.json" ]] || {
    echo "[TRAIN] ${DATASET} has no meta/modality.json; run scripts/bench/write_modality_json.py" >&2
    exit 1
}

# The sub_task channel is useless if the dataset carries no sub_tasks: GR00T assigns "" to every
# frame no span covers, and an all-empty language channel trains a no-language policy while
# claiming to train a phase-conditioned one.
if [[ "${LANG_KEY}" == "sub_task" ]]; then
    n_sub=$(python3 -c "
import json,sys
p='${DATASET}/meta/episodes.jsonl'
rows=[json.loads(l) for l in open(p) if l.strip()]
print(sum(1 for r in rows if r.get('sub_tasks')), len(rows))
")
    set -- ${n_sub}
    if [[ "$1" != "$2" || "$1" == "0" ]]; then
        echo "[TRAIN] LANG_KEY=sub_task but only $1/$2 episodes carry sub_tasks." >&2
        echo "        Run scripts/bench/emit_sub_tasks.py first -- it refuses to write a" >&2
        echo "        partially annotated dataset, which is why a partial one means it never ran." >&2
        exit 1
    fi
    echo "[TRAIN] sub_tasks present on $1/$2 episodes"
fi

# The spec agrees with this dataset's language, or cells.py will later speak sentences the
# checkpoint never saw. Cheap, and it fails before hours of GPU time rather than after.
python3 scripts/bench/check_language_fidelity.py --dataset "${DATASET}" >/dev/null || {
    echo "[TRAIN] language fidelity check failed on ${DATASET}; see" >&2
    echo "        scripts/bench/check_language_fidelity.py --dataset ${DATASET}" >&2
    exit 1
}
echo "[TRAIN] language fidelity: spec re-derives this dataset's instructions exactly"

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
if (( free_mib < 40000 )); then
    echo "[TRAIN] the busiest GPU has only ${free_mib} MiB free, and this run needs ~45 GB on" >&2
    echo "        each of four. Something else is training. Wait rather than OOM hours in." >&2
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >&2
    exit 1
fi

echo "[TRAIN] name      : ${NAME}"
echo "[TRAIN] dataset   : ${DATASET}"
echo "[TRAIN] channel   : ${LANG_KEY}   dropout p=${LANG_DROPOUT}   action=${ACTION_REP}"
echo "[TRAIN] steps     : ${STEPS} (save every ${SAVE_EVERY}, keep ${SAVE_LIMIT})"
echo
exec bash scripts/bench/run_finetune.sh
