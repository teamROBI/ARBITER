#!/usr/bin/env bash
# Fine-tune GR00T N1.7-3B on the v2 training set. Every non-obvious flag here was arrived at by
# measurement or by a failure, so they are documented rather than left to be rediscovered.
#
#   bash scripts/bench/run_finetune.sh                      # absolute joint targets
#   ACTION_REP=rel bash scripts/bench/run_finetune.sh       # relative arm joints
#   STEPS=4000 PER_GPU=128 bash scripts/bench/run_finetune.sh
#   USE_WANDB=0 bash scripts/bench/run_finetune.sh          # local only
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

ACTION_REP="${ACTION_REP:-abs}"
PER_GPU="${PER_GPU:-192}"
NGPU="${NGPU:-4}"
STEPS="${STEPS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-400}"
# Retained checkpoints. TANGO kept every one (STEPS/SAVE_EVERY + 1 = 11 at these defaults), and
# that alone is why its runs are 58-71G each and 515G of its 1.1T is checkpoints. Three is
# enough to hold the final plus two fallbacks; raise it deliberately, per run, if a mid-training
# checkpoint is actually going to be evaluated.
SAVE_LIMIT="${SAVE_LIMIT:-3}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-arbiter}"
DATASET="${DATASET:-${REPO_ROOT}/data/datasets/lerobot_merged/arbiter_all}"
NAME="${NAME:-n1d7_${ACTION_REP}_b$((PER_GPU*NGPU))_s${STEPS}}"
PORT="${PORT:-$((29500 + RANDOM % 400))}"

GLOBAL=$((PER_GPU*NGPU))
# Learning rate scales with batch, sqrt not linear. 1e-4 was chosen for global batch 64; linear
# scaling to 768 would give 1.2e-3, aggressive for fine-tuning a 3B model on 630 episodes.
LR="${LR:-$(python3 -c "print(f'{1e-4*((${GLOBAL}/64)**0.5):.3g}')")}"

OUT="${OUT:-${REPO_ROOT}/data/output/train/${NAME}}"
LOG="${REPO_ROOT}/data/logs/v2/${NAME}.log"
mkdir -p "$(dirname "${LOG}")" "${OUT}"

[[ -d "${DATASET}" ]] || { echo "[ERROR] no dataset at ${DATASET}" >&2; exit 1; }
[[ -f "${DATASET}/meta/modality.json" ]] || {
    echo "[ERROR] ${DATASET} has no meta/modality.json." >&2
    echo "        Run: python scripts/bench/write_modality_json.py" >&2; exit 1; }

echo "[TRAIN] action rep : ${ACTION_REP}"
echo "[TRAIN] batch      : ${PER_GPU}/gpu x ${NGPU} = ${GLOBAL}   lr ${LR}"
echo "[TRAIN] steps      : ${STEPS}  (save every ${SAVE_EVERY})"
echo "[TRAIN] dataset    : ${DATASET}"
echo "[TRAIN] output     : ${OUT}"
echo "[TRAIN] wandb      : $([[ "${USE_WANDB}" == "1" ]] && echo "${WANDB_PROJECT}/${NAME}" || echo off)"
echo "[TRAIN] log        : ${LOG}"

cd "${REPO_ROOT}/third_party/Isaac-GR00T"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/arbiter/policy/gr00t:${PWD}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"
export TOKENIZERS_PARALLELISM=false

# Selects the action representation in modality_config.py. Relative needs BOTH
# rep=RELATIVE on the modality keys and the model's use_relative_action; that module sets the
# second, since FinetuneConfig does not expose it.
export ARBITER_ACTION_REP="${ACTION_REP}"

# Read by modality_config.py in every rank worker. Exported rather than merely inherited so
# they appear in the run's own environment record: a checkpoint whose language channel or
# dropout rate is unknown afterwards cannot be used as either branch of a CAG mix.
export ARBITER_LANG_KEY="${LANG_KEY:-task}"
export ARBITER_LANG_DROPOUT="${LANG_DROPOUT:-0.0}"

# 192/gpu peaks at 45,398 MiB of 49,140 (92%) and 256 OOMs at 48,422. 160 uses the same memory
# as 192 but is 8% slower, so 192 is the ceiling worth taking. expandable_segments because 92%
# leaves ~3.7 GB and allocator fragmentation over thousands of steps is the realistic way this
# OOMs when a short probe did not.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
    WANDB_ARGS=(--use_wandb --wandb_project "${WANDB_PROJECT}")
else
    export WANDB_MODE=disabled
fi

# torchrun, not plain python. With num_gpus>1 GR00T selects DeepSpeed, which needs a launcher to
# set WORLD_SIZE/LOCAL_RANK; run as a single process it falls through to HF Trainer's
# DataParallel and dies on a CPU-resident parameter.
#
# stdbuf and a real file, not a pipe into tail: tail buffers until exit, which left a 2.4 h run
# with no visible loss or step count at all.
exec stdbuf -oL -eL "${REPO_ROOT}/venvs/gr00t/bin/torchrun" \
    --nproc_per_node="${NGPU}" --master_port="${PORT}" \
    -m gr00t.experiment.launch_finetune \
    --base_model_path "${BASE_MODEL:-nvidia/GR00T-N1.7-3B}" \
    --dataset_path "${DATASET}" \
    --embodiment_tag new_embodiment \
    --modality_config_path "${REPO_ROOT}/arbiter/policy/gr00t/modality_config.py" \
    --num_gpus "${NGPU}" --global_batch_size "${GLOBAL}" \
    --max_steps "${STEPS}" --save_steps "${SAVE_EVERY}" \
    --save_total_limit "${SAVE_LIMIT}" --save_only_model \
    --dataloader_num_workers "${WORKERS:-6}" \
    --learning_rate "${LR}" --warmup_ratio 0.05 \
    --output_dir "${OUT}" --experiment_name "${NAME}" \
    "${WANDB_ARGS[@]}" \
    2>&1 | tee "${LOG}"
