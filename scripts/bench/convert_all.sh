#!/usr/bin/env bash
# Archive -> LeRobot -> merged training set, for every axis present in the collect directory.
#
#   bash scripts/bench/convert_all.sh
#   COLLECT=data/collect ROOT=data/datasets/lerobot bash scripts/bench/convert_all.sh
#
# Runs in the **gr00t** venv: lerobot is installed there, not in venvs/sim. Conversion is
# CPU-bound video encoding, so axes run in parallel up to WORKERS.
#
# TOPOLOGY yields one dataset per (height, offset) because each is its own archive, and the
# merge takes them all -- the merged set is what training reads.
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

COLLECT="${COLLECT:-data/collect}"
ROOT="${ROOT:-data/datasets/lerobot}"
MERGED="${MERGED:-data/datasets/lerobot_merged/arbiter_all}"
WORKERS="${WORKERS:-6}"
PY_BIN="${REPO_ROOT}/venvs/gr00t/bin/python"
LOGDIR="${LOGDIR:-data/logs/v2/convert}"

[[ -x "${PY_BIN}" ]] || { echo "[ERROR] missing gr00t venv at ${PY_BIN}" >&2; exit 1; }
mkdir -p "${ROOT}" "${LOGDIR}"

mapfile -t ARCHIVES < <(ls -1 "${COLLECT}"/*.hdf5 2>/dev/null)
[[ ${#ARCHIVES[@]} -gt 0 ]] || { echo "[ERROR] no archives in ${COLLECT}" >&2; exit 1; }
echo "[CONVERT] ${#ARCHIVES[@]} archives -> ${ROOT}  (${WORKERS} parallel)"

FAILED="$(mktemp)"; : > "${FAILED}"
n=0
for arch in "${ARCHIVES[@]}"; do
    stem="$(basename "${arch}" .hdf5)"
    # The axis is the archive stem minus TOPOLOGY's _hNNNN_oNNNN tag.
    axis="${stem%%_h[0-9]*}"
    repo="arbiter_${stem}"
    (
        rm -rf "${ROOT}/${repo}"
        "${PY_BIN}" arbiter/splits/isaaclab2lerobot.py \
            --dataset_file "${arch}" --expected-axis "${axis}" \
            --split train --overwrite \
            --repo_id "${repo}" --root "${ROOT}" > "${LOGDIR}/${stem}.log" 2>&1
        rc=$?
        if [[ ${rc} -ne 0 ]]; then
            echo "  FAIL ${stem} (rc=${rc}) -> ${LOGDIR}/${stem}.log"
            echo "${stem}" >> "${FAILED}"
        else
            echo "  ok   ${stem}  $(grep -oE 'instructions: [0-9]+ re-derived[^,]*' "${LOGDIR}/${stem}.log" | tail -1)"
        fi
    ) &
    n=$((n + 1))
    (( n % WORKERS == 0 )) && wait
done
wait

if [[ -s "${FAILED}" ]]; then
    echo "[CONVERT] $(wc -l < "${FAILED}") archive(s) failed; not merging"
    rm -f "${FAILED}"; exit 1
fi
rm -f "${FAILED}"

# Every episode's language must come from the spec, not from the recorded copy. A conversion
# that fell back for all of them would mean a spec fix silently did not reach the dataset.
if grep -hq "no instruction was re-derived" "${LOGDIR}"/*.log 2>/dev/null; then
    echo "[ERROR] some archive produced no re-derived instruction; see ${LOGDIR}" >&2
    exit 1
fi

# Refuse to merge if ROOT holds a dataset with no matching archive. Such a directory is a
# leftover from an earlier conversion under an earlier spec, and merging it is silent
# TEST-SET CONTAMINATION: when TOPOLOGY's trained heights moved from [0.02 0.03 0.05 0.07] to
# [0.02 0.05 0.12 0.13], the h0030 and h0070 datasets stayed behind, a `arbiter_*` glob swept
# them into the merge, and 90 episodes at what are now TEST heights were trained on. Nothing
# downstream could have detected it -- the episodes are well-formed and labelled split=train,
# because they *were* train when they were written.
STALE=()
for d in "${ROOT}"/arbiter_*; do
    [[ -d "${d}" ]] || continue
    stem="$(basename "${d}")"; stem="${stem#arbiter_}"
    [[ -f "${COLLECT}/${stem}.hdf5" ]] || STALE+=("${stem}")
done
if [[ ${#STALE[@]} -gt 0 ]]; then
    echo "[ERROR] ${#STALE[@]} dataset(s) in ${ROOT} have no archive in ${COLLECT}:" >&2
    printf '    %s\n' "${STALE[@]}" >&2
    echo "        They predate the current spec. Move them aside, then re-run." >&2
    exit 1
fi

echo "[MERGE] -> ${MERGED}"
rm -rf "${MERGED}"
# Built from the archive list, not from a glob of the output directory, so the merge can only
# ever contain what this run converted.
PARTS=()
for arch in "${ARCHIVES[@]}"; do
    PARTS+=("${ROOT}/arbiter_$(basename "${arch}" .hdf5)")
done
"${PY_BIN}" arbiter/splits/merge_lerobot_datasets.py \
    --input_dirs "${PARTS[@]}" --output_dir "${MERGED}" || {
    echo "[ERROR] merge failed" >&2; exit 1; }

# modality.json is emitted from the merged set's own info.json so the two cannot drift; the
# trainer refuses to start without it.
"${PY_BIN}" scripts/bench/write_modality_json.py --root "${MERGED}" || {
    echo "[ERROR] write_modality_json failed" >&2; exit 1; }

echo "[DONE] merged dataset at ${MERGED}"
"${PY_BIN}" - "${MERGED}" <<'PY'
import json, sys
from pathlib import Path
m = Path(sys.argv[1]) / "meta"
info = json.load(open(m / "info.json"))
tasks = [json.loads(l)["task"] for l in open(m / "tasks.jsonl")]
print(f"  episodes {info['total_episodes']}  frames {info['total_frames']}  tasks {len(tasks)}")
for t in tasks:
    print(f"    {t}")
PY
