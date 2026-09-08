#!/usr/bin/env bash
# Phase 0.7 -- sweep CAG's guidance scale across the whole four-cell table.
#
# This is the go/no-go measurement. The claim it tests is structural: omega is a single global
# scale, so raising it to win the language-authority cells (L-AUTH, CONFLICT-VF) must also pull
# the same lever in the cells where vision is the correct authority (V-AUTH, CONFLICT-VT). One
# monotone gain cannot move two rows up and two rows down, so no value of omega should be correct
# on the whole table.
#
# Two outcomes, both worth having:
#
#   * A tradeoff appears -> the thesis holds, and the motivating figure exists on day one.
#   * No tradeoff -> a training-free method already handles all four cells, and ARBITER's method
#     needs rethinking before a collection is spent on it. Learning that here costs one retrain.
#
# ONE CHECKPOINT PER SWEEP, INCLUDING omega=1. The curve must vary only omega; if omega=1 came
# from a different checkpoint than omega=2, the baseline column would differ by checkpoint as
# well and the curve would have no origin. To also measure plain relabeling (no dropout), run
# this a second time with CKPT pointing at that checkpoint and OMEGAS=1 -- that is a separate
# point, not part of this curve.
#
# Usage:
#   CKPT=data/output/train/<dropout-run>/.../checkpoint-4000 bash scripts/bench/omega_sweep.sh
#   OMEGAS="1 2 3" HEIGHTS="0.10" bash scripts/bench/omega_sweep.sh
#
# Needs one GPU with ~20 GB, and boots Isaac once per (cell, arm, height).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CKPT="${CKPT:-}"
OUT="${OUT:-data/output/eval/omega_sweep}"
PORT="${PORT:-5556}"
EPISODES="${EPISODES:-6}"
OMEGAS="${OMEGAS:-0 0.5 1 1.5 2 3 5}"
HEIGHTS="${HEIGHTS:-0.08 0.10 0.13}"
OFFSET="${OFFSET:-0.0}"

# Every cell/arm pair in the table. The naming cells are PAIRED on purpose: compliance on one
# arm cannot separate obedience from habit, so both arms of a pair must be run or the cell
# reports nothing. score_arbitration.py says [PARTIAL] rather than quietly reporting a rate.
PAIRS="${PAIRS:-L-AUTH:habitual L-AUTH:requested V-AUTH:silent CONFLICT-VF:agrees CONFLICT-VF:contradicts CONFLICT-VT:names_sealed}"

[[ -n "${CKPT}" ]] || { echo "[SWEEP] set CKPT to the instruction-dropout checkpoint" >&2; exit 1; }
[[ "${CKPT}" = /* ]] || CKPT="${REPO_ROOT}/${CKPT}"
[[ -d "${CKPT}" ]] || { echo "[SWEEP] checkpoint not found: ${CKPT}" >&2; exit 1; }

# Three of the four cells speak a lane, and only the sub_task middle phase grounds one. On the
# task channel those phrases are out of vocabulary and the sweep would measure distribution
# shift. eval_policy.py refuses per-cell too; this fails once, up front, instead of per run.
export ARBITER_LANG_KEY="${ARBITER_LANG_KEY:-sub_task}"
if [[ "${ARBITER_LANG_KEY}" != "sub_task" ]]; then
    echo "[SWEEP] ARBITER_LANG_KEY=${ARBITER_LANG_KEY}, but L-AUTH, CONFLICT-VF and" >&2
    echo "        CONFLICT-VT are only interpretable on sub_task." >&2
    exit 1
fi

read -r GPU free_mib < <(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', *' '{print $2, $1}' | sort -rn | head -1 | awk '{print $2, $1}'
)
if (( free_mib < 20000 )); then
    echo "[SWEEP] the emptiest GPU (${GPU}) has ${free_mib} MiB free; needs ~20 GB." >&2
    exit 1
fi
export GPU CUDA_VISIBLE_DEVICES="${GPU}"

echo "[SWEEP] checkpoint : ${CKPT}"
echo "[SWEEP] gpu        : ${GPU} (${free_mib} MiB free)"
echo "[SWEEP] channel    : ${ARBITER_LANG_KEY}"
echo "[SWEEP] omegas     : ${OMEGAS}"
echo "[SWEEP] heights    : ${HEIGHTS}"

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "${p}" 2>/dev/null || true; done; }
trap cleanup EXIT

mkdir -p data/logs
echo "[SWEEP] starting policy server on :${PORT}"
PORT="${PORT}" GPU="${GPU}" \
    bash scripts/bench/run_policy_server.sh "${CKPT}" > data/logs/omega_server.log 2>&1 &
PIDS+=($!)
for _ in $(seq 1 90); do
    sleep 2
    grep -qiE 'ready|listening|server started' data/logs/omega_server.log && break
done
tail -3 data/logs/omega_server.log || true

# shellcheck source=/dev/null
source arbiter/sim/activate_sim_env.sh
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

for w in ${OMEGAS}; do
  # omega=1 short-circuits to the conditional policy with a single forward pass, so it needs no
  # assertion about the null branch. Everything else does.
  extra=(--cag-omega "${w}")
  [[ "${w}" != "1" ]] && extra+=(--cag-null-validated)

  for pair in ${PAIRS}; do
    cell="${pair%%:*}"; arm="${pair##*:}"
    for h in ${HEIGHTS}; do
      # V-AUTH is the only cell that spans below h*; the lane cells are around-class only and
      # eval_policy.py refuses an out-of-cell height rather than scoring an uninterpretable one.
      echo "[SWEEP] --- w=${w} ${cell}/${arm} h=${h}"
      timeout -k 30 1800 python arbiter/sim/tools/eval_policy.py \
          --cell "${cell}" --cell-arm "${arm}" \
          --split all --policy server --port "${PORT}" \
          --barrier-h "${h}" --barrier-offset "${OFFSET}" \
          --episodes "${EPISODES}" --out "${OUT}" \
          "${extra[@]}" || echo "[SWEEP] w=${w} ${cell}/${arm} h=${h} exited non-zero"
    done
  done
done

echo
echo "==================== SWEEP RESULTS ===================="
python scripts/bench/score_arbitration.py --dir "${OUT}"
