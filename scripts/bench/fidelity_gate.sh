#!/usr/bin/env bash
# Phase 0.3 -- the copy-fidelity gate.
#
# ARBITER's simulation stack was COPIED from TANGO rather than imported, and Phase 0 evaluates
# TANGO's checkpoints on it. That only works if the scene ARBITER builds is the scene those
# checkpoints were trained on. This reproduces two of their known results on the copied scene.
# If either misses, the copy drifted -- a prop dimension, a camera parameter, a lighting
# intensity, a spawn rotation -- and every number produced afterwards is measuring the drift
# rather than the policy.
#
# Run this BEFORE anything else depends on the copy.
#
# Expected, read off TANGO's stored eval JSONs rather than from prose (the write-up quotes
# subsets: 18/18 and 20/20 are partial grids, the full ones are below):
#
#   1. Control, no blocker, the six around-class heights 0.08-0.13 at 6 episodes each:
#        36/36 success, 36/36 route class "around", ALL on the -y (habitual) lane.
#
#   2. The height sweep across h*: 100% route-correct at every height, flipping from "over" to
#        "around" exactly at h* = 0.08. TANGO's v7 grid was 26 rollouts over 6 heights.
#
# Both run on the `task` language channel, which is what v7 was trained with. Do not set
# ARBITER_LANG_KEY=sub_task here: that is a different checkpoint's channel and would make a
# fidelity miss indistinguishable from a channel mismatch.
#
# Usage:
#   bash scripts/bench/fidelity_gate.sh [CHECKPOINT_DIR]
#
# Needs a free GPU: it boots Isaac once per height and serves a 3B policy.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CKPT="${1:-${REPO_ROOT}/data/checkpoints/n1d7_rel_v7-checkpoint-4000}"
OUT="${OUT:-data/output/eval/fidelity}"
PORT="${PORT:-5555}"
EPISODES="${EPISODES:-6}"
AROUND_HEIGHTS=(0.08 0.09 0.10 0.11 0.12 0.13)
SWEEP_HEIGHTS=(0.03 0.06 0.08 0.09 0.11 0.12)

[[ -d "${CKPT}" ]] || { echo "[GATE] checkpoint not found: ${CKPT}" >&2; exit 1; }

if [[ "${ARBITER_LANG_KEY:-task}" != "task" ]]; then
    echo "[GATE] ARBITER_LANG_KEY=${ARBITER_LANG_KEY} but the v7 checkpoint was trained on" >&2
    echo "       'task'. A mismatch here would look like a fidelity failure. Unset it." >&2
    exit 1
fi

# Pick the emptiest GPU and USE that one. Checking the emptiest while serving on GPU 0 would
# pass the check and then OOM, and an OOM mid-gate is indistinguishable from a fidelity failure.
read -r GPU free_mib < <(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', *' '{print $2, $1}' | sort -rn | head -1 | awk '{print $2, $1}'
)
export GPU
if (( free_mib < 20000 )); then
    echo "[GATE] the emptiest GPU (${GPU}) has only ${free_mib} MiB free; a 3B policy server" >&2
    echo "       plus Isaac's renderer needs ~20 GB. Wait rather than OOM into a false" >&2
    echo "       failure -- an OOM here looks exactly like the copy having drifted." >&2
    exit 1
fi
# Isaac's RTX renderer ignores --device and renders on GPU 0 regardless, so the evaluator is
# pinned with CUDA_VISIBLE_DEVICES rather than a flag (measured 2.4x throughput difference).
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "[GATE] checkpoint : ${CKPT}"
echo "[GATE] out        : ${OUT}"
echo "[GATE] channel    : task (v7's own)"
echo "[GATE] gpu        : ${GPU} (${free_mib} MiB free)"

PIDS=()
cleanup() {
    # `wait` on a bare job blocks on the policy server forever; collect PIDs and kill explicitly.
    for p in "${PIDS[@]:-}"; do kill "${p}" 2>/dev/null || true; done
}
trap cleanup EXIT

echo "[GATE] starting policy server on :${PORT} (gpu ${GPU:-0})"
# The checkpoint is POSITIONAL; PORT and GPU come from the environment. It runs the gr00t venv
# itself and sets ARBITER_LANG_KEY through to the modality config, so the channel the server
# reads and the channel the evaluator sends are the same variable.
PORT="${PORT}" GPU="${GPU:-0}" \
    bash scripts/bench/run_policy_server.sh "${CKPT}" > "data/logs/fidelity_server.log" 2>&1 &
PIDS+=($!)

for _ in $(seq 1 90); do
    sleep 2
    grep -qiE 'ready|listening|server started' "data/logs/fidelity_server.log" && break
done
echo "[GATE] server log tail:"; tail -3 "data/logs/fidelity_server.log" || true

run_height() {
    local h="$1" extra="$2" tag="$3"
    echo "[GATE] --- ${tag} h=${h}"
    # One process per height: the barrier is baked into the stage, and a second
    # SimulationContext in one process deadlocks with the GPU idle.
    # shellcheck disable=SC2086
    timeout -k 30 1800 python arbiter/sim/tools/eval_policy.py \
        --axis topology --split all --policy server --port "${PORT}" \
        --barrier-h "${h}" --barrier-offset 0.0 --episodes "${EPISODES}" \
        --out "${OUT}/${tag}" ${extra} || echo "[GATE] h=${h} exited non-zero"
}

# shellcheck source=/dev/null
source arbiter/sim/activate_sim_env.sh
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

for h in "${AROUND_HEIGHTS[@]}"; do run_height "${h}" "" "control"; done
for h in "${SWEEP_HEIGHTS[@]}"; do run_height "${h}" "" "sweep"; done

echo
echo "==================== GATE VERDICT ===================="
python scripts/bench/check_fidelity.py --control "${OUT}/control" --sweep "${OUT}/sweep"
