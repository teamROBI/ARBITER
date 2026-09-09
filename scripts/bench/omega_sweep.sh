#!/usr/bin/env bash
# Sweep CAG's guidance scale across the four-cell table, on every free GPU.
#
# This is the go/no-go measurement. The claim it tests is structural: omega is a single global
# scale, so raising it to win the language-authority cells (L-AUTH, CONFLICT-VF) must also pull
# the same lever in the cells where vision is the correct authority (V-AUTH, CONFLICT-VT). One
# monotone gain cannot move two rows up and two rows down, so no value of omega should be correct
# on the whole table.
#
# Two outcomes, both worth having:
#   * A tradeoff appears -> the thesis holds and the motivating figure exists on day one.
#   * No tradeoff -> a training-free method already handles all four cells, and the method needs
#     rethinking before a collection is spent on it. Learning that here costs one retrain.
#
# ONE CHECKPOINT PER SWEEP, INCLUDING omega=1. The curve must vary only omega; if omega=1 came
# from a different checkpoint than omega=2, the baseline column would differ by checkpoint as
# well and the curve would have no origin. To also measure plain relabeling (no dropout), run
# this a second time with CKPT pointing at that checkpoint and OMEGAS=1 -- a separate point, not
# part of this curve.
#
# PARALLELISM. One policy server per GPU, several eval clients per GPU sharing it, and a flock'd
# pull queue feeding them all.
#
# Sizing, from the fidelity-gate run: the 3B server is ~6.5 GB and one Isaac eval ~5.1 GB, so a
# 49 GB card fits a server plus roughly seven evals on memory alone. Memory is not what binds.
# The server answers at ~8 calls/s and each eval needs ~1.9 (one per 16-step chunk at 30 Hz), so
# ~4 clients saturate one server. WORKERS_PER_GPU therefore defaults to 3 -- under the throughput
# ceiling, with the memory check below as a backstop. Raise it if the server logs show idle time.
#
# A pull queue rather than a static split because job costs differ by more than 10x: a rollout
# that succeeds early exits in ~200 steps, one that stalls burns the full 600, and a static split
# leaves GPUs idle at the tail.
#
# Each worker is pinned with CUDA_VISIBLE_DEVICES, never `--device cuda:N`: Isaac's RTX renderer
# renders on GPU 0 regardless of the flag, measured at a 2.4x throughput difference.
#
# Usage:
#   CKPT=data/checkpoints/n1d7_rel_v8_subtask-checkpoint-4000 OMEGAS=1 bash scripts/bench/omega_sweep.sh
#   CKPT=<dropout ckpt> bash scripts/bench/omega_sweep.sh          # the full curve
#   GPUS="0 1" bash scripts/bench/omega_sweep.sh                   # restrict to some GPUs

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CKPT="${CKPT:-}"
OUT="${OUT:-data/output/eval/omega_sweep}"
PORT_BASE="${PORT_BASE:-5600}"
EPISODES="${EPISODES:-6}"
OMEGAS="${OMEGAS:-0 0.5 1 1.5 2 3 5}"
HEIGHTS="${HEIGHTS:-0.08 0.10 0.13}"
OFFSET="${OFFSET:-0.0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-3}"
#: Measured footprints, used to refuse an over-subscription rather than discover it as an OOM
#: halfway through a sweep.
SERVER_MIB="${SERVER_MIB:-6700}"
EVAL_MIB="${EVAL_MIB:-5300}"
JOB_TIMEOUT="${JOB_TIMEOUT:-1800}"

[[ -n "${CKPT}" ]] || { echo "[SWEEP] set CKPT to the checkpoint to serve" >&2; exit 1; }
[[ "${CKPT}" = /* ]] || CKPT="${REPO_ROOT}/${CKPT}"
[[ -d "${CKPT}" ]] || { echo "[SWEEP] checkpoint not found: ${CKPT}" >&2; exit 1; }

# Three of the four cells speak a lane, and only the sub_task middle phase grounds one. On the
# task channel those phrases are out of vocabulary and the sweep would measure distribution
# shift. eval_policy.py refuses per-cell too; this fails once, up front.
export ARBITER_LANG_KEY="${ARBITER_LANG_KEY:-sub_task}"
if [[ "${ARBITER_LANG_KEY}" != "sub_task" ]]; then
    echo "[SWEEP] ARBITER_LANG_KEY=${ARBITER_LANG_KEY}, but L-AUTH, CONFLICT-VF and" >&2
    echo "        CONFLICT-VT are only interpretable on sub_task." >&2
    exit 1
fi

# ── pick the GPUs that are actually free ─────────────────────────────────────
if [[ -n "${GPUS:-}" ]]; then
    CANDIDATES=(${GPUS})
else
    mapfile -t CANDIDATES < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
fi
USE_GPUS=()
for g in "${CANDIDATES[@]}"; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${g}")
    if (( free >= MIN_FREE_MIB )); then
        USE_GPUS+=("${g}")
    else
        echo "[SWEEP] skipping gpu ${g}: only ${free} MiB free (need ${MIN_FREE_MIB})"
    fi
done
if (( ${#USE_GPUS[@]} == 0 )); then
    echo "[SWEEP] no GPU has ${MIN_FREE_MIB} MiB free; a 3B server plus Isaac needs ~20 GB." >&2
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >&2
    exit 1
fi

# ── build the job queue from the spec, not from bash ─────────────────────────
# Each cell declares its own valid heights: the lane cells are around-class only, because below
# h* the route class is `over` and there is no lane to choose. V-AUTH spans the flip. Generating
# the queue from cells.py means that rule lives in one place.
QUEUE="$(mktemp)"; LOCK="$(mktemp)"
trap_files=("${QUEUE}" "${LOCK}")
export HEIGHTS   # read by the generator below; must be exported BEFORE the heredoc runs
PYTHONPATH="${REPO_ROOT}" python3 - "${QUEUE}" ${OMEGAS} <<'PY'
import sys
from arbiter.suites import cells as C
out, omegas = sys.argv[1], [float(x) for x in sys.argv[2:]]
import os
heights = [float(h) for h in os.environ["HEIGHTS"].split()]
rows = []
for w in omegas:
    for name, cell in C.CELLS.items():
        valid = [h for h in heights if any(abs(h - ch) < 1e-9 for ch in cell.heights)]
        for arm in cell.arms:
            for h in valid:
                rows.append(f"{w} {name} {arm.name} {h}")
with open(out, "w") as f:
    f.write("\n".join(rows) + "\n")
print(f"[SWEEP] {len(rows)} jobs "
      f"({len(omegas)} omega x cells/arms x heights, out-of-cell heights dropped)")
PY
N_JOBS=$(wc -l < "${QUEUE}")

echo "[SWEEP] checkpoint : ${CKPT}"
echo "[SWEEP] channel    : ${ARBITER_LANG_KEY}"
echo "[SWEEP] omegas     : ${OMEGAS}"
echo "[SWEEP] heights    : ${HEIGHTS}"
echo "[SWEEP] gpus       : ${USE_GPUS[*]}  (one server each, ${WORKERS_PER_GPU} eval workers per gpu)"
echo "[SWEEP] concurrency: $(( ${#USE_GPUS[@]} * WORKERS_PER_GPU )) eval processes"
echo "[SWEEP] jobs       : ${N_JOBS}"

# Refuse an over-subscription up front. An OOM twenty minutes into a sweep is indistinguishable
# from a policy failure in the results.
NEED_MIB=$(( SERVER_MIB + EVAL_MIB * WORKERS_PER_GPU ))
for g in "${USE_GPUS[@]}"; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${g}")
    if (( NEED_MIB > free )); then
        echo "[SWEEP] gpu ${g}: ${WORKERS_PER_GPU} workers need ~${NEED_MIB} MiB " \
             "(server ${SERVER_MIB} + ${WORKERS_PER_GPU} x ${EVAL_MIB}) but only ${free} free." >&2
        echo "        Lower WORKERS_PER_GPU." >&2
        exit 1
    fi
done
echo "[SWEEP] budget     : ~${NEED_MIB} MiB per gpu"

SERVER_PIDS=()
WORKER_PIDS=()
cleanup() {
    # kill -9, not TERM: Isaac's non-daemon threads ignore SIGTERM and the process survives
    # while the kill reports success. Explicit PIDs because a bare `wait` blocks on the servers.
    for p in "${WORKER_PIDS[@]:-}"; do kill -9 "${p}" 2>/dev/null || true; done
    for p in "${SERVER_PIDS[@]:-}"; do kill -9 "${p}" 2>/dev/null || true; done
    rm -f "${trap_files[@]}" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p data/logs "${OUT}"

# ── one server per GPU ───────────────────────────────────────────────────────
for g in "${USE_GPUS[@]}"; do
    port=$((PORT_BASE + g))
    echo "[SWEEP] server: gpu ${g} port ${port}"
    PORT="${port}" GPU="${g}" \
        bash scripts/bench/run_policy_server.sh "${CKPT}" \
        > "data/logs/omega_server_gpu${g}.log" 2>&1 &
    SERVER_PIDS+=($!)
done

echo "[SWEEP] waiting for servers to come up"
for g in "${USE_GPUS[@]}"; do
    for _ in $(seq 1 120); do
        grep -qiE 'listening|ready|server started' "data/logs/omega_server_gpu${g}.log" && break
        sleep 2
    done
    if grep -qiE 'listening|ready|server started' "data/logs/omega_server_gpu${g}.log"; then
        echo "[SWEEP]   gpu ${g}: up"
    else
        echo "[SWEEP]   gpu ${g}: NOT up -- see data/logs/omega_server_gpu${g}.log" >&2
        tail -3 "data/logs/omega_server_gpu${g}.log" >&2
        exit 1
    fi
done

# shellcheck source=/dev/null
source arbiter/sim/activate_sim_env.sh
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ── one worker per GPU, pulling from the queue ───────────────────────────────
worker() {
    local g="$1" k="${2:-1}" port=$((PORT_BASE + $1)) job
    while :; do
        job="$(flock "${LOCK}" -c "head -n1 '${QUEUE}'; sed -i '1d' '${QUEUE}'")"
        [[ -z "${job}" ]] && break
        set -- ${job}
        local w="$1" cell="$2" arm="$3" h="$4"
        local extra=(--cag-omega "${w}")
        # omega=1 short-circuits to the conditional policy with a single forward pass, so it
        # needs no assertion about the null branch. Everything else does.
        #
        # Compared NUMERICALLY: the generator emits "1.0", and a string test against "1" would
        # append --cag-null-validated at omega=1, writing cag_null_validated:true into the
        # provenance of a run served by a checkpoint that has no instruction dropout. Inert at
        # omega=1, but a false claim in the record is worse than a useless flag.
        if [[ "$(awk -v a="${w}" 'BEGIN{print (a==1)?"y":"n"}')" != "y" ]]; then
            extra+=(--cag-null-validated)
        fi
        echo "[gpu${g}.${k}] w=${w} ${cell}/${arm} h=${h}"
        CUDA_VISIBLE_DEVICES="${g}" timeout -k 30 "${JOB_TIMEOUT}" \
            python arbiter/sim/tools/eval_policy.py \
                --cell "${cell}" --cell-arm "${arm}" \
                --split all --policy server --port "${port}" \
                --barrier-h "${h}" --barrier-offset "${OFFSET}" \
                --episodes "${EPISODES}" --out "${OUT}" \
                "${extra[@]}" >> "data/logs/omega_worker_gpu${g}_${k}.log" 2>&1 \
            || echo "[gpu${g}.${k}] FAILED w=${w} ${cell}/${arm} h=${h}"
    done
    echo "[gpu${g}.${k}] queue empty, worker done"
}

for g in "${USE_GPUS[@]}"; do
    for k in $(seq 1 "${WORKERS_PER_GPU}"); do
        : > "data/logs/omega_worker_gpu${g}_${k}.log"
        worker "${g}" "${k}" &
        WORKER_PIDS+=($!)
    done
done
echo "[SWEEP] ${#WORKER_PIDS[@]} workers started"
wait "${WORKER_PIDS[@]}"

echo
echo "==================== SWEEP RESULTS ===================="
python scripts/bench/score_arbitration.py --dir "${OUT}"
