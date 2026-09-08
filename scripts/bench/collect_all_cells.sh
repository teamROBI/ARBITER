#!/usr/bin/env bash
# Collect every axis's TRAIN demonstrations across all GPUs.
#
#   bash scripts/bench/collect_all_cells.sh                       # 15 demos/condition, GPUs 0-3
#   DEMOS=15 GPUS="0 1" bash scripts/bench/collect_all_cells.sh
#   AXES="topology order" bash scripts/bench/collect_all_cells.sh  # re-collect a subset
#
# Three constraints from this repo's history shape the whole design:
#
#   * ONE SimulationContext PER PROCESS. A second in the same process deadlocks with the GPU at
#     0% and memory held, so every axis gets its own launch rather than a loop.
#   * TOPOLOGY needs one launch per (barrier height, lateral offset). Both are baked into the
#     stage at build time, and each pair writes its own archive: they briefly shared one file
#     tagged by height alone, and parallel workers then raced next_episode_index into
#     BlockingIOError, yielding 22/24/18/15 episodes where 45 were expected.
#   * CUDA_VISIBLE_DEVICES, not --device cuda:N. Isaac's RTX renderer defaults to GPU 0
#     regardless of --device, which capped four workers at 1.8x; masking took 5.3 -> 12.7
#     episodes/min.
#
# Work is pulled from a flock'd queue rather than partitioned up front, because job costs differ
# by more than an order of magnitude -- POSITION has 9 conditions and ORDER has 3, and a static
# split leaves GPUs idle at the tail. One axis is only ever claimed by one worker, so no two
# processes write the same archive.
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DEMOS="${DEMOS:-15}"
GPUS="${GPUS:-0 1 2 3}"
OUT="${OUT:-data/collect}"
AXES="${AXES:-}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
JOB_TIMEOUT="${JOB_TIMEOUT:-7200}"
LOGDIR="${LOGDIR:-data/logs/v2/collect}"

mkdir -p "${OUT}" "${LOGDIR}"

# Isaac needs its environment, not just its interpreter. Calling venvs/sim/bin/python directly
# fails at `import isaacsim`: activate_sim_env.sh is what locates the Isaac Sim install and
# prepends its extension directories to PYTHONPATH. Sourced once here, since the workers are
# subshells and inherit it.
# shellcheck disable=SC1091
source "${REPO_ROOT}/arbiter/sim/activate_sim_env.sh" > /dev/null 2>&1 || {
    echo "[ERROR] could not source arbiter/sim/activate_sim_env.sh" >&2; exit 1; }
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
python -c "import isaacsim" 2>/dev/null || {
    echo "[ERROR] isaacsim still not importable after activating the sim env" >&2; exit 1; }

# The job list is derived from the spec, never hand-written: TOPOLOGY's trained heights moved
# from all-OVER to two per homotopy class, and a literal list would have silently kept
# collecting the old ones.
mapfile -t JOBS < <(python3 - "${AXES}" <<'PY'
import sys
sys.path.insert(0, ".")
from arbiter.suites.spec import AXES, TOPOLOGY_OFFSETS, TOPOLOGY_TRAIN_H

want = [a for a in (sys.argv[1] or "").split() if a] or list(AXES)
for ax in want:
    if ax not in AXES:
        sys.exit(f"unknown axis {ax!r}; have {sorted(AXES)}")
    if ax == "topology":
        for h in TOPOLOGY_TRAIN_H:
            for off in TOPOLOGY_OFFSETS:
                print(f"--axis topology --barrier-h {h:.2f} --barrier-offset {off:+.2f}")
    else:
        print(f"--axis {ax}")
PY
) || { echo "[ERROR] could not build the job list" >&2; exit 1; }

QUEUE="$(mktemp "${TMPDIR:-/tmp}/arb_collect_queue.XXXXXX")"
LOCK="${QUEUE}.lock"
printf '%s\n' "${JOBS[@]}" > "${QUEUE}"
touch "${LOCK}"
trap 'rm -f "${QUEUE}" "${LOCK}"' EXIT

echo "[COLLECT] ${#JOBS[@]} jobs, ${DEMOS} demos/condition, GPUs: ${GPUS}"
echo "[COLLECT] archives -> ${OUT}"
echo "[COLLECT] logs     -> ${LOGDIR}"

FAILED="${QUEUE}.failed"
: > "${FAILED}"

worker() {
    local gpu="$1" job tag rc
    while :; do
        # Pop under the lock: print the head and delete it in one critical section, so two
        # workers cannot claim the same axis and write the same archive.
        job="$(flock "${LOCK}" -c "head -n1 '${QUEUE}'; sed -i '1d' '${QUEUE}'")"
        [[ -z "${job}" ]] && break
        tag="$(echo "${job}" | sed 's/--axis //; s/ --barrier-h /_h/; s/ --barrier-offset /_o/; s/[^A-Za-z0-9_.+-]/_/g')"
        echo "[GPU ${gpu}] start ${tag}"
        # timeout -k: SIGTERM alone will not kill a process holding Isaac's ~200 non-daemon
        # threads, and one run ignored a plain timeout for 2h50m at 116% CPU.
        CUDA_VISIBLE_DEVICES="${gpu}" timeout -k 30 "${JOB_TIMEOUT}" \
            python arbiter/sim/tools/collect_cell.py ${job} \
            --splits train --demos "${DEMOS}" --max-attempts "${MAX_ATTEMPTS}" \
            --out "${OUT}" > "${LOGDIR}/${tag}.log" 2>&1
        rc=$?
        if [[ ${rc} -ne 0 ]]; then
            echo "[GPU ${gpu}] FAIL  ${tag} (rc=${rc}) -> ${LOGDIR}/${tag}.log"
            flock "${LOCK}" -c "echo '${tag} rc=${rc}' >> '${FAILED}'"
        else
            echo "[GPU ${gpu}] done  ${tag}"
        fi
    done
}

for g in ${GPUS}; do worker "${g}" & done
wait

N_FAIL=$(wc -l < "${FAILED}")
echo
if [[ "${N_FAIL}" -gt 0 ]]; then
    echo "[COLLECT] ${N_FAIL} job(s) FAILED:"
    sed 's/^/    /' "${FAILED}"
else
    echo "[COLLECT] all ${#JOBS[@]} jobs succeeded"
fi
echo "[COLLECT] verify with: python3 scripts/bench/verify_collection.py"
rm -f "${FAILED}"
exit $(( N_FAIL > 0 ? 1 : 0 ))
