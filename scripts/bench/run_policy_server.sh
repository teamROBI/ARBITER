#!/usr/bin/env bash
# Serve a fine-tuned checkpoint over ZMQ for arbiter/sim/tools/eval_policy.py.
#
# Two processes are unavoidable, not a design choice: Isaac Lab needs Python 3.11 and GR00T
# needs 3.10, so the simulator and the policy cannot share an interpreter. The evaluator is the
# client; this is the server.
#
#   bash scripts/bench/run_policy_server.sh data/output/train/n1d7_all/n1d7_all/checkpoint-2000
#   PORT=5556 GPU=1 bash scripts/bench/run_policy_server.sh <ckpt>
#
# The embodiment tag and modality config must match what the checkpoint was trained with, or
# the server will happily accept observations whose joints are in a different order and return
# actions that look valid.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT="${1:-}"
PORT="${PORT:-5555}"
GPU="${GPU:-0}"
EMBODIMENT="${EMBODIMENT:-new_embodiment}"

if [[ -z "${CKPT}" ]]; then
    echo "[ERROR] usage: $0 <checkpoint-dir>" >&2
    echo "        available:" >&2
    ls -d "${REPO_ROOT}"/data/output/train/*/*/checkpoint-* 2>/dev/null | sed 's/^/          /' >&2
    exit 1
fi
# Resolve a relative path against the repo root. Without this, transformers takes a relative
# path for a HuggingFace repo id and dies in validate_repo_id -- which reads as a corrupt
# checkpoint rather than a path mistake, several stack frames deep inside huggingface_hub.
[[ "${CKPT}" = /* ]] || CKPT="${REPO_ROOT}/${CKPT}"
[[ -d "${CKPT}" ]] || { echo "[ERROR] no such checkpoint: ${CKPT}" >&2; exit 1; }

VENV="${REPO_ROOT}/venvs/gr00t/bin/python"
[[ -x "${VENV}" ]] || { echo "[ERROR] missing gr00t venv at ${VENV}" >&2; exit 1; }

echo "[SERVER] checkpoint : ${CKPT}"
echo "[SERVER] embodiment : ${EMBODIMENT}"
echo "[SERVER] port       : ${PORT}   gpu: ${GPU}"

cd "${REPO_ROOT}/third_party/Isaac-GR00T"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/arbiter/policy/gr00t:${PWD}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

# "${CKPT}", not "$@": the resolution above rewrites the variable, but the original positional
# argument is what reaches python if "$@" is forwarded. That was the whole bug -- the -d check
# and the echo used the resolved path while transformers got the relative one, so the guard
# passed and the failure surfaced as an unrelated HFValidationError about a repo id.
exec "${VENV}" - "${CKPT}" <<'PY'
import os
import sys

# Registers the 8-dim joint action/state config under new_embodiment. Import BEFORE the policy
# so the tag resolves to our layout rather than raising.
import modality_config  # noqa: F401
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyServer

ckpt = sys.argv[1]
port = int(os.environ.get("PORT", "5555"))
embodiment = os.environ.get("EMBODIMENT", "new_embodiment")

policy = Gr00tPolicy(
    embodiment_tag=embodiment,
    model_path=ckpt,
    # CUDA_VISIBLE_DEVICES already masks to one GPU, so cuda:0 is that GPU.
    device="cuda:0",
    strict=True,
)
print(f"[SERVER] loaded; modality keys: {sorted(policy.get_modality_config())}", flush=True)
print(f"[SERVER] listening on {port}", flush=True)
PolicyServer.start_server(policy, port=port, host="*")
PY
