#!/usr/bin/env bash
# Build the Isaac-side venv for ARBITER.
#
# Much shorter than TANGO's equivalent because two dependencies were dropped: CycloneDDS (built
# from source with cmake) and robotis_dds_python. Those exist for real-robot teleop over DDS,
# and ARBITER is sim-only -- nothing in the repo imports them. See arbiter/sim/pyproject.toml.
#
# Isaac Lab is installed *editable* from third_party/IsaacLab, pinned at the commit recorded in
# third_party/PINNED.md. Do not float it: Phase 0 evaluates a checkpoint trained under this
# exact runtime, and a physics or renderer change would be indistinguishable from a scene-copy
# error in the fidelity gate.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/setup/venv_layout.sh"
arb_resolve_venv "sim" "${SCRIPT_DIR}/.venv"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

command -v uv >/dev/null 2>&1 || { echo "[ERROR] uv is not on PATH." >&2; exit 1; }

if [[ ! -d "${REPO_ROOT}/third_party/IsaacLab/source/isaaclab" ]]; then
    echo "[ERROR] third_party/IsaacLab is missing. See third_party/PINNED.md." >&2
    exit 1
fi

echo "[INFO] venv        : ${VENV_DIR}"
echo "[INFO] python       : ${PYTHON_VERSION}"
echo "[INFO] isaac lab    : $(git -C "${REPO_ROOT}/third_party/IsaacLab" describe --tags --always)"

uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
cd "${SCRIPT_DIR}"
VIRTUAL_ENV="${VENV_DIR}" uv sync --extra dev --active

# The compatibility symlink activate_sim_env.sh resolves. Without it the venv exists on the
# data disk but nothing can find it.
arb_link_venv

echo
echo "[OK] sim env ready. Activate with:"
echo "     source arbiter/sim/activate_sim_env.sh"
