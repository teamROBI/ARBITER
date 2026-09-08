#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
GROOT_DIR="${REPO_ROOT}/third_party/Isaac-GR00T"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/setup/venv_layout.sh"
arb_resolve_venv "gr00t" "${PROJECT_DIR}/.venv"
VENV_PROMPT="${VENV_PROMPT:-gr00t}"
CLEAN_REINSTALL="${CLEAN_REINSTALL:-0}"

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv is not installed or not on PATH." >&2
    exit 1
fi

if [[ ! -d "${GROOT_DIR}" ]]; then
    echo "[ERROR] Isaac-GR00T dependency not found: ${GROOT_DIR}" >&2
    echo "        Run: git submodule update --init --recursive" >&2
    exit 1
fi

cd "${PROJECT_DIR}"

if [[ "${CLEAN_REINSTALL}" == "1" ]]; then
    rm -rf "${VENV_DIR}"
fi

uv python install "${PYTHON_VERSION}"
uv venv --python "${PYTHON_VERSION}" --prompt "${VENV_PROMPT}" "${VENV_DIR}"
arb_link_venv

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

uv pip install setuptools

# Install PyTorch cu128 (GR00T N1.7 requires torch 2.7.1)
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# Install Isaac-GR00T editable without deps (flash-attn needs special build)
uv pip install --no-deps --editable "${GROOT_DIR}"

# Install gr00t N1.7 deps (excluding flash-attn and torch/torchvision already installed)
uv pip install \
    "albumentations==1.4.18" "av==16.1.0" "diffusers==0.35.1" "dm-tree==0.1.8" \
    "lmdb==1.7.5" "msgpack==1.1.0" "msgpack-numpy==0.4.8" "pandas==2.2.3" \
    "peft==0.17.1" "termcolor==3.2.0" "transformers==4.57.3" "tyro==0.9.17" \
    "click==8.1.8" "datasets==3.6.0" "einops==0.8.1" "gymnasium==1.2.2" \
    "matplotlib==3.10.1" "numpy==1.26.4" "omegaconf==2.3.0" "scipy==1.15.3" \
    "torchcodec==0.4.0" "wandb==0.23.0" "pyzmq==27.0.1" "deepspeed==0.17.6" \
    "huggingface-hub[cli]" "cryptography==42.0.8" "gitpython==3.1.46" \
    "onnx>=1.20.0" \
    "h5py" "decord"

# flash-attn: prefer the matching prebuilt wheel, fall back to a source build.
#
# Isaac-GR00T's pyproject maps prebuilt wheel URLs only for cp310/cp312, and a source
# build here takes 40-90 minutes. The installer probes the venv's own python tag, torch
# minor version and C++ ABI and fetches the matching asset, so it gets the same
# flash-attn 2.7.4.post1 this pin asks for without the compile. It falls back to
# --no-build-isolation against the venv's torch if no asset matches.
bash "${SCRIPT_DIR}/install_flash_attn.sh" "${VENV_DIR}"

# Install lerobot v2.1 format (v0.3.3) without deps to avoid torch reinstall
uv pip install --no-deps lerobot==0.3.3
uv pip install jsonlines pyarrow

# Verify
python - <<'PY'
import torch
version = torch.__version__
cuda_version = torch.version.cuda
print(f"[INFO] torch version: {version}")
print(f"[INFO] CUDA version:  {cuda_version}")
if not version.startswith("2.7.1"):
    raise SystemExit(f"[ERROR] Expected torch 2.7.1, got {version}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
PY

echo ""
echo "[INFO] GR00T 1.7 environment ready"
echo "[INFO] Activate with: source ${VENV_DIR}/bin/activate"
echo "[INFO] Current prompt name: ${VENV_PROMPT}"
if [[ "${CLEAN_REINSTALL}" == "1" ]]; then
    echo "[INFO] Clean reinstall cleared: ${VENV_DIR}"
fi
