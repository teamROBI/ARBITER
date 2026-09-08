#!/usr/bin/env bash
# Install flash-attn into the combined RL venv.
#
# Isaac-GR00T pins flash-attn==2.7.4.post1 but its pyproject only maps prebuilt
# wheel URLs for cp310 and cp312 (plus an aarch64 cp310 local wheel). Our venv is
# cp311, because Isaac Sim 5.1 embeds CPython 3.11.
#
# The upstream release does publish a cp311 asset for that exact version -- it is
# simply not referenced by Isaac-GR00T -- so we install the same version SFT and
# the policy server use, rather than bumping to a different one:
#
#   flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
#
# The ABI suffix is probed from the venv's own torch rather than assumed, since a
# mismatch surfaces as an undefined-symbol ImportError at first use instead of at
# install time.
#
# Usage: bash install_flash_attn_cp311.sh <venv-dir>
set -euo pipefail

VENV="${1:?usage: install_flash_attn_cp311.sh <venv-dir>}"
PY="${VENV}/bin/python"
[[ -x "${PY}" ]] || { echo "[ERROR] no interpreter at ${PY}" >&2; exit 1; }

read -r PYTAG TORCHTAG ABITAG < <("${PY}" - <<'EOF'
import sys, torch
print(
    f"cp{sys.version_info.major}{sys.version_info.minor}",
    "torch" + ".".join(torch.__version__.split("+")[0].split(".")[:2]),
    "cxx11abiTRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "cxx11abiFALSE",
)
EOF
)
echo "[INFO] target ABI: ${PYTAG} / ${TORCHTAG} / ${ABITAG}"

if "${PY}" -c 'import flash_attn' 2>/dev/null; then
    echo "[INFO] flash_attn already importable: $("${PY}" -c 'import flash_attn;print(flash_attn.__version__)')"
    exit 0
fi

BASE="https://github.com/Dao-AILab/flash-attention/releases/download"
for VER in 2.7.4.post1 2.8.3; do
    WHL="flash_attn-${VER}+cu12${TORCHTAG}${ABITAG}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
    URL="${BASE}/v${VER}/${WHL}"
    CODE="$(curl -sIL -o /dev/null -w '%{http_code}' "${URL}" || echo 000)"
    if [[ "${CODE}" != "200" ]]; then
        echo "[INFO] no asset (HTTP ${CODE}): v${VER} ${PYTAG}/${TORCHTAG}/${ABITAG}"
        continue
    fi
    echo "[INFO] installing ${WHL}"
    if uv pip install --python "${PY}" --no-deps "${URL}"; then
        "${PY}" -c 'import flash_attn; print(f"[OK]   flash_attn {flash_attn.__version__}")'
        exit 0
    fi
    echo "[WARN] install/import failed for v${VER}; trying next" >&2
done

echo "[WARN] no usable prebuilt wheel; building from source (~40-90 min on 32 cores)." >&2
echo "[WARN] If this is too slow, skip it: GR00T runs on SDPA if you set" >&2
echo "       \"use_flash_attention\": false in OUR copy of the checkpoint config.json." >&2
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS:-16}" \
    uv pip install --python "${PY}" --no-build-isolation --no-deps 'flash-attn==2.7.4.post1'
"${PY}" -c 'import flash_attn; print(f"[OK]   flash_attn {flash_attn.__version__} (source build)")'
