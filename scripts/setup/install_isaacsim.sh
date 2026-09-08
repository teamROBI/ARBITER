#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Install Isaac Sim 5.1.0 (standalone) at /data1/jokim/simulation/isaacsim
# Copies the standalone zip from turing, rosen, or hinton, then installs.
# ============================================================

DATA_ROOT="${DATA_ROOT:-/data1}"
# Use jokim subdir if it already exists on this server, otherwise store directly under DATA_ROOT
if [[ -d "${DATA_ROOT}/jokim" ]]; then
    _SIM_BASE="${DATA_ROOT}/jokim/simulation"
else
    _SIM_BASE="${DATA_ROOT}/simulation"
fi
INSTALL_DIR="${_SIM_BASE}/isaacsim"
ZIP_NAME="isaac-sim-standalone-5.1.0-linux-x86_64.zip"
LOCAL_ZIP="${_SIM_BASE}/${ZIP_NAME}"   # kept outside INSTALL_DIR so reinstall doesn't wipe it
SERVERS=(turing rosen hinton)

# --- Check if already installed ---
if [[ -d "${INSTALL_DIR}" && -f "${INSTALL_DIR}/isaac-sim.sh" ]]; then
    version=$(cat "${INSTALL_DIR}/VERSION" 2>/dev/null || echo "unknown")
    echo "Isaac Sim ${version} is already installed at ${INSTALL_DIR}."
    read -rp "Reinstall? [y/N] " ans
    if [[ ! "${ans}" =~ ^[Yy]$ ]]; then
        echo "Aborting."
        exit 0
    fi
    echo "==> Removing existing installation..."
    rm -rf "${INSTALL_DIR}"
fi

# --- Copy zip if not already present ---
if [[ -f "${LOCAL_ZIP}" ]]; then
    echo "==> Zip already exists at ${LOCAL_ZIP}, skipping download."
else
    SELECTED=""
    REMOTE_ZIP=""
    for server in "${SERVERS[@]}"; do
        # Detect the remote server's data layout dynamically
        _remote_zip="$(ssh "${server}" "
            for dr in /data1 /data2 /data; do
                [[ -d \"\${dr}\" ]] || continue
                if [[ -d \"\${dr}/jokim\" ]]; then
                    base=\"\${dr}/jokim/simulation\"
                else
                    base=\"\${dr}/simulation\"
                fi
                zip=\"\${base}/isaacsim/${ZIP_NAME}\"
                if [[ -f \"\${zip}\" ]]; then echo \"\${zip}\"; exit 0; fi
            done
        " 2>/dev/null || true)"
        if [[ -n "${_remote_zip}" ]]; then
            echo "==> Checking ${server} ... [ok] Found at ${_remote_zip}"
            SELECTED="${server}"
            REMOTE_ZIP="${_remote_zip}"
            break
        else
            echo "==> Checking ${server} ... [skip] Not found"
        fi
    done

    if [[ -z "${SELECTED}" ]]; then
        echo "Error: ${ZIP_NAME} not found on any of: ${SERVERS[*]}"
        exit 1
    fi

    echo ""
    echo "==> Copying ${ZIP_NAME} from ${SELECTED}..."
    mkdir -p "$(dirname "${LOCAL_ZIP}")"
    scp "${SELECTED}:${REMOTE_ZIP}" "${LOCAL_ZIP}"
fi

# --- Unzip into install dir ---
echo ""
echo "==> Unzipping into ${INSTALL_DIR}..."
mkdir -p "${INSTALL_DIR}"
unzip -o "${LOCAL_ZIP}" -d "${INSTALL_DIR}"

# --- Post-install ---
echo ""
echo "==> Running post_install.sh..."
cd "${INSTALL_DIR}"
bash ./post_install.sh

echo ""
version=$(cat "${INSTALL_DIR}/VERSION" 2>/dev/null || echo "unknown")
echo "==> Isaac Sim ${version} installed at ${INSTALL_DIR}"
echo ""
echo "To launch, run:"
echo "  unset DISPLAY && ${INSTALL_DIR}/isaac-sim.sh --headless"
echo "  or set up a virtual display with Xvfb first."
