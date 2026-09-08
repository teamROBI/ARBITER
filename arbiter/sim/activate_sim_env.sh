#!/usr/bin/env bash

# This script must be sourced so the activated environment persists.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Please source this script: source arbiter/sim/activate_sim_env.sh" >&2
    exit 1
fi

# NOTE: `return` inside this function only leaves the function, so every call site must
# `return 1` itself. Before this was fixed the script printed "[ERROR] sim venv not found"
# and then "[INFO] Activated arbiter sim environment" in the same breath -- a guard that
# cannot stop anything, reporting success after failing.
_activate_sim_env_die() {
    echo "[ERROR] $*" >&2
    return 1
}

_prepend_env_path() {
    local var_name="$1"
    local path_value="$2"
    local current_value

    if [[ -z "${path_value}" || ! -e "${path_value}" ]]; then
        return 0
    fi

    current_value="${!var_name:-}"
    case ":${current_value}:" in
        *":${path_value}:"*) ;;
        *) export "${var_name}=${path_value}${current_value:+:${current_value}}" ;;
    esac
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SIM_VENV="${SIM_VENV:-${REPO_ROOT}/arbiter/sim/.venv}"
if [[ -z "${ISAACSIM_PATH:-}" ]]; then
    if [[ -d "/data/simulation/isaacsim" ]]; then
        ISAACSIM_PATH="/data/simulation/isaacsim"
    elif [[ -d "/data1/jokim/simulation/isaacsim" ]]; then
        ISAACSIM_PATH="/data1/jokim/simulation/isaacsim"
    elif [[ -d "/isaac-sim" ]]; then
        ISAACSIM_PATH="/isaac-sim"
    else
        ISAACSIM_PATH="${HOME}/isaacsim"
    fi
fi
ISAACLAB_ISAACSIM_LINK="${REPO_ROOT}/third_party/IsaacLab/_isaac_sim"
LIBSTDCXX_COMPAT_DIR="${LIBSTDCXX_COMPAT_DIR:-}"

if [[ ! -f "${SIM_VENV}/bin/activate" ]]; then
    _activate_sim_env_die "sim venv not found: ${SIM_VENV}"
    return 1
fi

if [[ ! -f "${ISAACSIM_PATH}/setup_conda_env.sh" ]]; then
    _activate_sim_env_die "Isaac Sim root not found or incomplete: ${ISAACSIM_PATH}"
    return 1
fi

if [[ ! -d "${REPO_ROOT}/third_party/IsaacLab" ]]; then
    _activate_sim_env_die "IsaacLab dependency not found: ${REPO_ROOT}/third_party/IsaacLab"
    return 1
fi

source "${SIM_VENV}/bin/activate"
unset LD_PRELOAD

mkdir -p "$(dirname -- "${ISAACLAB_ISAACSIM_LINK}")"
if [[ -L "${ISAACLAB_ISAACSIM_LINK}" ]]; then
    CURRENT_LINK_TARGET="$(readlink -f "${ISAACLAB_ISAACSIM_LINK}")"
else
    CURRENT_LINK_TARGET=""
fi

if [[ "${CURRENT_LINK_TARGET}" != "${ISAACSIM_PATH}" ]]; then
    ln -sfn "${ISAACSIM_PATH}" "${ISAACLAB_ISAACSIM_LINK}" \
        || { _activate_sim_env_die "failed to link ${ISAACLAB_ISAACSIM_LINK} -> ${ISAACSIM_PATH}"; return 1; }
fi

_had_nounset=0
if [[ $- == *u* ]]; then
    _had_nounset=1
    set +u
fi
source "${ISAACLAB_ISAACSIM_LINK}/setup_conda_env.sh"
if [[ ${_had_nounset} -eq 1 ]]; then
    set -u
fi

export ISAACSIM_PATH

if [[ -n "${LIBSTDCXX_COMPAT_DIR}" && -f "${LIBSTDCXX_COMPAT_DIR}/libstdc++.so.6" ]]; then
    _prepend_env_path LD_LIBRARY_PATH "${LIBSTDCXX_COMPAT_DIR}"
fi

USD_LIBS_EXT_DIR=""
if [[ -d "${ISAACSIM_PATH}/extscache" ]]; then
    for ext_dir in "${ISAACSIM_PATH}"/extscache/*; do
        [[ -d "${ext_dir}" && -d "${ext_dir}/pxr" ]] || continue
        if [[ "$(basename -- "${ext_dir}")" == omni.usd.libs-* && -z "${USD_LIBS_EXT_DIR}" ]]; then
            USD_LIBS_EXT_DIR="${ext_dir}"
        fi
    done

    if [[ -n "${USD_LIBS_EXT_DIR}" ]]; then
        _prepend_env_path PYTHONPATH "${USD_LIBS_EXT_DIR}"
        _prepend_env_path LD_LIBRARY_PATH "${USD_LIBS_EXT_DIR}/bin"
        _prepend_env_path LD_LIBRARY_PATH "${USD_LIBS_EXT_DIR}/lib"
    fi

    for ext_dir in "${ISAACSIM_PATH}"/extscache/*; do
        [[ -d "${ext_dir}" && -d "${ext_dir}/pxr" ]] || continue
        [[ "${ext_dir}" == "${USD_LIBS_EXT_DIR}" ]] && continue
        _prepend_env_path PYTHONPATH "${ext_dir}"
        _prepend_env_path LD_LIBRARY_PATH "${ext_dir}/bin"
        _prepend_env_path LD_LIBRARY_PATH "${ext_dir}/lib"
    done
fi

echo "[INFO] Activated arbiter sim environment"
echo "[INFO] VIRTUAL_ENV=${VIRTUAL_ENV}"
echo "[INFO] ISAACSIM_PATH=${ISAACSIM_PATH}"
if [[ -n "${LIBSTDCXX_COMPAT_DIR}" && -f "${LIBSTDCXX_COMPAT_DIR}/libstdc++.so.6" ]]; then
    echo "[INFO] LIBSTDCXX_COMPAT_DIR=${LIBSTDCXX_COMPAT_DIR}"
fi
if [[ -n "${USD_LIBS_EXT_DIR}" ]]; then
    echo "[INFO] USD_LIBS_EXT_DIR=${USD_LIBS_EXT_DIR}"
fi
