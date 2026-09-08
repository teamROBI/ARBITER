#!/usr/bin/env bash
# Central venv layout for ARBITER.
#
# All Python environments live under one directory on the data disk
# ($REPO_ROOT/venvs, itself a symlink) rather than scattered as .venv folders
# inside the source tree. It sits beside data/ rather than inside it, so a data
# sync can never touch the environments. The source tree keeps a symlink at each
# conventional location so that every wrapper script, doc and IDE integration
# continues to work unchanged.
#
#   venvs/sim  <- arbiter/sim/.venv
#   venvs/gr00t    <- arbiter/policy/gr00t/.venv
#   venvs/rl       <- arbiter/rl/.venv
#
# Sourced by each setup_*_env.sh after it has set REPO_ROOT.
#
# Overrides:
#   ARBITER_VENV_ROOT  relocate the whole set (default: $REPO_ROOT/venvs)
#   VENV_DIR         pin one env to an explicit path (skips the root entirely)

# Resolve VENV_DIR for one environment and remember where its compatibility
# symlink belongs. Call before creating the venv.
#
#   arb_resolve_venv <env_name> <legacy_link_path>
arb_resolve_venv() {
    local name="$1"
    local legacy="$2"

    ARBITER_VENV_ROOT="${ARBITER_VENV_ROOT:-${REPO_ROOT}/venvs}"
    _ARBITER_LEGACY_LINK="${legacy}"

    if [[ -n "${VENV_DIR:-}" ]]; then
        # Explicit per-env override: honour it and skip the shared root.
        _ARBITER_MANAGED=0
        return 0
    fi

    if [[ ! -d "${ARBITER_VENV_ROOT}" && ! -d "$(dirname -- "${ARBITER_VENV_ROOT}")" ]]; then
        echo "[ERROR] ${ARBITER_VENV_ROOT} is not reachable." >&2
        echo "        Environments are installed onto the data disk. Create the" >&2
        echo "        symlink first, e.g.:" >&2
        echo "          mkdir -p /data1/\$USER/projects/ARBITER/venvs" >&2
        echo "          ln -s /data1/\$USER/projects/ARBITER/venvs ${REPO_ROOT}/venvs" >&2
        echo "        Or set ARBITER_VENV_ROOT to install them somewhere else." >&2
        return 1
    fi

    VENV_DIR="${ARBITER_VENV_ROOT}/${name}"
    _ARBITER_MANAGED=1
    mkdir -p "${ARBITER_VENV_ROOT}"
}

# Point the in-tree conventional path at the real venv. Call right after the
# venv is created, so anything reading the conventional path mid-install works.
arb_link_venv() {
    local link="${_ARBITER_LEGACY_LINK:-}"
    [[ -z "${link}" ]] && return 0

    if [[ -e "${link}" && ! -L "${link}" ]]; then
        echo "[WARN] ${link} is a real directory, not a symlink." >&2
        echo "       Leaving it alone. Remove it and re-run to use ${VENV_DIR}." >&2
        return 0
    fi

    # Point at the venv through the repo's own paths. -s keeps this lexical, so
    # the link stays valid through $REPO_ROOT/venvs rather than being rewritten
    # to wherever that symlink currently lands.
    local target="${VENV_DIR}"
    local link_dir
    link_dir="$(dirname -- "${link}")"
    if command -v realpath >/dev/null 2>&1; then
        target="$(realpath -s --relative-to="${link_dir}" "${VENV_DIR}" 2>/dev/null || echo "${VENV_DIR}")"
    fi

    ln -sfn "${target}" "${link}"
    echo "[INFO] ${link} -> ${target}"
}
