#!/usr/bin/env bash
# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
#
# Run a script against Isaac Sim's bundled USD (pxr + PhysxSchema) WITHOUT booting Kit.
#
# The asset generators only author USD files — they never need a simulation app, and booting
# one takes minutes. But `pxr` is not pip-installed in any of the three project venvs
# (create_assets.py's docstring claims the sim venv has it; it does not, so that script
# was unrunnable). Isaac Sim ships the core USD libs and the PhysX schema as separate
# extscache extensions, and importing them needs three env vars set correctly.
#
# Paths are discovered, never hardcoded: extscache directory names carry version and build
# hashes that change on every Isaac Sim upgrade, so they are globbed.
#
# Usage:
#   arbiter/sim/tools/usd_python.sh arbiter/sim/tools/create_scene.py --help
#
# Override the Isaac Sim location with ISAACSIM_ROOT if it is not in a standard place.

set -euo pipefail

die() { echo "[ERROR] $*" >&2; exit 1; }

# ── locate Isaac Sim ─────────────────────────────────────────────────────────
if [[ -z "${ISAACSIM_ROOT:-}" ]]; then
  for cand in \
      "/data1/${USER}/simulation/isaacsim" \
      "${HOME}/isaacsim" \
      "/opt/isaacsim" \
      "/isaac-sim"; do
    if [[ -x "${cand}/kit/python/bin/python3" ]]; then
      ISAACSIM_ROOT="${cand}"
      break
    fi
  done
fi
[[ -n "${ISAACSIM_ROOT:-}" ]] || die "Isaac Sim not found. Set ISAACSIM_ROOT to its install dir."
[[ -x "${ISAACSIM_ROOT}/kit/python/bin/python3" ]] \
  || die "No kit python under ISAACSIM_ROOT=${ISAACSIM_ROOT}"

PY="${ISAACSIM_ROOT}/kit/python/bin/python3"
CACHE="${ISAACSIM_ROOT}/extscache"

# ── glob the two extensions that carry pxr ───────────────────────────────────
# omni.usd.libs        → pxr.Usd / UsdGeom / UsdPhysics / UsdShade / Gf / Sdf
# omni.usd.schema.physx → pxr.PhysxSchema (optional; generators fall back to raw attrs)
pick_newest() {
  # shellcheck disable=SC2012
  ls -d "${CACHE}"/"$1"-* 2>/dev/null | sort -V | tail -1
}

USD_LIBS="$(pick_newest 'omni.usd.libs' || true)"
[[ -n "${USD_LIBS}" && -d "${USD_LIBS}/pxr/Usd" ]] \
  || die "core USD not found under ${CACHE}/omni.usd.libs-*"

PHYSX_SCHEMA="$(pick_newest 'omni.usd.schema.physx' || true)"

PYPATH="${USD_LIBS}"
LDPATH="${USD_LIBS}/bin"
PLUGPATH=""
if [[ -n "${PHYSX_SCHEMA}" && -d "${PHYSX_SCHEMA}/pxr/PhysxSchema" ]]; then
  # PXR_PLUGINPATH_NAME must list the directories that actually contain plugInfo.json — the
  # per-schema `resources` dirs, and there are three of them (PhysxSchema,
  # PhysxSchemaAddition, OmniUsdPhysicsDeformableSchema). Pointing at the `plugins` parent
  # instead leaves the schema registry half-initialised, which is worse than not loading it
  # at all: the registry raises Tf errors that stay pending and then surface at the next
  # unrelated pxr call, so an innocent UsdGeom.Xform.Define dies with an empty
  # Tf.ErrorException. Globbed because the set of schema dirs changes between releases.
  PLUGPATH="$(find "${PHYSX_SCHEMA}/plugins" -maxdepth 2 -type d -name resources 2>/dev/null \
              | sort | paste -sd: -)"
  if [[ -n "${PLUGPATH}" ]]; then
    PYPATH="${PYPATH}:${PHYSX_SCHEMA}"
    LDPATH="${LDPATH}:${PHYSX_SCHEMA}/bin"
  else
    echo "[WARN] PhysxSchema found but no plugInfo.json resources dir; skipping it." >&2
  fi
else
  echo "[WARN] PhysxSchema not found; generators will write raw physx attributes instead." >&2
fi

# Repo root on the path so scripts can import arbiter.*
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export PYTHONPATH="${PYPATH}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${LDPATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
[[ -n "${PLUGPATH}" ]] && export PXR_PLUGINPATH_NAME="${PLUGPATH}${PXR_PLUGINPATH_NAME:+:${PXR_PLUGINPATH_NAME}}"

exec "${PY}" "$@"
