#!/usr/bin/env bash
# Static checks that must pass BEFORE a GPU run.
#
# Exists because of a specific, repeated failure. Four times in one day a change of mine failed
# silently rather than loudly: a name that was never imported (NameError ~90 s into an Isaac
# boot), an `exec` forwarding the unresolved argument while the guard checked the resolved one,
# a str.replace whose anchor did not exist so the edit was a no-op, and a string comparison
# against a float. Each cost a GPU run and, worse, two of them produced confident wrong numbers.
#
# The selectors below are the ones that break at runtime rather than offend style:
#   F821  undefined name          -- the NameError class
#   F811  redefinition            -- a shadowed import silently wins
#   F822  undefined name in __all__
#   (syntax errors surface as a parse failure regardless of selector)
#
# Unused imports (F401) are reported separately and do not fail: the copied stack inherited ~22
# of them and cleaning those is unrelated to correctness.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RUFF="${RUFF:-venvs/sim/bin/ruff}"
[[ -x "${RUFF}" ]] || { echo "[LINT] no ruff at ${RUFF}" >&2; exit 1; }

echo "[LINT] runtime-breaking checks (F821,F811,F822)"
if "${RUFF}" check --select F821,F811,F822 --output-format=concise arbiter scripts; then
    echo "[LINT] clean"
else
    echo "[LINT] FAIL -- these break at runtime; fix before spending a GPU run." >&2
    exit 1
fi

echo
echo "[LINT] advisory: unused imports (not fatal)"
"${RUFF}" check --select F401 --output-format=concise --statistics arbiter scripts || true
