#!/usr/bin/env bash
# Make third_party/act's action/state dimension configurable instead of hardcoded to 14.
#
# ACT ships for bimanual ALOHA: 14 = 2 arms x (6 joints + gripper). It already threads
# `state_dim` through DETRVAE.__init__ and the action head, but four input projections and two
# build functions hardcode the literal. TANGO is 8-dim single-arm (7 Franka joints + gripper),
# which is pinned by the plan and is not negotiable for a shipped default.
#
# Done as a patch SCRIPT rather than an in-place edit so the divergence from upstream
# (tonyzhaozh/act) is explicit, reproducible from a fresh clone, and reviewable. It is
# idempotent: running it twice is a no-op, and it prints what it changed.
#
# The alternative -- zero-padding TANGO's 8 dims out to 14 -- was rejected: it would spend model
# capacity on six constant channels and let the action head be scored on predicting zeros.
#
#   bash scripts/setup/patch_act_state_dim.sh            # patch to $ACT_STATE_DIM (default 8)
#   ACT_STATE_DIM=8 bash scripts/setup/patch_act_state_dim.sh
#   bash scripts/setup/patch_act_state_dim.sh --check    # report only, change nothing
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DIM="${ACT_STATE_DIM:-8}"
F="${REPO_ROOT}/third_party/act/detr/models/detr_vae.py"
CHECK=0
[[ "${1:-}" == "--check" ]] && CHECK=1

[[ -f "${F}" ]] || { echo "[ERROR] not found: ${F}" >&2; exit 1; }

hardcoded=$(grep -cE "nn\.Linear\(14,|output_dim=14|state_dim = 14" "${F}" || true)
already=$(grep -cE "ARBITER_STATE_DIM" "${F}" || true)

echo "[ACT] file            : ${F}"
echo "[ACT] target state_dim: ${DIM}"
echo "[ACT] hardcoded 14s   : ${hardcoded}"
echo "[ACT] already patched : $([[ "${already}" -gt 0 ]] && echo yes || echo no)"

if [[ "${already}" -gt 0 ]]; then
    echo "[ACT] nothing to do"
    exit 0
fi
if [[ "${CHECK}" -eq 1 ]]; then
    echo "[ACT] --check: no changes written"
    exit 0
fi

cp "${F}" "${F}.orig"          # keep pristine upstream beside it, for diffing

python3 - "${F}" "${DIM}" <<'PY'
import re, sys
path, dim = sys.argv[1], int(sys.argv[2])
s = open(path).read()

banner = (f"# --- TANGO patch: state/action dimension ---------------------------------------\n"
          f"# Upstream ACT hardcodes 14 (bimanual ALOHA). TANGO is {dim}-dim single-arm: 7 Franka\n"
          f"# joints + gripper. Applied by scripts/setup/patch_act_state_dim.sh; the untouched\n"
          f"# upstream file sits beside this one as detr_vae.py.orig.\n"
          f"ARBITER_STATE_DIM = {dim}\n"
          f"# -------------------------------------------------------------------------------\n")
# insert after the last import at module top
m = list(re.finditer(r"^(?:import|from)\s+\S+.*$", s, re.M))
i = m[-1].end()
s = s[:i] + "\n\n" + banner + s[i:]

s = s.replace("nn.Linear(14, hidden_dim)", "nn.Linear(ARBITER_STATE_DIM, hidden_dim)")
s = s.replace("output_dim=14", "output_dim=ARBITER_STATE_DIM")
s = re.sub(r"state_dim = 14\s*#\s*TODO hardcode", "state_dim = ARBITER_STATE_DIM", s)
open(path, "w").write(s)
print(f"  rewrote input projections, action head and build fns to ARBITER_STATE_DIM={dim}")
PY

left=$(grep -cE "nn\.Linear\(14,|output_dim=14|state_dim = 14" "${F}" || true)
echo "[ACT] hardcoded 14s remaining: ${left}"
[[ "${left}" -eq 0 ]] || { echo "[ERROR] some hardcoded dims survived" >&2; exit 1; }
python3 -c "import ast,sys; ast.parse(open('${F}').read()); print('[ACT] file still parses')"
echo "[ACT] done. Upstream preserved at ${F}.orig"
