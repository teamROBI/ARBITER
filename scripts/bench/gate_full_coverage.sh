#!/usr/bin/env bash
# Exhaustive achievability gate: EVERY condition of every axis, not a stratified sample.
#
# The sampled gate found no unachievable condition outside APPROACH, but POSITION was only
# 12 of 667 conditions (1.8%). A single unachievable condition left in the training set
# silently corrupts the collected distribution, which is exactly what this control exists to
# prevent -- so it has to be run exhaustively once before collecting anything.
#
# One launch per axis (a second SimulationContext deadlocks), and one launch per
# (height, offset) for TOPOLOGY since a prop cannot move after the stage is built.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source arbiter/sim/activate_sim_env.sh >/dev/null 2>&1
export PYTHONPATH=$PWD:$PYTHONPATH

OUT=${1:-data/output/achievability_full}
mkdir -p "$OUT"
BIG=100000        # larger than any axis, so --per-axis never truncates

for AX in order factorial extent approach direction position; do
  echo "=== $AX ==="
  timeout 14400 python arbiter/sim/tools/gate.py \
    --axis "$AX" --per-axis "$BIG" --out "$OUT" 2>&1 \
    | grep -E --line-buffered "achievable \(|VERDICT|failures by|-> "
done

# Heights come from the spec, not from a literal list. This line used to carry 0.14, which
# measurement had already removed (it failed at all three offsets, the only unachievable
# configuration in the sweep), and it would not have picked up the trained/tested re-split
# either.
HEIGHTS=$(python3 -c "import sys; sys.path.insert(0,'$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)'); from arbiter.suites.spec import barrier_heights; print(' '.join(f'{h:.2f}' for h in barrier_heights()))")
echo "[GATE] barrier heights from spec: ${HEIGHTS}"
for H in ${HEIGHTS}; do
  for O in -0.06 0.0 0.06; do
    echo "=== topology h=$H o=$O ==="
    timeout 1800 python arbiter/sim/tools/gate.py \
      --axis topology --barrier-h "$H" --barrier-offset "$O" --per-axis "$BIG" --out "$OUT" 2>&1 \
      | grep -E --line-buffered "achievable \(|-> "
  done
done
echo "=== FULL COVERAGE DONE ==="
