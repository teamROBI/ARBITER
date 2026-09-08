#!/usr/bin/env bash
# Render every axis's scene, one launch per axis.
#
# One scene per process is a hard constraint: building a second SimulationContext in the same
# process deadlocks -- clear_instance() does not fully tear down the stage and PhysX scene.
# Observed as a 256-target batch completing and a 48-target batch then hanging forever with the
# GPU at 0%. So this loops launches rather than axes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source arbiter/sim/activate_sim_env.sh >/dev/null 2>&1
export PYTHONPATH=$PWD:$PYTHONPATH

OUT=${1:-data/output/renders}
mkdir -p "$OUT"
for AX in position extent direction topology order factorial approach; do
  EXTRA=()
  # Show a barrier that demands the AROUND class, so the render shows the harder case.
  [[ "$AX" == "topology" ]] && EXTRA=(--barrier-h 0.11)
  echo "=== $AX ==="
  timeout 900 python arbiter/sim/tools/render_scene.py --axis "$AX" "${EXTRA[@]}" --out "$OUT" 2>&1 \
    | grep -E "^\[RENDER\]"
done
echo "=== RENDER ALL DONE ==="
