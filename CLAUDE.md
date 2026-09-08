# CLAUDE.md — ARBITER

## Project overview

**ARBITER** asks which channel decides. When a policy can demonstrably execute several
behaviours, does language get to choose, or do the pixels — and can language be made to win
without giving up the visual competence that currently does the choosing?

Paper title (working): **"Pixels or Words?"** Method name reserved: **ARBITER**.

Single-arm **Franka Panda**, sim-only, Isaac Sim 5.1 / Isaac Lab 2.3.2. No fixed deadline.

- **Language:** Python 3.11 sim side (Isaac Sim embeds CPython 3.11); 3.10 for GR00T.
- **Action/state convention:** 8-dim single arm = 7 arm joints + 1 gripper.
- **Branch:** `master` (note: `main` is the nominal default; nothing is pushed yet).

### Not the same question as TANGO

TANGO measures how far competence extends *past* the demonstrations — a generalization radius,
and a coverage cost derived from it. ARBITER holds capability **fixed** (both behaviours
demonstrated, at that condition) and asks which channel picks.

The two need opposite experimental setups: TANGO holds behaviours out to make them hard, ARBITER
must guarantee nothing is held out so that capability can never explain a failure. Support
distance, generalization radius and coverage cost are deliberately **not** in this repo.

### The four-cell table

`arbiter/suites/cells.py`. Every existing counterfactual benchmark has one kind of cell —
language names something and obeying it is right — so none can separate a policy that arbitrates
well from one that merely weights language more.

| cell | scene | spoken | should win | channel |
|---|---|---|---|---|
| `L-AUTH` | no blocker, both lanes open | names a side | **language** | `sub_task` |
| `V-AUTH` | barrier height swept across `h*` | route-silent | **vision** | either |
| `CONFLICT-VF` | ghost blocker: visible, **no collider** | names a side | **language** | `sub_task` |
| `CONFLICT-VT` | solid blocker: genuinely seals a lane | names the sealed lane | **vision** | `sub_task` |

`CONFLICT-VF` is the row competitors cannot build: the pixels are *lying*, so obeying them is a
factual error. `CONFLICT-VT` is safety arbitration and is reported **separately** — compliance
there is a collision.

**Why the shape is the argument.** CAG mixes one global scale, `pi_uncond + w*(pi_cond -
pi_uncond)`. A single monotone gain moves all four rows the same direction, so raising `w` to fix
`L-AUTH` and `CONFLICT-VF` necessarily erodes `V-AUTH` and `CONFLICT-VT`. No `w` is correct on
all four — provable before a rollout, and this table measures it.

## Provenance: copied from TANGO, not imported

The sim stack was **copied** and adapted (`git log` starts fresh here). Nothing may resolve into
`TANGO`, `R2-M2`, `ACS_ROBI` or any other checkout.

```bash
python scripts/bench/check_self_contained.py --strict
```

Two differences from TANGO's version of that checker, both deliberate:

- `TANGO` is in `PATTERNS`.
- It scans the **worktree**, not `git ls-files`. TANGO's missed 17 untracked files, including
  its whole ACT baseline and most of `scripts/bench`, so its green result was partly vacuous.

Baseline is **empty** — no live references, no legacy debt. Prose mentions of TANGO are declared
in `ALLOWED` with reasons; every one cites a measured number that justifies a design choice.

`third_party/` holds vendored upstream checkouts, gitignored, pinned in `third_party/PINNED.md`.
They were copied at TANGO's exact commits rather than cloned from `main`, because Phase 0
evaluates a borrowed checkpoint and a newer Isaac Lab could change physics or rendering — a
fidelity mismatch would then be indistinguishable from a scene-copy error.

## Layout

`data/` and `venvs/` are symlinks onto `/data1`. Nothing heavy on `/home`.

```
arbiter/suites/spec.py     geometry + conditions + instruction derivation (the copied axes)
arbiter/suites/cells.py    the four arbitration cells, built on spec
arbiter/suites/view.py     head-camera projection, pure geometry
arbiter/suites/layout_gen.py  cameras, workspace, floor markings
arbiter/sim/env/scene.py   build_scene / reset_condition_scene / assert_condition_realised
arbiter/sim/env/isaac_backend.py   DifferentialIK, DECIMATION=4
arbiter/sim/tools/         collect_cell, gate, eval_policy, create_scene, create_assets, renderers
arbiter/collect/           the route primitive, experts, success predicates, constants
arbiter/splits/            HDF5 -> LeRobot conversion and merge
arbiter/policy/cag.py      Counterfactual Action Guidance -- the baseline to beat
arbiter/policy/gr00t/      modality config, ARBITER_* env switches
scripts/bench/             drivers, scoring, verification, hygiene
```

## Quick start

```bash
source arbiter/sim/activate_sim_env.sh          # derives REPO_ROOT itself
export PYTHONPATH=$PWD:$PYTHONPATH

# props and graspable assets (pxr only, no Kit boot, no GPU)
bash arbiter/sim/tools/usd_python.sh arbiter/sim/tools/create_scene.py
bash arbiter/sim/tools/usd_python.sh arbiter/sim/tools/create_assets.py
bash arbiter/sim/tools/usd_python.sh arbiter/sim/tools/tests/test_scene_props.py

# stdlib-only tests (111 of them; needs nothing but pytest + numpy)
PYTHONPATH=. venvs/spec/bin/python -m pytest \
  arbiter/suites/tests arbiter/collect/tests arbiter/splits/tests arbiter/policy/tests \
  arbiter/sim/tools/tests/test_spawn_rotation.py arbiter/sim/env/tests -q
```

## Gotchas — read before changing anything here

### `arbiter/suites/spec.py` still holds all seven TANGO axes, on purpose

It is tempting to delete POSITION/EXTENT/DIRECTION/ORDER/APPROACH/FACTORIAL and keep only the
barrier geometry. Do not, until Phase 2. Two reasons:

1. The borrowed checkpoints were trained on datasets spanning all seven, so reproducing their
   control numbers (the copy-fidelity gate) needs the spec to enumerate those exact conditions.
2. The instruction vocabulary depends on them. `"right"` is grounded by DIRECTION and FACTORIAL;
   `"left"` by TOPOLOGY's own around-left demos. A word grounded on no axis is out of vocabulary,
   and speaking it measures distribution shift rather than arbitration.

They can go once ARBITER trains on its own balanced collection, when nothing depends on matching
an external training run.

### Only `sub_task` grounds a side, so three of four cells require it

Topology's **task-level** instruction is `"move the red block past the barrier and put it on the
target"` — route-silent, never names a side. So a side-naming instruction is out of vocabulary on
the `task` channel, and a null there is unsurprising rather than informative. Only the `sub_task`
middle phase names one: `"carry the red block around the {left|right} side of the barrier"`.

Consequence: `L-AUTH`, `CONFLICT-VF` and `CONFLICT-VT` are interpretable **only** on `sub_task`,
which makes evaluating the per-phase-language checkpoint a *prerequisite* rather than a bonus.
`cells.assert_interpretable` raises rather than letting an off-channel run read as a finding.

Note `ARBITER_LANG_KEY` selects `task` or `sub_task`, and **only the first modality key reaches
the model** — it is a choice between channels, never a way to have both.

### `--cell` speaks the middle phase from step 0

`arbiter/sim/tools/eval_policy.py --cell L-AUTH --cell-arm requested --barrier-h 0.10` sets the
axis, blocker flags and spoken text from `cells.py`, refuses conflicting manual flags, and
refuses a cell whose wording is out of vocabulary on the current `ARBITER_LANG_KEY`.

A cell's phrasing is three phase texts but the evaluator supplies **one** string per observation,
so `--cell` speaks the *middle* (route-carrying) phase for the whole episode. That is a
methodological choice with a measured basis: the policy commits to its route between step 17 and
step 45, well before the first observable motion at ~step 90 — inside what would be the *reach*
phase. Route text that only arrives when the carry phase begins arrives after the decision it is
meant to influence. The cost is the model hearing carry text during the reach, which is the
smaller of the two distortions.

### Validation happens before `AppLauncher`, and that is not a style preference

Cell resolution, the axis check and the CAG null check all run *before* the simulator boots. An
argument error after `AppLauncher` surfaces ~90 s in, and because everything past that point must
leave through `os._exit`, the failure is a bare exit with no result file — which is how a broken
control run once read as a silent success. This was re-learned the hard way here: the checks were
first written inside `main()` and a missing `--axis` spent a full Isaac boot before complaining.
`cells.py` is stdlib-only precisely so it can run up there.

### The invariant that caught a real error

`CONFLICT-VT` was first marked valid on `task`, reasoning that it is a *vision* cell. But its arm
speaks "around the left side", which is out of vocabulary on `task` whatever the cell tests —
authority and channel-requirement are independent properties. The rule is now derived from the
arms at import time (`cells._check_channels`), not restated in a literal.

### Nothing may be spoken that the training set does not ground

`cells.spoken_sub_tasks` builds phrases *from the spec*, never by formatting a sentence locally.
`test_every_spoken_phrase_is_grounded` generates the spec's whole phrase vocabulary across all
seven axes and asserts membership.

The receipts for why: TANGO measured an **empty** instruction at 0/6 *with no obstacle in the
scene at all*, and a scrambled one degraded toward no detour rather than toward the trained lane.
Both look like "language does not steer" and neither is. The load-bearing arm is always the
*fluent* one.

`V-AUTH` needs a route-silent middle phrase and topology has none, so it borrows EXTENT's
`"carry the red block out to the target"`. Both axes use `arb_cube_red`, so the sentence carries
no compositional novelty either — every word *and* the word pairing is in the training set.

### Every naming cell is paired, because one arm is never a result

Compliance alone cannot separate "obeyed the instruction" from "did what it always does and the
instruction agreed" — which is the failure mode under study. The reported quantity is the
difference between arms on a cloned simulator state. `test_paired_arms_expect_opposite_lanes`
fails any pair that expects the same outcome.

### The blocker's physics and framing criteria conflict by ~1 cm — and that is fine

`BLOCKER_HEIGHT_M = 0.40`, set from the measured end-effector apex range **0.222–0.349 m**: the
prop must clear the *arm*, not the barrier. TANGO's earlier 0.16 m choice ("taller than any swept
barrier") let 36/36 arms fly over, and the cube carried ~4 cm lower merely *grazed* the top —
which is 600 steps of grinding, not a forced detour. `BLOCKER_LEGACY_HEIGHT_M = 0.16` survives
only to reproduce the v6/v7 grids.

But the tallest blocker that fits **entirely** in the head frame is **0.339 m** (measured by
`arbiter/suites/view.py`), so 0.40 m is cropped: 85% of its face is in frame.

That is acceptable, and "fully in frame" was never the requirement. What a vision channel needs
is measured instead, and holds:

- it occludes the lane it seals (the cue is present);
- it occludes **nothing** the task depends on — object, target, all three barrier points and the
  forced lane all stay visible;
- ghost and solid are geometrically identical, so `CONFLICT-VF`'s control differs in collision
  **only**.

**Do not "fix" the cropping by moving or widening the camera.** The head camera reproduces
LIBERO's `agentview` (45° vertical), and Phase 0 evaluates a checkpoint trained under exactly
that view; changing it would invalidate every borrowed checkpoint and the mismatch would be
indistinguishable from a scene-copy error. Change the prop, never the camera.

### Camera-frame bearings

`right` is `+y`, `up` is `-x` — asserted in `test_view.py`, and the reason the spec's compass
words are what they are. TANGO had every DIRECTION instruction 90° out for weeks and *nothing the
benchmark reported would have revealed it*; it was caught by watching a rollout. Use "far"/"near",
never "top"/"bottom" ("move it to the top" invites "lift it"). Object phrases are curated, never
`replace("_", " ")`.

Related: the rename from `tango_*` to `arb_*` assets nearly broke this. `_obj_phrase` strips the
project prefix, and with the filter still naming `"tango"` every phrase would have rendered as
*"arb cube red"* and gone into the dataset language. The curation guard now asserts against
`"arb"`.

### Process traps inherited from TANGO

1. **`os._exit` after `AppLauncher`, plus `timeout -k`.** SIGTERM alone will not kill Isaac's
   ~200 non-daemon threads; one TANGO run ignored a plain `timeout` for 2h50m at 116% CPU having
   printed nothing.
2. **One `SimulationContext` per process; one launch per baked prop configuration** (barrier
   height, offset, blocker). Only collisionless kinematic props (pads) may move per condition.
3. **`CUDA_VISIBLE_DEVICES`, never `--device cuda:N`** — Isaac's RTX renderer renders on GPU 0
   regardless (2.4× throughput difference). Use a `flock`'d pull queue, not a static split.
4. **A bare `wait` blocks on the background policy server.** Collect PIDs and
   `wait "${PIDS[@]}"`. Empty log files read as "still running" — that is the tell.
5. **One HDF5 writer per file**, enforced by the filename.
6. **`DECIMATION = 4`** (30 Hz control over 120 Hz physics), or every phase times out
   centimetres short and it looks like bad IK.
7. **Rate-limited commands accumulate from the previous *command***, never from achieved state.
8. **`--save_total_limit 3`** on every fine-tune. TANGO's runs are 58–71G each purely from
   retaining ten checkpoints; that is where 515G of its 1.1T went.

### Measurement traps — the theme is checks that cannot fail

9. **An empty run must FAIL.** TANGO's gate once printed `VERDICT PASS: the axis is fully
   achievable` having tested *zero* conditions, because `n_ok == len(results)` is trivially true
   at zero. `cells.conditions` raises on an empty enumeration for the same reason. *A check that
   cannot fail is worse than no check, because it reads as evidence.*
10. **Success predicates only see the end state.** ORDER scored 20/20 while executing the
    *reverse* sequence; an "around" route flew *over* the barrier and reported success.
11. **A blocker number is meaningless without the route classifier.** The blocker seals a lateral
    lane and cannot seal the vertical one, so a policy that goes over posts a high score having
    never detoured — ACT does exactly that at every height. Always read
    `scripts/bench/score_route_class.py` beside success.
12. **The place margin must equal the visible marker.** 0.08 against a 3 cm pad inflated every
    place axis and superseded the numbers.
13. **Never conclude from one barrier height.** TANGO's scrambled-instruction arm looked clean at
    h=0.09 and reversed at h=0.10.
14. **Programmatic checks are blind to composition.** A barrier rotated 90°, a camera aimed at the
    shoulder and a robot floating 18 cm off the table all passed every assertion and were caught
    by rendering an image. `view.py` closes the quantitative questions so that looking at an image
    is spent on what arithmetic cannot see. Render after any scene change.
15. **Determinism needs jitter in a quantity no cell sweeps** (`jittered_home`, σ 0.02 rad, seeded
    from condition key + attempt).
16. **Floor markings** need `collider=False` (a 3 mm ledge catches the fingers) and
    `rot=YUP_TO_ZUP` (otherwise a 30×14 cm lane spawns as a 4 mm vertical *fin* that renders as a
    hairline and reads as "too faint"). Mark fixed geometry, never the swept quantity.

## Status

**Phase 0.1–0.2 done.** Repo scaffolded on `/data1`; TANGO's sim stack copied and adapted;
hygiene checker reworked and passing `--strict` with an empty baseline; the four-cell table
written; **111 stdlib-only tests passing**; 19 procedural props and 18 graspable assets generated
and validated through the USD prop test.

Constants confirmed to have survived the copy: `BLOCKER_HEIGHT_M = 0.40`,
`BLOCKER_LEGACY_HEIGHT_M = 0.16`, `bypass_side(0) = -y` with forced `+y`, `expected_homotopy`
flipping at `h* = 0.08`.

**Sim venv built** (`venvs/sim`, py3.11, torch 2.7.0+cu128, CUDA visible), Isaac Lab installed
editable from the pinned checkout. `pxr`, `AppLauncher` and every `arbiter` module import cleanly
after sourcing `activate_sim_env.sh`. A bare `import isaaclab` still fails on `omni.client` — that
is expected and is precisely why the codebase constructs `AppLauncher` *before* importing
`isaaclab`.

Two bugs fixed while getting there:

- `setup_sim_env.sh` never created the `arbiter/sim/.venv` compatibility symlink that
  `activate_sim_env.sh` resolves, so the venv existed on the data disk and nothing could find it.
- `_activate_sim_env_die` returns from *itself*, not from the sourced script, so every guard
  printed its error and then fell through to `[INFO] Activated arbiter sim environment`. A guard
  that reports success after failing is the same class of defect as the vacuous gate. Call sites
  now `return 1`.

**CAG baseline implemented** (`arbiter/policy/cag.py`, 21 tests). Wraps any policy: two forward
passes per chunk, `a = a_uncond + omega*(a_cond - a_uncond)`. `omega = 1` short-circuits to a
*single* pass returning the conditional action byte-for-byte, so the baseline column of the sweep
is identical to the unguided run rather than merely close to it. It **refuses to run** at any
other omega unless the caller asserts an instruction-dropout checkpoint, because an empty
instruction on a normally-trained checkpoint is out of distribution (TANGO: 0/6 with no obstacle
at all) and a tradeoff curve built against it would be an artifact.

### The fidelity gate's expected numbers, corrected

The plan and the write-up quote **18/18 LEFT ×18** and **20/20 route-correct**. Those are
*subsets*. Read off TANGO's stored eval JSONs, the full grids are:

- **control** (no blocker, six around-class heights 0.08-0.13 x 6 episodes): **36/36** success,
  36/36 route class `around`, **all on the `-y` lane**.
- **sweep** (across `h*`): **26 rollouts over 6 heights, 100% route-correct**, flipping from
  `over` to `around` exactly at `h* = 0.08`.

`scripts/bench/check_fidelity.py` asserts those and nothing looser. TANGO's own numbers are 100%
on both, so a tolerance would be tolerance for an unexplained difference — and the gate exists so
that an unexplained difference stops the project instead of propagating into it.

Validated in both directions against TANGO's stored data before ever running on a GPU: it
**passes** on the v7 control, and **fails** when fed the v7 blocker arm as the control, and on
empty directories.

### What the blocker arm actually did, and why `no_lane` matters

Re-scored with ARBITER's own copied classifier, TANGO's v7 blocker arm over 36 rollouts:
`{NO-CROSSING: 24, over: 10, around: 2}`, and **zero on `+y`** — the forced detour never happens.
So 34 of 36 rollouts made **no lane choice at all**: they either ground against the blocker for
the whole budget or lifted over the middle, which the blocker cannot seal.

`scripts/bench/score_arbitration.py` therefore counts those as `no_lane` and excludes them from
the side denominator rather than scoring them as "wrong lane". Scoring them as failures would
invent a decision the policy never made — and ACT does exactly this at every height.

**Phase 0.3 blocked on GPUs.** A TANGO fine-tune (`n1d7_rel_v7_seed1234`, the seed-1234
replication) holds all four A6000s at ~44.8/49.1 GiB. It saves every ~29 min.
`scripts/bench/fidelity_gate.sh` refuses to start unless the emptiest GPU has 20 GB free, because
an OOM mid-gate looks exactly like the copy having drifted.

### Phase 0 runbook

Everything below is written and validated as far as it can be without a GPU. The order matters:
each step's failure mode is cheaper to discover than the next step's.

```bash
# 0.3  copy fidelity -- reproduce TANGO's v7 control and height sweep on ARBITER's own scene
bash scripts/bench/fidelity_gate.sh
#      validated both ways against TANGO's stored data: PASSES on the v7 control,
#      FAILS on the v7 blocker arm and on empty dirs

# 0.4  the free result -- the per-phase-language checkpoint, never evaluated on DETOUR
ARBITER_LANG_KEY=sub_task bash scripts/bench/omega_sweep.sh   # with OMEGAS=1
#      CKPT=data/checkpoints/n1d7_rel_v8_subtask-checkpoint-4000

# 0.5  instruction-dropout retrain -- the only new training in Phase 0
bash scripts/bench/train_dropout.sh

# 0.7  the go/no-go: sweep omega across the whole four-cell table
CKPT=<dropout checkpoint> bash scripts/bench/omega_sweep.sh

# scoring
python scripts/bench/score_arbitration.py --dir data/output/eval/omega_sweep
```

Both launchers refuse to start when the GPUs are occupied (checked: they exit 1 with the memory
table). `train_dropout.sh` runs its cheap checks first — sub_task coverage, then language
fidelity — so a dataset problem surfaces in seconds rather than hours in.

### The language half of fidelity is already answered

`scripts/bench/check_language_fidelity.py` re-derives every instruction in a dataset from the
spec and compares. On `data/datasets/lerobot_merged/arbiter_v8_subtask` (615 episodes, copied
from TANGO and 51M in total): **615/615 identical at task level and 615/615 at phase level.**

So ARBITER's copied spec reproduces the language those checkpoints were trained with, byte for
byte. Only the visual/physics half of fidelity is still waiting on a GPU.

Getting there exposed a trap worth keeping. `_obj_phrase` strips the project prefix, and the
curated dictionary is keyed on `arb_*`, so a TANGO-collected sidecar renders `tango_cube_red` as
**"red tango cube"** — a phrase no annotator would write and nothing at inference reproduces. It
is silent, and it would go into every re-derived instruction. The copied dataset's sidecar was
therefore rewritten to `arb_conditions.jsonl` with `arb_*` keys, leaving the `task` strings
untouched because those define the task-index partition. The checker refuses a dataset that
still carries the old prefix.

### Not yet done

The GPU-dependent steps: the 0.3 gate run, the 0.4 v8 evaluation, the 0.5 retrain, the 0.7 sweep.
Nothing has been committed yet.
