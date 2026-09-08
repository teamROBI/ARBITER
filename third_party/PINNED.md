# Vendored third-party checkouts

These are upstream projects, not ARBITER code. The working trees are **not** tracked in this
repo (see `.gitignore`); this file is, so the pin is recorded and checkable.

| path | upstream | pin | commit |
|---|---|---|---|
| `IsaacLab` | https://github.com/isaac-sim/IsaacLab.git | `v2.3.2-13-gf4aa17f87` | `f4aa17f87e2e5db5484f0b5974918573e8918ce2` |
| `Isaac-GR00T` | https://github.com/jkim50104/Isaac-GR00T.git (fork) | `v1.6-77-g032d732` | `032d7323314cdad7e8d635cc3d27f8766e88b4e2` |

## Why these are pinned, and why they were copied rather than re-cloned

Phase 0.3 evaluates a **borrowed checkpoint** on a scene this repo builds itself, and reproducing
that checkpoint's known numbers is what proves the scene did not drift. Cloning upstream `main`
would pull a newer Isaac Lab, whose contact handling or renderer may differ from what the
checkpoint was trained under — and a fidelity mismatch would then be indistinguishable from
"ARBITER's copy of the scene is wrong". Pinning removes the variable.

Once ARBITER trains its own policies on its own balanced collection (Phase 2), the pin is free to
move: at that point nothing depends on matching an external training run.

## Verify the pin

```bash
for d in IsaacLab Isaac-GR00T; do
  printf "%-14s %s\n" "$d" "$(git -C third_party/$d rev-parse HEAD)"
done
```

`IsaacLab/_isaac_sim` is a symlink into the shared Isaac Sim 5.1.0 install
(`/data1/jokim/simulation/isaacsim`, 30G), created by `arbiter/sim/activate_sim_env.sh`, which
probes several standard locations. That is a machine-level dependency like a system library, not
a sibling-project reference — the install is not a checkout of another project.
