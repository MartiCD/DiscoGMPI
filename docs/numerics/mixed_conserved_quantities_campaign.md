# Mixed-order long-time invariant h/p experiments

This reproduces the six-quantity periodic traveling-wave experiments using
`DistributedMixedDG`, p_E=p, p_H=p+1, equispaced nodes, and ESPRK(p+2).
It does not implement the separate weak-charge / RK-stage high-precision audit.
The original equal-order workflows remain available.

The numerical integrals are evaluated in the actual two polynomial spaces using
the reusable mixed diagnostic workspace and degree 4(p_H)+2 quadrature. Energy,
optical chirality, linear/angular momentum and integrated D/B are sampled every
100 steps by default, with initial, final and checkpoint samples always included. The CSV retains
signed components, exact values and drift from the original numerical initial
state. Plots reuse the existing signed and absolute-error six-panel functions.
The historical plots call sums of integrated D/B components “charge”; these
are not divergence charge. The CSV separately includes integrated elementwise
divergence charge and divergence L2 norms. Angular momentum on a periodic box
is not a universal invariant; the analytic value here applies to this wave.

## Defaults and comparison

- Domain [0,2] × [0,1] × [0,1], epsilon=mu=1, k=omega=pi, magnetic amplitude 1.
- h study: p_E=2, p_H=3 on M1–M4. These match the current executable equal-order
  h manager's resolutions (3,2,2), (6,4,4), (12,8,8), (24,16,16).
  Its older prose documentation lists different settings.
- p study: p_E=1,2,3,4 on a fixed M2 mesh per family. `--p-mesh-level` changes
  that mesh. Each study includes both structured/unstructured and both fluxes.
- Both fields interpolated initially; `--electric-initialization curl-potential`
  and `--magnetic-initialization curl-potential` enable the existing discrete-curl
  initializers independently. Keep initialization identical when comparing runs.
- T=10000, CFL=0.1, linear mixed rule dt_limit=CFL*h_min/p_H, dt adjusted downward
  to hit T. This differs from the equal-order timestep rule. A CFL sweep is
  needed to separate temporal effects from spatial effects in invariant drift.
- Polynomial-aware ranks, 112 physical cores/ranks per full node, 36-hour
  allocations. Small explicit partial-node requests and meshes with fewer than
  112 elements can share a node. Multi-node rank counts must be multiples of 112.
- The work estimate is `Ne * Np_E * Np_H * (p_E+2) / (10*20*4)`, normalized to
  p_E=2. `--elements-per-rank 500` sets the reference work per rank. This is a
  heuristic for selecting a pilot; a measured benchmark decides whether to run
  production. `--elements-per-node 28000` adds a memory floor equivalent to the
  former 56-ranks * 500-elements layout. Per-rank runtime overhead still needs
  measurement. Rank count is capped by the element count.
- Sample every 100 steps, checkpoint every 1000, print every 1000. Two rotating
  checkpoint slots retain the previous committed generation. Final samples and
  checkpoints are also written on a cooperative wall-time exit. Setting
  `--checkpoint-every 0` disables periodic saves, but boundary saves remain.
- A ten-minute wall-time reserve permits a checkpoint before the 36-hour limit.
  The batch script has a 45-minute no-progress watchdog and a two-hour startup
  allowance. `--watchdog-seconds 0` disables the watchdog. Output locks prevent
  overlapping allocations from writing the same history/checkpoints.

The analytic reference integrals are energy=1, chirality=0, P=(1,0,0),
L=(0,1/2,-1/2), and integrated D=B=0 for this particular traveling wave.
The runner verifies the domain before using these references.

## MN5 preparation and submission

Upload the current code first. From the mixed-order repository on MN5:

```bash
mkdir -p output/slurm
sbatch examples/prepare_mn5_mixed_conserved_quantities.slurm
```

This prepares 32 independent cases under `output/mixed_conserved_quantities`:
16 h-study and 16 p-study jobs. The existing Gmsh split-refinement generator
and structured mesh writer are reused. Preparation uses the same module stack
as the existing equal-order h preparation script.

For an exact comparison on previously prepared h meshes, instead run:

```bash
python3 examples/manage_mixed_conserved_quantities.py prepare \
  --mesh-manifest output/conserved_quantities_h_exrank500/manifest.json
```

Choose one preparation method; existing roots are never overwritten. Case
paths are absolute, so prepare on MN5, or use an existing MN5 mesh manifest.
Physical parameters are frozen in case TOMLs during preparation. Submit-time
`--sample-every` and `--checkpoint-every` overrides are recorded in run metadata.
Changing sampling preserves the fields/time grid but can miss excursions between
samples. For oscillatory invariant histories, compare a short every-step pilot
with sampling every 100 steps before deciding that temporal resolution suffices.

To reproduce the older long-time p experiment's fixed structured mesh,
magnetic curl initialization and T=1000 instead of the unified defaults:

```bash
python3 examples/manage_mixed_conserved_quantities.py prepare \
  --root output/mixed_invariants_original_p \
  --studies p --families structured \
  --p-mesh examples/meshes/periodic_box_structured_2x1x1.vtk \
  --magnetic-initialization curl-potential --final-time 1000
```

This still uses mixed ESPRK(p+2) and CFL=0.1, not the old time integrator/rule.
Pass this alternative `--root` to subsequent submit/status/plot commands.

After successful preparation, first preview the short benchmarks:

```bash
python3 examples/manage_mixed_conserved_quantities.py benchmark
python3 examples/manage_mixed_conserved_quantities.py benchmark --submit
```

Each benchmark uses a separate output below `ROOT/benchmarks/CASE/FINGERPRINT`,
20 warmup steps followed by 1000 measured steps, and a two-hour allocation by
default. Its timestep comes from the full T=10000 case. Timing includes the RHS,
sampled diagnostics, reductions, writes and checkpoints after warmup; setup/JIT
is excluded. A terminal checkpoint also contributes, making short estimates
conservative. Increasing `--benchmark-steps` improves confidence.

Start with one costly case and inspect node utilization/memory:

```bash
python3 examples/manage_mixed_conserved_quantities.py benchmark \
  --studies p --families unstructured --fluxes centered --orders 4 --submit
bsc_load JOB_ID
```

Once those benchmarks finish:

```bash
python3 examples/manage_mixed_conserved_quantities.py report
python3 examples/manage_mixed_conserved_quantities.py submit --segments auto
python3 examples/manage_mixed_conserved_quantities.py submit --segments auto --submit
```

Production admission requires a benchmark matching the mesh/physics, source,
actual ranks, sampling and checkpoint interval. The projection includes a 20%
margin and must fit `--target-days 7` (queue time excluded). A missing or too-slow
benchmark prevents submission. This is an estimate, not a completion guarantee;
longer pilots are advisable near the threshold. All selected cases are checked
before any are submitted.

If a layout is too slow, benchmark another legal `--ntasks` count, then use that
same override for `report` and `submit`. Small partial-node jobs may use 16, 32
or 64 ranks; full-node jobs use 112, 224, 336, etc. Rank count alone does not
prove efficiency: check `bsc_load`, memory and step rate in the pilot. A long
allocation with spinning MPI ranks can still have high CPU usage and poor
scaling. `--ranks-per-node` accepts only 112.

`--segments auto` schedules enough dependent allocations to cover the measured
whole-case runtime plus margin. Each successor uses `afterok`, starts with
`--restart=auto`, and exits cheaply if the case already completed. A cooperative
wall-time checkpoint exits successfully. An OOM, watchdog timeout, or unexpected
failure stops the chain; inspect the logs and resubmit with `--resume` after
resolving the cause. Use `--segments N` to limit the chain explicitly. The default
is one allocation. An existing/recorded case needs `--resume`; active or pending
jobs recorded in the ledger are checked against `squeue` and skipped.
Do not run simultaneous submitters. `ACCOUNT`, `QOS`, `WALLTIME` override scheduler
settings; a benchmark defaults to 2h and production to 36h.

For a short independent pilot campaign:

```bash
OUTPUT_ROOT=output/mixed_invariants_pilot \
sbatch examples/prepare_mn5_mixed_conserved_quantities.slurm \
  --studies h --families structured --levels 1 \
  --final-time 1 --checkpoint-every 100
```

Wait for preparation, then run `benchmark` and `report` for this root before
`submit` as above. For a shortened physical case choose `--benchmark-steps` and
`--warmup-steps` that fit before its final step, and pass the same benchmark-step
requirement to `report` and `submit`.

## Status and plots

```bash
python3 examples/manage_mixed_conserved_quantities.py status
python3 examples/manage_mixed_conserved_quantities.py plot
# Partial completed selection:
python3 examples/manage_mixed_conserved_quantities.py plot \
  --studies h --families structured --fluxes centered --levels 1
```

Histories: ROOT/STUDY/FAMILY/FLUX/{mLEVEL|pORDER}/diagnostics/invariant_diagnostics.csv.
Every allocation writes an immutable `segments/UUID.csv`. Checkpoints record the
committed segment ends and the original initial diagnostic baseline. A final
streaming merge produces the canonical history above, discards crash suffixes,
and removes duplicate restart-boundary rows. Partial histories remain available
in `segments/`; plotting through the manager still requires completion.
`completed.toml` is written only after the final sample and checkpoint succeed.
Status distinguishes completed, partial, submitted and not-submitted; it does
not infer Slurm state. Plotting requires completion of every selected run and
limits rendering to about 50000 points per run without changing stored samples.
PDF, PNG and SVG plots go into each flux directory's `plots/` folder. These are
time histories under h/p refinement, not fitted convergence-rate tables.

## Existing campaigns and continuation

Upload the new scripts after old jobs stop, or stage them separately. Existing
TOMLs are accepted: the manager overrides sampling to 100 and checkpointing to
1000 when submitting. It also recomputes polynomial-aware ranks, so an old
campaign's saved `ranks` does not silently retain an inefficient policy.

To change rank counts, prepare a fresh campaign root while reusing the old mesh
bank. No old output/history is removed:

```bash
python3 examples/manage_mixed_conserved_quantities.py prepare \
  --root output/mixed_conserved_quantities_optimized \
  --mesh-manifest output/mixed_conserved_quantities/mesh_bank/manifest.json
python3 examples/manage_mixed_conserved_quantities.py benchmark \
  --root output/mixed_conserved_quantities_optimized --submit
# After the pilot jobs finish:
python3 examples/manage_mixed_conserved_quantities.py report \
  --root output/mixed_conserved_quantities_optimized
python3 examples/manage_mixed_conserved_quantities.py submit \
  --root output/mixed_conserved_quantities_optimized --segments auto --submit
```

If the old preparation used an external mesh manifest, supply that same manifest
instead of the mesh_bank path. Preserve any nondefault physics/initialization
settings in the new preparation. Filters (`--studies`, `--orders`, `--levels`,
`--families`, `--fluxes`, `--match`) can restrict pilots and submission.

To retain old work, keep its actual MPI count. For example, a four-rank case:

```bash
python3 examples/manage_mixed_conserved_quantities.py benchmark \
  --root output/mixed_conserved_quantities \
  --match mixed_inv_h_structured_centered_p2_m2 --ntasks 4 --submit
python3 examples/manage_mixed_conserved_quantities.py report \
  --root output/mixed_conserved_quantities \
  --match mixed_inv_h_structured_centered_p2_m2 --ntasks 4
python3 examples/manage_mixed_conserved_quantities.py submit \
  --root output/mixed_conserved_quantities \
  --match mixed_inv_h_structured_centered_p2_m2 --ntasks 4 \
  --resume --segments auto --submit
```

The runner discovers old `checkpoints/stepN` manifests as well as new slots,
validates checksums/history, and falls back to the previous valid checkpoint if
necessary. Legacy flat checkpoint metadata plus the flushed original CSV are
supported. All history files referenced by checkpoints must remain in place.
If committed checkpoints exist but none validate, it fails instead of silently
starting from zero. MPI count, ownership, mesh/basis/material signature and time
grid must match. The manager rejects rank changes in an existing checkpoint tree.

Direct execution supports `--restart=auto|none|DIR`, `--output-dir=DIR`,
`--sample-every=N`, `--checkpoint-every=N`, `--segment-steps=N`,
`--max-wall-seconds=N`, `--benchmark-steps=N`, `--warmup-steps=N`, and
`--heartbeat=FILE`, all with `=`. Direct Slurm submissions can forward these
arguments after the case TOML; the batch wrapper adds the wall-time deadline,
watchdog and output lock. `--segment-steps` is useful for deliberate recovery
pilots. The fixed tiny test below compares exact fields and histories, including
legacy migration and a corrupt newest checkpoint; it is not a convergence test:

```bash
python3 test/test_mixed_invariant_workflow.py
mpiexec -n 2 julia --project=. test/test_mixed_invariant_restart.jl
```
