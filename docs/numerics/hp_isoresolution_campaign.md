# Structured and unstructured space/time hp campaign

The extended campaign contains periodic plane-wave and PEC-cavity cases,
centered and alternating fluxes, equispaced and warp-blend nodes, p=1,2,3,4,
M1--M4, and structured and unstructured tetrahedral families. The baseline
time factor is one. By default M4 also receives factors two and four, giving
the fixed-mesh dt, dt/2 and dt/4 experiment. This produces 384 case files;
submit focused subsets with `--match` rather than launching the whole manifest.
Each production case runs one wave period with H-first ESPRK(p+1). Every time
step, including the initial state, contributes to the sampled maximum-in-time
spatial L2 errors of E, H, the combined field, and all six components.

## Resolution and meshes

| p | M1 | M2 | M3 | M4 |
|---|---:|---:|---:|---:|
| 4 | 3 | 6 | 12 | 24 |
| 3 | 4 | 8 | 16 | 32 |
| 2 | 6 | 12 | 24 | 48 |
| 1 | 12 | 24 | 48 | 96 |

Entries q are elements per wavelength along x for the periodic box [0,2]x[0,1]^2
(k=pi, lambda=2, c=1, period=2), and elements per edge for the unit PEC cube
(period=2/sqrt(3)). The invariant nominal resolution is p*q=12*2^(M-1).
This is a directional resolution proxy: actual DG scalar and field unknown
counts are recorded separately as Ne*binomial(p+3,3) and six times that value.

The two base families q=3 and q=4 are meshed independently. Each subsequent
mesh uses Gmsh uniform tetrahedral splitting (eight children per parent).
Only CAD edge segment counts are prescribed; triangular faces and tetrahedral
interiors are unstructured. q is a target spacing/edge subdivision count, not
an exact count of tetrahedra intersected by every interior ray. Rates use both
measured h_max, the largest tetrahedral edge, and
h_eff=(|Omega|/Ne)^(1/3).

Opposite surfaces are made periodic before generation. After every split,
translated boundary triangle connectivity is verified geometrically. Each
unique mesh is shared by matching p/M cases and both fluxes/nodal sets. Mesh
hashes, Gmsh version, seed, and quality metrics accompany the results.
Generation is single threaded with seed 1234. Netgen optimizes the base mesh;
refined meshes retain their nested split structure. The minimum tetrahedral
mean ratio must be at least 0.1; failure stops preparation.

The structured family uses transfinite box curves, surfaces and volumes and is
tetrahedralized by Gmsh. For the 2x1x1 periodic box an odd x resolution rounds
the transverse cell count to the nearest positive integer, so measured h_max
and h_eff—not nominal q—are used for comparisons. The PEC cube uses equal
counts on all edges.

Gmsh API reference: https://gmsh.info/doc/texinfo/#gmsh_002fmodel_002fmesh_002frefine

## Time resolution

The requested rule is dt_p = (p/4)*dt4 within each M, so lower orders use more
time steps per period. This differs from constant time-step counts across p.
With CFL=0.8/4=0.2, preparation computes

    dt4_limit = min_p (4 * CFL * hmin_p / p^2)
    n4 = 3 * ceil(T / (3 * dt4_limit))
    n_p = 4*n4/p,  dt_p = T/n_p.

hmin here is the solver CFL length min(3*volume/surface_area), the inradius;
it differs from the minimum diameter reported in the convergence CSV. Making
n4 divisible by three gives exact integer counts for p=3 as well as p=1,2,4.
The Julia runner independently enforces each mesh's CFL bound. No fixed dt
ratio across mesh levels is assumed because split tetrahedra can change shape.

CFL=0.2 and ESPRK(p+1) do not establish negligible temporal error by themselves.
Generated time-factor cases repeat M4 with two and four times the baseline
step count and distinct output paths. A discrepancy means temporal
contamination must be resolved before interpreting spatial rates.

Every case also computes the spatial L2 interpolation error at t=0, before any
ESPRK step. Electric, magnetic, total and component errors are written into
the result CSV.

## Prepare on MN5

The preparation interpreter needs NumPy and the Gmsh Python API. Select a
working environment through PYTHON if needed. Solver jobs require the existing
Julia project/depot and Intel MPI configuration. Preparation produces absolute
paths, so prepare on the filesystem used by the solver jobs.

From the repository root:

```bash
mkdir -p output/slurm
sbatch examples/prepare_mn5_hp_isoresolution.slurm
```

Alternatively, run directly on a suitable allocated node:

```bash
python3 examples/prepare_hp_isoresolution.py --output-dir output/hp_isoresolution
```

The extended diagnostic defaults are equivalent to:

```bash
python3 examples/prepare_hp_isoresolution.py \
  --output-dir output/hp_isoresolution_extended \
  --mesh-families unstructured structured \
  --time-factors 1 2 4 --temporal-levels 4
```

Use a fresh output directory for each campaign. Preparation refuses to
overwrite a directory. The manifest is published only after every mesh passes.
The defaults have large finest meshes: in the locally tested Gmsh 4.15.2 base
families, extrapolating exact eightfold splitting gives about 12.2 million
periodic and 37.0 million cavity tetrahedra for p=1,M4. The preparation Slurm
script reserves a full node's memory; production preparation and solve times
have not been benchmarked. Its 12-hour request and the solver's 36-hour request
are editable placeholders, not completion guarantees.

## Preview and submit

After preparation completes, preview all resource requests:

```bash
bash examples/submit_mn5_hp_isoresolution.sh --root output/hp_isoresolution
```

Then submit:

```bash
bash examples/submit_mn5_hp_isoresolution.sh --root output/hp_isoresolution --submit
```

Each case is a separate job. Rank requests are ceil(Ne/elements_per_rank),
capped by max_ranks (defaults 1000 and 448). One thread per rank is used; nodes
are ceil(ranks/112), configurable with --ranks-per-node. Element count, not p,
controls this requested resource scaling. Account, QOS, WALLTIME and
SBATCH_COMMAND can be overridden through environment variables. Use --match
to select case names. Extended names include the family and timestep factor,
for example `plane_wave_centered_equispaced_structured_p4_m4_dt2`.
Repeat `--match` to require several substrings. For example, all baseline
structured centered/equispaced plane-wave cases are selected with
`--match plane_wave_centered_equispaced_structured --match _dt1`.
Repeated --submit calls submit again; job IDs are appended to submissions.jsonl.

## Results and scope

```bash
python3 examples/manage_hp_isoresolution.py collect --root output/hp_isoresolution
```

The collector reports measured h_max and h_eff rates separately for each
physical case, flux, nodal set, mesh family, timestep factor and p. It also
reports t=0 interpolation rates. It does not bridge missing mesh levels. It writes
convergence.csv and pending_cases.json. Unexpected rates do not fail a job;
non-finite errors, invalid meshes and CFL violations do. No universal p+1 rate
is asserted by this campaign.

Each case CSV contains the established field-error and mesh-quality schema.
Its metadata TOML contains actual ranks, scalar/field DOFs, steps, RK stages,
partition RHS evaluation counts, initial/final broken-divergence L2 and signed
charge diagnostics, and relative energy drift. These are validity diagnostics,
not additional convergence targets. Broken divergence excludes face jumps.
Integration timing includes per-step error evaluations. Case wall time includes
setup and mesh loading, but excludes Julia startup/imports and final output.
The Slurm stdout reports launcher wall time including Julia startup. Process
hours are actual ranks times case wall seconds divided by 3600.

Generate initial, final-time and maximum-in-time tables for either family and
mesh metric with, for example:

```bash
python3 examples/postprocess_convergence_csv_tables.py \
  --input output/hp_isoresolution_extended/convergence.csv \
  --output-dir output/hp_isoresolution_extended/latex_tables/structured_hmax \
  --mesh-family structured --h-metric h_max --time-factor 1 \
  --nodal-set equispaced --error-modes initial final max-time
```

Repeat with `--h-metric h_eff` and/or `--mesh-family unstructured`. Fixed-mesh
temporal tables are generated with:

```bash
python3 examples/postprocess_hp_temporal_sensitivity_tables.py \
  --input output/hp_isoresolution_extended/convergence.csv \
  --output-dir output/hp_isoresolution_extended/latex_tables/temporal_structured \
  --mesh-family structured --nodal-set equispaced --level 4
```

## Small local smoke run

```bash
python3 examples/prepare_hp_isoresolution.py --output-dir /tmp/hp-smoke \
    --levels 1 --periods 0.001 --elements-per-rank 100000 --time-factors 1
mpiexec -n 2 julia --project=. examples/run_hp_isoresolution_case.jl \
    /tmp/hp-smoke/cases/plane_wave_centered_equispaced_structured_p4_m1_dt1.toml
mpiexec -n 2 julia --project=. examples/run_hp_isoresolution_case.jl \
    /tmp/hp-smoke/cases/pec_cavity_alternating_warp-blend_unstructured_p4_m1_dt1.toml
```

This checks mesh/solver/output plumbing; it does not measure asymptotic rates.
