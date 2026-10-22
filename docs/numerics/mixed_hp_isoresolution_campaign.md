# Mixed-order MN5 hp-isoresolution campaign

This workflow mirrors `hp_isoresolution_campaign.md` with E in P_p, H in
P_(p+1), and H-first ESPRK order p+2, for p=1,2,3,4. It runs from a plain
rsync copy; Git is not required. The older eight-job
`submit_mn5_mixed_order.sh` is a separate structured plane-wave example.
Use the `mixed_hp_isoresolution` scripts below for the complete hp workflow.

For repeated source uploads, use the repository helper from the local checkout:

```bash
bash examples/upload_mn5.sh --dry-run
bash examples/upload_mn5.sh --upload
```

The first command connects but changes nothing. The second incrementally copies
changed files to `/gpfs/projects/bsc115/bsc021336/DiscoGMPI-mixed-order/`.
It excludes `.git/`, `.julia_depot/`, `output/`, generated visualization and
job-log files, and never uses `--delete`. Set `MN5_HOST`, `MN5_DESTINATION`,
`MN5_SSH_PORT`, or `MN5_JUMP_HOST` when the site connection differs from the
defaults. `--host`, `--destination`, `--port`, and `--jump` provide the same
settings on the command line. The local branch is copied as ordinary files;
MN5 does not need a Git checkout.

## Campaign definition

Defaults cover periodic plane waves on [0,2]x[0,1]^2 and PEC cavity modes
on [0,1]^3; centered and alternating fluxes; equispaced and warp-blend nodes;
structured and unstructured tetrahedra; and levels M1--M4. Both families
reuse the established Gmsh mesh generation, periodic triangle verification,
SHA256 provenance and minimum mean-ratio quality threshold. Each level has
nominal electric resolution p*q=12*2^(M-1). Magnetic resolution is (p+1)*q
and is deliberately reported separately. The two fields do not have equal
DOF budgets: their counts are 3*Ne*binomial(p+3,3) and
3*Ne*binomial(p+4,3). `order` in legacy-compatible tables denotes p_E.

Each case runs one period: 2 for the plane wave and 2/sqrt(3) for the cavity.
M4 also runs timestep factors 2 and 4, giving 384 independent cases in total
(256 baseline cases plus 128 temporal checks). Both factors and selected
levels can be changed at preparation. With `--levels 1`, the default temporal
level becomes M1. Initial-only jobs deduplicate across flux and time factor.

Preparation preserves dt_p=p*dt4/4. For the mixed solver it uses

    limit_p = CFL * inradius_p / (p+1)  # linear (default), p_H=p+1
    dt4_limit = min_p (4*limit_p/p)
    n4 = 3*ceil(T/(3*dt4_limit))
    nsteps_p = (4*n4/p)*time_factor

The default CFL is 0.1. The inradius is no larger than minimum altitude,
so this is conservative relative to the solver's altitude-based mixed CFL
estimate. The runner checks that estimate independently. Neither estimate
proves ESPRK stability on all meshes or negligible temporal error.
`--cfl-degree-rule quadratic` selects the older denominator `(p+2)^2`.

All nine error quantities (E, H, total and six components) use common volume
quadrature but separate electric and magnetic basis evaluations. Evaluations
are cached; error quadrature degree is 4*(p+1)+2 by default. Initial, final,
and sampled maximum-in-time L2 errors are written. Every timestep, including
t=0, is sampled. The total maximum is sampled directly, not constructed from
independently maximized E and H. Nonpolynomial exact solutions still require
quadrature sensitivity checks before publication; `cubature_degree` is an
explicit per-case parameter that can be increased in a separate campaign.

## Transfer and environment

Copy the updated repository into the separate MN5 directory
`/gpfs/projects/bsc115/bsc021336/DiscoGMPI-mixed-order/`, including all shared
hp Python utilities and scripts, not only files containing `mixed` in their
names. Exclude local `LocalPreferences.toml` unless it is already configured
for MN5. Keep the working MN5 MPI preferences and Julia depot there.

Preparation uses exactly the existing Gmsh module setup from
`prepare_mn5_hp_isoresolution.slurm`. Solver jobs use the same Intel MPI and
Julia modules as the existing hp workflow. Those modules and the MPI.jl
library must be compatible on MN5. No dependencies are installed by jobs.
Set `MIXED_CODE_ID` to a snapshot identifier for Git-free provenance.

Before submitting any solver case, instantiate once on an MN5 login node with
the same modules used by the jobs:

```bash
module purge
module load intel/2023.0 impi/2021.10.0 julia/1.10.0
cd /gpfs/projects/bsc115/bsc021336/DiscoGMPI-mixed-order
export JULIA_DEPOT_PATH="$PWD/.julia_depot:"
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. -e 'using MPI, DiscoGMPI; MPI.versioninfo()'
```

The last command must load both packages and report an MPI library compatible
with the Intel MPI `mpiexec`. If this checkout has no network access, reuse the
already working MN5 DiscoGMPI depot by placing it after the new writable depot,
for example `JULIA_DEPOT_PATH="$PWD/.julia_depot:/path/to/old/.julia_depot:"`,
then instantiate again. An MN5 `LocalPreferences.toml` from the working old
checkout may be copied only when it was created for the same Intel MPI module.
Do not copy laptop MPI preferences to MN5.

The production launcher performs the same package/MPI import once before
starting `mpiexec`; it exits with one readable setup error when the environment
is incomplete. It does not install packages inside a compute job.

## Prepare a small workflow check first

On MN5, from the copied repository root:

```bash
cd /gpfs/projects/bsc115/bsc021336/DiscoGMPI-mixed-order
mkdir -p output/slurm
export MIXED_CODE_ID=mixed-hp-snapshot-2026-10-16
export OUTPUT_ROOT=output/mixed_hp_smoke
sbatch examples/prepare_mn5_mixed_hp_isoresolution.slurm \
  --levels 1 --periods 0.000001 --mesh-families structured --time-factors 1
```

Wait for preparation to complete successfully and `manifest.json` to exist.
Then preview and submit a single case:

```bash
bash examples/submit_mn5_mixed_hp_isoresolution.sh \
  --root output/mixed_hp_smoke \
  --match plane_wave_centered_equispaced_structured_p4_m1_dt1

bash examples/submit_mn5_mixed_hp_isoresolution.sh \
  --root output/mixed_hp_smoke \
  --match plane_wave_centered_equispaced_structured_p4_m1_dt1 --submit
```

This tests setup, integration and output, not convergence rates.

## Prepare the full campaign

Choose a fresh output root; preparation refuses an existing directory. Paths
in manifests are absolute, so prepare on MN5 where the solver will run.

```bash
export OUTPUT_ROOT=output/mixed_hp_isoresolution
export LEVELS=4
export PERIODS=1
export CFL_EFFECTIVE=0.1
export ELEMENTS_PER_RANK=1000
unset MAX_RANKS
sbatch examples/prepare_mn5_mixed_hp_isoresolution.slurm
```

Optional preparation arguments are forwarded to the shared Python preparer:
`--mesh-families structured unstructured`, `--time-factors 1 2 4`, and
`--temporal-levels 4`. The preparation job requests one full node (112 CPUs)
for mesh memory and runs Gmsh single-threaded. The finest unstructured meshes
can contain tens of millions of tetrahedra. The mixed solver currently loads
global topology on root and stores dense operators per local/ghost element.
Element-based rank estimates are inherited from the old workflow, not a
validated mixed-order memory model. Check smaller cases' memory and timing
before requesting all finest levels; the default walltimes are placeholders.

## Preview, submit and monitor

After preparation succeeds:

```bash
# Preview one baseline subset (16 jobs).
bash examples/submit_mn5_mixed_hp_isoresolution.sh \
  --root "$OUTPUT_ROOT" \
  --match plane_wave_centered_equispaced_structured --match _dt1

# Submit that subset.
bash examples/submit_mn5_mixed_hp_isoresolution.sh \
  --root "$OUTPUT_ROOT" \
  --match plane_wave_centered_equispaced_structured --match _dt1 --submit

# Alternatively submit ALL 384 cases, once resources have been assessed.
bash examples/submit_mn5_mixed_hp_isoresolution.sh --root "$OUTPUT_ROOT" --submit

squeue -u "$USER"
sacct -X -u "$USER" --starttime today --format=JobID,JobName,State,Elapsed,ExitCode
```

Repeated `--match` filters require all substrings. Each case is a separate
job; ranks are ceil(Ne/elements_per_rank), capped by max_ranks. Nodes are
ceil(ranks/112), configurable with `--ranks-per-node`; submission also sets
the corresponding `--ntasks-per-node` placement explicitly. Override `ACCOUNT`,
`QOS` and `WALLTIME` through environment variables. Commands preview by
default; only `--submit` calls sbatch. Job IDs are appended to
`submissions.jsonl`, and logs are in the campaign's `slurm/` directory.
Repeated full submissions submit again and can overwrite completed case
results; choose subsets intentionally and do not overlap submissions.

Initial-only interpolation jobs are available with `--initial-only` (add
`--submit` to execute). They write `initial_results/` and can be reused across
flux/time-factor duplicates. These jobs do not produce evolution results.

## Checkpoint, restart and stalled-job protection

The mixed runner now checkpoints every 1000 completed timesteps and at the
final timestep. Existing prepared case files need no changes. Every-step
error sampling remains enabled. Each checkpoint saves the owned E/H fields,
step/time/dt, initial errors and diagnostics, all nine accumulated max-time
errors, and the number of samples. Restart continues at the next timestep
without resetting any error maximum. Rebuilding the mesh/operators and
workspaces is still required on each launch; those caches are not stored.

Storage is bounded to two rotating generations per case:

```text
ROOT/checkpoints/CASE/slot-a/mixed_manifest.toml
ROOT/checkpoints/CASE/slot-a/mixed_rank0000.bin ...
ROOT/checkpoints/CASE/slot-b/mixed_manifest.toml
ROOT/checkpoints/CASE/slot-b/mixed_rank0000.bin ...
ROOT/checkpoints/CASE/latest.toml
```

Before reusing the inactive slot, its manifest is invalidated. Each rank writes
an atomic field piece; the manifest commits only after all pieces and checksums
succeed. The active slot is preserved during this operation. Automatic restart
checks committed slots newest-first and falls back to the previous valid slot
if a piece is corrupt. If committed checkpoints exist but neither is compatible
and valid, the job fails with an explanation rather than silently discarding
its history. A final-step checkpoint can regenerate the result CSV after a
failure during final diagnostics/output.

The default `--restart auto` starts at t=0 when no committed checkpoint exists.
Resubmitting the same selected case later resumes automatically. Restart requires
the same case, time grid, mesh, polynomial spaces, flux, quadrature and MPI rank
count/partition; node placement may change. To change the number of ranks, start
a fresh run with a separate checkpoint directory. An explicit `--restart DIR`
must name a slot directory and requires selection of exactly one case.
`--restart none` refuses an existing checkpoint history; use a fresh checkpoint
directory for an intentional independent rerun.

The Slurm launcher uses an external Python supervisor (standard library only,
no packages or network access). It allows 7200 seconds for startup, then stops
the MPI launcher if the heartbeat has not advanced for 2700 seconds (45 minutes).
Rank zero updates the heartbeat at integration start, the first resumed step,
every 100 steps, checkpoint boundaries, and finalization. Thus the timeout also
covers a stalled checkpoint or final diagnostics. Allow enough time for 100
steps, checkpoint I/O and final diagnostics on the selected case. A watchdog
timeout prints `WATCHDOG_TIMEOUT` and exits 124; the batch allocation then ends
and Slurm cleans up its tasks. There is no automatic requeue or resubmission.
The launcher also takes a per-case `flock` to reject overlapping batch runs;
the kernel releases the lock when the batch script exits. Direct Julia launches
must likewise avoid simultaneous writes to a case's checkpoint/result paths.

Preview the four missing unstructured PEC M4 dt1 cases:

```bash
for p in 2 4; do
  bash examples/submit_mn5_mixed_hp_isoresolution.sh \
    --root output/mixed_hp_isoresolution \
    --match 'pec_cavity' --match "_equispaced_unstructured_p${p}_m4_dt1" \
    --checkpoint-every 1000 --restart auto --watchdog-seconds 2700
done
```

Add `--submit` to those commands after reviewing the requests and ensuring
the selected cases have no active jobs. Old runs created before checkpointing
must begin at step zero once; their printed errors/timesteps do not contain
restartable fields or the full error history. Upload all changed sources before
submitting. Uploading does not retrofit checkpoints into already-running jobs.

Overrides available on the mixed submitter:

```bash
--checkpoint-every 500    # more frequent recovery points
--checkpoint-every 0      # disable periodic and final checkpoint writes
--startup-timeout 14400   # longer operator/cache construction allowance
--watchdog-seconds 5400   # longer interval without a heartbeat
--watchdog-seconds 0      # disable external watchdog
```

For direct `sbatch` calls, the same defaults are set by the launcher; override
`MIXED_HP_CHECKPOINT_EVERY`, `MIXED_HP_RESTART`, `MIXED_HP_WATCHDOG_SECONDS`,
or `MIXED_HP_STARTUP_TIMEOUT` in the submitted environment. Extra arguments to
the Slurm script are passed through to Julia, including `--checkpoint-dir=DIR`,
`--restart=DIR`, and `--stop-after-step=N`. The last option checkpoints and exits
without publishing a completed result, allowing a deliberate restart test.

For a new pilot case only, using the same rank resources both times:

```bash
sbatch --ntasks=2 --nodes=1 --ntasks-per-node=2 \
  examples/run_mn5_mixed_hp_isoresolution.slurm CASE.toml \
  --checkpoint-every=10 --stop-after-step=20
# Wait for the deliberate stop, then:
sbatch --ntasks=2 --nodes=1 --ntasks-per-node=2 \
  examples/run_mn5_mixed_hp_isoresolution.slurm CASE.toml --checkpoint-every=10
```

Choose a pilot with more than 20 steps. A partial run produces no convergence
CSV. The resumed final CSV covers all nsteps+1 samples; collect and table
generation commands remain the same. Runtime metadata records the restart slot
and step. Integration time accumulates successful checkpointed segments; case
walltime/process hours describe the current allocation only, so use Slurm
accounting to include discarded work and setup across all attempts.

Checkpointing limits lost work; it does not establish or repair the cause of a
stall. Startup time is excluded from `integration_seconds`, so short integration
timings alone do not prove when an old allocation stopped progressing. Inspect
the job's CPU/memory timeline and heartbeat on new runs before changing resources.

Local resilience checks (tiny runs, not convergence studies):

```bash
mpiexec -n 2 julia --project=. test/test_mixed_hp_restart.jl
python3 -m unittest discover -s test -p test_progress_watchdog.py
```

## Collect and generate tables

```bash
python3 examples/manage_mixed_hp_isoresolution.py collect --root "$OUTPUT_ROOT"

python3 examples/postprocess_convergence_csv_tables.py \
  --input "$OUTPUT_ROOT/convergence.csv" \
  --output-dir "$OUTPUT_ROOT/latex_tables/structured_hmax" \
  --mesh-family structured --h-metric h_max --time-factor 1 \
  --nodal-set equispaced --error-modes initial final max-time

python3 examples/postprocess_hp_temporal_sensitivity_tables.py \
  --input "$OUTPUT_ROOT/convergence.csv" \
  --output-dir "$OUTPUT_ROOT/latex_tables/temporal_structured" \
  --mesh-family structured --nodal-set equispaced --level 4
```

Repeat for unstructured meshes, warp-blend nodes and h_eff as needed. The
shared table tools detect and validate the mixed degree columns and label
captions with p_E=p, p_H=p+1, ESPRK(p+2). The collector writes pending_cases.json,
does not bridge missing levels, and checks result metadata against the mixed
manifest. It reports measured slopes without an expected-rate verdict.
Temporal tables require complete factors 1,2,4 for a given p/level.

Results include separate degrees, nodal counts and DOFs. Metadata includes
mesh hashes, the snapshot ID, MPI rank count, Julia version, Slurm job ID,
step/stage counts, integration/error time, process-hours, energy drift,
and initial/final broken divergence and signed charges. Broken divergence
excludes face jumps. Error integration is included in reported integration
timing; it performs a global reduction at each sample.

## Local validation

### Every-step error performance and resources

The runner retains the initial, every-step, final and sampled maximum errors.
The error cache stores harmonic exact-field amplitudes but no unused physical
coordinate array. Its six-component global reduction buffer is reused every step.
Initial and final validity diagnostics share the error cache's quadrature points,
weights and interpolation matrices; gradient and value scratch arrays are reused
across elements and both calls. MPI trace exchanges reuse buffer-retaining
`MPI.MultiRequest` slots after `Waitall` completes.

Results are still published only after final validity diagnostics succeed. This
keeps the completed CSV/metadata contract intact; the final-step checkpoint
supports recovery if those diagnostics or output fail. Reduced allocation pressure does not establish a fix for
the MN5 Julia GC segmentation faults, which require a separate reproduction.
It caches owned element Jacobians and two harmonic
spatial amplitudes for the plane wave or cavity solution. Each sample uses a
single sincos pair, field evaluations and contiguous weighted error sums.
No quadrature degree or sampling interval is reduced. At p=4 (2197 quadrature
points), Jacobian and harmonic arrays cost about 0.20 MiB per owned element,
in addition to DG operators and field storage. Rank 0 prints its cache size
and timestep progress every 100 steps. The maximum is over discrete time
samples, not a certified continuous-time supremum.

Preparation still defaults to 1000 elements per rank. Use
`--elements-per-rank 500` during preparation (or `ELEMENTS_PER_RANK=500` for
the preparation Slurm wrapper) to change this. A 72-element case gets one
rank with either threshold. To benchmark two ranks on an existing case,
use a narrow filter and `--ntasks 2` on the submission script; omit `--submit`
to preview. Do not resubmit a case concurrently to the same output file.
Extra ranks should be selected from measured wall time and process-hours;
every-step error reductions can limit scaling on small meshes.

Mixed campaign preparation has no default rank cap: the requested ranks are
`ceil(nelements/elements_per_rank)`, and submission converts that count to MN5
nodes using `--ranks-per-node` (112 by default). Site/QoS limits still apply.
Set `MAX_RANKS=448` in the preparation environment or pass `--max-ranks 448`
to restore a ceiling. Increasing `ELEMENTS_PER_RANK` requests fewer ranks and
allows more work per rank; removing the rank cap instead keeps the target near
1000 elements per rank on the largest meshes.

The default preparation rule `--cfl-degree-rule linear --cfl 0.1` uses
`C*h_min/(p_H*c)`. The former quadratic factor `(p_H+1)^2` remains available
with `--cfl-degree-rule quadratic`.
The selected rule is recorded in each mixed case and honored by the runner's
geometry check. Cases without the key use the linear rule.
Use a new output directory to regenerate cases; copying source code alone
does not change existing TOML step counts. The preparation wrapper forwards
these command-line flags. The campaign's h_min is the minimum **inradius**;
the runtime check uses minimum altitude. The hp time-ratio constraint and
integer rounding can further reduce the actual dt below the estimate.

For p_H=5 the linear rule's nominal timestep is 7.2 times the quadratic
compatibility estimate at fixed geometry and C. The geometry check enforces
the selected rule; temporal-sensitivity results remain the evidence for
separating spatial and temporal error. No stability or convergence claim
follows from the error-kernel regression test.

`julia --project=. test/test_mixed_hp_error_kernel.jl` compares cached errors
with direct physical integration for both analytical solutions at several
times, using noncontiguous owned elements, and checks the CFL rule ratio.

`python3 test/test_mixed_hp_workflow.py` checks mixed DOFs, the timestep bound
and ratios, manifest separation, submission previews, initial deduplication,
and preservation of equal-order defaults using synthetic mesh metadata.
Shell syntax and Python compilation are checked separately. Small fixed-mesh
MPI smoke checks exercise the actual mesh loader, solver and output; they
do not establish convergence rates or MN5 scaling.

Validation on 2026-10-16 passed four Python contract tests, real one-level
Gmsh preparation for both mesh families, and these fixed-mesh MPI smoke paths:
structured periodic/centered/equispaced and PEC/alternating/warp-blend at two
ranks, unstructured periodic/alternating/warp-blend at four ranks, and
structured periodic initial-only output at one rank. Evolution smoke checks
used p_E=4, p_H=5, ESPRK6 and three steps over 0.000001 periods. Collection,
initial-result reuse and spatial table generation also passed. Temporal table
formatting was checked with synthetic data; temporal refinement was not run.
The existing high-degree basis orthonormality warning (~1.4e-9) remains visible.
No MN5 jobs were submitted by local validation.
