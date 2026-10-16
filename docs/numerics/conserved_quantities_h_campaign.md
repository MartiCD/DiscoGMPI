# Long-time conserved quantities under h refinement

The campaign uses the periodic [0,2] × [0,1] × [0,1] traveling-wave
driver, p=4, ESPRK5, equispaced nodes (the driver's default), interpolation
of both fields, final time 10000 and CFL 0.2. The driver chooses its step
from its mesh-dependent CFL estimator and adjusts it to reach final time.

The four target resolutions are (3,2,2), (6,3,3), (12,6,6), (24,12,12).
Structured cells are divided into tetrahedra. Unstructured meshes are
generated independently by the existing Gmsh periodic-box generator:
counts prescribe box-edge subdivisions, not the total tetrahedron count.
Both families validate opposite-face triangulation matching. The first
transition is not a uniform factor-two refinement in every direction.

Both centered and alternating fluxes run separately on every mesh, making
16 jobs. Preparation records actual tetrahedron counts and SHA256 hashes.
Rank counts scale as ceil(nelements/1000), capped at 448. These are tunable
resource defaults, not measured optimal performance or runtime guarantees.
Requested nodes are ceil(ranks/112). WALLTIME defaults to 36 hours; the
T=10000 runs need pilot timings to assess whether that allocation suffices.

## MN5

From the repository root:

```bash
mkdir -p output/slurm
sbatch examples/prepare_mn5_conserved_quantities_h.slurm
```

Preparation requires Julia with the project dependencies and the Gmsh
executable; it does not use the Gmsh Python bindings or install packages.
The Slurm module stack includes the GCC 13.2 runtime required by MN5 Gmsh.
Wait for successful preparation and manifest.json before submission.

```bash
# Preview all 16 jobs, then submit explicitly.
python3 examples/manage_conserved_quantities_h.py submit
python3 examples/manage_conserved_quantities_h.py submit --submit

# Or select a pilot (do not submit it twice).
python3 examples/manage_conserved_quantities_h.py submit \
  --families structured --fluxes centered --levels 1 --submit
```

Use --levels 2 3 4 to submit the remaining levels of that selection.
ACCOUNT, QOS and WALLTIME override submission defaults. --root selects
another campaign directory for every action. Preparation refuses to
overwrite an existing directory; choose a fresh root for a new campaign.
Do not submit the same pending selection twice: output directories may
not exist until jobs begin, and submission does not deduplicate queued jobs.

Lightweight diagnostics are written every step, full diagnostics every
10000 steps, terminal output every 1000, checkpoints every 10000, and
ParaView every 1000000000 steps, as in the long-time p workflow. Existing
driver behavior at initial/final times also applies. Environment variables
ENERGY_EVERY, FULL_DIAGNOSTICS_EVERY, PRINT_EVERY, CHECKPOINT_EVERY,
PARAVIEW_EVERY and FINAL_TIME permit shorter pilot runs. Case run metadata
records actual settings. Each job has a private mesh/partition cache.

## Local preparation and postprocessing

```bash
python3 examples/manage_conserved_quantities_h.py prepare
python3 examples/manage_conserved_quantities_h.py status
python3 examples/manage_conserved_quantities_h.py plot
```

Local preparation requires Julia/project dependencies and gmsh in PATH.
Status reports file presence only, not completion. Inspect the driver log
and final recorded time to verify completion before interpreting plots.

Results live in ROOT/FAMILY/FLUX/mLEVEL/diagnostics/invariant_diagnostics.csv.
Plots go to ROOT/FAMILY/FLUX/plots, with signed and log-y six-panel figures
in PNG, PDF and SVG. The existing p-study plot functions are reused:
energy, optical chirality, momentum norms and summed integrated D/B.
Charge labels retain the existing convention. Log-y figures use the same
error definitions as the p plots; no h-convergence rates are computed.
Labels show M1–M4 and target counts instead of assuming h halves exactly.
Plotting requires all selected levels' diagnostics; select a subset with
--levels for partial results. The lightweight driver retains vector
components in CSVs so momentum norms can be computed during plotting.
