# DiscoGMPI

DiscoGMPI is the distributed-memory version of DiscoG3D. It is a separate
Julia package and Git repository, so development here does not overwrite or
remove the original DiscoG3D codebase.

The package currently combines:

- The serial and threaded DG/Maxwell solver inherited from DiscoG3D.
- Tetrahedral mesh partitioning and owned/ghost element bookkeeping.
- MPI interface-face discovery and trace exchange.
- Face-node permutation handling across MPI partitions.
- Owned-element Maxwell RHS assembly and distributed RK time integration.
- Rank-local distributed mesh files for scalable startup.
- Atomic per-rank Maxwell checkpoints and restart validation.
- Parallel VTK time-series output and run/partition metadata.
- Piecewise-constant spatial permittivity and permeability from mesh material IDs.
- PEC, PMC, first-order absorbing, and distributed periodic boundaries.
- Six-field nonlinear PML damping without auxiliary variables.
- Distributed energy, charge, linear-momentum, and angular-momentum diagnostics.
- Partition visualization utilities.

## Layout

- `src/solver/`: serial/threaded solver kernels migrated from DiscoG3D.
- `src/DistributedMesh3D.jl`: distributed mesh and halo topology.
- `src/TraceMaps.jl`: MPI trace communication and face permutations.
- `src/DistributedDG.jl`: rank-local DG construction, Maxwell halo exchange,
  owned-only RHS updates, global energy reduction, and distributed time stepping.
- `src/DistributedIO.jl`: rank-local mesh persistence, checkpoint/restart, and
  run/partition metadata.
- `src/NonlinearPML.jl`: Abarbanel--Gottlieb--Hesthaven nonlinear PML
  profiles, source terms, and serial/distributed RK wrappers.
- `src/MetisIO.jl`: METIS partition input/output helpers.
- `examples/solver/`: migrated DiscoG3D examples using `DiscoGMPI`.
- `examples/meshes/`: sample VTK meshes, METIS partition files, and
  parameterized Gmsh geometries for structured/unstructured studies.
- `benchmark/solver/`: migrated serial/threaded benchmarks.
- `test/test_solver.jl`: inherited solver regression tests.
- `test/test_mpi_*.jl`: tests intended to run under `mpiexec`.

## Serial and threaded tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## MPI tests

```bash
mpiexec -n 2 julia --project=. test/test_mpi_face_permutation.jl
mpiexec -n 2 julia --project=. test/test_mpi_distributed_maxwell.jl
mpiexec -n 4 julia --project=. test/test_mpi_distributed_maxwell.jl
mpiexec -n 2 julia --project=. test/test_mpi_physical_coverage.jl
mpiexec -n 2 julia --project=. test/test_mpi_nonlinear_pml.jl
```

## Distributed Maxwell example

```bash
mpiexec -n 2 julia --project=. examples/distributed_maxwell.jl
```

The complete Poisson-bracket experiment includes convergence diagnostics,
checkpointing, rank-local mesh loading, and ParaView time-series output:

```bash
mpiexec -n 2 julia --project=. \
  examples/distributed_poisson_bracket_maxwell.jl \
  --final-time=0.25 \
  --paraview-every=10 \
  --checkpoint-every=50
```

The output directory contains the same production artifact family as the
periodic driver: `energy.csv`, `quadrature_diagnostics.csv`,
`run_metadata.toml`, `partition_metadata.csv`, `paraview_series.csv`,
`fields.pvd`, parallel VTU field pieces, final `integration_points_rankNNNN.csv`
files, and restart checkpoints. The quadrature diagnostics include field and
energy errors, electric and magnetic charge, linear and angular momentum, and
the optical chirality
`0.5 * (epsilon * E dot curl(E) + mu * H dot curl(H))`. ParaView output uses
high-order VTK Lagrange tetrahedra and includes numerical, analytical, and
error vectors: `ElectricField`, `ExactElectricField`, `ElectricFieldError`,
and their magnetic counterparts.

On the first run, rank zero reads the global mesh and partition and prepares
one owned-plus-halo mesh file per rank. Later runs load those files
independently:

```bash
mpiexec -n 2 julia --project=. \
  examples/distributed_poisson_bracket_maxwell.jl \
  --distributed-mesh-dir=output/distributed_mesh_cache/tet_mesh_ranks2
```

Use `--rebuild-distributed-mesh` after changing the source mesh or partition.

Restart from any completed checkpoint directory:

```bash
mpiexec -n 2 julia --project=. \
  examples/distributed_poisson_bracket_maxwell.jl \
  --restart=output/distributed_poisson_bracket/checkpoints/step00000100 \
  --final-time=0.5
```

The restart currently requires the same MPI rank count, DG order, and element
partition. `--final-time` is the absolute target time, not an additional
duration.

Select perfect-magnetic-conductor walls with:

```bash
mpiexec -n 2 julia --project=. \
  examples/distributed_poisson_bracket_maxwell.jl \
  --boundary-condition=pmc
```

The PMC validation uses the electromagnetic dual of the PEC cavity mode.
Enable a nonlinear PML on all six sides of the same Poisson-bracket
discretization with:

```bash
mpiexec -n 2 julia --project=. \
  examples/distributed_poisson_bracket_maxwell.jl \
  --boundary-condition=pmc \
  --pml-width=0.2 --pml-sigma-max=12 \
  --pml-degree=2 --esprk-order=4
```

The production driver keeps the H-first ESPRK update when PML is active and
passes the PML source through the same Poisson-bracket RHS closure. Scalar
`--epsilon` and `--mu` values are supported. The undamped analytical cavity
mode remains in the diagnostic files as a reference, but is no longer the
exact solution once damping is active.

Run the distributed convergence study with PEC boundaries, the default, or
select the electromagnetic-dual PMC cavity mode:

```bash
mpiexec -n 2 julia --project=. \
  examples/convergence_distributed_poisson_bracket_maxwell.jl \
  --boundary-condition=pmc \
  --cells=2,3,4,5 --orders=1,2,3 \
  --output=output/convergence_distributed_poisson_bracket_pmc.csv
```

The convergence CSV includes a `boundary_condition` column. The Python
post-processor also accepts older CSV files without that column and treats
them as PEC results.

The same cavity convergence driver can also generate an unstructured Gmsh
mesh family on the unit cube:

```bash
mpiexec -n 2 julia --project=. \
  examples/convergence_distributed_poisson_bracket_maxwell.jl \
  --mesh-family=unstructured \
  --nx-targets=2,4,8,16 \
  --orders=2,3,4 \
  --periods=1 \
  --output=output/convergence_distributed_poisson_bracket_unstructured.csv
```

Rank zero invokes Gmsh on `examples/meshes/unit_cube_unstructured.geo`. The
`NxTarget` values request target sizes `Lx/NxTarget`; rates are still computed
from the measured `characteristic_h=(volume / number_of_tetrahedra)^(1/3)`.
The CSV includes `mesh_family` so structured and unstructured cavity studies
can be post-processed from the same schema. Use `--mesh-dir=PATH` to keep or
inspect the generated VTK files.

Run the corresponding periodic plane-wave convergence study with:

```bash
mpiexec -n 2 julia --project=. \
  examples/convergence_distributed_periodic_poisson_bracket_maxwell.jl \
  --nx-targets=4,8,16 --orders=2,3,4 --periods=1 \
  --output=output/convergence_distributed_periodic_poisson_bracket.csv
```

Rank zero invokes Gmsh on
`examples/meshes/periodic_box_unstructured.geo` with the requested
`NxTarget` values. `target_h=Lx/NxTarget` is recorded as Gmsh-request
metadata, while observed convergence rates are computed with the measured
volume-based size
`characteristic_h=(volume / number_of_tetrahedra)^(1/3)`. This distinction is
important for independently generated unstructured Gmsh meshes: doubling
`NxTarget` requests a half-size mesh, but the actual tetrahedron count and
`characteristic_h` ratio may differ, especially on coarse levels. Regenerate
older periodic convergence CSV files before comparing rates, because older
outputs used the requested target size in the rate denominator.

The generated unstructured periodic tetrahedral meshes are partitioned across
the launched MPI ranks and use the production periodic face exchange. The CSV
includes aggregate electric, magnetic, and total L2 errors; component errors
and rates for `Ex`, `Ey`, `Ez`, `Hx`, `Hy`, and `Hz`; `h_min`, `h_max`, and
`characteristic_h`; and energy, charge, and sampled `Linf` diagnostics. For
the plane wave, the active exact components are `Ez` and `Hy`; `Ex`, `Ey`,
`Hx`, and `Hz` are exact-zero components and therefore measure transverse
polarization leakage. On general unstructured tetrahedral meshes, that leakage
can converge closer to order `N` and can reduce the aggregate vector rate even
when the active plane-wave components behave correctly. The finest refinement
pair is still checked against the strict aggregate targets `E=N+1` and `H=N`,
but the component and leakage diagnostics should be inspected when the strict
electric aggregate check fails.

### Strict periodic space-time isoresolution study

The strict isoresolution driver uses the fixed periodic domain
`[0,1] x [0,0.25] x [0,0.25]` and the existing `+x` plane wave with
`k=2*pi` and wavelength `lambda=1`. It runs the `P1`, `P2`, and `P4`
discretizations on four levels each. `P4M0` starts at four uniform mesh
intervals per wavelength; resolution doubles between consecutive mesh levels
and between `P4`, `P2`, and `P1` at a fixed level.

```bash
mpiexec -n 2 julia --project=. \
  examples/convergence_distributed_periodic_isoresolution.jl \
  --final-time=0.25 \
  --output=output/convergence_distributed_periodic_isoresolution.csv
```

The driver constructs six deterministic structured meshes shared by the
twelve `P/M` cases. It requires the independently measured Cartesian spacing
to equal the target spacing to roundoff, and it enforces the same factor-two
hierarchy for the time step. Inspect resource and time-step planning without
solving with `--dry-run`. A small MPI check is available with:

```bash
mpiexec -n 2 julia --project=. \
  examples/convergence_distributed_periodic_isoresolution.jl --smoke
```

The temporal base step is prescribed independently of the legacy CFL
estimator:

```text
dt0 = C * h(P4M0) / 4,  C = 0.8,  h(P4M0) = 0.25
dt0 = 0.05

       M0  M1  M2  M3       (divisor applied to dt0)
P4      1   2   4   8
P2      2   4   8  16
P1      4   8  16  32
```

The final time must be an integer multiple of `dt0`. The CSV records the old
CFL estimate and whether each requested step satisfies it, but that estimate
does not override this prescribed matrix. Exceeding it produces a warning and
may cause the separate numerical-convergence verdict to fail.

For every field component, the electric and magnetic vector fields, and the
combined field, the CSV reports the space-time errors
`L2(0,T; L2(Omega))` and `Linf(0,T; L2(Omega))`. Spatial L2 errors are
evaluated with the existing distributed cubature diagnostics at every time-step
endpoint. The temporal L2 norm uses the composite trapezoidal rule applied to
the squared spatial L2 error; the temporal Linf norm is the maximum sampled at
the time-step endpoints. Rates compare consecutive `M` levels at fixed `P`
using the exact measured Cartesian spacing ratio. Rates for analytically zero
components describe numerical polarization leakage and can be dominated by
roundoff once those errors become very small.

Each completed case is atomically saved under `<output-stem>_cases/`. Resume a
compatible interrupted run with `--resume`; use `--checkpoint-dir=PATH` to
choose another location. The driver atomically rewrites the partial CSV after
every fresh case. Structural isoresolution and numerical convergence are
reported separately, with the latter checking finite and monotonically
decreasing active-component errors and finest-pair `Ez`/`Hy` rates for both
temporal norms. The machine-readable summary is `<output-stem>_verdict.toml`.

Use the distributed validation matrix as the release correctness gate before
major changes:

```bash
julia --project=. examples/validate_distributed_maxwell_matrix.jl \
  --profile=standard
```

By default this runs PEC, PMC, periodic plane-wave, and PML smoke cases over
1, 2, and 4 MPI ranks. It reuses the public convergence and production drivers,
compares rank-1 diagnostics against multi-rank runs, checks expected convergence
rates where they are meaningful, enforces invariant tolerances for energy,
charge, momentum, angular momentum, and optical chirality, and launches the
standalone MPI regression tests from the same orchestrator. The detailed matrix
is `validation_matrix.csv`; the official machine-readable release summary is
`validation_summary.json` under `output/validation_matrix/`. Use
`--profile=smoke` for a shorter local gate and `--profile=strict` for the more
expensive certification run.

## Materials and boundary conditions

Spatially varying isotropic material properties are assigned per element from
the distributed mesh material ID:

```julia
materials = maxwell_element_materials(
    distributed_dg,
    Dict(
        1 => MaxwellMaterial(1.0, 1.0),
        2 => MaxwellMaterial(4.0, 2.0),
    ),
)

registry = MaxwellBoundaryRegistry(
    Dict(
        10 => MaxwellBC_PEC,
        20 => MaxwellBC_PMC,
        30 => MaxwellBC_Absorbing,
    ),
)

maxwell_rhs!(
    rhs,
    U,
    distributed_dg,
    registry,
    PoissonBracketFormulation(),
    materials,
)
```

## Nonlinear six-field PML

`src/NonlinearPML.jl` implements the primary nonlinear PML system from
Abarbanel, Gottlieb, and Hesthaven (2006). For homogeneous unit material
coefficients, the implemented source reduces to

```text
dE/dt = curl(H) + (sigma .* P_H) x H
dH/dt = -curl(E) + (sigma .* P_E) x E

P_H =  (E x H) / (a*|H|^2 + (1-a)*|E|^2 + regularization)
P_E = -(E x H) / (a*|E|^2 + (1-a)*|H|^2 + regularization).
```

The default is `a=0.5` with a positive denominator regularization of
`1e-12`. The magnetic-source sign follows Appendix A and the energy-decay
identity in the paper; those are inconsistent with the sign printed in its
vector equation (2.10).

Build nodal damping profiles and advance one distributed ESPRK step with:

```julia
pml = build_maxwell_nonlinear_pml(
    distributed_dg;
    sigma_x = (x, y, z) -> polynomial_pml_sigma(
        x,
        1.5,
        2.0;
        sigma_max = 12.0,
        degree = 2,
    ),
    regularization = 1e-12,
)

scheme = explicit_partitioned_symplectic_rk_scheme(4; first_partition = :H)
work = MaxwellPartitionedRKWorkspace(U, scheme)
distributed_maxwell_nonlinear_pml_partitioned_symplectic_rk_step!(
    U,
    work,
    scheme,
    dt,
    distributed_dg,
    registry,
    PoissonBracketFormulation(),
    pml;
    ε = 2.0,
    μ = 3.0,
)
```

Use `distributed_periodic_maxwell_nonlinear_pml_partitioned_symplectic_rk_step!`
when some boundary pairs are periodic. Explicit-RK PML helper functions remain
available for non-Hamiltonian experiments. The PML source is evaluated only on
owned elements; it adds no auxiliary fields and requires no additional MPI
traces. Both PEC and PMC outer boundaries are supported. Heterogeneous
piecewise-constant `MaxwellElementMaterials` are supported by the material-aware
PML RHS and timestep overloads.

Run the distributed Gaussian-pulse demonstration with:

```bash
mpiexec -n 2 julia --project=. \
  examples/distributed_nonlinear_pml_maxwell.jl \
  --nx=8 --order=2 --rk-order=4 \
  --final-time=1.3 --cfl=0.15 \
  --pml-width=0.5 --sigma-max=12
```

The box is periodic in `y` and `z`, has nonlinear PML layers at both `x`
ends, and writes sampled global energy to
`output/distributed_nonlinear_pml/energy.csv`.

The first release-gate validation for the nonlinear PML is the empty-domain
incident-wave reflection test:

```bash
mpiexec -n 2 julia --project=. \
  examples/validate_empty_domain_incident_pml.jl \
  --max-reflection-ratio=1e-2 \
  --max-final-energy-ratio=2.5e-1
```

By default this gate uses
`examples/meshes/periodic_box_structured_nx16_ny8_nz8.vtk`, order `p=2`,
ESPRK order `3`, `lambda0=0.5`, a right-going sine-modulated Gaussian pulse,
periodic `y/z` boundaries, and nonlinear PML layers of width `0.5` on the two
`x` ends. It writes
`output/validation_sequence/empty_domain_incident_pml/empty_domain_incident_pml_summary.csv`
and `.json`. The gate fails if the left-monitor reflected energy ratio exceeds
`--max-reflection-ratio`, if the final total energy ratio exceeds
`--max-final-energy-ratio`, or if the PML run grows energy beyond
`--max-energy-growth`. Any option accepted by the production PML sweep driver
can also be passed to this gate, for example `--mesh`, `--flux`,
`--pml-width`, or `--sigma-max`.

The next validation-sequence gate checks the incident-aware PEC condition on
the metallic sphere:

```bash
mpiexec -n 2 julia --project=. \
  examples/validate_pec_sphere_boundary_residual.jl \
  --max-final-relative-rms=1e-1
```

The scattering driver now writes
`diagnostics/pec_boundary_residual.csv`, containing the sphere-surface
diagnostic `||n x E_total||`, its RMS value, incident-amplitude-normalized RMS,
and pointwise maximum. The gate writes
`output/validation_sequence/pec_sphere_boundary_residual/pec_sphere_boundary_residual_summary.csv`
and `.json`, and exits nonzero if the configured residual threshold fails.
Use `--max-final-l2`, `--max-final-relative-max`, and
`--max-sampled-relative-rms` to enable stricter checks. The gate defaults to
`examples/meshes/metallic_sphere_scattering_validation.vtk`, a coarse
validation mesh with about `2.9e4` tetrahedra. The higher-resolution
`metallic_sphere_scattering.vtk` mesh has about `3.6e5` tetrahedra and is
intended for production scattering/RCS studies, not for the quick residual
gate.

The qualitative scattered-field gate prepares the ParaView datasets used to
inspect the scattered `E_x` and `H_y` fields:

```bash
mpiexec -n 2 julia --project=. \
  examples/validate_scattered_ex_hy_fields.jl
```

This gate uses the same coarse validation sphere mesh, runs a short scattering
case, writes `fields.pvd`, and verifies that the final PVTU snapshot contains
the vector arrays `E_scat` and `H_scat`. In ParaView, open the reported
`fields.pvd` file and display the components `E_scat_x` and `H_scat_y`. The
gate also writes `scattered_ex_hy_fields_summary.csv` and `.json` with the
final PVD/PVTU paths and array checks.

The next validation-sequence gate runs the coarse PEC-sphere RCS extraction
and compares the angular samples against the exact PEC Mie series:

The production workflow is documented in detail in
`docs/numerics/rcs_metallic_sphere.md`. The implementation is centered on
`examples/distributed_metallic_sphere_scattering.jl`, which solves for the
scattered field of a PEC sphere under a `+z` propagating, `x` polarized
incident plane wave. The physical field is reconstructed as
`E_total = E_scat + E_inc` and `H_total = H_scat + H_inc`; the metallic sphere
enforces `n x E_total = 0`. When `--enable-rcs` is active, the driver
integrates the PEC surface current `J_s = n x H_total`, performs a
time-windowed Fourier extraction at the incident frequency, reconstructs the
far field, and writes
`sigma(theta,phi) = 4*pi*|E_infinity_scat|^2/|E_inc|^2` to
`diagnostics/rcs.csv`.

```bash
mpiexec -n 2 julia --project=. \
  examples/validate_coarse_sphere_rcs.jl
```

By default it uses `metallic_sphere_scattering_validation.vtk`, order `p=1`,
RCS extraction over `37` polar angles at `phi=0`, and no ParaView output. It
writes the numerical RCS table to `diagnostics/rcs.csv`, the Mie comparison to
`diagnostics/coarse_mie_rcs_comparison.csv`, and machine-readable summaries to
`coarse_sphere_rcs_summary.csv` and `.json`. The default pass/fail check is a
coarse release-gate check: finite rows, at least one accumulated RCS sample,
positive window weight, and nonzero numerical RCS. Add
`--max-absolute-normalized-error`, `--max-rms-relative-error`, or
`--max-db-error` when a stricter quantitative RCS gate is appropriate for the
chosen mesh, final time, and transient window.

The main RCS controls are:

- `--rcs-start-time`: ignore samples before this time. For quantitative Mie
  comparison it should be after the incident-wave startup transient has left
  the near field.
- `--rcs-every`: time-step interval for Fourier/RCS samples. Values such as
  `5` or `10` are usually enough for long runs; `1` is useful for short smoke
  tests but adds overhead.
- `--rcs-theta-count`, `--rcs-theta-min-degrees`,
  `--rcs-theta-max-degrees`: polar scattering angle grid.
- `--rcs-phi-degrees`: comma-separated azimuth angles. The default `0`
  produces the standard E-plane comparison for this incident polarization.

For a lighter high-order smoke run, generate the EPW4 mesh and run the same
gate with `p=4`/ESPRK5:

```bash
gmsh examples/meshes/metallic_sphere_scattering.geo -3 -format vtk \
  -o examples/meshes/metallic_sphere_scattering_epw4.vtk \
  -setnumber ElementsPerWavelength 4 \
  -setnumber SphereMeshSizeFactor 1.0 \
  -setnumber PMLMeshSizeFactor 3.0 \
  -setnumber FarMeshSizeFactor 4.0 \
  -setnumber CurvatureSamples 8 \
  -setnumber PMLWidth 1.0 \
  -setnumber AirBuffer 1.0 \
  -setnumber OptimizeMesh 0

mpiexec -n 2 -genv OPENBLAS_NUM_THREADS 1 julia --project=. \
  examples/validate_coarse_sphere_rcs.jl \
  --mesh=examples/meshes/metallic_sphere_scattering_epw4.vtk \
  --order=4 --esprk-order=5 \
  --final-time=1e-6 --rcs-start-time=1e-6 \
  --rcs-every=1 --diagnostics-every=1 \
  --min-max-rcs=0 \
  --output-dir=output/validation_sequence/coarse_sphere_rcs_epw4_p4_smoke
```

The existing Mie postprocessor can be used directly on any RCS run directory:

```bash
python3 examples/postprocess_metallic_sphere_mie_rcs.py \
  --run-dir output/validation_sequence/coarse_sphere_rcs_epw4_p4_smoke
```

It writes `diagnostics/mie_rcs_comparison.csv`,
`tables/mie_rcs_comparison.tex`, and `plots/mie_rcs_comparison.pdf`. A
one-step run validates the high-order RCS/output/postprocessing path only; use
a longer final time and a transient-free `--rcs-start-time` for quantitative
Mie agreement.

Interpret the two-panel postprocessor figure as follows. The top panel compares
`10 log10(sigma/(pi a^2))` against the exact Mie curve. The bottom panel shows
`|sigma_h-sigma_Mie|/|sigma_Mie|`. Large errors in a one-step or startup-window
run are expected and do not by themselves indicate a broken RCS implementation.
A quantitative validation should show stable RCS curves under later
`--rcs-start-time`, smaller `--rcs-every`, mesh refinement, and increased
polynomial order.

For production Poisson-bracket/ESPRK PML validation, use the separate sweep
driver:

```bash
mpiexec -n 2 julia --project=. \
  examples/distributed_poisson_bracket_pml_validation.jl \
  --mesh=examples/meshes/periodic_box_structured_nx16_ny8_nz8.vtk \
  --order=2 --esprk-order=3 \
  --final-time=3.0 --cfl=0.15 \
  --central-wavelength=0.5 \
  --fluxes=centered,alternating \
  --pml-widths=0.25,0.5 \
  --sigma-maxes=8,12 --sigma-degrees=2,3 \
  --paraview-every=25
```

Each sweep case is isolated under
`output/poisson_bracket_pml_validation/<case>/`. The case directory contains
`diagnostics/energy.csv`, `diagnostics/reflection_diagnostics.csv`,
`fields.pvd` with parallel VTU time-series data, resolved run configuration,
the input manifest, run metadata, and partition metadata. The sweep root also
writes `sweep_summary.csv` with the final energy ratio and measured reflection
ratio. Alternating Poisson-bracket fluxes currently require PEC outer
boundaries; centered fluxes can be run with PEC, PMC, or first-order absorbing
outer boundaries. Use `--meshes=mesh1.vtk,mesh2.vtk` to sweep over multiple
axis-aligned periodic cuboid meshes. The default carrier is a sine-modulated
Gaussian with `lambda0=0.5`; for the `nx16` mesh this gives four x-elements
per central wavelength and `f0=2` when `epsilon=mu=1`.

Periodic boundary IDs are paired geometrically across all ranks. Pass
`materials` while building the exchange when periodic partners may have
different material properties:

```julia
periodic = build_distributed_periodic_maxwell_exchange(
    distributed_dg,
    default_unit_box_periodic_specs();
    materials = materials,
)
```

Run the traveling- and standing-wave invariant validation with:

```bash
mpiexec -n 2 julia --project=. \
  examples/distributed_periodic_maxwell_validation.jl \
  --cells=2 --order=2 --steps=20 --dt=0.002
```

The result
`output/distributed_periodic_validation/periodic_analytical_validation.csv`
contains field and energy L2 errors, energy drift, electric and magnetic
charge, and linear and angular momentum together with analytical targets.

## Periodic Poisson-bracket production driver

The periodic traveling-wave driver defaults to
`examples/meshes/periodic_box_2x1x1_nx4.vtk`. It accepts a conforming
tetrahedral mesh of an axis-aligned box whose opposite surface triangulations
match under translation. The lower and upper coordinate bounds are inferred
from the mesh, boundary tags `1` through `6` are reconstructed from the six
box planes, and the periodic translations are derived from the three inferred
lengths. Coordinate bounds therefore do not need to be passed on the command
line.

The driver applies the periodic pairs `1<->2`, `3<->4`, and `5<->6`, and by
default advances

```text
E = (0, 0, -sin(2*pi*x - 2*pi*t))
H = (0, sin(2*pi*x - 2*pi*t), 0).
```

For a box whose x minimum is not zero, `x` in this expression is replaced by
`x - xmin`. The default wave number is `2*pi`; select another positive value
with `--wave-number=K`. The driver verifies that `K*Lx/(2*pi)` is an integer,
so the analytical field is periodic over the inferred x extent. For example,
`--wave-number=3.141592653589793` gives one wavelength on an x extent of
length 2.

The supplied mesh generator accepts the number of structured cells in the
x direction. Each x cell is split into six tetrahedra:

```bash
julia --project=. \
  examples/meshes/generate_periodic_box_2x1x1.jl \
  --nx=8
```

This writes `examples/meshes/periodic_box_2x1x1_nx8.vtk`, containing 48
tetrahedra. Use `--output=PATH` to select another VTK path. The default
remains `--nx=4`, which reproduces the supplied 24-tetrahedron mesh.

Launch one MPI rank per requested partition:

```bash
mpiexec -n 2 julia --project=. \
  examples/distributed_periodic_poisson_bracket_maxwell.jl \
  --partitions=2
```

The value passed to `--partitions` must equal the rank count passed to
`mpiexec`. `--partitions=N` is the partition count, whereas
`--partition=PATH` selects an existing METIS partition file. Therefore,
`--partition=2` is not equivalent to `--partitions=2`; it attempts to read a
file named `2`.

Run the generated `nx8` mesh on two ranks with:

```bash
mpiexec -n 2 julia --project=. \
  examples/distributed_periodic_poisson_bracket_maxwell.jl \
  --partitions=2 \
  --mesh=examples/meshes/periodic_box_2x1x1_nx8.vtk \
  --repartition \
  --rebuild-distributed-mesh \
  --order=2 \
  --esprk-order=3 \
  --final-time=10.0 \
  --cfl=0.5
```

The automatically generated partition and distributed-mesh cache include the
mesh basename, so `nx4` and `nx8` data are stored separately. Use
`--repartition` and `--rebuild-distributed-mesh` when replacing a mesh while
retaining the same filename.

For example, a mesh with bounds
`[0,1] x [0,0.25] x [0,0.25]` works without a domain option:

```bash
mpiexec -n 2 julia --project=. \
  examples/distributed_periodic_poisson_bracket_maxwell.jl \
  --mesh=examples/meshes/periodic_box_unstructured.vtk \
  --partitions=2 \
  --repartition \
  --rebuild-distributed-mesh \
  --order=4 \
  --esprk-order=5 \
  --cfl=0.8 \
  --final-time=1.0
```

For example, a four-way run is:

```bash
mpiexec -n 4 julia --project=. \
  examples/distributed_periodic_poisson_bracket_maxwell.jl \
  --partitions=4 --repartition
```

When `--partition` is omitted, rank zero converts the VTK tetrahedra using
`write_metis_mesh_from_vtk`, runs `mpmetis`, and stores the partition under
`output/partitions/`. Use `--repartition` to regenerate it. The driver then
uses DiscoGMPI's rank-local distributed mesh cache.

An existing partition can be selected explicitly:

```bash
mpiexec -n 2 julia --project=. \
  examples/distributed_periodic_poisson_bracket_maxwell.jl \
  --partitions=2 \
  --partition=output/partitions/periodic_box_2x1x1_nx4.mesh.epart.2
```

DG order 2 or higher is required by this driver. For the default `nx4` mesh,
order-1 DG nodes all lie at zeros of the prescribed sine wave.

The output directory
`output/distributed_periodic_poisson_bracket/` contains energy and quadrature
diagnostics, including the distributed optical chirality
`0.5 * (epsilon * E dot curl(E) + mu * H dot curl(H))`, ParaView time-series
files, rank metadata, integration-point CSV files, and restart checkpoints.
For this linearly polarized analytical wave, the exact optical chirality is
zero.

Resume from a checkpoint while retaining the same MPI rank count and
partition:

```bash
mpiexec -n 2 julia --project=. \
  examples/distributed_periodic_poisson_bracket_maxwell.jl \
  --partitions=2 \
  --restart=output/distributed_periodic_poisson_bracket/checkpoints/step00000100 \
  --final-time=0.5
```

Use an MPI launcher compatible with the library reported by
`julia --project=. -e 'using MPI; MPI.versioninfo()'`. The MPI.jl helper
`mpiexecjl` can be installed or used when the system `mpiexec` targets a
different MPI implementation.

Important production outputs are:

- `run_metadata.toml`: configuration, timing, mesh and solver parameters.
- `partition_metadata.csv`: owned/ghost counts, neighbors and interface faces.
- `fields.pvd`: ParaView time-series collection.

The Poisson-bracket production drivers write each DG element as a high-order
VTK Lagrange tetrahedron rather than reducing it to four corner values. The
numerical, exact, and error vectors are available as `ElectricField`,
`ExactElectricField`, `ElectricFieldError`, and their magnetic counterparts.
- `checkpoints/stepNNNNNNNN/`: checksummed per-rank restart files.

For scalable loading, keep the global mesh and partition vector only on rank 0:

```julia
root_mesh = rank == 0 ? read_vtu_mesh("mesh.vtu") : nothing
root_partition = rank == 0 ? read_metis_epart("mesh.epart.4") : nothing

distributed_dg = build_distributed_dg_from_root(
    root_mesh,
    root_partition,
    order;
    comm = MPI.COMM_WORLD,
)
```

The file-based overload also reads both files only on rank 0:

```julia
distributed_dg = build_distributed_dg_from_root(
    "mesh.vtu",
    "mesh.epart.4",
    order;
    comm = MPI.COMM_WORLD,
)
```

Each rank receives only its owned elements, one face halo, required
coordinates, material and boundary metadata, and MPI interface tables. The
older `build_distributed_dg` constructor remains available when the global mesh
is already replicated. Before every Maxwell RHS evaluation, the six field
traces are exchanged with nonblocking MPI communication. Time integration
updates only owned elements; ghost values are refreshed before each stage or
substep.

For repeated production runs, persist and reload the rank-local mesh directly:

```julia
prepare_distributed_mesh_partition(
    root_mesh,
    root_partition,
    "mesh_cache";
    comm = MPI.COMM_WORLD,
)

distributed_dg = build_distributed_dg_from_partition(
    "mesh_cache",
    order;
    comm = MPI.COMM_WORLD,
)
```

Checkpoint APIs operate on owned degrees of freedom and refresh ghost traces
after loading:

```julia
write_distributed_checkpoint(
    "checkpoints/step00000100",
    U,
    distributed_dg;
    step = 100,
    time = 0.1,
    dt = 1e-3,
)

U, state = load_distributed_checkpoint(
    "checkpoints/step00000100",
    distributed_dg,
)
```
