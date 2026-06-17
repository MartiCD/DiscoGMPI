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
  --pml-degree=2 --rk-order=4
```

PML runs use explicit RK instead of ESPRK. The undamped analytical cavity
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

Run the distributed validation matrix to collect these checks across one rank
and multiple ranks:

```bash
julia --project=. examples/validate_distributed_maxwell_matrix.jl \
  --profile=standard \
  --cases=cavity-pec,cavity-pmc,periodic \
  --ranks=1,2
```

The validation matrix reuses the public convergence and production drivers.
It writes `validation_matrix.csv` under `output/validation_matrix/`, compares
rank-1 CSV diagnostics against each multi-rank run within configurable
tolerances, and records explicit expected-rate checks. For cavity PEC/PMC
cases the aggregate targets are `E=N+1` and `H=N`; for the periodic plane wave
the matrix additionally checks the active components `Ez=N+1` and `Hy=N`.
Use `--profile=smoke` for quick rank-equivalence checks and `--profile=strict`
for more expensive certification runs.

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
Abarbanel, Gottlieb, and Hesthaven (2006). In the current dimensionless
vacuum implementation,

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

Build nodal damping profiles and advance one distributed explicit RK step
with:

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

run_distributed_maxwell_nonlinear_pml_time_steps!(
    U,
    distributed_dg,
    registry,
    PoissonBracketFormulation(),
    pml;
    rk_order = 4,
    dt = dt,
    nsteps = 100,
    energy_every = 10,
)
```

Use `distributed_periodic_maxwell_nonlinear_pml_rk_step!` when some boundary
pairs are periodic. The PML source is evaluated only on owned elements; it
adds no auxiliary fields and requires no additional MPI traces. Use an
explicit RK method rather than ESPRK because damping destroys the conservative
Poisson-bracket structure. Both PEC and PMC outer boundaries are supported.
Heterogeneous material coupling is not yet implemented for this PML path.

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
