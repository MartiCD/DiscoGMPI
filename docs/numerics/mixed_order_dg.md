# Mixed-order Maxwell DG

The mixed API evolves three electric components in `P_p` and three magnetic
components in `P_(p+1)` on affine tetrahedra. It is selected explicitly through
`MixedDGDiscretization` or `build_distributed_mixed_dg_from_root`; existing
equal-order drivers remain available as regression baselines. Passing a mixed
field to an equal-order driver is unsupported. No implicit truncation or padding
is used.

## Spaces and operators

For each element, electric arrays have `binomial(p+3,3)` rows and magnetic arrays
have `binomial(p+4,3)` rows. `MixedReferenceTet` owns separate reference elements,
quadrature mass matrices, evaluations, and lifts. Its rectangular matrices are

```
S_EH[a][i,j] = integral(l_E[i] * derivative_a(l_H[j]))
S_HE[a][j,i] = integral(l_H[j] * derivative_a(l_E[i]))
```

They satisfy `S_EH[a] + S_HE[a]' = sum_faces n[a] F_E' W_face F_H`.
The volume Maxwell blocks use `S_EH` and its transpose with the curl component
signs. After material-weighted mass multiplication the complete source-free
operator is skew-adjoint. The opposite Maxwell block is not assembled from an
interpolation of E into the H space.

Mixed derivatives have total degree at most `2p`, mixed face products `2p+1`,
and the H mass integrand `2p+2`. Positive Gauss/Duffy rules integrate the volume
through `2p+2` and the faces through `2p+1`. The reference tetrahedron uses
vertices `(-1,-1,-1)`, `(1,-1,-1)`, `(-1,1,-1)`, `(-1,-1,1)`.

The implementation follows equation (21) of the draft. In particular, the E
surface term is the magnetic **jump correction**, while the H surface term
uses the electric trace. Treating the hatted notation in the earlier bracket
form as a conventional H numerical trace would produce a different method.

## Faces, boundaries, MPI, and threading

Each directed face has one physical quadrature grid, two local evaluation
matrices, two neighbor evaluation matrices, and separate electric/magnetic lifts.
There is no assumption that electric and magnetic nodal sets match. There is
also no assumption that a Duffy quadrature grid is invariant under a triangle
permutation: the neighbor polynomial is evaluated at the actual receiver points.

Centered and alternating fluxes use the existing orientation convention.
Unpaired physical boundaries are PEC. Periodic specifications identify paired
triangles by translated vertices. Invalid or duplicate matches are rejected.
For prescribed PEC data, `boundary_electric(tag,x,y,z,t)` supplies an electric
trace; only its tangential component enters the H equation. The default is zero.

The distributed builder accepts a global mesh and zero-based element partition
on root and `nothing` elsewhere. Root distributes mesh/face metadata and each
rank assembles its own operators. Root still holds global mesh topology during
setup, as in the existing root mesh loader. During evolution each rank retains
only owned elements and ordinary/periodic face halos. Root does not retain the
global solution. Every rank must own at least one element.

MPI messages contain six trace columns per directed receiving face, each with
the common face quadrature count. They are ordered deterministically by receiver
element and local face. Thus the message size is based on quadrature, not either
field's nodal count. Nonblocking neighbor exchange overlaps elements with no
remote faces. Ghost volume coefficients are not used for remote residuals.
There are no global reductions in an RHS call. Diagnostics and timestep selection
perform reductions outside the stage kernels.

`ThreadedBackend()` parallelizes owned element writes with per-thread workspaces;
MPI calls remain outside threaded regions. Source/boundary callbacks must be
thread-safe. RHS closures are workspace-owning and are not reentrant.

## Usage

```julia
using DiscoGMPI, MPI
MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)
nranks = MPI.Comm_size(MPI.COMM_WORLD)
mesh = rank == 0 ? structured_box_mesh(2,2,2;
    boundary_id=periodic_box_boundary_id) : nothing
parts = rank == 0 ? [mod(i-1,nranks) for i in axes(mesh.tets,2)] : nothing
dg = build_distributed_mixed_dg_from_root(mesh,parts,2;
    periodic=DiscoGMPI.box_periodic_specs(1.,1.,1.))
U = interpolate_maxwell_field(dg,
    (x,y,z)->(0.,sin(2pi*x),0.), (x,y,z)->(0.,0.,sin(2pi*x)))
rhs! = DiscoGMPI.make_maxwell_rhs_function(dg;flux_kind=MaxwellFlux_Central)
scheme = mixed_esprk_scheme(2) # p_E=2, p_H=3, ESPRK4, H first
work = MaxwellPartitionedRKWorkspace(U,scheme)
dt = mixed_cfl_dt(dg;CFL=0.1)
DiscoGMPI.partitioned_symplectic_rk_step!(U,work,scheme,dt,rhs!)
diagnostics = mixed_diagnostic_values(mixed_diagnostics(U,dg))
MPI.Finalize()
```

`project_maxwell_field` supplies independent quadrature L2 projection instead of
nodal interpolation. Potential initialization uses degree `p_E+1` for electric
potential and `p_H+1` for magnetic potential. Each initializer changes only its
target field and constructs the polynomial curl at that field's nodes.

## Time, sources, and CFL

`mixed_esprk_scheme(p)` selects the existing order `p+2` method with H first.
The convergence driver supports `p=1:4`; requests outside that range fail instead
of reducing the temporal order. The timestep estimate is

```
dt = CFL * min_elements(h_min_altitude / ((p_H+1)^2 * wave_speed))
```

This is a conservative heuristic, not a universal stability theorem. The small
matrix spectrum audit reports the semidiscrete and ESPRK amplification spectra.
Actual convergence studies should also halve the timestep.

`with_mixed_source(rhs!, dg, source)` evaluates both fields at common volume
quadrature points and projects electric/magnetic source values into their own
spaces. The callback signature is `(x,y,z,t,E,H,epsilon,mu)` and returns
`(electric=(...), magnetic=(...))` in field/time units. For prescribed current
use `electric=-J/epsilon`. `with_mixed_nonlinear_pml` evaluates the existing
nonlinear PML formula on that same quadrature grid. Nonpolynomial sources need
a quadrature sensitivity check; their projection is not claimed exact.

Use ordinary `mixed_rk_step!` with an `ExplicitRKScheme` for driven or dissipative
systems with general coupled sources. It supplies the correct RK stage times.
The source-free ESPRK shear argument does not generally apply to nonlinear PML.

## Diagnostics and storage

`mixed_diagnostics` returns additive quadrature sums; `mixed_diagnostic_values`
names the energy, independent E/H L2 errors, integrated D/B, broken divergence
charges, divergence L2 norms, momentum, angular momentum, and chirality.
Exact solution callbacks accept `(x,y,z,t)`. Broken divergence diagnostics are
not automatically conserved Gauss constraints. Energy conservation does not
establish a discrete de Rham property or eliminate spurious stationary modes.

Checkpoints have a separate mixed manifest, both degrees and nodal counts,
ownership and geometry/basis/material fingerprints, and SHA-256 piece checksums.
Restart requires the same mesh, basis, material coefficients, partition, and
rank count. Equal-order checkpoints are rejected. Restart fields contain owned
coefficients; subsequent residual evaluation refreshes remote traces.

VTK output evaluates both source fields on an equispaced `p_H` visualization
grid. This interpolation is only for output and never feeds back into the state.
Both degrees are stored as cell data. Integration-point CSV output likewise
records both degrees and the physical quadrature weights.

## Convergence and verification

Spatial and temporal convergence validation is deferred to future MN5 runs.
The opt-in temporal refinement test is `test/test_mixed_order_temporal.jl`;
it is deliberately not included in the default test suite. No production
convergence rate is asserted for either field.

### MN5 launchers

From the repository root, preview commands without running Julia or submitting:

```bash
bash examples/run_mixed_order.sh --dry-run
bash examples/submit_mn5_mixed_order.sh --ntasks=8
```

When ready on MN5, submit eight independent jobs (four degree pairs, two fluxes):

```bash
bash examples/submit_mn5_mixed_order.sh --submit --ntasks=8
```

Review the account, QoS, modules, wall time and MPI.jl library configuration in
`examples/run_mn5_mixed_order.slurm` first. These follow existing repository
launchers but have not been validated on MN5. Dependencies must already be
installed in the selected Julia depot. The script prints `MPI.versioninfo()`
for checking compatibility with the loaded Intel MPI launcher.
Override resources with sbatch options; `--cpus-per-task` also controls Julia
threads and selects the threaded backend when greater than one.

Campaign defaults can be overridden with `MIXED_OUTPUT_ROOT`, `MIXED_LEVELS`
(default `iso:1,2,3,4`), `MIXED_FINAL_TIME`, `MIXED_CFL`, and
`MIXED_INITIALIZATION` (`interpolate` or `project`). Outputs use separate
degree/flux filenames under a timestamped directory. Existing CSVs are refused;
each CSV has a TOML provenance sidecar. Use a distinct output root for a
half-CFL repeat. The submission wrapper creates the Slurm log directory.

### Study definition

The campaign supports either cubic h refinement or directional isoresolution.
For `iso:1,2,3`, it uses `n_x=12*2^(level-1)/p`, `n_y=n_z=2^(level-1)` on a
`1 × 1/4 × 1/4` periodic box. Thus `p_E*n_x` matches across p, but
`p_H*n_x` and actual DOF counts do not. Each p family is uniformly refined.
The CSV reports both resolutions and both actual DOF counts. This is a
structured affine-tetrahedral campaign; it does not replace an unstructured
mesh or a long-time cavity study.

Errors are explicitly labeled **final-time L2**, not the maximum-in-time norm
used by some older drivers. No slope is enforced. Run the same study with half
CFL and compare the two errors before assigning a spatial rate.

```bash
julia --project=. test/runtests.jl
julia --project=. --threads=2 test/test_mixed_order_solver.jl
mpiexec -n 1 julia --project=. test/test_mpi_mixed_order.jl
mpiexec -n 2 julia --project=. test/test_mpi_mixed_order.jl
mpiexec -n 4 julia --project=. test/test_mpi_mixed_order.jl
julia --project=. examples/audit_mixed_spectrum.jl
julia --project=. examples/benchmark_mixed_order.jl
mpiexec -n 2 julia --project=. examples/convergence_mixed_order.jl \
  /tmp/mixed-isoresolution.csv 1,2,3,4 iso:1,2,3 0.05 0.1
python3 examples/postprocess_mixed_order.py /tmp/mixed-isoresolution.csv
```

The constructor and test API intentionally keep the old equal-order solver
separate. Old experiment manifests, RCS drivers, and equal-order diagnostic
workspaces are not silently relabeled as mixed. Use the explicit mixed API and
the new campaign driver when selecting this discretization.

## Local validation record (2026-10-16)

After the final launcher/integration changes, these non-convergence checks passed:

```bash
julia --project=. --threads=2 -e 'using LinearAlgebra; BLAS.set_num_threads(1); include("test/test_mixed_order_reference.jl"); include("test/test_mixed_order_solver.jl")'
mpiexec -n 1 julia --project=. test/test_mpi_mixed_order.jl
mpiexec -n 2 julia --project=. test/test_mpi_mixed_order.jl
mpiexec -n 4 julia --project=. test/test_mpi_mixed_order.jl
bash -n examples/run_mixed_order.sh examples/run_mn5_mixed_order.slurm examples/submit_mn5_mixed_order.sh
bash examples/run_mixed_order.sh --dry-run
bash examples/submit_mn5_mixed_order.sh --ntasks=8
julia --project=. -e 'include("examples/convergence_mixed_order.jl"); println("Campaign module loaded; no study executed")'
```

Reference tests: 1,974 assertions. Serial/threaded tests: 305 assertions.
MPI tests: 317 assertions at one rank, 221 per rank at two ranks, and 129
per rank at four ranks. MPI checks include residual equivalence, checkpoint
continuation, incompatible/corrupted checkpoint rejection, and output.
The existing modal basis builder warns of an orthonormality defect about
`1.4e-9` at high degree; mixed quadrature and Green-identity tests pass without
relaxing tolerances. This remains a conditioning limitation to monitor.

No spatial/temporal refinement study or Slurm submission was executed in this
validation pass. The complete legacy suite was not rerun in this pass because
convergence-related validation was explicitly deferred. MN5 module/MPI
compatibility, production scaling, CFL suitability, and separate E/H rates
remain to be established there. The CFL formula is an estimate, not a general
stability proof. No production-readiness claim follows from these unit tests.
