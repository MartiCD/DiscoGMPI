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
- Partition visualization utilities.

## Layout

- `src/solver/`: serial/threaded solver kernels migrated from DiscoG3D.
- `src/DistributedMesh3D.jl`: distributed mesh and halo topology.
- `src/TraceMaps.jl`: MPI trace communication and face permutations.
- `src/DistributedDG.jl`: rank-local DG construction, Maxwell halo exchange,
  owned-only RHS updates, global energy reduction, and distributed time stepping.
- `src/MetisIO.jl`: METIS partition input/output helpers.
- `examples/solver/`: migrated DiscoG3D examples using `DiscoGMPI`.
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
```

## Distributed Maxwell example

```bash
mpiexec -n 2 julia --project=. examples/distributed_maxwell.jl
```

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
