#!/usr/bin/env julia

# Run from the repository root, for example:
#   mpiexec -n 2 julia --project=. examples/distributed_maxwell.jl

using MPI
using DiscoGMPI

include(joinpath(@__DIR__, "solver", "stationary_pec_cube.jl"))

const MPI_WAS_INITIALIZED = MPI.Initialized()
MPI_WAS_INITIALIZED || MPI.Init()

const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NRANKS = MPI.Comm_size(COMM)

root_mesh = RANK == 0 ? build_pec_cube_mesh(2) : nothing
root_partition = if RANK == 0
    mod.(0:(size(root_mesh.tets, 2) - 1), NRANKS)
else
    nothing
end

distributed_dg = build_distributed_dg_from_root(
    root_mesh,
    root_partition,
    1;
    comm = COMM,
)

U = stationary_maxwell_field(
    distributed_dg.dg.mesh,
    distributed_dg.dg.ref,
)

registry = MaxwellBoundaryRegistry(
    Dict(PEC_BOUNDARY_ID => DiscoGMPI.MaxwellBC_PEC),
)
formulation = HesthavenWarburtonFormulation(MaxwellFlux_Upwind)

println(
    "rank ", RANK,
    ": owned = ", length(distributed_dg.distributed_mesh.partition.owned),
    ", ghosts = ", length(distributed_dg.distributed_mesh.partition.ghosts),
    ", neighbors = ", distributed_dg.exchange.neighbors,
)

run_distributed_maxwell_time_steps!(
    U,
    distributed_dg,
    registry,
    formulation;
    rk_order = 3,
    dt = 1e-4,
    nsteps = 2,
)

MPI.Barrier(COMM)

if !MPI_WAS_INITIALIZED && !MPI.Finalized()
    MPI.Finalize()
end
