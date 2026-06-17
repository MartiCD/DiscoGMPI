using Test
using MPI
using DiscoGMPI

include(
    joinpath(
        @__DIR__,
        "..",
        "examples",
        "convergence_distributed_periodic_isoresolution.jl",
    ),
)

initialized_here = !MPI.Initialized()
initialized_here && MPI.Init()

try
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    root_mesh = rank == 0 ? build_periodic_isoresolution_mesh(4) : nothing
    root_partition = rank == 0 ?
                     periodic_balanced_spatial_partition(root_mesh, nranks) :
                     nothing
    distributed_dg = build_distributed_dg_from_root(
        root_mesh,
        root_partition,
        1;
        comm = comm,
    )
    box = distributed_periodic_box(distributed_dg)
    wave = PlaneWaveParameters(2.0 * pi, 2.0 * pi, 1.0, box.lower[1])
    electric, magnetic = exact_periodic_wave_functions(
        0.0;
        epsilon = 1.0,
        mu = 1.0,
        wave = wave,
    )
    U = interpolate_maxwell_field(distributed_dg, electric, magnetic)
    cubature_order = 6

    uncached = distributed_periodic_component_l2_errors(
        U,
        distributed_dg,
        0.0,
        cubature_order;
        epsilon = 1.0,
        mu = 1.0,
        wave = wave,
    )
    workspace = DistributedMaxwellComponentL2Workspace(
        distributed_dg,
        cubature_order,
    )
    cached = distributed_periodic_component_l2_errors!(
        workspace,
        U,
        distributed_dg,
        0.0;
        epsilon = 1.0,
        mu = 1.0,
        wave = wave,
    )
    cached_again = distributed_periodic_component_l2_errors!(
        workspace,
        U,
        distributed_dg,
        0.0;
        epsilon = 1.0,
        mu = 1.0,
        wave = wave,
    )

    @test collect(cached) ≈ collect(uncached) rtol = 1e-14 atol = 1e-14
    @test cached_again == cached
finally
    initialized_here && !MPI.Finalized() && MPI.Finalize()
end
