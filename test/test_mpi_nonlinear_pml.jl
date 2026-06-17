using Test
using MPI
using DiscoGMPI

# Run with:
#   mpiexec -n 2 julia --project=. test/test_mpi_nonlinear_pml.jl

const MPI_WAS_INITIALIZED = MPI.Initialized()
MPI_WAS_INITIALIZED || MPI.Init()

const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NPROCS = MPI.Comm_size(COMM)

function two_tet_mesh()
    points = [
        0.0 1.0 0.0 0.0 1.0
        0.0 0.0 1.0 0.0 1.0
        0.0 0.0 0.0 1.0 1.0
    ]
    tets = [
        1 2
        2 3
        3 4
        4 5
    ]
    tris = [
        1 1 1 3 2 2
        4 2 3 4 5 3
        3 4 2 5 4 5
    ]
    return RawVTUMesh(
        points,
        tets,
        tris,
        [1, 2],
        collect(3:8),
        Dict{String, Any}(
            "boundary_id" => [0, 0, 10, 10, 10, 10, 10, 10],
        ),
    )
end

try
    if NPROCS == 2
        root_mesh = RANK == 0 ? two_tet_mesh() : nothing
        root_partition = RANK == 0 ? [0, 1] : nothing
        distributed_dg = build_distributed_dg_from_root(
            root_mesh,
            root_partition,
            2;
            comm = COMM,
        )
        amplitude = 2.0
        U = interpolate_maxwell_field(
            distributed_dg.dg.mesh,
            distributed_dg.dg.ref,
            (x, y, z) -> (0.0, 0.0, amplitude),
            (x, y, z) -> (0.0, -amplitude, 0.0),
        )
        pml = build_maxwell_nonlinear_pml(
            distributed_dg;
            sigma_x = (x, y, z) -> 3.0,
        )
        base_rhs = DiscoGMPI.similar_maxwell_rhs(U)
        rhs = DiscoGMPI.similar_maxwell_rhs(U)
        registry = empty_maxwell_boundary_registry()
        formulation = PoissonBracketFormulation()

        maxwell_rhs!(
            base_rhs,
            U,
            distributed_dg,
            registry,
            formulation,
        )
        maxwell_nonlinear_pml_rhs!(
            rhs,
            U,
            distributed_dg,
            registry,
            formulation,
            pml,
        )

        damping =
            3.0 * amplitude^2 /
            (amplitude^2 + pml.regularization)
        local_error = 0.0
        for elem in distributed_dg.distributed_mesh.partition.owned
            local_error = max(
                local_error,
                maximum(
                    abs.(
                        (@view rhs.rhsEz[:, elem]) .-
                        (@view base_rhs.rhsEz[:, elem]) .+
                        damping * amplitude
                    ),
                ),
                maximum(
                    abs.(
                        (@view rhs.rhsHy[:, elem]) .-
                        (@view base_rhs.rhsHy[:, elem]) .-
                        damping * amplitude
                    ),
                ),
                maximum(
                    abs.(
                        (@view rhs.rhsEx[:, elem]) .-
                        (@view base_rhs.rhsEx[:, elem])
                    ),
                ),
                maximum(
                    abs.(
                        (@view rhs.rhsEy[:, elem]) .-
                        (@view base_rhs.rhsEy[:, elem])
                    ),
                ),
                maximum(
                    abs.(
                        (@view rhs.rhsHx[:, elem]) .-
                        (@view base_rhs.rhsHx[:, elem])
                    ),
                ),
                maximum(
                    abs.(
                        (@view rhs.rhsHz[:, elem]) .-
                        (@view base_rhs.rhsHz[:, elem])
                    ),
                ),
            )
        end
        ghost_error = 0.0
        for elem in distributed_dg.distributed_mesh.partition.ghosts
            ghost_error = max(
                ghost_error,
                maximum(abs.(@view rhs.rhsEx[:, elem])),
                maximum(abs.(@view rhs.rhsEy[:, elem])),
                maximum(abs.(@view rhs.rhsEz[:, elem])),
                maximum(abs.(@view rhs.rhsHx[:, elem])),
                maximum(abs.(@view rhs.rhsHy[:, elem])),
                maximum(abs.(@view rhs.rhsHz[:, elem])),
            )
        end

        @test MPI.Allreduce(local_error, max, COMM) < 1e-12
        @test MPI.Allreduce(ghost_error, max, COMM) == 0.0

        scheme = explicit_rk_scheme(4)
        global_dg = DGDiscretization(two_tet_mesh(), 2)
        serial_U = interpolate_maxwell_field(
            global_dg.mesh,
            global_dg.ref,
            (x, y, z) -> (0.0, 0.0, amplitude),
            (x, y, z) -> (0.0, -amplitude, 0.0),
        )
        serial_pml = build_maxwell_nonlinear_pml(
            global_dg;
            sigma_x = (x, y, z) -> 3.0,
        )
        serial_work = MaxwellRKWorkspace(serial_U, scheme)
        maxwell_nonlinear_pml_rk_step!(
            serial_U,
            serial_work,
            scheme,
            0.01,
            global_dg,
            registry,
            formulation,
            serial_pml,
        )

        work = MaxwellRKWorkspace(U, scheme)
        distributed_maxwell_nonlinear_pml_rk_step!(
            U,
            work,
            scheme,
            0.01,
            distributed_dg,
            registry,
            formulation,
            pml,
        )

        local_step_error = 0.0
        for elem in distributed_dg.distributed_mesh.partition.owned
            global_elem =
                distributed_dg.distributed_mesh.elements.global_ids[elem]
            for (local_component, global_component) in zip(
                (U.Ex, U.Ey, U.Ez, U.Hx, U.Hy, U.Hz),
                (
                    serial_U.Ex,
                    serial_U.Ey,
                    serial_U.Ez,
                    serial_U.Hx,
                    serial_U.Hy,
                    serial_U.Hz,
                ),
            )
                local_step_error = max(
                    local_step_error,
                    maximum(
                        abs.(
                            (@view local_component[:, elem]) .-
                            (@view global_component[:, global_elem])
                        ),
                    ),
                )
            end
        end
        @test MPI.Allreduce(local_step_error, max, COMM) < 1e-11
    else
        @test_skip "requires exactly two MPI ranks"
    end
finally
    if !MPI_WAS_INITIALIZED && !MPI.Finalized()
        MPI.Finalize()
    end
end
