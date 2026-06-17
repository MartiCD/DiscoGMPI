using Test
using MPI
using DiscoGMPI

# Run with:
#   mpiexec -n 2 julia --project=. test/test_mpi_distributed_maxwell.jl

const MPI_WAS_INITIALIZED = MPI.Initialized()
MPI_WAS_INITIALIZED || MPI.Init()

const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NPROCS = MPI.Comm_size(COMM)

function two_tet_pec_mesh()
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

    tet_cell_ids = [1, 2]
    tri_cell_ids = collect(3:8)
    boundary_id = [0, 0, 10, 10, 10, 10, 10, 10]

    return RawVTUMesh(
        points,
        tets,
        tris,
        tet_cell_ids,
        tri_cell_ids,
        Dict{String, Any}("boundary_id" => boundary_id),
    )
end

function chain_pec_mesh()
    points = [
        0.0 1.0 0.0 0.0 1.0 2.0 2.0
        0.0 0.0 1.0 0.0 1.0 1.0 0.0
        0.0 0.0 0.0 1.0 1.0 0.0 1.0
    ]

    tets = [
        1 2 3 4
        2 3 4 5
        3 4 5 6
        4 5 6 7
    ]'

    face_counts = Dict{NTuple{3, Int}, Int}()
    face_nodes = Dict{NTuple{3, Int}, NTuple{3, Int}}()

    for elem in axes(tets, 2)
        for local_face in DiscoGMPI.TET_FACES
            nodes = (
                tets[local_face[1], elem],
                tets[local_face[2], elem],
                tets[local_face[3], elem],
            )
            key = Tuple(sort(collect(nodes)))
            face_counts[key] = get(face_counts, key, 0) + 1
            face_nodes[key] = nodes
        end
    end

    boundary_faces = [
        face_nodes[key]
        for (key, count) in face_counts
        if count == 1
    ]
    tris = reduce(hcat, collect.(boundary_faces))
    ntets = size(tets, 2)
    ntris = size(tris, 2)
    tet_cell_ids = collect(1:ntets)
    tri_cell_ids = collect((ntets + 1):(ntets + ntris))
    boundary_id = zeros(Int, ntets + ntris)
    boundary_id[tri_cell_ids] .= 10

    return RawVTUMesh(
        points,
        tets,
        tris,
        tet_cell_ids,
        tri_cell_ids,
        Dict{String, Any}("boundary_id" => boundary_id),
    )
end

function deterministic_maxwell_field(ref::ReferenceTet, nelements::Int)
    components = [zeros(Float64, ref.Np, nelements) for _ in 1:6]

    for elem in 1:nelements
        for node in 1:ref.Np
            for component in 1:6
                components[component][node, elem] =
                    100.0 * component + 10.0 * elem + 0.125 * node
            end
        end
    end

    return MaxwellField(components...)
end

function copy_field(U::MaxwellField)
    return MaxwellField(
        copy(U.Ex),
        copy(U.Ey),
        copy(U.Ez),
        copy(U.Hx),
        copy(U.Hy),
        copy(U.Hz),
    )
end

function rhs_components(rhs::MaxwellRHS)
    return (
        rhs.rhsEx,
        rhs.rhsEy,
        rhs.rhsEz,
        rhs.rhsHx,
        rhs.rhsHy,
        rhs.rhsHz,
    )
end

function field_components(U::MaxwellField)
    return (U.Ex, U.Ey, U.Ez, U.Hx, U.Hy, U.Hz)
end

function poison_ghosts!(U::MaxwellField, distributed_dg)
    for elem in distributed_dg.distributed_mesh.partition.ghosts
        for component in field_components(U)
            fill!(@view(component[:, elem]), -1.0e9)
        end
    end
    return U
end

function owned_rhs_error(global_rhs, local_rhs, distributed_dg)
    error = 0.0

    for local_elem in distributed_dg.distributed_mesh.partition.owned
        global_elem =
            distributed_dg.distributed_mesh.elements.global_ids[local_elem]

        for (global_component, local_component) in
            zip(rhs_components(global_rhs), rhs_components(local_rhs))
            error = max(
                error,
                maximum(
                    abs.(
                        global_component[:, global_elem] .-
                        local_component[:, local_elem]
                    ),
                ),
            )
        end
    end

    return error
end

function owned_field_error(global_field, local_field, distributed_dg)
    error = 0.0

    for local_elem in distributed_dg.distributed_mesh.partition.owned
        global_elem =
            distributed_dg.distributed_mesh.elements.global_ids[local_elem]

        for (global_component, local_component) in
            zip(field_components(global_field), field_components(local_field))
            error = max(
                error,
                maximum(
                    abs.(
                        global_component[:, global_elem] .-
                        local_component[:, local_elem]
                    ),
                ),
            )
        end
    end

    return error
end

function local_owned_field_error(left, right, distributed_dg)
    error = 0.0
    owned = distributed_dg.distributed_mesh.partition.owned
    for (left_component, right_component) in
        zip(field_components(left), field_components(right))
        error = max(
            error,
            maximum(
                abs.(
                    left_component[:, owned] .-
                    right_component[:, owned]
                ),
            ),
        )
    end
    return error
end

function local_owned_rhs_difference(left, right, distributed_dg)
    error = 0.0
    owned = distributed_dg.distributed_mesh.partition.owned
    for (left_component, right_component) in
        zip(rhs_components(left), rhs_components(right))
        error = max(
            error,
            maximum(
                abs.(
                    left_component[:, owned] .-
                    right_component[:, owned]
                ),
            ),
        )
    end
    return error
end

function max_ghost_rhs(rhs, distributed_dg)
    value = 0.0

    for elem in distributed_dg.distributed_mesh.partition.ghosts
        for component in rhs_components(rhs)
            value = max(value, maximum(abs.(@view component[:, elem])))
        end
    end

    return value
end

function exchanged_ghost_trace_error(global_U, local_U, distributed_dg)
    error = 0.0

    for neighbor in distributed_dg.exchange.neighbors
        for face in distributed_dg.exchange.faces[neighbor]
            global_elem =
                distributed_dg.distributed_mesh.elements.global_ids[face.ghost_elem]

            for (global_component, local_component) in
                zip(field_components(global_U), field_components(local_U))
                error = max(
                    error,
                    maximum(
                        abs.(
                            global_component[face.ghost_nodes, global_elem] .-
                            local_component[face.ghost_nodes, face.ghost_elem]
                        ),
                    ),
                )
            end
        end
    end

    return error
end

function compare_rhs(
    global_U,
    global_dg,
    distributed_dg,
    registry,
    formulation,
)
    global_rhs = DiscoGMPI.similar_maxwell_rhs(global_U)
    maxwell_rhs!(
        global_rhs,
        global_U,
        global_dg,
        registry,
        formulation,
    )

    local_U = localize_maxwell_field(global_U, distributed_dg)
    poison_ghosts!(local_U, distributed_dg)
    local_rhs = DiscoGMPI.similar_maxwell_rhs(local_U)

    maxwell_rhs!(
        local_rhs,
        local_U,
        distributed_dg,
        registry,
        formulation,
    )

    local_trace_error = exchanged_ghost_trace_error(
        global_U,
        local_U,
        distributed_dg,
    )
    local_error = owned_rhs_error(
        global_rhs,
        local_rhs,
        distributed_dg,
    )
    global_error = MPI.Allreduce(local_error, max, COMM)
    ghost_rhs = MPI.Allreduce(
        max_ghost_rhs(local_rhs, distributed_dg),
        max,
        COMM,
    )
    trace_error = MPI.Allreduce(local_trace_error, max, COMM)

    return global_error, ghost_rhs, trace_error
end

@testset "Root mesh validation fails collectively" begin
    caught_validation_error = try
        build_distributed_dg_from_root(
            nothing,
            nothing,
            1;
            comm = COMM,
        )
        false
    catch error
        occursin(
            "root rank must provide global_mesh",
            sprint(showerror, error),
        )
    end

    @test caught_validation_error
end

if NPROCS == 2
    mesh = two_tet_pec_mesh()
    order = 2
    global_dg = DGDiscretization(mesh, order)
    root_mesh = RANK == 0 ? mesh : nothing
    root_partition = RANK == 0 ? [0, 1] : nothing
    distributed_dg = build_distributed_dg_from_root(
        root_mesh,
        root_partition,
        order;
        comm = COMM,
    )
    registry = MaxwellBoundaryRegistry(
        Dict(10 => DiscoGMPI.MaxwellBC_PEC),
    )
    global_U = deterministic_maxwell_field(global_dg.ref, 2)

    @testset "Distributed DG construction" begin
        @test length(distributed_dg.distributed_mesh.partition.owned) == 1
        @test length(distributed_dg.distributed_mesh.partition.ghosts) == 1
        @test distributed_dg.exchange.neighbors == [1 - RANK]
        @test length(distributed_dg.exchange.faces[1 - RANK]) == 1
    end

    @testset "Distributed mesh and checkpoint I/O" begin
        io_root = MPI.bcast(
            RANK == 0 ? mktempdir() : "",
            COMM;
            root = 0,
        )
        mesh_dir = joinpath(io_root, "mesh")
        checkpoint_dir = joinpath(io_root, "checkpoint")
        metadata_dir = joinpath(io_root, "metadata")

        prepared_mesh = prepare_distributed_mesh_partition(
            RANK == 0 ? mesh : nothing,
            RANK == 0 ? [0, 1] : nothing,
            mesh_dir;
            comm = COMM,
            metadata = Dict("test_case" => "two_tet_pec"),
        )
        cached_dg = build_distributed_dg_from_partition(
            mesh_dir,
            order;
            comm = COMM,
        )

        @test prepared_mesh.elements.global_ids ==
              cached_dg.distributed_mesh.elements.global_ids
        @test prepared_mesh.nodes.global_ids ==
              cached_dg.distributed_mesh.nodes.global_ids
        @test cached_dg.exchange.neighbors == [1 - RANK]

        checkpoint_U = localize_maxwell_field(global_U, cached_dg)
        write_distributed_checkpoint(
            checkpoint_dir,
            checkpoint_U,
            cached_dg;
            step = 7,
            time = 0.125,
            dt = 0.005,
            metadata = Dict("initial_energy_total" => 3.25),
        )
        restored_U, state =
            load_distributed_checkpoint(checkpoint_dir, cached_dg)
        restore_error = MPI.Allreduce(
            local_owned_field_error(
                checkpoint_U,
                restored_U,
                cached_dg,
            ),
            max,
            COMM,
        )

        @test restore_error == 0.0
        @test state.step == 7
        @test state.time == 0.125
        @test state.dt == 0.005
        @test state.metadata["initial_energy_total"] == 3.25

        write_distributed_run_metadata(
            metadata_dir,
            cached_dg;
            configuration = Dict("case" => "I/O regression"),
            runtime = Dict("status" => "complete"),
        )
        metadata_exists = MPI.bcast(
            RANK == 0 &&
            isfile(joinpath(metadata_dir, "run_metadata.toml")) &&
            isfile(joinpath(metadata_dir, "partition_metadata.csv")),
            COMM;
            root = 0,
        )
        @test metadata_exists
        metadata_header_has_quality = MPI.bcast(
            RANK == 0 &&
            all(
                field -> occursin(
                    field,
                    first(eachline(joinpath(metadata_dir, "partition_metadata.csv"))),
                ),
                (
                    "local_elements",
                    "ghost_nodes",
                    "ghost_dofs",
                    "interface_faces",
                    "interface_faces_per_owned_element",
                    "halo_elements_per_owned_element",
                    "halo_dofs_per_owned_element",
                    "send_values",
                    "recv_values",
                ),
            ),
            COMM;
            root = 0,
        )
        @test metadata_header_has_quality

        MPI.Barrier(COMM)
        RANK == 0 && rm(io_root; recursive = true, force = true)
        MPI.Barrier(COMM)
    end

    @testset "Distributed Maxwell RHS profiling" begin
        formulation = PoissonBracketFormulation()
        profiled_U = localize_maxwell_field(global_U, distributed_dg)
        reference_U = localize_maxwell_field(global_U, distributed_dg)
        poison_ghosts!(profiled_U, distributed_dg)
        poison_ghosts!(reference_U, distributed_dg)
        profiled_rhs = DiscoGMPI.similar_maxwell_rhs(profiled_U)
        reference_rhs = DiscoGMPI.similar_maxwell_rhs(reference_U)

        timing = profile_distributed_maxwell_rhs!(
            profiled_rhs,
            profiled_U,
            distributed_dg,
            registry,
            formulation;
            tag = 25017,
        )
        maxwell_rhs!(
            reference_rhs,
            reference_U,
            distributed_dg,
            registry,
            formulation,
        )

        rhs_difference = MPI.Allreduce(
            local_owned_rhs_difference(
                profiled_rhs,
                reference_rhs,
                distributed_dg,
            ),
            max,
            COMM,
        )
        @test rhs_difference <= 1e-10
        @test MPI.Allreduce(timing.halo_exchange_seconds, min, COMM) >= 0.0
        @test MPI.Allreduce(timing.rhs_assembly_seconds, min, COMM) >= 0.0
        @test MPI.Allreduce(timing.ghost_zero_seconds, min, COMM) >= 0.0
        @test MPI.Allreduce(timing.total_seconds, max, COMM) > 0.0
    end

    @testset "Distributed Maxwell RHS matches serial" begin
        formulations = (
            HesthavenWarburtonFormulation(MaxwellFlux_Central),
            HesthavenWarburtonFormulation(MaxwellFlux_Upwind),
            PoissonBracketFormulation(),
        )

        for formulation in formulations
            error, ghost_rhs, trace_error = compare_rhs(
                global_U,
                global_dg,
                distributed_dg,
                registry,
                formulation,
            )

            @test error <= 1e-10
            @test ghost_rhs == 0.0
            @test trace_error <= 1e-12
        end
    end

    @testset "Distributed PMC Maxwell RHS matches serial" begin
        pmc_registry = MaxwellBoundaryRegistry(
            Dict(10 => DiscoGMPI.MaxwellBC_PMC),
        )
        for formulation in (
            HesthavenWarburtonFormulation(MaxwellFlux_Central),
            PoissonBracketFormulation(),
        )
            error, ghost_rhs, trace_error = compare_rhs(
                global_U,
                global_dg,
                distributed_dg,
                pmc_registry,
                formulation,
            )

            @test error <= 1e-10
            @test ghost_rhs == 0.0
            @test trace_error <= 1e-12
        end
    end

    @testset "Distributed Maxwell energy matches serial" begin
        local_U = localize_maxwell_field(global_U, distributed_dg)
        serial_energy = maxwell_energy(
            global_U,
            global_dg.ref,
            global_dg.mappings,
        )
        distributed_energy = distributed_maxwell_energy(
            local_U,
            distributed_dg,
        )

        @test distributed_energy.electric ≈ serial_energy.electric atol = 1e-12
        @test distributed_energy.magnetic ≈ serial_energy.magnetic atol = 1e-12
        @test distributed_energy.total ≈ serial_energy.total atol = 1e-12
    end

    @testset "Distributed explicit RK step matches serial" begin
        scheme = explicit_rk_scheme(3)
        dt = 1e-4
        formulation =
            HesthavenWarburtonFormulation(MaxwellFlux_Upwind)

        serial_U = copy_field(global_U)
        serial_work = DiscoGMPI.MaxwellRKWorkspace(serial_U, scheme)
        serial_rhs! = DiscoGMPI.make_maxwell_rhs_function(
            global_dg,
            registry,
            formulation,
        )
        DiscoGMPI.rk_step!(
            serial_U,
            serial_work,
            scheme,
            dt,
            serial_rhs!,
        )

        local_U = localize_maxwell_field(global_U, distributed_dg)
        local_work = DiscoGMPI.MaxwellRKWorkspace(local_U, scheme)
        distributed_rk_step!(
            local_U,
            local_work,
            scheme,
            dt,
            distributed_dg,
            registry,
            formulation,
        )

        error = MPI.Allreduce(
            owned_field_error(serial_U, local_U, distributed_dg),
            max,
            COMM,
        )
        @test error <= 1e-10
    end

    @testset "Distributed partitioned RK step matches serial" begin
        scheme = explicit_partitioned_symplectic_rk_scheme(
            2;
            first_partition = :H,
        )
        dt = 1e-4
        formulation = PoissonBracketFormulation()

        serial_U = copy_field(global_U)
        serial_work = MaxwellPartitionedRKWorkspace(serial_U, scheme)
        serial_rhs! = DiscoGMPI.make_maxwell_rhs_function(
            global_dg,
            registry,
            formulation,
        )
        partitioned_symplectic_rk_step!(
            serial_U,
            serial_work,
            scheme,
            dt,
            serial_rhs!,
        )

        local_U = localize_maxwell_field(global_U, distributed_dg)
        local_work = MaxwellPartitionedRKWorkspace(local_U, scheme)
        distributed_partitioned_symplectic_rk_step!(
            local_U,
            local_work,
            scheme,
            dt,
            distributed_dg,
            registry,
            formulation,
        )

        error = MPI.Allreduce(
            owned_field_error(serial_U, local_U, distributed_dg),
            max,
            COMM,
        )
        @test error <= 1e-10
    end
elseif NPROCS == 4
    mesh = chain_pec_mesh()
    order = 2
    global_dg = DGDiscretization(mesh, order)
    root_mesh = RANK == 0 ? mesh : nothing
    root_partition = RANK == 0 ? [0, 1, 2, 3] : nothing
    distributed_dg = build_distributed_dg_from_root(
        root_mesh,
        root_partition,
        order;
        comm = COMM,
    )
    registry = MaxwellBoundaryRegistry(
        Dict(10 => DiscoGMPI.MaxwellBC_PEC),
    )
    global_U = deterministic_maxwell_field(global_dg.ref, 4)

    @testset "Four-rank distributed Maxwell RHS" begin
        @test length(distributed_dg.distributed_mesh.partition.owned) == 1
        @test length(distributed_dg.exchange.neighbors) == (RANK in (0, 3) ? 1 : 2)

        error, ghost_rhs, trace_error = compare_rhs(
            global_U,
            global_dg,
            distributed_dg,
            registry,
            HesthavenWarburtonFormulation(MaxwellFlux_Upwind),
        )

        @test error <= 1e-10
        @test ghost_rhs == 0.0
        @test trace_error <= 1e-12
    end

    @testset "Four-rank distributed RK step" begin
        scheme = explicit_rk_scheme(3)
        dt = 1e-4
        formulation =
            HesthavenWarburtonFormulation(MaxwellFlux_Upwind)

        serial_U = copy_field(global_U)
        serial_work = DiscoGMPI.MaxwellRKWorkspace(serial_U, scheme)
        serial_rhs! = DiscoGMPI.make_maxwell_rhs_function(
            global_dg,
            registry,
            formulation,
        )
        DiscoGMPI.rk_step!(
            serial_U,
            serial_work,
            scheme,
            dt,
            serial_rhs!,
        )

        local_U = localize_maxwell_field(global_U, distributed_dg)
        local_work = DiscoGMPI.MaxwellRKWorkspace(local_U, scheme)
        distributed_rk_step!(
            local_U,
            local_work,
            scheme,
            dt,
            distributed_dg,
            registry,
            formulation,
        )

        error = MPI.Allreduce(
            owned_field_error(serial_U, local_U, distributed_dg),
            max,
            COMM,
        )
        @test error <= 1e-10
    end
else
    @testset "MPI size guard" begin
        @info "Run this test with exactly two or four MPI ranks." NPROCS
        @test false
    end
end

MPI.Barrier(COMM)

if !MPI_WAS_INITIALIZED && !MPI.Finalized()
    MPI.Finalize()
end
