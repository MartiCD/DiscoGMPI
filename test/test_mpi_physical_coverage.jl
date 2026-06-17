using Test
using MPI
using DiscoGMPI

const MPI_WAS_INITIALIZED = MPI.Initialized()
MPI_WAS_INITIALIZED || MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NPROCS = MPI.Comm_size(COMM)

function periodic_cube_mesh()
    points = [
        0.0 1.0 1.0 0.0 0.0 1.0 1.0 0.0
        0.0 0.0 1.0 1.0 0.0 0.0 1.0 1.0
        0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0
    ]
    tets = [
        1 1 1 1 1 1
        2 3 4 8 5 6
        3 4 8 5 6 2
        7 7 7 7 7 7
    ]

    counts = Dict{NTuple{3, Int}, Int}()
    oriented = Dict{NTuple{3, Int}, NTuple{3, Int}}()
    for elem in axes(tets, 2), face in DiscoGMPI.TET_FACES
        nodes = (
            tets[face[1], elem],
            tets[face[2], elem],
            tets[face[3], elem],
        )
        key = Tuple(sort(collect(nodes)))
        counts[key] = get(counts, key, 0) + 1
        oriented[key] = nodes
    end

    boundary_faces = [
        oriented[key] for (key, count) in counts if count == 1
    ]
    tris = reduce(hcat, collect.(boundary_faces))
    ntets = size(tets, 2)
    ntris = size(tris, 2)
    boundary_id = zeros(Int, ntets + ntris)

    for tri in axes(tris, 2)
        centroid = sum(points[:, tris[:, tri]]; dims = 2)[:, 1] / 3
        tag = if abs(centroid[1]) < 1e-12
            1
        elseif abs(centroid[1] - 1.0) < 1e-12
            2
        elseif abs(centroid[2]) < 1e-12
            3
        elseif abs(centroid[2] - 1.0) < 1e-12
            4
        elseif abs(centroid[3]) < 1e-12
            5
        elseif abs(centroid[3] - 1.0) < 1e-12
            6
        else
            error("Could not classify cube boundary triangle.")
        end
        boundary_id[ntets + tri] = tag
    end

    material_id = zeros(Int, ntets + ntris)
    for elem in 1:ntets
        centroid = sum(points[:, tets[:, elem]]; dims = 2)[:, 1] / 4
        material_id[elem] = centroid[1] < 0.5 ? 1 : 2
    end

    return RawVTUMesh(
        points,
        tets,
        tris,
        collect(1:ntets),
        collect((ntets + 1):(ntets + ntris)),
        Dict{String, Any}(
            "boundary_id" => boundary_id,
            "material_id" => material_id,
        ),
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

function owned_rhs_error(global_rhs::MaxwellRHS, local_rhs::MaxwellRHS, distributed_dg)
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

@testset "Distributed periodic physical coverage" begin
    NPROCS >= 2 ||
        error("Run this test with at least two MPI ranks.")

    mesh = RANK == 0 ? periodic_cube_mesh() : nothing
    partition = RANK == 0 ? mod.(0:5, NPROCS) : nothing
    distributed_dg = build_distributed_dg_from_root(
        mesh,
        partition,
        2;
        comm = COMM,
        root = 0,
        material_tag_name = "material_id",
    )
    material_table = Dict(
        1 => MaxwellMaterial(1.0, 1.0),
        2 => MaxwellMaterial(1.0, 1.0),
    )
    materials = maxwell_element_materials(distributed_dg, material_table)
    periodic = build_distributed_periodic_maxwell_exchange(
        distributed_dg;
        materials = materials,
    )
    registry = MaxwellBoundaryRegistry(
        Dict(id => MaxwellBC_None for id in 1:6),
    )

    local_face_count = length(periodic.faces)
    @test MPI.Allreduce(local_face_count, +, COMM) == 12

    zero = interpolate_maxwell_field(
        distributed_dg,
        (x, y, z) -> (0.0, 0.0, 0.0),
        (x, y, z) -> (0.0, 0.0, 0.0),
    )
    rhs = DiscoGMPI.similar_maxwell_rhs(zero)
    maxwell_rhs_periodic!(
        rhs,
        zero,
        distributed_dg,
        periodic,
        registry,
        PoissonBracketFormulation(),
        materials,
    )
    owned = distributed_dg.distributed_mesh.partition.owned
    local_maximum = maximum(
        maximum(abs, component[:, owned])
        for component in (
            rhs.rhsEx,
            rhs.rhsEy,
            rhs.rhsEz,
            rhs.rhsHx,
            rhs.rhsHy,
            rhs.rhsHz,
        )
    )
    @test MPI.Allreduce(local_maximum, max, COMM) < 1e-13

    serial_mesh = periodic_cube_mesh()
    serial_dg = DGDiscretization(serial_mesh, 2)
    serial_periodic = build_periodic_flux_faces(
        serial_mesh,
        serial_dg.ref,
        serial_dg.flux_faces,
        default_unit_box_periodic_specs(),
    )
    nonzero_periodic = interpolate_maxwell_field(
        serial_mesh,
        serial_dg.ref,
        (x, y, z) -> (
            sin(2pi * x) + 0.25 * cos(2pi * y),
            cos(2pi * y) + 0.20 * sin(2pi * z),
            sin(2pi * z) + 0.15 * cos(2pi * x),
        ),
        (x, y, z) -> (
            cos(2pi * z) - 0.10 * sin(2pi * y),
            sin(2pi * x) + 0.30 * cos(2pi * z),
            cos(2pi * y) - 0.20 * sin(2pi * x),
        ),
    )
    serial_rhs = DiscoGMPI.similar_maxwell_rhs(nonzero_periodic)
    maxwell_rhs_periodic!(
        serial_rhs,
        nonzero_periodic,
        serial_dg,
        serial_periodic,
        registry,
        PoissonBracketFormulation(),
    )

    local_periodic = localize_maxwell_field(nonzero_periodic, distributed_dg)
    poison_ghosts!(local_periodic, distributed_dg)
    local_rhs = DiscoGMPI.similar_maxwell_rhs(local_periodic)
    maxwell_rhs_periodic!(
        local_rhs,
        local_periodic,
        distributed_dg,
        periodic,
        registry,
        PoissonBracketFormulation(),
        materials,
    )
    local_rhs_norm = maximum(
        maximum(abs, component[:, owned])
        for component in rhs_components(local_rhs)
    )
    local_periodic_error =
        owned_rhs_error(serial_rhs, local_rhs, distributed_dg)
    @test MPI.Allreduce(local_rhs_norm, max, COMM) > 1e-8
    @test MPI.Allreduce(local_periodic_error, max, COMM) < 1e-10

    plane_wave = interpolate_maxwell_field(
        distributed_dg,
        (x, y, z) -> (0.0, cos(2pi * x), 0.0),
        (x, y, z) -> (0.0, 0.0, cos(2pi * x)),
    )
    diagnostics =
        distributed_maxwell_invariants(plane_wave, distributed_dg, materials)
    @test diagnostics.energy.total > 0.0
    @test abs(diagnostics.electric_charge) < 1e-10
    @test abs(diagnostics.magnetic_charge) < 1e-10
    @test diagnostics.linear_momentum[1] > 0.0
    @test abs(diagnostics.linear_momentum[2]) < 1e-12
    @test abs(diagnostics.linear_momentum[3]) < 1e-12
end

MPI.Barrier(COMM)
MPI_WAS_INITIALIZED || MPI.Finalize()
