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
