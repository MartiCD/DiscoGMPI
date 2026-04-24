using Test
using MPI

# MPI tests should be launched explicitly with mpiexec, for example:
#   mpiexec -n 2 julia --project=. test/test_mpi_metis_partition.jl
#   mpiexec -n 4 julia --project=. test/test_mpi_metis_partition.jl

include(joinpath(@__DIR__, "..", "src", "DiscoGMPI.jl"))
using .DiscoGMPI
using .DiscoGMPI.DistributedMesh3D
using .DiscoGMPI.TraceMaps

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM) + 1   # Julia-side ranks are 1-based in the mesh layer
const NPROCS = MPI.Comm_size(COMM)

function sample_chain_mesh()
    coords = [
        0.0 1.0 0.0 0.0 1.0 2.0 2.0;
        0.0 0.0 1.0 0.0 1.0 1.0 0.0;
        0.0 0.0 0.0 1.0 1.0 0.0 1.0;
    ]

    elems = [
        1 2 3 4;
        2 3 4 5;
        3 4 5 6;
        4 5 6 7;
    ]'

    return coords, elems
end

function write_epart(parts_zero_based::Vector{Int})
    dir = mktempdir()
    path = joinpath(dir, "mesh.epart.$(maximum(parts_zero_based) + 1)")
    open(path, "w") do io
        for p in parts_zero_based
            println(io, p)
        end
    end
    return path
end

encode_value(global_node::Int, owner_global_elem::Int) = 1000.0 * owner_global_elem + global_node

function fill_owned_field(mesh::DistributedMesh{T}) where {T}
    field = fill(T(-1), size(mesh.nodes.coords, 2), length(mesh.elements.global_ids))
    for le in mesh.partition.owned
        ge = mesh.elements.global_ids[le]
        for ln in 1:size(field, 1)
            gnode = mesh.nodes.global_ids[ln]
            field[ln, le] = T(encode_value(gnode, ge))
        end
    end
    return field
end

function verify_interface_roundtrip(mesh, ghost_field)
    for nrank in mesh.mpi.neighbors
        for recv_face in mesh.mpi.comms[nrank].recv_faces
            ge_owner = recv_face.neighbor_global_elem
            n1, n2, n3 = recv_face.face_nodes_local
            g1 = mesh.nodes.global_ids[n1]
            g2 = mesh.nodes.global_ids[n2]
            g3 = mesh.nodes.global_ids[n3]
            @test ghost_field[n1, recv_face.local_elem] == encode_value(g1, ge_owner)
            @test ghost_field[n2, recv_face.local_elem] == encode_value(g2, ge_owner)
            @test ghost_field[n3, recv_face.local_elem] == encode_value(g3, ge_owner)
        end
    end
end

@testset "METIS partition reader" begin
    parts0 = [0, 1, 1, 0]
    parts1 = [1, 2, 2, 1]
    epart = write_epart(parts0)
    @test read_metis_epart(epart) == parts0
    @test read_metis_epart(epart; one_based_parts=true) == parts1
end

coords, elems = sample_chain_mesh()

if NPROCS == 2
    parts = [0, 0, 1, 1]
    epart = write_epart(parts)
    mesh = build_distributed_mesh_from_metis(coords, elems, epart, RANK)

    @testset "2-rank distributed mesh from METIS" begin
        if RANK == 1
            @test length(mesh.partition.owned) == 2
            @test length(mesh.partition.ghosts) == 1
            @test mesh.mpi.neighbors == [2]
            @test length(mesh.mpi.comms[2].send_faces) == 1
            @test length(mesh.mpi.comms[2].recv_faces) == 1
            @test first(mesh.mpi.comms[2].send_faces).neighbor_global_elem == 3
            @test first(mesh.mpi.comms[2].send_faces).orientation == 1
            @test issorted([DistributedMesh3D.interface_sort_key(f) for f in mesh.mpi.comms[2].send_faces])
            @test issorted([DistributedMesh3D.interface_sort_key(f) for f in mesh.mpi.comms[2].recv_faces])
        elseif RANK == 2
            @test length(mesh.partition.owned) == 2
            @test length(mesh.partition.ghosts) == 1
            @test mesh.mpi.neighbors == [1]
            @test length(mesh.mpi.comms[1].send_faces) == 1
            @test length(mesh.mpi.comms[1].recv_faces) == 1
            @test first(mesh.mpi.comms[1].send_faces).neighbor_global_elem == 2
            @test first(mesh.mpi.comms[1].send_faces).orientation == 1
            @test issorted([DistributedMesh3D.interface_sort_key(f) for f in mesh.mpi.comms[1].send_faces])
            @test issorted([DistributedMesh3D.interface_sort_key(f) for f in mesh.mpi.comms[1].recv_faces])
        end
    end

    @testset "2-rank trace exchange round-trip" begin
        owned_field = fill_owned_field(mesh)
        ghost_field = trace_exchange_roundtrip!(owned_field, mesh, COMM)
        verify_interface_roundtrip(mesh, ghost_field)
    end
elseif NPROCS == 4
    parts = [0, 1, 2, 3]
    epart = write_epart(parts)
    mesh = build_distributed_mesh_from_metis(coords, elems, epart, RANK)

    @testset "4-rank distributed mesh from METIS" begin
        @test length(mesh.partition.owned) == 1
        if RANK == 1
            @test length(mesh.partition.ghosts) == 1
            @test mesh.mpi.neighbors == [2]
            @test length(mesh.mpi.comms[2].send_faces) == 1
            @test first(mesh.mpi.comms[2].send_faces).neighbor_global_elem == 2
            @test first(mesh.mpi.comms[2].send_faces).orientation == 1
        elseif RANK == 2
            @test length(mesh.partition.ghosts) == 2
            @test mesh.mpi.neighbors == [1, 3]
            @test length(mesh.mpi.comms[1].send_faces) == 1
            @test length(mesh.mpi.comms[3].send_faces) == 1
            @test all(f.orientation == 1 for f in mesh.mpi.comms[1].send_faces)
            @test all(f.orientation == 1 for f in mesh.mpi.comms[3].send_faces)
        elseif RANK == 3
            @test length(mesh.partition.ghosts) == 2
            @test mesh.mpi.neighbors == [2, 4]
            @test length(mesh.mpi.comms[2].send_faces) == 1
            @test length(mesh.mpi.comms[4].send_faces) == 1
            @test all(f.orientation == 1 for f in mesh.mpi.comms[2].send_faces)
            @test all(f.orientation == 1 for f in mesh.mpi.comms[4].send_faces)
        elseif RANK == 4
            @test length(mesh.partition.ghosts) == 1
            @test mesh.mpi.neighbors == [3]
            @test length(mesh.mpi.comms[3].send_faces) == 1
            @test first(mesh.mpi.comms[3].send_faces).neighbor_global_elem == 3
            @test first(mesh.mpi.comms[3].send_faces).orientation == 1
        end

        for nrank in mesh.mpi.neighbors
            @test issorted([DistributedMesh3D.interface_sort_key(f) for f in mesh.mpi.comms[nrank].send_faces])
            @test issorted([DistributedMesh3D.interface_sort_key(f) for f in mesh.mpi.comms[nrank].recv_faces])
        end
    end

    @testset "4-rank trace exchange round-trip" begin
        owned_field = fill_owned_field(mesh)
        ghost_field = trace_exchange_roundtrip!(owned_field, mesh, COMM)
        verify_interface_roundtrip(mesh, ghost_field)
    end
else
    @testset "MPI size guard" begin
        @info "Run this test suite with exactly 2 or 4 MPI ranks." NPROCS
        @test false
    end
end

MPI.Barrier(COMM)
MPI.Finalize()