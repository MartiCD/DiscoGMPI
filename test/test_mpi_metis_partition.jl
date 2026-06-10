using Test
using MPI
using DiscoGMPI
using DiscoGMPI.DistributedMesh3D
using DiscoGMPI.TraceMaps

# Run with:
#   mpiexec -n 2 julia --project=. test/test_mpi_metis_partition.jl
#   mpiexec -n 4 julia --project=. test/test_mpi_metis_partition.jl

const MPI_WAS_INITIALIZED = MPI.Initialized()
MPI_WAS_INITIALIZED || MPI.Init()

const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NPROCS = MPI.Comm_size(COMM)

function sample_chain_mesh()
    coords = [
        0.0 1.0 0.0 0.0 1.0 2.0 2.0
        0.0 0.0 1.0 0.0 1.0 1.0 0.0
        0.0 0.0 0.0 1.0 1.0 0.0 1.0
    ]

    elems = [
        1 2 3 4
        2 3 4 5
        3 4 5 6
        4 5 6 7
    ]'

    return coords, elems
end

function write_epart(parts::Vector{Int})
    dir = mktempdir()
    path = joinpath(dir, "mesh.epart")
    open(path, "w") do io
        foreach(part -> println(io, part), parts)
    end
    return path
end

function send_sort_key(mesh, face)
    return (
        mesh.elements.global_ids[face.local_elem],
        face.local_face,
        face.neighbor_global_elem,
        face.neighbor_face,
    )
end

function recv_sort_key(mesh, face)
    return (
        face.neighbor_global_elem,
        face.neighbor_face,
        mesh.elements.global_ids[face.local_elem],
        face.local_face,
    )
end

function verify_trace_exchange(mesh)
    Nfp = 1
    Nvars = 1
    face_centroid = reshape([1 / 3, 1 / 3, 1 / 3], 1, 3)

    trace_maps, ghost_traces, face_lookup, perm_maps =
        build_mpi_trace_maps_buffers_and_permutations(
            mesh,
            Nfp,
            Nvars,
            face_centroid,
            Float64,
        )

    trace = zeros(Float64, Nfp, 4, length(mesh.elements.global_ids), Nvars)

    for neighbor in mesh.mpi.neighbors
        for face in mesh.mpi.comms[neighbor].send_faces
            global_elem = mesh.elements.global_ids[face.local_elem]
            trace[1, face.local_face, face.local_elem, 1] = global_elem
        end
    end

    exchange_face_traces_blocking!(trace, ghost_traces, trace_maps, COMM)

    for neighbor in mesh.mpi.neighbors
        for face in mesh.mpi.comms[neighbor].send_faces
            uM, uP, face_type = get_face_traces(
                mesh,
                trace,
                ghost_traces,
                face_lookup,
                perm_maps,
                face.local_elem,
                face.local_face,
            )

            @test face_type == :mpi
            @test uM[1, 1] == mesh.elements.global_ids[face.local_elem]
            @test uP[1, 1] == face.neighbor_global_elem
        end
    end
end

@testset "METIS partition reader" begin
    parts_zero_based = [0, 1, 1, 0]
    parts_one_based = parts_zero_based .+ 1

    @test read_metis_epart(write_epart(parts_zero_based)) == parts_zero_based
    @test read_metis_epart(
        write_epart(parts_one_based);
        one_based_parts = true,
    ) == parts_zero_based
end

coords, elems = sample_chain_mesh()

if NPROCS == 2
    mesh = build_distributed_mesh_from_metis(
        coords,
        elems,
        write_epart([0, 0, 1, 1]),
        RANK,
    )

    @testset "2-rank distributed mesh from METIS" begin
        @test length(mesh.partition.owned) == 2
        @test length(mesh.partition.ghosts) == 1

        neighbor = 1 - RANK
        @test mesh.mpi.neighbors == [neighbor]
        @test length(mesh.mpi.comms[neighbor].send_faces) == 1
        @test length(mesh.mpi.comms[neighbor].recv_faces) == 1

        send_face = only(mesh.mpi.comms[neighbor].send_faces)
        @test send_face.neighbor_global_elem == (RANK == 0 ? 3 : 2)
        @test send_face.orientation == 1
        @test issorted(
            mesh.mpi.comms[neighbor].send_faces;
            by = face -> send_sort_key(mesh, face),
        )
        @test issorted(
            mesh.mpi.comms[neighbor].recv_faces;
            by = face -> recv_sort_key(mesh, face),
        )
    end

    @testset "2-rank trace exchange" begin
        verify_trace_exchange(mesh)
    end
elseif NPROCS == 4
    mesh = build_distributed_mesh_from_metis(
        coords,
        elems,
        write_epart([0, 1, 2, 3]),
        RANK,
    )

    expected_neighbors = (
        [1],
        [0, 2],
        [1, 3],
        [2],
    )
    expected_ghosts = (1, 2, 2, 1)

    @testset "4-rank distributed mesh from METIS" begin
        @test length(mesh.partition.owned) == 1
        @test length(mesh.partition.ghosts) == expected_ghosts[RANK + 1]
        @test mesh.mpi.neighbors == expected_neighbors[RANK + 1]

        for neighbor in mesh.mpi.neighbors
            @test length(mesh.mpi.comms[neighbor].send_faces) == 1
            @test length(mesh.mpi.comms[neighbor].recv_faces) == 1
            @test only(mesh.mpi.comms[neighbor].send_faces).orientation == 1
            @test issorted(
                mesh.mpi.comms[neighbor].send_faces;
                by = face -> send_sort_key(mesh, face),
            )
            @test issorted(
                mesh.mpi.comms[neighbor].recv_faces;
                by = face -> recv_sort_key(mesh, face),
            )
        end
    end

    @testset "4-rank trace exchange" begin
        verify_trace_exchange(mesh)
    end
else
    @testset "MPI size guard" begin
        @info "Run this test suite with exactly 2 or 4 MPI ranks." NPROCS
        @test false
    end
end

MPI.Barrier(COMM)

if !MPI_WAS_INITIALIZED && !MPI.Finalized()
    MPI.Finalize()
end
