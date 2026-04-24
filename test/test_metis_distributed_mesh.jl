using Test
using DiscoGMPI
using DiscoGMPI.DistributedMesh3D

include("../src/MetisIO.jl")

@testset "METIS distributed mesh consistency" begin
    vtk_path = joinpath(@__DIR__,"../","examples", "meshes", "tet_mesh.vtk")
    epart_path = joinpath(@__DIR__,"../","examples", "meshes", "tet_mesh.mesh.epart.2")

    coords, elems = read_mesh_file_tet_vtk(vtk_path)

    if size(elems, 1) != 4 && size(elems, 2) == 4
        elems = permutedims(elems)
    end

    nparts = 2
    meshes = [
        build_distributed_mesh_from_metis(coords, elems, epart_path, rank)
        for rank in 0:nparts-1
    ]

    elem_to_rank = read_metis_epart(epart_path)

    @testset "owned elements" begin
        for rank in 0:nparts-1
            expected_owned = findall(==(rank), elem_to_rank)
            @test sort(meshes[rank+1].partition.owned) == expected_owned
        end
    end

    @testset "neighbors are symmetric" begin
        for r in 0:nparts-1
            mesh_r = meshes[r+1]

            for n in mesh_r.mpi.neighbors
                mesh_n = meshes[n+1]

                @test r in mesh_n.mpi.neighbors
            end
        end
    end

    @testset "interface faces are cross-rank" begin
        for r in 0:nparts-1
            mesh = meshes[r+1]

            for n in mesh.mpi.neighbors
                comm = mesh.mpi.comms[n]

                for f in comm.send_faces
                    @test elem_to_rank[f.global_elem] == r
                    @test elem_to_rank[f.neighbor_global_elem] == n
                    @test f.orientation in 1:6
                end
            end
        end
    end

    @testset "interface pairing symmetry" begin
        for r in 0:nparts-1
            mesh_r = meshes[r+1]

            for n in mesh_r.mpi.neighbors
                mesh_n = meshes[n+1]

                faces_rn = [
                    (
                        f.global_elem,
                        f.local_face,
                        f.neighbor_global_elem,
                        f.neighbor_face,
                    )
                    for f in mesh_r.mpi.comms[n].send_faces
                ]

                faces_nr = [
                    (
                        f.neighbor_global_elem,
                        f.neighbor_face,
                        f.global_elem,
                        f.local_face,
                    )
                    for f in mesh_n.mpi.comms[r].send_faces
                ]

                @test sort(faces_rn) == sort(faces_nr)
            end
        end
    end
end