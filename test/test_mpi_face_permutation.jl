using Test
using MPI

include("../src/DistributedMesh3D.jl")
using .DistributedMesh3D

include("../src/TraceMaps.jl")
using .TraceMaps

# ------------------------------------------------------------
# Reference face nodes: 6-node triangle in barycentric coords
# ------------------------------------------------------------
function reference_face_nodes_p2_barycentric()
    return [
        1.0  0.0  0.0
        0.0  1.0  0.0
        0.0  0.0  1.0
        0.5  0.5  0.0
        0.0  0.5  0.5
        0.5  0.0  0.5
    ]  # (Nfp, 3)
end

# ------------------------------------------------------------
# Smooth linear test field
# Returns Nvars values at (x,y,z)
# ------------------------------------------------------------
@inline function linear_field(x::Float64, y::Float64, z::Float64)
    return (
        x + 2y + 3z,
        -2x + y + 0.5z,
        x - y + 4z + 1,
    )
end

# ------------------------------------------------------------
# Physical coordinates of the Nfp nodes on local face (e,f)
# from barycentric face reference nodes
# ------------------------------------------------------------
function physical_face_points(
    mesh,
    e::Int,
    f::Int,
    face_nodes_reference::AbstractMatrix{<:Real},
)
    Nfp = size(face_nodes_reference, 1)

    lv = tet_face_local_vertices[f]
    n1 = mesh.elements.vertices[lv[1], e]
    n2 = mesh.elements.vertices[lv[2], e]
    n3 = mesh.elements.vertices[lv[3], e]

    x1 = mesh.nodes.coords[:, n1]
    x2 = mesh.nodes.coords[:, n2]
    x3 = mesh.nodes.coords[:, n3]

    X = Matrix{Float64}(undef, Nfp, 3)

    for i in 1:Nfp
        λ1 = face_nodes_reference[i, 1]
        λ2 = face_nodes_reference[i, 2]
        λ3 = face_nodes_reference[i, 3]

        X[i, 1] = λ1 * x1[1] + λ2 * x2[1] + λ3 * x3[1]
        X[i, 2] = λ1 * x1[2] + λ2 * x2[2] + λ3 * x3[2]
        X[i, 3] = λ1 * x1[3] + λ2 * x2[3] + λ3 * x3[3]
    end

    return X
end

# ------------------------------------------------------------
# Fill one local face trace from the linear physical field
# trace has shape (Nfp, Nfaces, Ne, Nvars)
# ------------------------------------------------------------
function fill_trace_on_face!(
    trace::Array{Float64,4},
    mesh,
    e::Int,
    f::Int,
    face_nodes_reference::AbstractMatrix{<:Real},
)
    X = physical_face_points(mesh, e, f, face_nodes_reference)
    Nfp = size(X, 1)

    for i in 1:Nfp
        vals = linear_field(X[i, 1], X[i, 2], X[i, 3])
        for v in 1:length(vals)
            trace[i, f, e, v] = vals[v]
        end
    end

    return nothing
end

# ------------------------------------------------------------
# Find the unique owned MPI face on this rank
# For this test mesh, each rank owns one element and one MPI face
# ------------------------------------------------------------
function get_owned_mpi_face(mesh)
    @test length(mesh.mpi.neighbors) == 1
    nbr = only(mesh.mpi.neighbors)
    comm = mesh.mpi.comms[nbr]
    @test length(comm.send_faces) == 1
    sf = comm.send_faces[1]
    return nbr, sf.local_elem, sf.local_face
end

# ------------------------------------------------------------
# Main test body
# ------------------------------------------------------------
function run_mpi_face_permutation_test()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    @test nranks == 2

    if nranks != 2
        if rank == 0
            @warn "Skipping MPI face permutation test: run with exactly 2 MPI ranks."
        end
        return
    end

    # --------------------------------------------------------
    # Global mesh:
    #
    # Tet 1: [1,2,3,4]
    # Tet 2: [1,3,2,5]
    #
    # Shared face is global nodes {1,2,3}, but with different
    # local ordering on each element, which forces a nontrivial
    # face permutation.
    # --------------------------------------------------------
    global_node_coords = [
         0.0   1.0   0.0   0.0   0.0
         0.0   0.0   1.0   0.0   0.0
         0.0   0.0   0.0   1.0  -1.0
    ]  # (3,5)

    global_elem_vertices = [
        1  1
        2  3
        3  2
        4  5
    ]  # (4,2)

    elem_to_rank = [0, 1]
    material_id_global = [1, 1]

    mesh = build_distributed_mesh(
        global_node_coords,
        global_elem_vertices,
        elem_to_rank,
        rank;
        material_id_global = material_id_global,
    )

    face_nodes_reference = reference_face_nodes_p2_barycentric()
    Nfp = size(face_nodes_reference, 1)
    Nvars = 3
    Ne = length(mesh.elements.global_ids)
    Nfaces = 4

    trace_maps, ghost_traces, face_lookup, perm_maps =
        build_mpi_trace_maps_buffers_and_permutations(
            mesh,
            Nfp,
            Nvars,
            face_nodes_reference,
            Float64,
        )

    trace = zeros(Float64, Nfp, Nfaces, Ne, Nvars)

    # Fill only the owned MPI face on this rank
    nbr, e, f = get_owned_mpi_face(mesh)
    fill_trace_on_face!(trace, mesh, e, f, face_nodes_reference)

    # Exchange raw face traces
    exchange_face_traces_blocking!(trace, ghost_traces, trace_maps, comm)

    # --------------------------------------------------------
    # Check 1: local minus trace exists
    # --------------------------------------------------------
    uM = @view trace[:, f, e, :]

    # Raw received trace before permutation
    rank_nbr, k = face_lookup.lookup[(e, f)]
    @test rank_nbr == nbr
    uP_raw = @view ghost_traces[nbr][:, k, :]

    # Reordered trace through production accessor
    uM2, uP, facetype = get_face_traces(
        mesh,
        trace,
        ghost_traces,
        face_lookup,
        perm_maps,
        e,
        f,
    )

    @test facetype == :mpi
    @test uM === uM2

    # --------------------------------------------------------
    # Check 2: permutation should be nontrivial for this setup
    # --------------------------------------------------------
    perm = perm_maps.lookup[(e, f)]
    @test perm != collect(1:Nfp)

    # --------------------------------------------------------
    # Check 3: raw receive should not match node-by-node
    # --------------------------------------------------------
    raw_matches = all(isapprox.(uM, uP_raw; atol=1e-12, rtol=1e-12))
    @test !raw_matches

    # --------------------------------------------------------
    # Check 4: reordered receive must match node-by-node
    # --------------------------------------------------------
    @test all(isapprox.(uM, uP; atol=1e-12, rtol=1e-12))

    for i in 1:Nfp, v in 1:Nvars
        @test isapprox(uM[i, v], uP[i, v]; atol=1e-12, rtol=1e-12)
    end

    MPI.Barrier(comm)

    if rank == 0
        @info "MPI face permutation sanity test passed."
    end
end

# ------------------------------------------------------------
# Test entry point
# ------------------------------------------------------------
mpi_was_initialized = MPI.Initialized()
if !mpi_was_initialized
    MPI.Init()
end

@testset "DiscoG MPI face permutation" begin
    run_mpi_face_permutation_test()
end

MPI.Barrier(MPI.COMM_WORLD)

if !mpi_was_initialized && !MPI.Finalized()
    MPI.Finalize()
end