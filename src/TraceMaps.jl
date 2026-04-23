module TraceMaps

# We already have DistributedMesh3D 
# We assume we have local DG trace arrays, something like:
# traceEx::Array{Float64,3}   # (Nfp, Nfaces, nlocal_elems)
# traceEy::Array{Float64,3}
# traceEz::Array{Float64,3}
# traceHx::Array{Float64,3}
# traceHy::Array{Float64,3}
# traceHz::Array{Float64,3}
# 
# or more generally one trace container with Nvars variables 

# This is Stage 2 communication layer:
# - identify MPI interface faces from mesh.mpi
# - build linear DOF indices for those faces 
# - pack face trace values from owned elements 
# - exchange with neighbor ranks using blocking MPI.Sendrecv!
# - unpack received values into a ghost-trace buffer
# - let your flux kernel read:
#  - local trace = minus side
#  - received ghost trace = plus side for MPI faces

# Bellow we assume:
# trace::Array{T,4}   # (Nfp, Nfaces, nlocal_elems, Nvars)
# ----------------------------------------------------------------

# PRACTICAL USAGE 
# 1) After building mesh and trace exchange objects:
# trace_maps, ghost_traces, face_lookup, perm_maps =
#     build_mpi_trace_maps_buffers_and_permutations(
#         mesh,
#         Nfp,
#         Nvars,
#         face_nodes_reference,
#         Float64,
#     )
#
# 2)Then, per stage:
# exchange_face_traces_blocking!(trace, ghost_traces, trace_maps, MPI.COMM_WORLD)
# 
# 3) Then in face flux loop:
# uM, uP, facetype = get_face_traces(
#     mesh,
#     trace,
#     ghost_traces,
#     face_lookup,
#     perm_maps,
#     e,
#     f,
# )
# If facetype == :mpi, uP is already reordered into the local minus-side node order.
#
#
# VERY IMPORTANT!
# This assumes that:
# - trace[:, f, e, :] uses the same canonical face-node ordering for every tetra face, and
# - face_nodes_reference matches that ordering exactly
# If tetra face trace nodes are generated differently on different faces, then we need one more layer:
# a face-specific embedding from the canonical triangle to each tetra face

using MPI
using ..DistributedMesh3D: ElementOwnership,
       FaceNeighborInfo,
       LocalNodeGeometry,
       LocalElementTopology,
       ElementPartition,
       InterfaceFace,
       NeighborComm,
       MPICommunicationTable,
       DistributedMesh,
       build_distributed_mesh,
       tet_face_local_vertices,
       TRI_PERMUTATIONS

export tet_face_local_vertices,
        build_mpi_trace_maps_buffers_and_permutations,
        exchange_face_traces_blocking!,
        get_face_traces


struct MPIFaceTraceMap
    rank::Int
    send_faces_local::Vector{Int}
    recv_faces_local::Vector{Int}
    send_dof_indices::Vector{Int}
    recv_dof_indices::Vector{Int}
    nsend::Int
    nrecv::Int
end

struct MPITraceExchangeTable
    neighbors::Vector{Int}
    maps::Dict{Int, MPIFaceTraceMap}
end

struct MPIRemoteFaceLookup
    lookup::Dict{Tuple{Int,Int}, Tuple{Int,Int}}
end

@inline function trace_linear_index(i::Int, f::Int, e::Int, v::Int,
                                    Nfp::Int, Nfaces::Int, Ne::Int)
    return i + (f-1)*Nfp + (e-1)*Nfp*Nfaces + (v-1)*Nfp*Nfaces*Ne
end

@inline function ghost_trace_linear_index(i::Int, k::Int, v::Int,
                                          Nfp::Int, nrecvfaces::Int)
    return i + (k-1)*Nfp + (v-1)*Nfp*nrecvfaces
end

function build_mpi_trace_maps(
    mesh::DistributedMesh,
    Nfp::Int,
    Nvars::Int,
)
    Ne = length(mesh.elements.global_ids)
    Nfaces = 4

    maps = Dict{Int, MPIFaceTraceMap}()

    for rank in mesh.mpi.neighbors
        comm = mesh.mpi.comms[rank]

        send_faces_local = collect(eachindex(comm.send_faces))
        recv_faces_local = collect(eachindex(comm.recv_faces))

        nsendfaces = length(send_faces_local)
        nrecvfaces = length(recv_faces_local)

        send_dof_indices = Vector{Int}(undef, Nfp * Nvars * nsendfaces)
        recv_dof_indices = Vector{Int}(undef, Nfp * Nvars * nrecvfaces)

        p = 1
        for sfidx in send_faces_local
            iface = comm.send_faces[sfidx]
            e = iface.local_elem
            f = iface.local_face
            for v in 1:Nvars
                for i in 1:Nfp
                    send_dof_indices[p] = trace_linear_index(i, f, e, v, Nfp, Nfaces, Ne)
                    p += 1
                end
            end
        end

        p = 1
        for (k, _) in enumerate(recv_faces_local)
            for v in 1:Nvars
                for i in 1:Nfp
                    recv_dof_indices[p] = ghost_trace_linear_index(i, k, v, Nfp, nrecvfaces)
                    p += 1
                end
            end
        end

        maps[rank] = MPIFaceTraceMap(
            rank,
            send_faces_local,
            recv_faces_local,
            send_dof_indices,
            recv_dof_indices,
            length(send_dof_indices),
            length(recv_dof_indices),
        )
    end

    return MPITraceExchangeTable(mesh.mpi.neighbors, maps)
end

function allocate_neighbor_ghost_traces(
    trace_maps::MPITraceExchangeTable,
    Nfp::Int,
    Nvars::Int,
    T::Type{<:Real}=Float64,
)
    ghost_traces = Dict{Int, Array{T,3}}()
    for rank in trace_maps.neighbors
        m = trace_maps.maps[rank]
        nrecvfaces = length(m.recv_faces_local)
        ghost_traces[rank] = zeros(T, Nfp, nrecvfaces, Nvars)
    end
    return ghost_traces
end

function pack_trace_buffer!(
    sendbuf::AbstractVector{T},
    trace::Array{T,4},
    send_dof_indices::AbstractVector{Int},
) where {T}
    tr = vec(trace)
    @inbounds for i in eachindex(send_dof_indices)
        sendbuf[i] = tr[send_dof_indices[i]]
    end
    return nothing
end

function unpack_trace_buffer!(
    ghost_trace::Array{T,3},
    recvbuf::AbstractVector{T},
    recv_dof_indices::AbstractVector{Int},
) where {T}
    gt = vec(ghost_trace)
    @inbounds for i in eachindex(recv_dof_indices)
        gt[recv_dof_indices[i]] = recvbuf[i]
    end
    return nothing
end

function exchange_face_traces_blocking!(
    trace::Array{T,4},
    ghost_traces::Dict{Int, Array{T,3}},
    trace_maps::MPITraceExchangeTable,
    comm_mpi::MPI.Comm;
    tag::Int = 1001,
) where {T}

    for rank in trace_maps.neighbors
        m = trace_maps.maps[rank]

        sendbuf = Vector{T}(undef, m.nsend)
        recvbuf = Vector{T}(undef, m.nrecv)

        pack_trace_buffer!(sendbuf, trace, m.send_dof_indices)

        MPI.Sendrecv!(
            sendbuf, rank, tag,
            recvbuf, rank, tag,
            comm_mpi,
        )

        unpack_trace_buffer!(ghost_traces[rank], recvbuf, m.recv_dof_indices)
    end

    return nothing
end

function build_mpi_remote_face_lookup(mesh::DistributedMesh)
    lut = Dict{Tuple{Int,Int}, Tuple{Int,Int}}()
    for rank in mesh.mpi.neighbors
        comm = mesh.mpi.comms[rank]
        for (k, sf) in enumerate(comm.send_faces)
            lut[(sf.local_elem, sf.local_face)] = (rank, k)
        end
    end
    return MPIRemoteFaceLookup(lut)
end

@inline function get_mpi_plus_trace(
    ghost_traces::Dict{Int, Array{T,3}},
    face_lookup::MPIRemoteFaceLookup,
    local_elem::Int,
    local_face::Int,
) where {T}
    rank, k = face_lookup.lookup[(local_elem, local_face)]
    return @view ghost_traces[rank][:, k, :]
end

# Old without reordering of plus traces:
# function build_mpi_trace_maps_and_buffers(
#     mesh::DistributedMesh,
#     Nfp::Int,
#     Nvars::Int,
#     T::Type{<:Real}=Float64,
# )
#     trace_maps = build_mpi_trace_maps(mesh, Nfp, Nvars)
#     ghost_traces = allocate_neighbor_ghost_traces(trace_maps, Nfp, Nvars, T)
#     face_lookup = build_mpi_remote_face_lookup(mesh)
#     return trace_maps, ghost_traces, face_lookup
# end

# New with precomputed permutation maps for plus traces:
# Function assumes face_node_reference describes the nodal points on one canonical triangular face 
# in the local face ordering used by DG face trace arrays
#
# Examples:
# 1) If we have 2D reference coordinates on the face, we can use:
# face_nodes_reference :: Matrix{Float64}   # (Nfp, 2)
# where each row is (r,s) on the triangle with vertices (0,0), (1,0), (0,1)
#
# 2) If we already have barycentric coordinates for the face nodes, we can use:
# face_nodes_reference :: Matrix{Float64}   # (Nfp, 3)
function build_mpi_trace_maps_buffers_and_permutations(
    mesh::DistributedMesh,
    Nfp::Int,
    Nvars::Int,
    face_nodes_reference::AbstractMatrix,
    T::Type{<:Real}=Float64,
)
    trace_maps = build_mpi_trace_maps(mesh, Nfp, Nvars)
    ghost_traces = allocate_neighbor_ghost_traces(trace_maps, Nfp, Nvars, T)
    face_lookup = build_mpi_remote_face_lookup(mesh)
    perm_maps = build_mpi_face_permutation_maps(mesh, Nfp, face_nodes_reference)
    return trace_maps, ghost_traces, face_lookup, perm_maps
end

# Old without reordering of plus traces:
# function get_face_traces(
#     mesh::DistributedMesh,
#     trace::Array{T,4},
#     ghost_traces::Dict{Int, Array{T,3}},
#     face_lookup::MPIRemoteFaceLookup,
#     e::Int,
#     f::Int,
# ) where {T}

#     uM = @view trace[:, f, e, :]
#     fi = mesh.elements.faceinfo[f, e]

#     if fi.neighbor_local_elem == 0
#         return uM, nothing, :boundary
#     end

#     ne = fi.neighbor_local_elem
#     nrank = mesh.elements.owner_rank[ne]

#     if nrank == mesh.elements.owner_rank[e]
#         uP = @view trace[:, fi.neighbor_local_face, ne, :]
#         return uM, uP, :local
#     else
#         uP = get_mpi_plus_trace(ghost_traces, face_lookup, e, f)
#         return uM, uP, :mpi
#     end
# end

struct MPIFacePermutationMap
    # key: (local_elem, local_face) on the owned/minus side
    # value: permutation vector perm such that
    # local_plus[i,:] = received_neighbor[perm[i],:]
    lookup::Dict{Tuple{Int,Int}, Vector{Int}}
end

@inline function get_mpi_plus_trace_permuted(
    ghost_traces::Dict{Int, Array{T,3}},
    face_lookup::MPIRemoteFaceLookup,
    perm_maps::MPIFacePermutationMap,
    local_elem::Int,
    local_face::Int,
) where {T}
    rank, k = face_lookup.lookup[(local_elem, local_face)]
    uP_raw = @view ghost_traces[rank][:, k, :]   # (Nfp, Nvars)
    perm = perm_maps.lookup[(local_elem, local_face)]
    return @view uP_raw[perm, :]
end

function get_face_traces(
    mesh::DistributedMesh,
    trace::Array{T,4},
    ghost_traces::Dict{Int, Array{T,3}},
    face_lookup::MPIRemoteFaceLookup,
    perm_maps::MPIFacePermutationMap,
    e::Int,
    f::Int,
) where {T}

    uM = @view trace[:, f, e, :]
    fi = mesh.elements.faceinfo[f, e]

    if fi.neighbor_local_elem == 0
        return uM, nothing, :boundary
    end

    ne = fi.neighbor_local_elem
    nrank = mesh.elements.owner_rank[ne]

    if nrank == mesh.elements.owner_rank[e]
        uP = @view trace[:, fi.neighbor_local_face, ne, :]
        return uM, uP, :local
    else
        uP = get_mpi_plus_trace_permuted(
            ghost_traces,
            face_lookup,
            perm_maps,
            e,
            f,
        )
        return uM, uP, :mpi
    end
end


# ------------------------------------------------------------
# Helpers: normalize reference face nodes into barycentric form
# ------------------------------------------------------------

@inline function _as_nfp_by_dim(A::AbstractMatrix, Nfp::Int, dim::Int)
    if size(A, 1) == Nfp && size(A, 2) == dim
        return Matrix(A)
    elseif size(A, 1) == dim && size(A, 2) == Nfp
        return permutedims(Matrix(A))
    else
        throw(ArgumentError("face_nodes_reference must have shape (Nfp,$dim) or ($dim,Nfp)"))
    end
end

function face_nodes_to_barycentric(
    face_nodes_reference::AbstractMatrix,
    Nfp::Int;
    atol::Real = 1e-12,
)
    # Accept either:
    #   - barycentric coordinates (Nfp,3)/(3,Nfp)
    #   - 2D triangle coordinates (Nfp,2)/(2,Nfp)
    if size(face_nodes_reference, 1) == Nfp || size(face_nodes_reference, 2) == Nfp
        # Try 3D barycentric first
        try
            B = _as_nfp_by_dim(face_nodes_reference, Nfp, 3)
            return B
        catch
        end

        # Then 2D Cartesian on reference triangle:
        # vertex1=(0,0), vertex2=(1,0), vertex3=(0,1)
        try
            X = _as_nfp_by_dim(face_nodes_reference, Nfp, 2)
            B = Matrix{Float64}(undef, Nfp, 3)
            for i in 1:Nfp
                x = X[i, 1]
                y = X[i, 2]
                λ1 = 1 - x - y
                λ2 = x
                λ3 = y
                B[i, 1] = λ1
                B[i, 2] = λ2
                B[i, 3] = λ3
            end
            return B
        catch
        end
    end

    throw(ArgumentError(
        "Could not interpret face_nodes_reference. Expected barycentric (Nfp,3)/(3,Nfp) or 2D triangle coords (Nfp,2)/(2,Nfp)."
    ))
end

# ------------------------------------------------------------
# Permute barycentric coordinates according to orientation code
#
# TRI_PERMUTATIONS[code] = p
# where neighbor_face_nodes[p] == local_face_nodes
#
# For a local barycentric point λ_local, the same physical point
# in neighbor ordering has barycentrics μ_neighbor with:
#   μ[p[1]] = λ[1]
#   μ[p[2]] = λ[2]
#   μ[p[3]] = λ[3]
# ------------------------------------------------------------

@inline function permute_barycentric_to_neighbor(
    λ::NTuple{3,Float64},
    p::NTuple{3,Int},
)::NTuple{3,Float64}
    μ = zeros(Float64, 3)
    μ[p[1]] = λ[1]
    μ[p[2]] = λ[2]
    μ[p[3]] = λ[3]
    return (μ[1], μ[2], μ[3])
end

@inline function barycentric_close(
    a::NTuple{3,Float64},
    b::NTuple{3,Float64},
    atol::Float64,
)::Bool
    return abs(a[1] - b[1]) ≤ atol &&
           abs(a[2] - b[2]) ≤ atol &&
           abs(a[3] - b[3]) ≤ atol
end

# ------------------------------------------------------------
# Build permutation for one orientation code
#
# Returns perm with:
#   local_plus[i,:] = received_neighbor[perm[i],:]
# ------------------------------------------------------------

function build_face_permutation_for_orientation(
    bary_nodes::AbstractMatrix,
    orientation::Int;
    atol::Float64 = 1e-10,
)
    Nfp = size(bary_nodes, 1)
    1 ≤ orientation ≤ length(TRI_PERMUTATIONS) ||
        throw(ArgumentError("Invalid orientation code $orientation"))

    p = TRI_PERMUTATIONS[orientation]

    local_bary = Vector{NTuple{3,Float64}}(undef, Nfp)
    for i in 1:Nfp
        local_bary[i] = (
            Float64(bary_nodes[i, 1]),
            Float64(bary_nodes[i, 2]),
            Float64(bary_nodes[i, 3]),
        )
    end

    neighbor_bary = Vector{NTuple{3,Float64}}(undef, Nfp)
    for i in 1:Nfp
        neighbor_bary[i] = permute_barycentric_to_neighbor(local_bary[i], p)
    end

    perm = Vector{Int}(undef, Nfp)

    # For each local node i, find which neighbor node j corresponds
    # to the same physical point under the vertex permutation.
    for i in 1:Nfp
        target = neighbor_bary[i]
        found = 0
        for j in 1:Nfp
            if barycentric_close(local_bary[j], target, atol)
                found = j
                break
            end
        end
        if found == 0
            error("Could not build face permutation for orientation=$orientation at local face node $i. Check face_nodes_reference ordering/tolerance.")
        end
        perm[i] = found
    end

    return perm
end

# ------------------------------------------------------------
# Precompute the six orientation permutations once
# ------------------------------------------------------------

function build_all_orientation_permutations(
    Nfp::Int,
    face_nodes_reference::AbstractMatrix;
    atol::Float64 = 1e-10,
)
    bary_nodes = face_nodes_to_barycentric(face_nodes_reference, Nfp; atol=atol)

    perms = Dict{Int, Vector{Int}}()
    for ori in 1:length(TRI_PERMUTATIONS)
        perms[ori] = build_face_permutation_for_orientation(bary_nodes, ori; atol=atol)
    end
    return perms
end

# ------------------------------------------------------------
# Main user-facing builder
# ------------------------------------------------------------

function build_mpi_face_permutation_maps(
    mesh::DistributedMesh,
    Nfp::Int,
    face_nodes_reference::AbstractMatrix;
    atol::Float64 = 1e-10,
)
    orientation_perms = build_all_orientation_permutations(
        Nfp,
        face_nodes_reference;
        atol=atol,
    )

    lut = Dict{Tuple{Int,Int}, Vector{Int}}()

    for rank in mesh.mpi.neighbors
        comm = mesh.mpi.comms[rank]

        for sf in comm.send_faces
            # sf.orientation is the mapping from the owned/minus face
            # to the neighbor/plus face ordering.
            lut[(sf.local_elem, sf.local_face)] = orientation_perms[sf.orientation]
        end
    end

    return MPIFacePermutationMap(lut)
end

function get_face_traces(
    mesh::DistributedMesh,
    trace::Array{T,4},
    ghost_traces::Dict{Int, Array{T,3}},
    face_lookup::MPIRemoteFaceLookup,
    perm_maps::MPIFacePermutationMap,
    e::Int,
    f::Int,
) where {T}

    uM = @view trace[:, f, e, :]
    fi = mesh.elements.faceinfo[f, e]

    if fi.neighbor_local_elem == 0
        return uM, nothing, :boundary
    end

    ne = fi.neighbor_local_elem
    nrank = mesh.elements.owner_rank[ne]

    if nrank == mesh.elements.owner_rank[e]
        uP = @view trace[:, fi.neighbor_local_face, ne, :]
        return uM, uP, :local
    else
        uP = get_mpi_plus_trace_permuted(
            ghost_traces,
            face_lookup,
            perm_maps,
            e,
            f,
        )
        return uM, uP, :mpi
    end
end

end # module TraceMaps