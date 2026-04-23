module DistributedMesh3D

export ElementOwnership,
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


# ============================================================
# Ownership
# ============================================================

@enum ElementOwnership OWNED GHOST

# ============================================================
# Tetrahedron face convention
#
# Face 1: (1,3,2)
# Face 2: (1,2,4)
# Face 3: (1,4,3)
# Face 4: (2,3,4)
#
# This matches the convention discussed previously.
# ============================================================

# This is SeisSol ordering, which is a common convention in many codes. It is not the only possible convention, but it is a reasonable choice for Stage 1. The key point is
# const tet_face_local_vertices = (
#     (1, 3, 2),
#     (1, 2, 4),
#     (1, 4, 3),
#     (2, 3, 4),
# )

# This is Hesthaven \& Warburton ordering, which is more standard in DG contexts.
const tet_face_local_vertices = (
    (1,2,3),
    (1,2,4),
    (2,3,4),
    (1,3,4),
)

# ------------------------------------------------------------
# Helper: orientation of one triangular face ordering relative
# to another. Returns an integer in 1:6.
#
# Convention:
#   orientation = index of permutation p such that
#   nbr_face[p] == face
#
# This is enough for Stage 1. Later you can replace it with
# a more DG-specific permutation/sign convention if needed.
# ------------------------------------------------------------
const TRI_PERMUTATIONS = (
    (1, 2, 3),
    (1, 3, 2),
    (2, 1, 3),
    (2, 3, 1),
    (3, 1, 2),
    (3, 2, 1),
)

# ============================================================
# Core structs
# ============================================================

struct FaceNeighborInfo
    neighbor_local_elem::Int   # 0 if boundary
    neighbor_local_face::Int   # 0 if boundary
    orientation::Int           # 0 if boundary
    bc_tag::Int                # 0 interior, >0 boundary tag
end

struct LocalNodeGeometry{T}
    coords::Matrix{T}                  # (3, nlocal_nodes)
    global_ids::Vector{Int}            # local node -> global node
    global_to_local::Dict{Int, Int}    # global node -> local node
end

struct LocalElementTopology
    vertices::Matrix{Int}              # (4, nlocal_elems), local node ids
    global_ids::Vector{Int}            # local elem -> global elem
    global_to_local::Dict{Int, Int}    # global elem -> local elem
    owner_rank::Vector{Int}            # owner rank of each local elem
    ownership::Vector{ElementOwnership}
    material_id::Vector{Int}           # zone/material per element
    faceinfo::Matrix{FaceNeighborInfo} # (4, nlocal_elems)
end

struct ElementPartition
    owned::Vector{Int}                 # local ids of owned elems
    ghosts::Vector{Int}                # local ids of ghost elems
end

struct InterfaceFace
    local_elem::Int
    local_face::Int
    neighbor_rank::Int
    neighbor_global_elem::Int
    neighbor_face::Int
    orientation::Int
    face_nodes_local::NTuple{3, Int}
    face_nodes_global::NTuple{3, Int}
end

struct NeighborComm
    rank::Int
    send_faces::Vector{InterfaceFace}
    recv_faces::Vector{InterfaceFace}
end

struct MPICommunicationTable
    neighbors::Vector{Int}
    comms::Dict{Int, NeighborComm}
end

struct DistributedMesh{T}
    dim::Int
    elemtype::Symbol
    nodes::LocalNodeGeometry{T}
    elements::LocalElementTopology
    partition::ElementPartition
    volumes::Vector{T}
    barycenters::Matrix{T}             # (3, nlocal_elems)
    mpi::MPICommunicationTable
end

# ============================================================
# Internal helper structs
# ============================================================

struct FaceRecord
    local_elem::Int
    local_face::Int
    global_elem::Int
    owner_rank::Int
    face_nodes_global::NTuple{3, Int}  # unsorted, local face ordering
end

# ============================================================
# Basic tetra geometry
# ============================================================

@inline function tetra_volume_and_barycenter(
    x1::NTuple{3, T},
    x2::NTuple{3, T},
    x3::NTuple{3, T},
    x4::NTuple{3, T},
) where {T<:Real}
    ax, ay, az = x2[1] - x1[1], x2[2] - x1[2], x2[3] - x1[3]
    bx, by, bz = x3[1] - x1[1], x3[2] - x1[2], x3[3] - x1[3]
    cx, cy, cz = x4[1] - x1[1], x4[2] - x1[2], x4[3] - x1[3]

    detJ = ax * (by * cz - bz * cy) -
           ay * (bx * cz - bz * cx) +
           az * (bx * cy - by * cx)

    vol = abs(detJ) / 6
    bary = (
        (x1[1] + x2[1] + x3[1] + x4[1]) / 4,
        (x1[2] + x2[2] + x3[2] + x4[2]) / 4,
        (x1[3] + x2[3] + x3[3] + x4[3]) / 4,
    )

    return vol, bary
end

# ============================================================
# Face helpers
# ============================================================

@inline function sorted_face_key(a::Int, b::Int, c::Int)
    if a > b
        a, b = b, a
    end
    if b > c
        b, c = c, b
    end
    if a > b
        a, b = b, a
    end
    return (a, b, c)
end

@inline function element_face_global_nodes(
    elem_global_vertices::NTuple{4, Int},
    face::Int,
)::NTuple{3, Int}
    lv = tet_face_local_vertices[face]
    return (
        elem_global_vertices[lv[1]],
        elem_global_vertices[lv[2]],
        elem_global_vertices[lv[3]],
    )
end

@inline function element_face_local_nodes(
    elem_local_vertices::NTuple{4, Int},
    face::Int,
)::NTuple{3, Int}
    lv = tet_face_local_vertices[face]
    return (
        elem_local_vertices[lv[1]],
        elem_local_vertices[lv[2]],
        elem_local_vertices[lv[3]],
    )
end

function triangle_orientation_code(
    face_nodes::NTuple{3, Int},
    neighbor_face_nodes::NTuple{3, Int},
)::Int
    # Returns which permutation of neighbor_face_nodes reproduces face_nodes
    # after matching vertex identities.
    for (icode, p) in enumerate(TRI_PERMUTATIONS)
        if (
            neighbor_face_nodes[p[1]] == face_nodes[1] &&
            neighbor_face_nodes[p[2]] == face_nodes[2] &&
            neighbor_face_nodes[p[3]] == face_nodes[3]
        )
            return icode
        end
    end
    error("Invalid triangle orientation: faces do not contain the same three global nodes.")
end

# ============================================================
# Set builders
# ============================================================

function build_owned_global_elements(elem_to_rank::AbstractVector{Int}, myrank::Int)
    return findall(==(myrank), elem_to_rank)
end

function build_face_to_global_elements(
    global_elem_vertices::AbstractMatrix{Int},
)
    ntotelem = size(global_elem_vertices, 2)
    face_to_elems = Dict{NTuple{3, Int}, Vector{Int}}()

    for ge in 1:ntotelem
        verts = (
            global_elem_vertices[1, ge],
            global_elem_vertices[2, ge],
            global_elem_vertices[3, ge],
            global_elem_vertices[4, ge],
        )
        for f in 1:4
            fg = element_face_global_nodes(verts, f)
            key = sorted_face_key(fg...)
            push!(get!(face_to_elems, key, Int[]), ge)
        end
    end

    return face_to_elems
end

function build_ghost_global_elements_face_halo(
    owned_global::AbstractVector{Int},
    global_elem_vertices::AbstractMatrix{Int},
    face_to_elems::Dict{NTuple{3, Int}, Vector{Int}},
)
    owned_set = Set(owned_global)
    ghost_set = Set{Int}()

    for ge in owned_global
        verts = (
            global_elem_vertices[1, ge],
            global_elem_vertices[2, ge],
            global_elem_vertices[3, ge],
            global_elem_vertices[4, ge],
        )
        for f in 1:4
            fg = element_face_global_nodes(verts, f)
            key = sorted_face_key(fg...)
            neighbors = face_to_elems[key]
            for nge in neighbors
                if nge != ge && !(nge in owned_set)
                    push!(ghost_set, nge)
                end
            end
        end
    end

    return sort!(collect(ghost_set))
end

# ============================================================
# Local node / element maps
# ============================================================

function build_local_node_geometry(
    global_node_coords::AbstractMatrix{T},
    global_elem_vertices::AbstractMatrix{Int},
    stored_global_elems::AbstractVector{Int},
) where {T<:Real}
    node_set = Set{Int}()

    for ge in stored_global_elems
        for k in 1:4
            push!(node_set, global_elem_vertices[k, ge])
        end
    end

    local_node_globals = sort!(collect(node_set))
    nlocal_nodes = length(local_node_globals)

    global_to_local = Dict{Int, Int}()
    for (ln, gn) in enumerate(local_node_globals)
        global_to_local[gn] = ln
    end

    coords = Matrix{T}(undef, 3, nlocal_nodes)
    for (ln, gn) in enumerate(local_node_globals)
        coords[:, ln] .= global_node_coords[:, gn]
    end

    return LocalNodeGeometry{T}(coords, local_node_globals, global_to_local)
end

function build_local_element_topology(
    global_elem_vertices::AbstractMatrix{Int},
    elem_to_rank::AbstractVector{Int},
    material_id_global::AbstractVector{Int},
    stored_global_elems::AbstractVector{Int},
    owned_global_set::Set{Int},
    nodes::LocalNodeGeometry,
)
    nlocal_elems = length(stored_global_elems)

    global_to_local = Dict{Int, Int}()
    for (le, ge) in enumerate(stored_global_elems)
        global_to_local[ge] = le
    end

    vertices = Matrix{Int}(undef, 4, nlocal_elems)
    owner_rank = Vector{Int}(undef, nlocal_elems)
    ownership = Vector{ElementOwnership}(undef, nlocal_elems)
    material_id = Vector{Int}(undef, nlocal_elems)

    default_face = FaceNeighborInfo(0, 0, 0, 0)
    faceinfo = fill(default_face, 4, nlocal_elems)

    for (le, ge) in enumerate(stored_global_elems)
        for k in 1:4
            gnode = global_elem_vertices[k, ge]
            vertices[k, le] = nodes.global_to_local[gnode]
        end
        owner_rank[le] = elem_to_rank[ge]
        ownership[le] = (ge in owned_global_set) ? OWNED : GHOST
        material_id[le] = material_id_global[ge]
    end

    return LocalElementTopology(
        vertices,
        collect(stored_global_elems),
        global_to_local,
        owner_rank,
        ownership,
        material_id,
        faceinfo,
    )
end

function build_partition(elements::LocalElementTopology)
    owned = Int[]
    ghosts = Int[]
    for le in eachindex(elements.global_ids)
        if elements.ownership[le] == OWNED
            push!(owned, le)
        else
            push!(ghosts, le)
        end
    end
    return ElementPartition(owned, ghosts)
end

# ============================================================
# Face dictionary over local stored mesh
# ============================================================

function build_local_face_records(
    elements::LocalElementTopology,
    global_elem_vertices::AbstractMatrix{Int},
)
    face_dict = Dict{NTuple{3, Int}, Vector{FaceRecord}}()

    nlocal_elems = length(elements.global_ids)

    for le in 1:nlocal_elems
        ge = elements.global_ids[le]
        gverts = (
            global_elem_vertices[1, ge],
            global_elem_vertices[2, ge],
            global_elem_vertices[3, ge],
            global_elem_vertices[4, ge],
        )

        for f in 1:4
            face_nodes_global = element_face_global_nodes(gverts, f)
            key = sorted_face_key(face_nodes_global...)
            rec = FaceRecord(
                le,
                f,
                ge,
                elements.owner_rank[le],
                face_nodes_global,
            )
            push!(get!(face_dict, key, FaceRecord[]), rec)
        end
    end

    return face_dict
end

# ============================================================
# Neighbor construction
# ============================================================

function build_local_face_neighbors!(
    elements::LocalElementTopology,
    global_elem_vertices::AbstractMatrix{Int};
    boundary_face_tags::Dict{NTuple{3, Int}, Int}=Dict{NTuple{3, Int}, Int}(),
)
    face_dict = build_local_face_records(elements, global_elem_vertices)

    for (key, records) in face_dict
        nrec = length(records)

        if nrec == 1
            # Boundary face
            r = records[1]
            bc = get(boundary_face_tags, key, 0)
            elements.faceinfo[r.local_face, r.local_elem] = FaceNeighborInfo(0, 0, 0, bc)

        elseif nrec == 2
            r1, r2 = records[1], records[2]

            ori12 = triangle_orientation_code(r1.face_nodes_global, r2.face_nodes_global)
            ori21 = triangle_orientation_code(r2.face_nodes_global, r1.face_nodes_global)

            elements.faceinfo[r1.local_face, r1.local_elem] = FaceNeighborInfo(
                r2.local_elem,
                r2.local_face,
                ori12,
                0,
            )
            elements.faceinfo[r2.local_face, r2.local_elem] = FaceNeighborInfo(
                r1.local_elem,
                r1.local_face,
                ori21,
                0,
            )

        else
            error("Non-manifold face detected for face key $key with $nrec adjacent tetrahedra.")
        end
    end

    return nothing
end

# ============================================================
# MPI interface construction
# ============================================================

function build_mpi_interface_table(
    elements::LocalElementTopology,
    myrank::Int,
)
    comms = Dict{Int, NeighborComm}()

    nlocal_elems = length(elements.global_ids)

    for le in 1:nlocal_elems
        if elements.ownership[le] != OWNED
            continue
        end

        ge = elements.global_ids[le]
        local_verts = (
            elements.vertices[1, le],
            elements.vertices[2, le],
            elements.vertices[3, le],
            elements.vertices[4, le],
        )

        for lf in 1:4
            fi = elements.faceinfo[lf, le]
            ne = fi.neighbor_local_elem

            if ne == 0
                continue
            end

            nrank = elements.owner_rank[ne]
            if nrank == myrank
                continue
            end

            nge = elements.global_ids[ne]
            nlf = fi.neighbor_local_face
            ori = fi.orientation

            nbr_local_verts = (
                elements.vertices[1, ne],
                elements.vertices[2, ne],
                elements.vertices[3, ne],
                elements.vertices[4, ne],
            )

            iface = InterfaceFace(
                le,
                lf,
                nrank,
                nge,
                nlf,
                ori,
                element_face_local_nodes(local_verts, lf),
                # global node ids are recovered indirectly later if needed,
                # but for Stage 1 let's reconstruct from local-global node map externally.
                # Here we temporarily store local face nodes projected through global ids below.
                # This field is overwritten in the post-processing loop.
                (0, 0, 0),
            )

            if !haskey(comms, nrank)
                comms[nrank] = NeighborComm(nrank, InterfaceFace[], InterfaceFace[])
            end

            push!(comms[nrank].send_faces, iface)

            # For Stage 1, we also record the owned-side view in recv_faces
            # as a placeholder description of interfaces involving ghost elems.
            ghost_iface = InterfaceFace(
                ne,
                nlf,
                myrank,
                ge,
                lf,
                triangle_orientation_code(
                    element_face_local_nodes(nbr_local_verts, nlf),
                    element_face_local_nodes(local_verts, lf),
                ),
                element_face_local_nodes(nbr_local_verts, nlf),
                (0, 0, 0),
            )
            push!(comms[nrank].recv_faces, ghost_iface)
        end
    end

    neighbors = sort!(collect(keys(comms)))
    return MPICommunicationTable(neighbors, comms)
end

function attach_global_face_nodes!(
    mpi::MPICommunicationTable,
    nodes::LocalNodeGeometry,
)
    for rank in mpi.neighbors
        comm = mpi.comms[rank]

        for i in eachindex(comm.send_faces)
            f = comm.send_faces[i]
            fg = (
                nodes.global_ids[f.face_nodes_local[1]],
                nodes.global_ids[f.face_nodes_local[2]],
                nodes.global_ids[f.face_nodes_local[3]],
            )
            comm.send_faces[i] = InterfaceFace(
                f.local_elem,
                f.local_face,
                f.neighbor_rank,
                f.neighbor_global_elem,
                f.neighbor_face,
                f.orientation,
                f.face_nodes_local,
                fg,
            )
        end

        for i in eachindex(comm.recv_faces)
            f = comm.recv_faces[i]
            fg = (
                nodes.global_ids[f.face_nodes_local[1]],
                nodes.global_ids[f.face_nodes_local[2]],
                nodes.global_ids[f.face_nodes_local[3]],
            )
            comm.recv_faces[i] = InterfaceFace(
                f.local_elem,
                f.local_face,
                f.neighbor_rank,
                f.neighbor_global_elem,
                f.neighbor_face,
                f.orientation,
                f.face_nodes_local,
                fg,
            )
        end
    end

    return nothing
end

# ============================================================
# Geometry
# ============================================================

function compute_element_geometry(
    nodes::LocalNodeGeometry{T},
    elements::LocalElementTopology,
) where {T<:Real}
    nlocal_elems = length(elements.global_ids)
    volumes = Vector{T}(undef, nlocal_elems)
    barycenters = Matrix{T}(undef, 3, nlocal_elems)

    for le in 1:nlocal_elems
        n1 = elements.vertices[1, le]
        n2 = elements.vertices[2, le]
        n3 = elements.vertices[3, le]
        n4 = elements.vertices[4, le]

        x1 = (nodes.coords[1, n1], nodes.coords[2, n1], nodes.coords[3, n1])
        x2 = (nodes.coords[1, n2], nodes.coords[2, n2], nodes.coords[3, n2])
        x3 = (nodes.coords[1, n3], nodes.coords[2, n3], nodes.coords[3, n3])
        x4 = (nodes.coords[1, n4], nodes.coords[2, n4], nodes.coords[3, n4])

        vol, bary = tetra_volume_and_barycenter(x1, x2, x3, x4)
        volumes[le] = vol
        barycenters[:, le] .= bary
    end

    return volumes, barycenters
end

# ============================================================
# Main constructor
# ============================================================

"""
    build_distributed_mesh(
        global_node_coords,
        global_elem_vertices,
        elem_to_rank,
        myrank;
        material_id_global = ones(Int, ntotelem),
        boundary_face_tags = Dict{NTuple{3,Int},Int}(),
    )

Build a Stage-1 distributed tetrahedral mesh for rank `myrank`.

Inputs
------
- `global_node_coords :: AbstractMatrix{T}` with size `(3, ntotnode)`
- `global_elem_vertices :: AbstractMatrix{Int}` with size `(4, ntotelem)`
- `elem_to_rank :: AbstractVector{Int}` of length `ntotelem`
- `myrank :: Int`

Keyword arguments
-----------------
- `material_id_global`: global material/zone id per element
- `boundary_face_tags`: dictionary mapping sorted global face node triples
  to boundary condition ids

Returns
-------
- `DistributedMesh{T}`

Stage 1 includes:
- owned elements
- face-halo ghost elements
- local/global node and element maps
- local face neighbors
- MPI interface faces
"""
function build_distributed_mesh(
    global_node_coords::AbstractMatrix{T},
    global_elem_vertices::AbstractMatrix{Int},
    elem_to_rank::AbstractVector{Int},
    myrank::Int;
    material_id_global::AbstractVector{Int}=ones(Int, size(global_elem_vertices, 2)),
    boundary_face_tags::Dict{NTuple{3, Int}, Int}=Dict{NTuple{3, Int}, Int}(),
) where {T<:Real}

    # ---------------------------
    # Basic checks
    # ---------------------------
    size(global_node_coords, 1) == 3 ||
        throw(ArgumentError("global_node_coords must have size (3, ntotnode)"))

    size(global_elem_vertices, 1) == 4 ||
        throw(ArgumentError("global_elem_vertices must have size (4, ntotelem) for tetrahedra"))

    ntotelem = size(global_elem_vertices, 2)
    length(elem_to_rank) == ntotelem ||
        throw(ArgumentError("elem_to_rank must have length ntotelem"))

    length(material_id_global) == ntotelem ||
        throw(ArgumentError("material_id_global must have length ntotelem"))

    # ---------------------------
    # Phase 1: owned + ghost globals
    # ---------------------------
    owned_global = build_owned_global_elements(elem_to_rank, myrank)
    owned_global_set = Set(owned_global)

    face_to_elems = build_face_to_global_elements(global_elem_vertices)

    ghost_global = build_ghost_global_elements_face_halo(
        owned_global,
        global_elem_vertices,
        face_to_elems,
    )

    stored_global_elems = vcat(owned_global, ghost_global)

    # ---------------------------
    # Phase 2: local nodes
    # ---------------------------
    nodes = build_local_node_geometry(
        global_node_coords,
        global_elem_vertices,
        stored_global_elems,
    )

    # ---------------------------
    # Phase 3: local elements
    # ---------------------------
    elements = build_local_element_topology(
        global_elem_vertices,
        elem_to_rank,
        material_id_global,
        stored_global_elems,
        owned_global_set,
        nodes,
    )

    partition = build_partition(elements)

    # ---------------------------
    # Phase 4: neighbors
    # ---------------------------
    build_local_face_neighbors!(
        elements,
        global_elem_vertices;
        boundary_face_tags=boundary_face_tags,
    )

    # ---------------------------
    # Phase 5: MPI interfaces
    # ---------------------------
    mpi = build_mpi_interface_table(elements, myrank)
    attach_global_face_nodes!(mpi, nodes)

    # ---------------------------
    # Phase 6: geometry
    # ---------------------------
    volumes, barycenters = compute_element_geometry(nodes, elements)

    return DistributedMesh(
        3,
        :tet,
        nodes,
        elements,
        partition,
        volumes,
        barycenters,
        mpi,
    )
end

end # module