struct DistributedDGInterfaceFace
    face_key::NTuple{3, Int}
    neighbor_rank::Int
    owned_elem::Int
    owned_face::Int
    owned_nodes::Vector{Int}
    ghost_elem::Int
    ghost_face::Int
    ghost_nodes::Vector{Int}
end

struct DistributedMaxwellExchange
    neighbors::Vector{Int}
    faces::Dict{Int, Vector{DistributedDGInterfaceFace}}
    send_buffers::Dict{Int, Vector{Float64}}
    recv_buffers::Dict{Int, Vector{Float64}}
end

struct DistributedDGDiscretization{
    B<:AbstractBackend,
    T<:Real,
}
    distributed_mesh::DistributedMesh3D.DistributedMesh{T}
    dg::DGDiscretization{B}
    exchange::DistributedMaxwellExchange
    comm::MPI.Comm
end

function global_boundary_face_tags(
    mesh::RawVTUMesh,
    boundary_tag_name::String,
)
    tags = tri_data(mesh, boundary_tag_name)
    boundary_faces = Dict{NTuple{3, Int}, Int}()

    for i in axes(mesh.tris, 2)
        nodes = (
            mesh.tris[1, i],
            mesh.tris[2, i],
            mesh.tris[3, i],
        )
        boundary_faces[sorted_face_key(nodes)] = Int(tags[i])
    end

    return boundary_faces
end

function local_boundary_face_tags(
    mesh::DistributedMesh3D.DistributedMesh,
)
    boundary_faces = Dict{NTuple{3, Int}, Int}()

    for elem in axes(mesh.elements.vertices, 2)
        local_vertices = (
            mesh.elements.vertices[1, elem],
            mesh.elements.vertices[2, elem],
            mesh.elements.vertices[3, elem],
            mesh.elements.vertices[4, elem],
        )

        for face in 1:4
            face_info = mesh.elements.faceinfo[face, elem]
            face_info.neighbor_local_elem == 0 || continue

            local_nodes = DistributedMesh3D.element_face_local_nodes(
                local_vertices,
                face,
            )
            global_nodes = (
                mesh.nodes.global_ids[local_nodes[1]],
                mesh.nodes.global_ids[local_nodes[2]],
                mesh.nodes.global_ids[local_nodes[3]],
            )
            boundary_faces[sorted_face_key(global_nodes)] = face_info.bc_tag
        end
    end

    return boundary_faces
end

function local_mesh_from_distributed_mesh(
    mesh::DistributedMesh3D.DistributedMesh,
    boundary_face_tags::Dict{NTuple{3, Int}, Int};
    boundary_tag_name::String = "boundary_id",
)
    face_records = Dict{
        NTuple{3, Int},
        Vector{NTuple{3, Int}},
    }()

    for elem in axes(mesh.elements.vertices, 2)
        for face in 1:4
            local_vertices = TET_FACES[face]
            local_nodes = (
                mesh.elements.vertices[local_vertices[1], elem],
                mesh.elements.vertices[local_vertices[2], elem],
                mesh.elements.vertices[local_vertices[3], elem],
            )
            global_nodes = (
                mesh.nodes.global_ids[local_nodes[1]],
                mesh.nodes.global_ids[local_nodes[2]],
                mesh.nodes.global_ids[local_nodes[3]],
            )
            key = sorted_face_key(global_nodes)
            push!(get!(face_records, key, NTuple{3, Int}[]), local_nodes)
        end
    end

    boundary_tris = NTuple{3, Int}[]
    boundary_ids = Int[]

    for (key, records) in face_records
        if length(records) == 1
            push!(boundary_tris, only(records))
            push!(boundary_ids, get(boundary_face_tags, key, 0))
        elseif length(records) != 2
            error(
                "Non-manifold local face $key has $(length(records)) adjacent elements.",
            )
        end
    end

    tris = isempty(boundary_tris) ?
           Matrix{Int}(undef, 3, 0) :
           reduce(hcat, collect.(boundary_tris))

    ntets = size(mesh.elements.vertices, 2)
    ntris = size(tris, 2)
    tet_cell_ids = collect(1:ntets)
    tri_cell_ids = collect((ntets + 1):(ntets + ntris))

    cell_boundary_ids = zeros(Int, ntets + ntris)
    cell_boundary_ids[tri_cell_ids] .= boundary_ids

    return RawVTUMesh(
        Matrix{Float64}(mesh.nodes.coords),
        copy(mesh.elements.vertices),
        tris,
        tet_cell_ids,
        tri_cell_ids,
        Dict{String, Any}(boundary_tag_name => cell_boundary_ids),
    )
end

function solver_face_key(
    mesh::DistributedMesh3D.DistributedMesh,
    elem::Int,
    face::Int,
)
    local_vertices = TET_FACES[face]
    local_nodes = (
        mesh.elements.vertices[local_vertices[1], elem],
        mesh.elements.vertices[local_vertices[2], elem],
        mesh.elements.vertices[local_vertices[3], elem],
    )
    global_nodes = (
        mesh.nodes.global_ids[local_nodes[1]],
        mesh.nodes.global_ids[local_nodes[2]],
        mesh.nodes.global_ids[local_nodes[3]],
    )
    return sorted_face_key(global_nodes)
end

function find_solver_face(
    mesh::DistributedMesh3D.DistributedMesh,
    elem::Int,
    face_key::NTuple{3, Int},
)
    for face in 1:4
        if solver_face_key(mesh, elem, face) == face_key
            return face
        end
    end

    error(
        "Could not find solver face $face_key on local element $elem " *
        "(global element $(mesh.elements.global_ids[elem])).",
    )
end

function build_distributed_maxwell_exchange(
    mesh::DistributedMesh3D.DistributedMesh,
    dg::DGDiscretization,
)
    faces = Dict{Int, Vector{DistributedDGInterfaceFace}}()
    send_buffers = Dict{Int, Vector{Float64}}()
    recv_buffers = Dict{Int, Vector{Float64}}()
    nfp = length(dg.fops.face_nodes[1])
    ncomponents = 6

    for neighbor in mesh.mpi.neighbors
        neighbor_faces = DistributedDGInterfaceFace[]

        for interface in mesh.mpi.comms[neighbor].send_faces
            key = sorted_face_key(interface.face_nodes_global)
            owned_elem = interface.local_elem
            ghost_elem =
                mesh.elements.global_to_local[interface.neighbor_global_elem]

            owned_face = find_solver_face(mesh, owned_elem, key)
            ghost_face = find_solver_face(mesh, ghost_elem, key)

            push!(
                neighbor_faces,
                DistributedDGInterfaceFace(
                    key,
                    neighbor,
                    owned_elem,
                    owned_face,
                    copy(dg.fops.face_nodes[owned_face]),
                    ghost_elem,
                    ghost_face,
                    copy(dg.fops.face_nodes[ghost_face]),
                ),
            )
        end

        sort!(neighbor_faces; by = face -> face.face_key)
        faces[neighbor] = neighbor_faces

        nvalues = length(neighbor_faces) * ncomponents * nfp
        send_buffers[neighbor] = Vector{Float64}(undef, nvalues)
        recv_buffers[neighbor] = Vector{Float64}(undef, nvalues)
    end

    return DistributedMaxwellExchange(
        copy(mesh.mpi.neighbors),
        faces,
        send_buffers,
        recv_buffers,
    )
end

function build_distributed_dg_from_local_mesh(
    distributed_mesh::DistributedMesh3D.DistributedMesh,
    order::Int;
    boundary_tag_name::String = "boundary_id",
    boundary_face_tags::Dict{NTuple{3, Int}, Int} =
        local_boundary_face_tags(distributed_mesh),
    trace_tol::Float64 = 1e-10,
    backend::AbstractBackend = SerialBackend(),
    comm::MPI.Comm = MPI.COMM_WORLD,
)
    local_mesh = local_mesh_from_distributed_mesh(
        distributed_mesh,
        boundary_face_tags;
        boundary_tag_name = boundary_tag_name,
    )

    dg = DGDiscretization(
        local_mesh,
        order;
        boundary_tag_name = boundary_tag_name,
        trace_tol = trace_tol,
        backend = backend,
    )

    exchange = build_distributed_maxwell_exchange(distributed_mesh, dg)

    return DistributedDGDiscretization(
        distributed_mesh,
        dg,
        exchange,
        comm,
    )
end

function build_distributed_dg(
    global_mesh::RawVTUMesh,
    elem_to_rank::AbstractVector{Int},
    order::Int;
    comm::MPI.Comm = MPI.COMM_WORLD,
    boundary_tag_name::String = "boundary_id",
    material_tag_name::Union{Nothing, String} = nothing,
    trace_tol::Float64 = 1e-10,
    backend::AbstractBackend = SerialBackend(),
)
    MPI.Initialized() ||
        throw(ArgumentError("MPI.Init() must be called before build_distributed_dg."))

    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    length(elem_to_rank) == size(global_mesh.tets, 2) ||
        throw(ArgumentError("elem_to_rank must contain one entry per tetrahedron."))

    all(part -> 0 <= part < nranks, elem_to_rank) ||
        throw(
            ArgumentError(
                "elem_to_rank entries must use zero-based MPI ranks in 0:$(nranks - 1).",
            ),
        )

    boundary_face_tags =
        global_boundary_face_tags(global_mesh, boundary_tag_name)

    material_ids = if material_tag_name === nothing
        ones(Int, size(global_mesh.tets, 2))
    else
        Int.(tet_data(global_mesh, material_tag_name))
    end

    distributed_mesh = build_distributed_mesh(
        global_mesh.points,
        global_mesh.tets,
        elem_to_rank,
        rank;
        material_id_global = material_ids,
        boundary_face_tags = boundary_face_tags,
    )

    return build_distributed_dg_from_local_mesh(
        distributed_mesh,
        order;
        boundary_tag_name = boundary_tag_name,
        boundary_face_tags = boundary_face_tags,
        trace_tol = trace_tol,
        backend = backend,
        comm = comm,
    )
end

const DISTRIBUTED_MESH_TRANSFER_TAG = 7301

function root_mesh_distribution_data(
    global_mesh::RawVTUMesh,
    elem_to_rank::AbstractVector{Int},
    nranks::Int;
    boundary_tag_name::String,
    material_tag_name::Union{Nothing, String},
)
    length(elem_to_rank) == size(global_mesh.tets, 2) ||
        throw(ArgumentError("elem_to_rank must contain one entry per tetrahedron."))

    all(part -> 0 <= part < nranks, elem_to_rank) ||
        throw(
            ArgumentError(
                "elem_to_rank entries must use zero-based MPI ranks in 0:$(nranks - 1).",
            ),
        )

    boundary_face_tags =
        global_boundary_face_tags(global_mesh, boundary_tag_name)
    material_ids = if material_tag_name === nothing
        ones(Int, size(global_mesh.tets, 2))
    else
        Int.(tet_data(global_mesh, material_tag_name))
    end
    face_to_elems =
        DistributedMesh3D.build_face_to_global_elements(global_mesh.tets)

    return boundary_face_tags, material_ids, face_to_elems
end

function distribute_mesh_from_root(
    global_mesh::Union{Nothing, RawVTUMesh},
    elem_to_rank::Union{Nothing, AbstractVector{Int}};
    comm::MPI.Comm,
    root::Int,
    boundary_tag_name::String,
    material_tag_name::Union{Nothing, String},
)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    0 <= root < nranks ||
        throw(ArgumentError("root must be an MPI rank in 0:$(nranks - 1)."))

    distribution_data = nothing
    validation_error = nothing

    if rank == root
        try
            global_mesh === nothing &&
                throw(ArgumentError("The root rank must provide global_mesh."))
            elem_to_rank === nothing &&
                throw(ArgumentError("The root rank must provide elem_to_rank."))
            distribution_data = root_mesh_distribution_data(
                global_mesh,
                elem_to_rank,
                nranks;
                boundary_tag_name = boundary_tag_name,
                material_tag_name = material_tag_name,
            )
        catch error
            validation_error = sprint(showerror, error)
        end
    end

    validation_error = MPI.bcast(validation_error, comm; root = root)
    validation_error === nothing ||
        throw(ArgumentError("Root mesh validation failed: $validation_error"))

    local_packet = nothing
    distribution_error = nothing

    if rank == root
        boundary_face_tags, material_ids, face_to_elems = distribution_data

        for target_rank in 0:(nranks - 1)
            packet = if distribution_error === nothing
                try
                    distributed_mesh = build_distributed_mesh(
                        global_mesh.points,
                        global_mesh.tets,
                        elem_to_rank,
                        target_rank;
                        material_id_global = material_ids,
                        boundary_face_tags = boundary_face_tags,
                        face_to_elems_global = face_to_elems,
                    )
                    (
                        mesh = distributed_mesh,
                        error = nothing,
                    )
                catch error
                    distribution_error = sprint(showerror, error)
                    (
                        mesh = nothing,
                        error = distribution_error,
                    )
                end
            else
                (
                    mesh = nothing,
                    error = distribution_error,
                )
            end

            if target_rank == root
                local_packet = packet
            else
                MPI.send(
                    packet,
                    comm;
                    dest = target_rank,
                    tag = DISTRIBUTED_MESH_TRANSFER_TAG,
                )
            end
        end
    else
        local_packet = MPI.recv(
            comm;
            source = root,
            tag = DISTRIBUTED_MESH_TRANSFER_TAG,
        )
    end

    local_success = local_packet.error === nothing ? 1 : 0
    all_succeeded = MPI.Allreduce(local_success, min, comm) == 1

    if !all_succeeded
        distribution_error = MPI.bcast(
            rank == root ? distribution_error : nothing,
            comm;
            root = root,
        )
        error("Rank-local mesh distribution failed: $distribution_error")
    end

    return local_packet.mesh
end

"""
    build_distributed_dg_from_root(global_mesh, elem_to_rank, order; root=0, ...)

Build a distributed DG discretization while keeping the global mesh and
partition vector only on `root`. Non-root ranks should pass `nothing` for both
arguments. Each rank receives only its owned elements, one face halo, local
coordinates, material ids, boundary tags, and communication metadata.
"""
function build_distributed_dg_from_root(
    global_mesh::Union{Nothing, RawVTUMesh},
    elem_to_rank::Union{Nothing, AbstractVector{Int}},
    order::Int;
    comm::MPI.Comm = MPI.COMM_WORLD,
    root::Int = 0,
    boundary_tag_name::String = "boundary_id",
    material_tag_name::Union{Nothing, String} = nothing,
    trace_tol::Float64 = 1e-10,
    backend::AbstractBackend = SerialBackend(),
)
    MPI.Initialized() ||
        throw(
            ArgumentError(
                "MPI.Init() must be called before build_distributed_dg_from_root.",
            ),
        )

    distributed_mesh = distribute_mesh_from_root(
        global_mesh,
        elem_to_rank;
        comm = comm,
        root = root,
        boundary_tag_name = boundary_tag_name,
        material_tag_name = material_tag_name,
    )

    return build_distributed_dg_from_local_mesh(
        distributed_mesh,
        order;
        boundary_tag_name = boundary_tag_name,
        trace_tol = trace_tol,
        backend = backend,
        comm = comm,
    )
end

function build_distributed_dg(
    global_mesh::RawVTUMesh,
    epart_path::AbstractString,
    order::Int;
    one_based_parts::Bool = false,
    kwargs...,
)
    elem_to_rank = read_metis_epart(
        epart_path;
        one_based_parts = one_based_parts,
    )
    return build_distributed_dg(
        global_mesh,
        elem_to_rank,
        order;
        kwargs...,
    )
end

"""
    build_distributed_dg_from_root(mesh_path, epart_path, order; root=0, ...)

Read a VTU mesh and METIS element partition only on `root`, then distribute
rank-local owned-plus-halo mesh packages.
"""
function build_distributed_dg_from_root(
    mesh_path::AbstractString,
    epart_path::AbstractString,
    order::Int;
    comm::MPI.Comm = MPI.COMM_WORLD,
    root::Int = 0,
    one_based_parts::Bool = false,
    kwargs...,
)
    MPI.Initialized() ||
        throw(
            ArgumentError(
                "MPI.Init() must be called before build_distributed_dg_from_root.",
            ),
        )

    rank = MPI.Comm_rank(comm)
    global_mesh = nothing
    elem_to_rank = nothing
    load_error = nothing

    if rank == root
        try
            global_mesh = read_vtu_mesh(mesh_path)
            elem_to_rank = read_metis_epart(
                epart_path;
                one_based_parts = one_based_parts,
            )
        catch error
            load_error = sprint(showerror, error)
        end
    end

    load_error = MPI.bcast(load_error, comm; root = root)
    load_error === nothing ||
        error("Root mesh loading failed: $load_error")

    return build_distributed_dg_from_root(
        global_mesh,
        elem_to_rank,
        order;
        comm = comm,
        root = root,
        kwargs...,
    )
end

function interpolate_maxwell_field(
    distributed_dg::DistributedDGDiscretization,
    Efun::Function,
    Hfun::Function,
)
    return interpolate_maxwell_field(
        distributed_dg.dg.mesh,
        distributed_dg.dg.ref,
        Efun,
        Hfun,
    )
end

function localize_maxwell_field(
    global_field::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
)
    global_ids = distributed_dg.distributed_mesh.elements.global_ids
    nelements = size(global_field.Ex, 2)

    all(global_elem -> 1 <= global_elem <= nelements, global_ids) ||
        throw(ArgumentError("Distributed mesh contains an invalid global element id."))

    return MaxwellField(
        copy(global_field.Ex[:, global_ids]),
        copy(global_field.Ey[:, global_ids]),
        copy(global_field.Ez[:, global_ids]),
        copy(global_field.Hx[:, global_ids]),
        copy(global_field.Hy[:, global_ids]),
        copy(global_field.Hz[:, global_ids]),
    )
end

function pack_maxwell_interface!(
    buffer::Vector{Float64},
    U::MaxwellField,
    faces::Vector{DistributedDGInterfaceFace},
)
    components = (U.Ex, U.Ey, U.Ez, U.Hx, U.Hy, U.Hz)
    offset = 1

    for face in faces
        for component in components
            for node in face.owned_nodes
                buffer[offset] = component[node, face.owned_elem]
                offset += 1
            end
        end
    end

    return buffer
end

function unpack_maxwell_interface!(
    U::MaxwellField,
    buffer::Vector{Float64},
    faces::Vector{DistributedDGInterfaceFace},
)
    components = (U.Ex, U.Ey, U.Ez, U.Hx, U.Hy, U.Hz)
    offset = 1

    for face in faces
        for component in components
            for node in face.ghost_nodes
                component[node, face.ghost_elem] = buffer[offset]
                offset += 1
            end
        end
    end

    return U
end

function exchange_maxwell_ghost_traces!(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization;
    tag::Int = 24017,
)
    exchange = distributed_dg.exchange
    requests = MPI.Request[]

    for neighbor in exchange.neighbors
        faces = exchange.faces[neighbor]
        send_buffer = exchange.send_buffers[neighbor]
        pack_maxwell_interface!(send_buffer, U, faces)
    end

    for neighbor in exchange.neighbors
        push!(
            requests,
            MPI.Irecv!(
                exchange.recv_buffers[neighbor],
                distributed_dg.comm;
                source = neighbor,
                tag = tag,
            ),
        )
    end

    for neighbor in exchange.neighbors
        push!(
            requests,
            MPI.Isend(
                exchange.send_buffers[neighbor],
                distributed_dg.comm;
                dest = neighbor,
                tag = tag,
            ),
        )
    end

    MPI.Waitall(requests)

    for neighbor in exchange.neighbors
        unpack_maxwell_interface!(
            U,
            exchange.recv_buffers[neighbor],
            exchange.faces[neighbor],
        )
    end

    return U
end

function zero_ghost_maxwell_rhs!(
    rhs::MaxwellRHS,
    distributed_dg::DistributedDGDiscretization,
)
    components = (
        rhs.rhsEx,
        rhs.rhsEy,
        rhs.rhsEz,
        rhs.rhsHx,
        rhs.rhsHy,
        rhs.rhsHz,
    )

    for elem in distributed_dg.distributed_mesh.partition.ghosts
        for component in components
            fill!(@view(component[:, elem]), 0.0)
        end
    end

    return rhs
end

function profile_distributed_maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    tag::Int = 24017,
)
    start_ns = time_ns()
    exchange_maxwell_ghost_traces!(U, distributed_dg; tag = tag)
    after_halo_ns = time_ns()

    maxwell_rhs!(
        rhs,
        U,
        distributed_dg.dg,
        registry,
        formulation;
        ε = ε,
        μ = μ,
    )
    after_rhs_ns = time_ns()

    zero_ghost_maxwell_rhs!(rhs, distributed_dg)
    after_zero_ns = time_ns()

    return (
        halo_exchange_seconds = (after_halo_ns - start_ns) * 1.0e-9,
        rhs_assembly_seconds = (after_rhs_ns - after_halo_ns) * 1.0e-9,
        ghost_zero_seconds = (after_zero_ns - after_rhs_ns) * 1.0e-9,
        total_seconds = (after_zero_ns - start_ns) * 1.0e-9,
    )
end

function profile_distributed_maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    materials::MaxwellElementMaterials;
    tag::Int = 24017,
)
    start_ns = time_ns()
    exchange_maxwell_ghost_traces!(U, distributed_dg; tag = tag)
    after_halo_ns = time_ns()

    maxwell_rhs!(
        rhs,
        U,
        distributed_dg.dg,
        registry,
        formulation,
        materials,
    )
    after_rhs_ns = time_ns()

    zero_ghost_maxwell_rhs!(rhs, distributed_dg)
    after_zero_ns = time_ns()

    return (
        halo_exchange_seconds = (after_halo_ns - start_ns) * 1.0e-9,
        rhs_assembly_seconds = (after_rhs_ns - after_halo_ns) * 1.0e-9,
        ghost_zero_seconds = (after_zero_ns - after_rhs_ns) * 1.0e-9,
        total_seconds = (after_zero_ns - start_ns) * 1.0e-9,
    )
end

function maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    exchange_maxwell_ghost_traces!(U, distributed_dg)

    maxwell_rhs!(
        rhs,
        U,
        distributed_dg.dg,
        registry,
        formulation;
        ε = ε,
        μ = μ,
    )

    return zero_ghost_maxwell_rhs!(rhs, distributed_dg)
end

function maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
)
    return maxwell_rhs!(
        rhs,
        U,
        distributed_dg,
        registry,
        HesthavenWarburtonFormulation(flux_kind);
        ε = ε,
        μ = μ,
    )
end

function make_distributed_maxwell_rhs_function(
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    return function rhs_function!(rhs::MaxwellRHS, U::MaxwellField)
        return maxwell_rhs!(
            rhs,
            U,
            distributed_dg,
            registry,
            formulation;
            ε = ε,
            μ = μ,
        )
    end
end

function make_distributed_maxwell_rhs_function(
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
)
    return make_distributed_maxwell_rhs_function(
        distributed_dg,
        registry,
        HesthavenWarburtonFormulation(flux_kind);
        ε = ε,
        μ = μ,
    )
end

function distributed_maxwell_energy(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    M = distributed_dg.dg.ref.M
    mappings = distributed_dg.dg.mappings.tet_mappings
    local_components = zeros(Float64, 6)

    for elem in distributed_dg.distributed_mesh.partition.owned
        J = mappings[elem].absdetJ

        local_components[1] +=
            0.5 * ε * J * mass_quadratic_form(M, @view U.Ex[:, elem])
        local_components[2] +=
            0.5 * ε * J * mass_quadratic_form(M, @view U.Ey[:, elem])
        local_components[3] +=
            0.5 * ε * J * mass_quadratic_form(M, @view U.Ez[:, elem])
        local_components[4] +=
            0.5 * μ * J * mass_quadratic_form(M, @view U.Hx[:, elem])
        local_components[5] +=
            0.5 * μ * J * mass_quadratic_form(M, @view U.Hy[:, elem])
        local_components[6] +=
            0.5 * μ * J * mass_quadratic_form(M, @view U.Hz[:, elem])
    end

    components = MPI.Allreduce(local_components, +, distributed_dg.comm)
    electric = sum(@view components[1:3])
    magnetic = sum(@view components[4:6])

    return MaxwellEnergy(
        electric,
        magnetic,
        electric + magnetic,
        components[1],
        components[2],
        components[3],
        components[4],
        components[5],
        components[6],
    )
end

function distributed_rk_step!(
    U::MaxwellField,
    work::MaxwellRKWorkspace,
    scheme::ExplicitRKScheme,
    dt::Float64,
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    rhs_function! = make_distributed_maxwell_rhs_function(
        distributed_dg,
        registry,
        formulation;
        ε = ε,
        μ = μ,
    )

    return rk_step!(U, work, scheme, dt, rhs_function!)
end

function distributed_partitioned_symplectic_rk_step!(
    U::MaxwellField,
    work::MaxwellPartitionedRKWorkspace,
    scheme::ExplicitPartitionedSymplecticRKScheme,
    dt::Float64,
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    rhs_function! = make_distributed_maxwell_rhs_function(
        distributed_dg,
        registry,
        formulation;
        ε = ε,
        μ = μ,
    )

    return partitioned_symplectic_rk_step!(
        U,
        work,
        scheme,
        dt,
        rhs_function!,
    )
end

function run_distributed_maxwell_time_steps!(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::HesthavenWarburtonFormulation;
    rk_order::Int = 4,
    dt::Float64,
    nsteps::Int,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    energy_every::Int = 1,
)
    scheme = explicit_rk_scheme(rk_order)
    work = MaxwellRKWorkspace(U, scheme)
    rank = MPI.Comm_rank(distributed_dg.comm)
    energy0 = distributed_maxwell_energy(
        U,
        distributed_dg;
        ε = ε,
        μ = μ,
    )

    if rank == 0
        println("Distributed Maxwell time marching")
        println("---------------------------------")
        println("MPI ranks:        ", MPI.Comm_size(distributed_dg.comm))
        println("RK scheme:        ", scheme.name)
        println("dt:               ", dt)
        println("nsteps:           ", nsteps)
        println("initial energy:   ", energy0.total)
    end

    for step in 1:nsteps
        distributed_rk_step!(
            U,
            work,
            scheme,
            dt,
            distributed_dg,
            registry,
            formulation;
            ε = ε,
            μ = μ,
        )

        if step % energy_every == 0 || step == nsteps
            energy = distributed_maxwell_energy(
                U,
                distributed_dg;
                ε = ε,
                μ = μ,
            )

            if rank == 0
                relative_drift =
                    (energy.total - energy0.total) /
                    max(energy0.total, eps(Float64))
                println(
                    "step = ", step,
                    ", time = ", step * dt,
                    ", energy = ", energy.total,
                    ", rel ΔE = ", relative_drift,
                )
            end
        end
    end

    return U
end

function run_distributed_maxwell_time_steps!(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry;
    flux_kind::MaxwellFluxKind = MaxwellFlux_Central,
    kwargs...,
)
    return run_distributed_maxwell_time_steps!(
        U,
        distributed_dg,
        registry,
        HesthavenWarburtonFormulation(flux_kind);
        kwargs...,
    )
end

function run_distributed_maxwell_partitioned_symplectic_time_steps!(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation;
    psrk_order::Int = 2,
    first_partition::Symbol = :H,
    dt::Float64,
    nsteps::Int,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    energy_every::Int = 1,
)
    scheme = explicit_partitioned_symplectic_rk_scheme(
        psrk_order;
        first_partition = first_partition,
    )
    work = MaxwellPartitionedRKWorkspace(U, scheme)
    rank = MPI.Comm_rank(distributed_dg.comm)
    energy0 = distributed_maxwell_energy(
        U,
        distributed_dg;
        ε = ε,
        μ = μ,
    )

    if rank == 0
        println("Distributed Poisson-bracket Maxwell time marching")
        println("--------------------------------------------------")
        println("MPI ranks:        ", MPI.Comm_size(distributed_dg.comm))
        println("PSRK scheme:      ", scheme.name)
        println("dt:               ", dt)
        println("nsteps:           ", nsteps)
        println("initial energy:   ", energy0.total)
    end

    for step in 1:nsteps
        distributed_partitioned_symplectic_rk_step!(
            U,
            work,
            scheme,
            dt,
            distributed_dg,
            registry,
            formulation;
            ε = ε,
            μ = μ,
        )

        if step % energy_every == 0 || step == nsteps
            energy = distributed_maxwell_energy(
                U,
                distributed_dg;
                ε = ε,
                μ = μ,
            )

            if rank == 0
                relative_drift =
                    (energy.total - energy0.total) /
                    max(energy0.total, eps(Float64))
                println(
                    "step = ", step,
                    ", time = ", step * dt,
                    ", energy = ", energy.total,
                    ", rel ΔE = ", relative_drift,
                )
            end
        end
    end

    return U
end
