using Dates
using Serialization
using SHA
using TOML

const DISTRIBUTED_MESH_FORMAT = "DiscoGMPI distributed mesh"
const DISTRIBUTED_MESH_FORMAT_VERSION = 1
const DISTRIBUTED_CHECKPOINT_FORMAT = "DiscoGMPI Maxwell checkpoint"
const DISTRIBUTED_CHECKPOINT_FORMAT_VERSION = 1

struct DistributedCheckpointState
    step::Int
    time::Float64
    dt::Float64
    metadata::Dict{String, Any}
end

function rank_file_label(rank::Int)
    return lpad(string(rank), 4, '0')
end

function atomic_serialize(path::AbstractString, value)
    mkpath(dirname(path))
    temporary = path * ".tmp.$(getpid())"
    try
        open(temporary, "w") do io
            serialize(io, value)
            flush(io)
        end
        mv(temporary, path; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return path
end

function atomic_toml(path::AbstractString, values::AbstractDict)
    mkpath(dirname(path))
    temporary = path * ".tmp.$(getpid())"
    try
        open(temporary, "w") do io
            TOML.print(io, values)
            flush(io)
        end
        mv(temporary, path; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return path
end

function atomic_text(writer::Function, path::AbstractString)
    mkpath(dirname(path))
    temporary = path * ".tmp.$(getpid())"
    try
        open(temporary, "w") do io
            writer(io)
            flush(io)
        end
        mv(temporary, path; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return path
end

function file_sha256(path::AbstractString)
    return bytes2hex(open(SHA.sha256, path))
end

function toml_value(value)
    if value === nothing
        return ""
    elseif value isa Symbol
        return String(value)
    elseif value isa AbstractString || value isa Number || value isa Bool
        return value
    elseif value isa AbstractVector || value isa Tuple
        return [toml_value(entry) for entry in value]
    elseif value isa AbstractDict
        return Dict(
            string(key) => toml_value(entry)
            for (key, entry) in value
        )
    end
    return string(value)
end

function collective_io_error(
    local_error::Union{Nothing, String},
    comm::MPI.Comm,
    context::AbstractString,
)
    rank = MPI.Comm_rank(comm)
    errors = MPI.gather(local_error, comm; root = 0)
    message = nothing

    if rank == 0
        failures = [
            "rank $(index - 1): $error"
            for (index, error) in enumerate(errors)
            if error !== nothing
        ]
        if !isempty(failures)
            message = context * " failed:\n" * join(failures, "\n")
        end
    end

    message = MPI.bcast(message, comm; root = 0)
    message === nothing || error(message)
    return nothing
end

function mesh_piece_metadata(
    path::AbstractString,
    distributed_mesh::DistributedMesh3D.DistributedMesh,
    rank::Int,
)
    return Dict{String, Any}(
        "rank" => rank,
        "file" => basename(path),
        "sha256" => file_sha256(path),
        "owned_elements" => length(distributed_mesh.partition.owned),
        "ghost_elements" => length(distributed_mesh.partition.ghosts),
        "local_nodes" => length(distributed_mesh.nodes.global_ids),
        "neighbors" => copy(distributed_mesh.mpi.neighbors),
    )
end

"""
    write_distributed_mesh_partition(output_dir, distributed_mesh; ...)

Persist one owned-plus-halo mesh piece per MPI rank. Each rank writes its own
piece, and rank zero publishes `mesh_manifest.toml` only after every piece has
been written successfully. The resulting directory can be loaded without any
rank reading or storing the global mesh.
"""
function write_distributed_mesh_partition(
    output_dir::AbstractString,
    distributed_mesh::DistributedMesh3D.DistributedMesh;
    comm::MPI.Comm = MPI.COMM_WORLD,
    metadata::AbstractDict = Dict{String, Any}(),
)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    if rank == 0
        mkpath(output_dir)
    end
    MPI.Barrier(comm)

    filename = "mesh_rank$(rank_file_label(rank)).bin"
    path = joinpath(output_dir, filename)
    local_error = nothing
    local_piece = nothing

    try
        payload = (
            format = DISTRIBUTED_MESH_FORMAT,
            version = DISTRIBUTED_MESH_FORMAT_VERSION,
            rank = rank,
            mpi_ranks = nranks,
            mesh = distributed_mesh,
        )
        atomic_serialize(path, payload)
        local_piece = mesh_piece_metadata(path, distributed_mesh, rank)
    catch error
        local_error = sprint(showerror, error)
    end

    collective_io_error(
        local_error,
        comm,
        "Distributed mesh piece writing",
    )
    pieces = MPI.gather(local_piece, comm; root = 0)

    manifest_error = nothing
    if rank == 0
        try
            sort!(pieces; by = piece -> piece["rank"])
            global_elements =
                sum(piece["owned_elements"] for piece in pieces)
            manifest = Dict{String, Any}(
                "format" => DISTRIBUTED_MESH_FORMAT,
                "version" => DISTRIBUTED_MESH_FORMAT_VERSION,
                "created_utc" => string(Dates.now(Dates.UTC)),
                "julia_version" => string(VERSION),
                "mpi_ranks" => nranks,
                "global_elements" => global_elements,
                "metadata" => toml_value(metadata),
                "pieces" => pieces,
            )
            atomic_toml(
                joinpath(output_dir, "mesh_manifest.toml"),
                manifest,
            )
        catch error
            manifest_error = sprint(showerror, error)
        end
    end
    collective_io_error(
        rank == 0 ? manifest_error : nothing,
        comm,
        "Distributed mesh manifest writing",
    )
    MPI.Barrier(comm)
    return joinpath(output_dir, "mesh_manifest.toml")
end

"""
    prepare_distributed_mesh_partition(global_mesh, elem_to_rank, output_dir; ...)

Create rank-local distributed mesh pieces from root-owned global inputs.
Non-root ranks pass `nothing`. This is a one-time preprocessing operation;
subsequent runs should call `load_distributed_mesh_partition`.
"""
function prepare_distributed_mesh_partition(
    global_mesh::Union{Nothing, RawVTUMesh},
    elem_to_rank::Union{Nothing, AbstractVector{Int}},
    output_dir::AbstractString;
    comm::MPI.Comm = MPI.COMM_WORLD,
    root::Int = 0,
    boundary_tag_name::String = "boundary_id",
    material_tag_name::Union{Nothing, String} = nothing,
    metadata::AbstractDict = Dict{String, Any}(),
)
    distributed_mesh = distribute_mesh_from_root(
        global_mesh,
        elem_to_rank;
        comm = comm,
        root = root,
        boundary_tag_name = boundary_tag_name,
        material_tag_name = material_tag_name,
    )
    write_distributed_mesh_partition(
        output_dir,
        distributed_mesh;
        comm = comm,
        metadata = metadata,
    )
    return distributed_mesh
end

"""
    prepare_distributed_mesh_partition_collective(global_mesh, elem_to_rank, output_dir; ...)

Create rank-local mesh pieces without a rank-zero distribution phase. Every
rank must provide the same global mesh and partition vector. This trades memory
for setup scalability and is useful for generated meshes or benchmark cases
where global inputs are already available on all ranks.
"""
function prepare_distributed_mesh_partition_collective(
    global_mesh::RawVTUMesh,
    elem_to_rank::AbstractVector{Int},
    output_dir::AbstractString;
    comm::MPI.Comm = MPI.COMM_WORLD,
    boundary_tag_name::String = "boundary_id",
    material_tag_name::Union{Nothing, String} = nothing,
    metadata::AbstractDict = Dict{String, Any}(),
)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    boundary_face_tags, material_ids, face_to_elems =
        root_mesh_distribution_data(
            global_mesh,
            elem_to_rank,
            nranks;
            boundary_tag_name = boundary_tag_name,
            material_tag_name = material_tag_name,
        )
    distributed_mesh = build_distributed_mesh(
        global_mesh.points,
        global_mesh.tets,
        elem_to_rank,
        rank;
        material_id_global = material_ids,
        boundary_face_tags = boundary_face_tags,
        face_to_elems_global = face_to_elems,
    )
    write_distributed_mesh_partition(
        output_dir,
        distributed_mesh;
        comm = comm,
        metadata = metadata,
    )
    return distributed_mesh
end

function manifest_piece(
    manifest::AbstractDict,
    rank::Int,
)
    pieces = get(manifest, "pieces", Any[])
    matches = [
        piece
        for piece in pieces
        if Int(piece["rank"]) == rank
    ]
    length(matches) == 1 ||
        error("Manifest does not contain exactly one piece for rank $rank.")
    return only(matches)
end

"""
    load_distributed_mesh_partition(output_dir; comm=MPI.COMM_WORLD)

Load one mesh shard independently on every rank and collectively validate the
manifest, checksum, rank count, and payload identity.
"""
function load_distributed_mesh_partition(
    output_dir::AbstractString;
    comm::MPI.Comm = MPI.COMM_WORLD,
)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    local_error = nothing
    distributed_mesh = nothing

    try
        manifest_path = joinpath(output_dir, "mesh_manifest.toml")
        isfile(manifest_path) ||
            error("Distributed mesh manifest not found: $manifest_path")
        manifest = TOML.parsefile(manifest_path)
        get(manifest, "format", "") == DISTRIBUTED_MESH_FORMAT ||
            error("Unsupported distributed mesh format.")
        Int(get(manifest, "version", 0)) ==
            DISTRIBUTED_MESH_FORMAT_VERSION ||
            error("Unsupported distributed mesh format version.")
        Int(get(manifest, "mpi_ranks", -1)) == nranks ||
            error(
                "Distributed mesh was created for " *
                "$(get(manifest, "mpi_ranks", "unknown")) ranks, " *
                "but this run uses $nranks.",
            )

        piece = manifest_piece(manifest, rank)
        path = joinpath(output_dir, String(piece["file"]))
        isfile(path) || error("Distributed mesh piece not found: $path")
        file_sha256(path) == String(piece["sha256"]) ||
            error("Checksum mismatch for distributed mesh piece $path")

        payload = open(deserialize, path)
        payload.format == DISTRIBUTED_MESH_FORMAT ||
            error("Invalid mesh payload format in $path")
        payload.version == DISTRIBUTED_MESH_FORMAT_VERSION ||
            error("Invalid mesh payload version in $path")
        payload.rank == rank ||
            error("Mesh payload rank $(payload.rank) does not match rank $rank")
        payload.mpi_ranks == nranks ||
            error("Mesh payload MPI size does not match this run")
        distributed_mesh = payload.mesh
    catch error
        local_error = sprint(showerror, error)
    end

    collective_io_error(
        local_error,
        comm,
        "Distributed mesh loading",
    )
    return distributed_mesh
end

function build_distributed_dg_from_partition(
    output_dir::AbstractString,
    order::Int;
    comm::MPI.Comm = MPI.COMM_WORLD,
    boundary_tag_name::String = "boundary_id",
    trace_tol::Float64 = 1e-10,
    backend::AbstractBackend = SerialBackend(),
)
    distributed_mesh = load_distributed_mesh_partition(
        output_dir;
        comm = comm,
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

function maxwell_checkpoint_components(U::MaxwellField)
    return (U.Ex, U.Ey, U.Ez, U.Hx, U.Hy, U.Hz)
end

function checkpoint_piece_metadata(
    path::AbstractString,
    rank::Int,
    owned_global_ids::Vector{Int},
)
    return Dict{String, Any}(
        "rank" => rank,
        "file" => basename(path),
        "sha256" => file_sha256(path),
        "owned_elements" => length(owned_global_ids),
        "first_global_element" =>
            isempty(owned_global_ids) ? 0 : minimum(owned_global_ids),
        "last_global_element" =>
            isempty(owned_global_ids) ? 0 : maximum(owned_global_ids),
    )
end

"""
    write_distributed_checkpoint(checkpoint_dir, U, distributed_dg; ...)

Write owned Maxwell degrees of freedom independently on each rank. A restart
manifest is atomically published only after all rank files are complete.
"""
function write_distributed_checkpoint(
    checkpoint_dir::AbstractString,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization;
    step::Int,
    time::Float64,
    dt::Float64,
    metadata::AbstractDict = Dict{String, Any}(),
)
    comm = distributed_dg.comm
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    owned = distributed_dg.distributed_mesh.partition.owned
    owned_global_ids =
        distributed_dg.distributed_mesh.elements.global_ids[owned]

    if rank == 0
        mkpath(checkpoint_dir)
    end
    MPI.Barrier(comm)

    filename = "state_rank$(rank_file_label(rank)).bin"
    path = joinpath(checkpoint_dir, filename)
    local_error = nothing
    local_piece = nothing

    try
        components = tuple(
            (
                copy(component[:, owned])
                for component in maxwell_checkpoint_components(U)
            )...,
        )
        payload = (
            format = DISTRIBUTED_CHECKPOINT_FORMAT,
            version = DISTRIBUTED_CHECKPOINT_FORMAT_VERSION,
            rank = rank,
            mpi_ranks = nranks,
            dg_order = distributed_dg.dg.ref.N,
            nodes_per_element = distributed_dg.dg.ref.Np,
            owned_global_ids = collect(owned_global_ids),
            components = components,
        )
        atomic_serialize(path, payload)
        local_piece =
            checkpoint_piece_metadata(path, rank, collect(owned_global_ids))
    catch error
        local_error = sprint(showerror, error)
    end

    collective_io_error(
        local_error,
        comm,
        "Distributed checkpoint piece writing",
    )
    pieces = MPI.gather(local_piece, comm; root = 0)

    manifest_error = nothing
    if rank == 0
        try
            sort!(pieces; by = piece -> piece["rank"])
            manifest = Dict{String, Any}(
                "format" => DISTRIBUTED_CHECKPOINT_FORMAT,
                "version" => DISTRIBUTED_CHECKPOINT_FORMAT_VERSION,
                "created_utc" => string(Dates.now(Dates.UTC)),
                "julia_version" => string(VERSION),
                "mpi_ranks" => nranks,
                "dg_order" => distributed_dg.dg.ref.N,
                "nodes_per_element" => distributed_dg.dg.ref.Np,
                "step" => step,
                "time" => time,
                "dt" => dt,
                "metadata" => toml_value(metadata),
                "pieces" => pieces,
            )
            atomic_toml(
                joinpath(checkpoint_dir, "checkpoint_manifest.toml"),
                manifest,
            )
        catch error
            manifest_error = sprint(showerror, error)
        end
    end
    collective_io_error(
        rank == 0 ? manifest_error : nothing,
        comm,
        "Distributed checkpoint manifest writing",
    )
    MPI.Barrier(comm)
    return joinpath(checkpoint_dir, "checkpoint_manifest.toml")
end

"""
    load_distributed_checkpoint(checkpoint_dir, distributed_dg)

Load owned fields from a checkpoint. Restart currently requires the same MPI
rank count, DG order, and element partition. Ghost face traces are refreshed
after loading.
"""
function load_distributed_checkpoint(
    checkpoint_dir::AbstractString,
    distributed_dg::DistributedDGDiscretization,
)
    comm = distributed_dg.comm
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    local_error = nothing
    U = nothing
    state = nothing

    try
        manifest_path =
            joinpath(checkpoint_dir, "checkpoint_manifest.toml")
        isfile(manifest_path) ||
            error("Checkpoint manifest not found: $manifest_path")
        manifest = TOML.parsefile(manifest_path)
        get(manifest, "format", "") == DISTRIBUTED_CHECKPOINT_FORMAT ||
            error("Unsupported checkpoint format.")
        Int(get(manifest, "version", 0)) ==
            DISTRIBUTED_CHECKPOINT_FORMAT_VERSION ||
            error("Unsupported checkpoint format version.")
        Int(get(manifest, "mpi_ranks", -1)) == nranks ||
            error(
                "Checkpoint requires $(get(manifest, "mpi_ranks", "unknown")) " *
                "MPI ranks, but this run uses $nranks.",
            )
        Int(get(manifest, "dg_order", -1)) == distributed_dg.dg.ref.N ||
            error(
                "Checkpoint DG order $(get(manifest, "dg_order", "unknown")) " *
                "does not match order $(distributed_dg.dg.ref.N).",
            )

        piece = manifest_piece(manifest, rank)
        path = joinpath(checkpoint_dir, String(piece["file"]))
        isfile(path) || error("Checkpoint piece not found: $path")
        file_sha256(path) == String(piece["sha256"]) ||
            error("Checksum mismatch for checkpoint piece $path")
        payload = open(deserialize, path)
        payload.format == DISTRIBUTED_CHECKPOINT_FORMAT ||
            error("Invalid checkpoint payload format in $path")
        payload.version == DISTRIBUTED_CHECKPOINT_FORMAT_VERSION ||
            error("Invalid checkpoint payload version in $path")
        payload.rank == rank ||
            error("Checkpoint payload rank does not match this rank")
        payload.mpi_ranks == nranks ||
            error("Checkpoint payload MPI size does not match this run")
        payload.dg_order == distributed_dg.dg.ref.N ||
            error("Checkpoint payload DG order does not match this run")

        owned = distributed_dg.distributed_mesh.partition.owned
        expected_global_ids =
            distributed_dg.distributed_mesh.elements.global_ids[owned]
        payload.owned_global_ids == expected_global_ids ||
            error(
                "Checkpoint partition does not match the current mesh " *
                "on rank $rank.",
            )

        np = distributed_dg.dg.ref.Np
        nlocal = length(distributed_dg.distributed_mesh.elements.global_ids)
        arrays = [zeros(Float64, np, nlocal) for _ in 1:6]
        for (array, stored) in zip(arrays, payload.components)
            size(stored) == (np, length(owned)) ||
                error("Invalid checkpoint field dimensions in $path")
            array[:, owned] .= stored
        end
        U = MaxwellField(arrays...)
        state = DistributedCheckpointState(
            Int(manifest["step"]),
            Float64(manifest["time"]),
            Float64(manifest["dt"]),
            Dict{String, Any}(
                string(key) => value
                for (key, value) in get(
                    manifest,
                    "metadata",
                    Dict{String, Any}(),
                )
            ),
        )
    catch error
        local_error = sprint(showerror, error)
    end

    collective_io_error(
        local_error,
        comm,
        "Distributed checkpoint loading",
    )
    exchange_maxwell_ghost_traces!(U, distributed_dg)
    return U, state
end

function partition_metadata_record(
    distributed_dg::DistributedDGDiscretization,
)
    distributed_mesh = distributed_dg.distributed_mesh
    rank = MPI.Comm_rank(distributed_dg.comm)
    owned_elements = length(distributed_mesh.partition.owned)
    ghost_elements = length(distributed_mesh.partition.ghosts)
    local_elements = owned_elements + ghost_elements
    local_nodes = length(distributed_mesh.nodes.global_ids)
    ghost_nodes = Set{Int}()
    for elem in distributed_mesh.partition.ghosts
        for node in @view(distributed_mesh.elements.vertices[:, elem])
            push!(ghost_nodes, node)
        end
    end
    owned_global_ids =
        distributed_mesh.elements.global_ids[distributed_mesh.partition.owned]
    interface_faces = sum(
        length(distributed_dg.exchange.faces[neighbor])
        for neighbor in distributed_dg.exchange.neighbors;
        init = 0,
    )
    send_faces = sum(
        length(distributed_mesh.mpi.comms[neighbor].send_faces)
        for neighbor in distributed_mesh.mpi.neighbors;
        init = 0,
    )
    recv_faces = sum(
        length(distributed_mesh.mpi.comms[neighbor].recv_faces)
        for neighbor in distributed_mesh.mpi.neighbors;
        init = 0,
    )
    send_values = sum(
        length(distributed_dg.exchange.send_buffers[neighbor])
        for neighbor in distributed_dg.exchange.neighbors;
        init = 0,
    )
    recv_values = sum(
        length(distributed_dg.exchange.recv_buffers[neighbor])
        for neighbor in distributed_dg.exchange.neighbors;
        init = 0,
    )
    nodes_per_element = distributed_dg.dg.ref.Np
    owned_scale = max(owned_elements, 1)

    return (
        rank = rank,
        owned_elements = owned_elements,
        ghost_elements = ghost_elements,
        local_elements = local_elements,
        local_nodes = local_nodes,
        ghost_nodes = length(ghost_nodes),
        owned_dofs = owned_elements * nodes_per_element,
        ghost_dofs = ghost_elements * nodes_per_element,
        local_dofs = local_elements * nodes_per_element,
        neighbors = length(distributed_dg.exchange.neighbors),
        neighbor_ranks = join(distributed_dg.exchange.neighbors, ';'),
        interface_faces = interface_faces,
        interface_faces_per_owned_element =
            interface_faces / owned_scale,
        send_faces = send_faces,
        recv_faces = recv_faces,
        send_values = send_values,
        recv_values = recv_values,
        halo_elements_per_owned_element = ghost_elements / owned_scale,
        halo_dofs_per_owned_element =
            (ghost_elements * nodes_per_element) / owned_scale,
        send_values_per_owned_element = send_values / owned_scale,
        recv_values_per_owned_element = recv_values / owned_scale,
        first_global_element =
            isempty(owned_global_ids) ? 0 : minimum(owned_global_ids),
        last_global_element =
            isempty(owned_global_ids) ? 0 : maximum(owned_global_ids),
    )
end

"""
    write_distributed_run_metadata(output_dir, distributed_dg; ...)

Write `run_metadata.toml` and `partition_metadata.csv`. Configuration and
runtime dictionaries are included verbatim after conversion to TOML values.
"""
function write_distributed_run_metadata(
    output_dir::AbstractString,
    distributed_dg::DistributedDGDiscretization;
    configuration::AbstractDict = Dict{String, Any}(),
    runtime::AbstractDict = Dict{String, Any}(),
)
    comm = distributed_dg.comm
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    local_record = partition_metadata_record(distributed_dg)
    records = MPI.gather(local_record, comm; root = 0)
    local_error = nothing

    if rank == 0
        try
            mkpath(output_dir)
            sort!(records; by = record -> record.rank)
            atomic_text(
                joinpath(output_dir, "partition_metadata.csv"),
            ) do io
                println(
                    io,
                    "rank,owned_elements,ghost_elements,local_elements," *
                    "local_nodes,ghost_nodes,owned_dofs,ghost_dofs," *
                    "local_dofs,neighbors,neighbor_ranks,interface_faces," *
                    "interface_faces_per_owned_element," *
                    "send_faces,recv_faces,send_values,recv_values," *
                    "halo_elements_per_owned_element," *
                    "halo_dofs_per_owned_element," *
                    "send_values_per_owned_element," *
                    "recv_values_per_owned_element," *
                    "first_global_element,last_global_element",
                )
                for record in records
                    println(
                        io,
                        join(
                            (
                                record.rank,
                                record.owned_elements,
                                record.ghost_elements,
                                record.local_elements,
                                record.local_nodes,
                                record.ghost_nodes,
                                record.owned_dofs,
                                record.ghost_dofs,
                                record.local_dofs,
                                record.neighbors,
                                record.neighbor_ranks,
                                record.interface_faces,
                                record.interface_faces_per_owned_element,
                                record.send_faces,
                                record.recv_faces,
                                record.send_values,
                                record.recv_values,
                                record.halo_elements_per_owned_element,
                                record.halo_dofs_per_owned_element,
                                record.send_values_per_owned_element,
                                record.recv_values_per_owned_element,
                                record.first_global_element,
                                record.last_global_element,
                            ),
                            ',',
                        ),
                    )
                end
            end

            owned_counts = [record.owned_elements for record in records]
            ghost_counts = [record.ghost_elements for record in records]
            local_counts = [record.local_elements for record in records]
            interface_counts = [record.interface_faces for record in records]
            neighbor_counts = [record.neighbors for record in records]
            ghost_dof_counts = [record.ghost_dofs for record in records]
            send_value_counts = [record.send_values for record in records]
            recv_value_counts = [record.recv_values for record in records]
            halo_ratios = [
                record.halo_elements_per_owned_element
                for record in records
            ]
            interface_ratios = [
                record.interface_faces_per_owned_element
                for record in records
            ]
            owned_minimum = minimum(owned_counts)
            owned_maximum = maximum(owned_counts)
            metadata = Dict{String, Any}(
                "format" => "DiscoGMPI run metadata",
                "version" => 1,
                "created_utc" => string(Dates.now(Dates.UTC)),
                "julia_version" => string(VERSION),
                "mpi_ranks" => nranks,
                "dg_order" => distributed_dg.dg.ref.N,
                "nodes_per_element" => distributed_dg.dg.ref.Np,
                "global_elements" => sum(owned_counts),
                "owned_elements_minimum" => owned_minimum,
                "owned_elements_maximum" => owned_maximum,
                "owned_elements_average" =>
                    sum(owned_counts) / max(nranks, 1),
                "owned_elements_imbalance" =>
                    owned_minimum == 0 ? 0.0 : owned_maximum / owned_minimum,
                "local_elements_total" => sum(local_counts),
                "local_elements_maximum" => maximum(local_counts),
                "local_elements_average" =>
                    sum(local_counts) / max(nranks, 1),
                "ghost_elements_total" => sum(ghost_counts),
                "ghost_elements_maximum" => maximum(ghost_counts),
                "ghost_elements_average" =>
                    sum(ghost_counts) / max(nranks, 1),
                "ghost_dofs_total" => sum(ghost_dof_counts),
                "ghost_dofs_maximum" => maximum(ghost_dof_counts),
                "interface_faces_total" => sum(interface_counts),
                "interface_faces_maximum" => maximum(interface_counts),
                "interface_faces_average" =>
                    sum(interface_counts) / max(nranks, 1),
                "halo_elements_per_owned_element_maximum" =>
                    maximum(halo_ratios),
                "interface_faces_per_owned_element_maximum" =>
                    maximum(interface_ratios),
                "neighbor_count_maximum" => maximum(neighbor_counts),
                "send_values_total" => sum(send_value_counts),
                "send_values_maximum" => maximum(send_value_counts),
                "send_values_average" =>
                    sum(send_value_counts) / max(nranks, 1),
                "recv_values_total" => sum(recv_value_counts),
                "recv_values_maximum" => maximum(recv_value_counts),
                "recv_values_average" =>
                    sum(recv_value_counts) / max(nranks, 1),
                "configuration" => toml_value(configuration),
                "runtime" => toml_value(runtime),
            )
            atomic_toml(
                joinpath(output_dir, "run_metadata.toml"),
                metadata,
            )
        catch error
            local_error = sprint(showerror, error)
        end
    end

    collective_io_error(
        rank == 0 ? local_error : nothing,
        comm,
        "Run metadata writing",
    )
    MPI.Barrier(comm)
    return (
        joinpath(output_dir, "run_metadata.toml"),
        joinpath(output_dir, "partition_metadata.csv"),
    )
end
