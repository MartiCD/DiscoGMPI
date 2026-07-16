const DEFAULT_CAVITY_BOUNDARY_ID = 10
const DRIVER_TET_FACE_NODE_IDS = (
    (2, 3, 4),
    (1, 4, 3),
    (1, 2, 4),
    (1, 3, 2),
)
const DEFAULT_MAXWELL_CONVERGENCE_RATE_TOLERANCE = 0.5
const DEFAULT_COMPONENT_RATE_ERROR_FLOOR = 1.0e-12
const REFERENCE_TETRAHEDRON_VOLUME = 4.0 / 3.0


struct DistributedMaxwellExperimentConfig
    mesh_path::String
    partition_path::String
    distributed_mesh_dir::String
    rebuild_distributed_mesh::Bool
    collective_distributed_mesh_prep::Bool
    output_dir::String
    polynomial_order::Int
    esprk_order::Int
    rk_order::Int
    final_time::Float64
    cfl::Float64
    epsilon::Float64
    mu::Float64
    boundary_condition::Symbol
    flux_kind::MaxwellFluxKind
    pml_width::Float64
    pml_sigma_max::Float64
    pml_degree::Int
    pml_a::Float64
    pml_regularization::Float64
    energy_every::Int
    cubature_order::Int
    paraview_every::Int
    checkpoint_every::Int
    checkpoint_dir::String
    restart_path::String
end

struct DistributedPeriodicMaxwellExperimentConfig
    mesh_path::String
    partition_path::String
    partition_count::Int
    repartition::Bool
    distributed_mesh_dir::String
    rebuild_distributed_mesh::Bool
    collective_distributed_mesh_prep::Bool
    output_dir::String
    polynomial_order::Int
    esprk_order::Int
    final_time::Float64
    cfl::Float64
    epsilon::Float64
    mu::Float64
    wave_number::Float64
    flux_kind::MaxwellFluxKind
    energy_every::Int
    cubature_order::Int
    paraview_every::Int
    checkpoint_every::Int
    checkpoint_dir::String
    restart_path::String
end

struct DistributedConvergenceConfig
    mesh_parameters::Vector{Int}
    mesh_family::Symbol
    mesh_dir::String
    orders::Vector{Int}
    final_time::Float64
    periods::Union{Nothing, Float64}
    cfl::Float64
    cfl_divisor::Float64
    jitter::Float64
    seed::Int
    epsilon::Float64
    mu::Float64
    boundary_condition::Symbol
    flux_kind::MaxwellFluxKind
    output::String
end

struct DistributedPeriodicConvergenceConfig
    mesh_family::Symbol
    nx_targets::Vector{Int}
    orders::Vector{Int}
    final_time::Float64
    periods::Union{Nothing, Float64}
    cfl::Float64
    cfl_divisor::Float64
    epsilon::Float64
    mu::Float64
    wave_number::Float64
    flux_kind::MaxwellFluxKind
    geo_path::String
    mesh_dir::String
    output::String
end

struct AxisAlignedBox
    lower::NTuple{3, Float64}
    upper::NTuple{3, Float64}
end

struct DiagnosticHistoryFiles
    energy_io::Union{Nothing, IO}
    quadrature_io::Union{Nothing, IO}
    write_header::Bool
end

struct RestartTimePlan
    dt::Float64
    nsteps::Int
end

struct ConvergenceTableColumn
    name::String
    getter::Function
end

function sorted_tet_face_key(nodes::NTuple{3, Int})
    values = sort(collect(nodes))
    return (values[1], values[2], values[3])
end

function build_tetrahedral_boundary_triangles(
    tets::AbstractMatrix{<:Integer};
    face_node_ids = DRIVER_TET_FACE_NODE_IDS,
)
    counts = Dict{NTuple{3, Int}, Int}()
    oriented_nodes = Dict{NTuple{3, Int}, NTuple{3, Int}}()

    for elem in axes(tets, 2)
        tet = @view tets[:, elem]
        for ids in face_node_ids
            nodes = (Int(tet[ids[1]]), Int(tet[ids[2]]), Int(tet[ids[3]]))
            key = sorted_tet_face_key(nodes)
            counts[key] = get(counts, key, 0) + 1
            oriented_nodes[key] = nodes
        end
    end

    boundary_faces = NTuple{3, Int}[]
    for (key, count) in counts
        if count == 1
            push!(boundary_faces, oriented_nodes[key])
        elseif count != 2
            error("Non-manifold tetrahedral face $key appears $count times.")
        end
    end

    sort!(boundary_faces; by = sorted_tet_face_key)
    return isempty(boundary_faces) ?
           Matrix{Int}(undef, 3, 0) :
           reduce(hcat, collect.(boundary_faces))
end

function raw_mesh_with_boundary_triangles(
    points::AbstractMatrix{<:Real},
    tets::AbstractMatrix{<:Integer},
    tris::AbstractMatrix{<:Integer},
    boundary_ids::AbstractVector{<:Integer};
    boundary_tag_name::String = "boundary_id",
)
    ntets = size(tets, 2)
    ntris = size(tris, 2)
    length(boundary_ids) == ntris ||
        throw(ArgumentError("boundary_ids must contain one value per triangle."))
    tet_cell_ids = collect(1:ntets)
    tri_cell_ids = collect((ntets + 1):(ntets + ntris))
    cell_boundary_ids = zeros(Int, ntets + ntris)
    cell_boundary_ids[tri_cell_ids] .= Int.(boundary_ids)

    mesh = RawVTUMesh(
        Matrix{Float64}(points),
        Matrix{Int}(tets),
        Matrix{Int}(tris),
        tet_cell_ids,
        tri_cell_ids,
        Dict{String, Any}(boundary_tag_name => cell_boundary_ids),
    )
    check_mesh_consistency(mesh)
    return mesh
end

function load_box_boundary_mesh(
    mesh_path::AbstractString;
    boundary_id::Integer = DEFAULT_CAVITY_BOUNDARY_ID,
    expected_lower::Union{Nothing, NTuple{3, Float64}} = nothing,
    expected_upper::Union{Nothing, NTuple{3, Float64}} = nothing,
    tolerance::Float64 = 1.0e-10,
    boundary_tag_name::String = "boundary_id",
)
    points, tets = read_mesh_file_tet_vtk(mesh_path)
    size(tets, 2) > 0 || error("The mesh '$mesh_path' contains no tetrahedra.")

    if expected_lower !== nothing && expected_upper !== nothing
        bounds = [
            (minimum(@view points[dimension, :]),
             maximum(@view points[dimension, :]))
            for dimension in 1:3
        ]
        all(
            dimension ->
                abs(bounds[dimension][1] - expected_lower[dimension]) <= tolerance &&
                abs(bounds[dimension][2] - expected_upper[dimension]) <= tolerance,
            1:3,
        ) || error(
            "The analytical cavity mode requires bounds " *
            "$expected_lower to $expected_upper; coordinate bounds are $bounds.",
        )
    end

    tris = build_tetrahedral_boundary_triangles(tets)
    return raw_mesh_with_boundary_triangles(
        points,
        tets,
        tris,
        fill(Int(boundary_id), size(tris, 2));
        boundary_tag_name = boundary_tag_name,
    )
end

function infer_axis_aligned_box(points::AbstractMatrix{<:Real})
    size(points, 2) > 0 || error("Cannot infer bounds from an empty mesh.")
    lower_values = vec(minimum(points; dims = 2))
    upper_values = vec(maximum(points; dims = 2))
    lower = Tuple(Float64.(lower_values))
    upper = Tuple(Float64.(upper_values))
    lengths = axis_aligned_box_lengths(AxisAlignedBox(lower, upper))
    all(>(0.0), lengths) ||
        error("The mesh has degenerate coordinate bounds $lower to $upper.")
    return AxisAlignedBox(lower, upper)
end

function axis_aligned_box_lengths(box::AxisAlignedBox)
    return ntuple(dimension -> box.upper[dimension] - box.lower[dimension], 3)
end

function periodic_box_boundary_id(
    points::AbstractMatrix{<:Real},
    triangle::AbstractVector{<:Integer};
    box::AxisAlignedBox,
    tolerance::Float64 = 1.0e-10,
)
    centroid = (
        sum(@view points[1, triangle]) / 3.0,
        sum(@view points[2, triangle]) / 3.0,
        sum(@view points[3, triangle]) / 3.0,
    )
    abs(centroid[1] - box.lower[1]) <= tolerance && return 1
    abs(centroid[1] - box.upper[1]) <= tolerance && return 2
    abs(centroid[2] - box.lower[2]) <= tolerance && return 3
    abs(centroid[2] - box.upper[2]) <= tolerance && return 4
    abs(centroid[3] - box.lower[3]) <= tolerance && return 5
    abs(centroid[3] - box.upper[3]) <= tolerance && return 6

    error(
        "Boundary triangle $(collect(triangle)) at centroid $centroid is not " *
        "on one of the inferred box planes $(box.lower) to $(box.upper). " *
        "Periodic box loading requires an axis-aligned box mesh.",
    )
end

function load_periodic_box_mesh(
    mesh_path::AbstractString;
    boundary_tag_name::String = "boundary_id",
)
    points, tets = read_mesh_file_tet_vtk(mesh_path)
    size(tets, 2) > 0 || error("The mesh '$mesh_path' contains no tetrahedra.")

    box = infer_axis_aligned_box(points)
    tolerance = max(1.0e-10, 1.0e-10 * maximum(axis_aligned_box_lengths(box)))
    tris = build_tetrahedral_boundary_triangles(tets)
    boundary_ids = Vector{Int}(undef, size(tris, 2))
    for triangle in axes(tris, 2)
        boundary_ids[triangle] = periodic_box_boundary_id(
            points,
            @view(tris[:, triangle]);
            box = box,
            tolerance = tolerance,
        )
    end

    return raw_mesh_with_boundary_triangles(
        points,
        tets,
        tris,
        boundary_ids;
        boundary_tag_name = boundary_tag_name,
    )
end

function periodic_box_specs(box::AxisAlignedBox)
    Lx, Ly, Lz = axis_aligned_box_lengths(box)
    return box_periodic_specs(Lx, Ly, Lz)
end

function distributed_axis_aligned_box(
    distributed_dg::DistributedDGDiscretization,
)
    points = distributed_dg.dg.mesh.points
    local_lower = vec(minimum(points; dims = 2))
    local_upper = vec(maximum(points; dims = 2))
    global_lower = MPI.Allreduce(local_lower, min, distributed_dg.comm)
    global_upper = MPI.Allreduce(local_upper, max, distributed_dg.comm)
    return AxisAlignedBox(Tuple(global_lower), Tuple(global_upper))
end

function plane_wave_for_periodic_box(
    box::AxisAlignedBox;
    wave_number::Float64,
    epsilon::Float64 = 1.0,
    mu::Float64 = 1.0,
    magnetic_amplitude::Float64 = 1.0,
    atol::Float64 = 1.0e-10,
)
    wave_number > 0.0 || throw(ArgumentError("wave_number must be positive."))
    epsilon > 0.0 || throw(ArgumentError("epsilon must be positive."))
    mu > 0.0 || throw(ArgumentError("mu must be positive."))
    Lx = axis_aligned_box_lengths(box)[1]
    periods = wave_number * Lx / (2.0 * pi)
    nearest_periods = round(Int, periods)
    isapprox(periods, nearest_periods; rtol = atol, atol = atol) ||
        throw(
            ArgumentError(
                "wave_number=$wave_number gives $periods periods over " *
                "the x extent $Lx. A periodic analytical wave requires an " *
                "integer number of periods. For one period use " *
                "wave_number=$(2.0 * pi / Lx).",
            ),
        )
    nearest_periods >= 1 ||
        throw(ArgumentError("The periodic wave must contain at least one x period."))

    angular_frequency = wave_number / sqrt(epsilon * mu)
    return PlaneWaveParameters(
        wave_number,
        angular_frequency,
        magnetic_amplitude,
        box.lower[1],
    )
end

function distributed_mesh_bounds(
    distributed_dg::DistributedDGDiscretization,
)
    box = distributed_axis_aligned_box(distributed_dg)
    return collect(box.lower), collect(box.upper)
end

function build_six_sided_nonlinear_pml(
    distributed_dg::DistributedDGDiscretization;
    width::Float64,
    sigma_max::Float64,
    degree::Int = 2,
    a::Float64 = 0.5,
    regularization::Float64 = 1.0e-12,
)
    width > 0.0 && sigma_max > 0.0 || return nothing
    lower, upper = distributed_mesh_bounds(distributed_dg)
    lengths = upper .- lower
    minimum(lengths) > 0.0 ||
        throw(ArgumentError("The distributed mesh bounds are degenerate."))
    2.0 * width < minimum(lengths) ||
        throw(
            ArgumentError(
                "PML width $width must be smaller than half the shortest " *
                "domain extent $(minimum(lengths)).",
            ),
        )

    directional_profile = function (coordinate, direction)
        left_interface = lower[direction] + width
        right_interface = upper[direction] - width
        return max(
            polynomial_pml_sigma(
                coordinate,
                left_interface,
                lower[direction];
                sigma_max = sigma_max,
                degree = degree,
            ),
            polynomial_pml_sigma(
                coordinate,
                right_interface,
                upper[direction];
                sigma_max = sigma_max,
                degree = degree,
            ),
        )
    end

    return build_maxwell_nonlinear_pml(
        distributed_dg;
        sigma_x = (x, y, z) -> directional_profile(x, 1),
        sigma_y = (x, y, z) -> directional_profile(y, 2),
        sigma_z = (x, y, z) -> directional_profile(z, 3),
        a = a,
        regularization = regularization,
    )
end

function resolved_maxwell_cubature_order(
    polynomial_order::Integer,
    requested_order::Integer = 0;
    maximum_order::Integer = 20,
)
    order = requested_order == 0 ? max(2, 2 * Int(polynomial_order) + 4) : Int(requested_order)
    2 <= order <= maximum_order ||
        throw(
            ArgumentError(
                "DG order $polynomial_order with requested cubature " *
                "$requested_order resolves to cubature order $order, but " *
                "available Jaskowiec-Sukumar rules are 2:$maximum_order.",
            ),
        )
    return order
end

function validate_element_partition(
    partition::AbstractVector{<:Integer},
    mesh::RawVTUMesh,
    nranks::Integer;
    partition_count::Integer = nranks,
    partition_path::AbstractString = "",
    mesh_path::AbstractString = "",
    require_all_ranks::Bool = true,
)
    length(partition) == size(mesh.tets, 2) ||
        error(
            "Partition$(isempty(partition_path) ? "" : " file $partition_path") " *
            "has $(length(partition)) entries, but mesh" *
            "$(isempty(mesh_path) ? "" : " $mesh_path") has " *
            "$(size(mesh.tets, 2)) tetrahedra.",
        )
    all(part -> 0 <= part < partition_count, partition) ||
        error("Partition entries must be zero-based ranks in 0:$(partition_count - 1).")
    require_all_ranks &&
        all(part -> any(==(part), partition), 0:(partition_count - 1)) ||
        error("Every MPI rank must own at least one tetrahedron.")
    partition_count == nranks ||
        throw(
            ArgumentError(
                "partition_count=$partition_count must equal the MPI rank " *
                "count $nranks for this driver path.",
            ),
        )
    return nothing
end

function collect_mpi_error_message(
    local_error::Union{Nothing, String},
    comm::MPI.Comm,
    context::AbstractString;
    root::Int = 0,
)
    rank = MPI.Comm_rank(comm)
    errors = MPI.gather(local_error, comm; root = root)
    message = nothing
    if rank == root
        failures = [
            "rank $(index - 1): $error"
            for (index, error) in enumerate(errors)
            if error !== nothing
        ]
        !isempty(failures) &&
            (message = string(context, " failed:\n", join(failures, "\n")))
    end
    message = MPI.bcast(message, comm; root = root)
    return message
end

function throw_if_mpi_error(
    local_error::Union{Nothing, String},
    comm::MPI.Comm,
    context::AbstractString;
    root::Int = 0,
)
    message = collect_mpi_error_message(local_error, comm, context; root = root)
    message === nothing || error(message)
    return nothing
end

function open_diagnostic_history_files(
    energy_path::AbstractString,
    quadrature_path::AbstractString;
    restarting::Bool,
)
    energy_io = nothing
    quadrature_io = nothing
    try
        energy_append = restarting && isfile(energy_path) && filesize(energy_path) > 0
        quadrature_append =
            restarting && isfile(quadrature_path) && filesize(quadrature_path) > 0
        energy_append == quadrature_append ||
            error("Energy and quadrature history append states disagree.")
        energy_io = open(energy_path, energy_append ? "a" : "w")
        quadrature_io = open(quadrature_path, quadrature_append ? "a" : "w")
        return DiagnosticHistoryFiles(energy_io, quadrature_io, !energy_append)
    catch error
        energy_io !== nothing && close(energy_io)
        quadrature_io !== nothing && close(quadrature_io)
        rethrow(error)
    end
end

function close_diagnostic_history_files(files::DiagnosticHistoryFiles)
    files.energy_io !== nothing && close(files.energy_io)
    files.quadrature_io !== nothing && close(files.quadrature_io)
    return nothing
end

function restart_initial_energy(
    state::DistributedCheckpointState,
    fallback::Float64,
)
    value = get(state.metadata, "initial_energy_total", fallback)
    return value isa Number ? Float64(value) : parse(Float64, string(value))
end

function restart_metadata_number(
    state::DistributedCheckpointState,
    name::AbstractString,
)
    haskey(state.metadata, name) || return nothing
    value = state.metadata[name]
    return value isa Number ? Float64(value) : parse(Float64, string(value))
end

function validate_restart_numbers(
    state::DistributedCheckpointState,
    expected_pairs;
    atol::Float64 = 1.0e-14,
)
    for (name, expected) in expected_pairs
        stored = restart_metadata_number(state, string(name))
        stored === nothing && continue
        isapprox(stored, Float64(expected); rtol = 0.0, atol = atol) ||
            throw(
                ArgumentError(
                    "Checkpoint $name=$stored does not match the requested " *
                    "value $expected.",
                ),
            )
    end
    return nothing
end

function validate_restart_strings(state::DistributedCheckpointState, expected_pairs)
    for (name, expected) in expected_pairs
        key = string(name)
        haskey(state.metadata, key) || continue
        string(state.metadata[key]) == string(expected) ||
            throw(
                ArgumentError(
                    "Checkpoint $key=$(state.metadata[key]) does not match " *
                    "the requested value $expected.",
                ),
            )
    end
    return nothing
end

function restart_time_plan(
    final_time::Float64,
    start_time::Float64,
    checkpoint_dt::Float64,
)
    remaining = final_time - start_time
    remaining > 0.0 ||
        throw(
            ArgumentError(
                "--final-time ($final_time) must be greater than the " *
                "checkpoint time ($start_time).",
            ),
        )
    ratio = remaining / checkpoint_dt
    nearest = round(Int, ratio)
    if nearest >= 1 && isapprox(ratio, nearest; rtol = 1.0e-10, atol = 1.0e-12)
        return RestartTimePlan(checkpoint_dt, nearest)
    end
    nsteps = max(1, ceil(Int, ratio))
    return RestartTimePlan(remaining / nsteps, nsteps)
end

function write_cavity_parallel_fields(
    output_basename::AbstractString,
    distributed_dg::DistributedDGDiscretization,
    U::MaxwellField;
    time::Float64,
    epsilon::Float64,
    mu::Float64,
    boundary_condition::Symbol = :pec,
)
    exact_electric, exact_magnetic = exact_cavity_mode_functions(
        time;
        epsilon = epsilon,
        mu = mu,
        boundary_condition = boundary_condition,
    )
    return write_parallel_maxwell_fields(
        String(output_basename),
        distributed_dg,
        U;
        time = time,
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
    )
end

function write_periodic_parallel_fields(
    output_basename::AbstractString,
    distributed_dg::DistributedDGDiscretization,
    U::MaxwellField;
    time::Float64,
    epsilon::Float64,
    mu::Float64,
    wave::PlaneWaveParameters,
)
    exact_electric, exact_magnetic = exact_periodic_wave_functions(
        time;
        epsilon = epsilon,
        mu = mu,
        wave = wave,
    )
    return write_parallel_maxwell_fields(
        String(output_basename),
        distributed_dg,
        U;
        time = time,
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
    )
end

function write_cavity_paraview_snapshot!(
    entries,
    output_dir::AbstractString,
    distributed_dg::DistributedDGDiscretization,
    U::MaxwellField,
    step::Int,
    time::Float64;
    epsilon::Float64,
    mu::Float64,
    boundary_condition::Symbol = :pec,
)
    exact_electric, exact_magnetic = exact_cavity_mode_functions(
        time;
        epsilon = epsilon,
        mu = mu,
        boundary_condition = boundary_condition,
    )
    return write_maxwell_paraview_snapshot!(
        entries,
        String(output_dir),
        distributed_dg,
        U,
        step,
        time;
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
    )
end

function write_periodic_paraview_snapshot!(
    entries,
    output_dir::AbstractString,
    distributed_dg::DistributedDGDiscretization,
    U::MaxwellField,
    step::Int,
    time::Float64,
    wave::PlaneWaveParameters;
    epsilon::Float64,
    mu::Float64,
)
    exact_electric, exact_magnetic = exact_periodic_wave_functions(
        time;
        epsilon = epsilon,
        mu = mu,
        wave = wave,
    )
    return write_maxwell_paraview_snapshot!(
        entries,
        String(output_dir),
        distributed_dg,
        U,
        step,
        time;
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
    )
end

function write_cavity_integration_points(
    output_dir::AbstractString,
    distributed_dg::DistributedDGDiscretization,
    U::MaxwellField,
    time::Float64,
    cubature_order::Int;
    epsilon::Float64,
    mu::Float64,
    boundary_condition::Symbol = :pec,
)
    exact_electric, exact_magnetic = exact_cavity_mode_functions(
        time;
        epsilon = epsilon,
        mu = mu,
        boundary_condition = boundary_condition,
    )
    exact_curl_electric, exact_curl_magnetic =
        exact_cavity_mode_curl_functions(
            time;
            epsilon = epsilon,
            mu = mu,
            boundary_condition = boundary_condition,
        )
    return write_maxwell_integration_points(
        String(output_dir),
        distributed_dg,
        U,
        cubature_order;
        epsilon = epsilon,
        mu = mu,
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
        exact_curl_electric = exact_curl_electric,
        exact_curl_magnetic = exact_curl_magnetic,
    )
end

function write_periodic_integration_points(
    output_dir::AbstractString,
    distributed_dg::DistributedDGDiscretization,
    U::MaxwellField,
    time::Float64,
    cubature_order::Int;
    epsilon::Float64,
    mu::Float64,
    wave::PlaneWaveParameters,
)
    exact_electric, exact_magnetic = exact_periodic_wave_functions(
        time;
        epsilon = epsilon,
        mu = mu,
        wave = wave,
    )
    exact_curl_electric, exact_curl_magnetic =
        exact_periodic_wave_curl_functions(
            time;
            epsilon = epsilon,
            mu = mu,
            wave = wave,
        )
    return write_maxwell_integration_points(
        String(output_dir),
        distributed_dg,
        U,
        cubature_order;
        epsilon = epsilon,
        mu = mu,
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
        exact_curl_electric = exact_curl_electric,
        exact_curl_magnetic = exact_curl_magnetic,
    )
end

function distributed_mesh_characteristic_h(
    distributed_dg::DistributedDGDiscretization;
    reference_volume::Float64 = REFERENCE_TETRAHEDRON_VOLUME,
)
    local_volume = 0.0
    for elem in distributed_dg.distributed_mesh.partition.owned
        local_volume +=
            reference_volume *
            distributed_dg.dg.mappings.tet_mappings[elem].absdetJ
    end
    global_volume = MPI.Allreduce(local_volume, +, distributed_dg.comm)
    nelements = MPI.Allreduce(
        length(distributed_dg.distributed_mesh.partition.owned),
        +,
        distributed_dg.comm,
    )
    return (global_volume / nelements)^(1.0 / 3.0)
end

function balanced_spatial_partition(mesh::RawVTUMesh, nranks::Int)
    nelements = size(mesh.tets, 2)
    nelements >= nranks ||
        throw(
            ArgumentError(
                "Mesh has $nelements tetrahedra for $nranks MPI ranks.",
            ),
        )

    centroids = Vector{NTuple{3, Float64}}(undef, nelements)
    for elem in 1:nelements
        nodes = @view mesh.tets[:, elem]
        centroids[elem] = (
            sum(@view mesh.points[1, nodes]) / 4.0,
            sum(@view mesh.points[2, nodes]) / 4.0,
            sum(@view mesh.points[3, nodes]) / 4.0,
        )
    end

    order = sortperm(1:nelements; by = elem -> centroids[elem])
    partition = zeros(Int, nelements)
    for (position, elem) in enumerate(order)
        partition[elem] =
            min(div((position - 1) * nranks, nelements), nranks - 1)
    end
    return partition
end

function structured_box_node_id(i::Int, j::Int, k::Int, nx::Int, ny::Int)
    return 1 + i + (nx + 1) * (j + (ny + 1) * k)
end

function deterministic_unit_interval(seed::Integer, values::Integer...)
    state = reinterpret(UInt64, Int64(seed))
    for value in values
        mix = reinterpret(UInt64, Int64(value))
        state = xor(
            state,
            mix + 0x9e3779b97f4a7c15 + (state << 6) + (state >> 2),
        )
    end
    state = xor(state, state >> 30)
    state *= 0xbf58476d1ce4e5b9
    state = xor(state, state >> 27)
    state *= 0x94d049bb133111eb
    state = xor(state, state >> 31)
    return Float64(state >> 11) / 9007199254740992.0
end

function structured_box_points(
    nx::Int,
    ny::Int,
    nz::Int;
    lower::NTuple{3, Float64} = (0.0, 0.0, 0.0),
    upper::NTuple{3, Float64} = (1.0, 1.0, 1.0),
    jitter::Float64 = 0.0,
    seed::Int = 0,
)
    nx >= 1 && ny >= 1 && nz >= 1 ||
        throw(ArgumentError("Structured mesh dimensions must be positive."))
    0.0 <= jitter < 0.25 ||
        throw(ArgumentError("jitter must lie in [0, 0.25)."))
    xs = collect(range(lower[1], upper[1]; length = nx + 1))
    ys = collect(range(lower[2], upper[2]; length = ny + 1))
    zs = collect(range(lower[3], upper[3]; length = nz + 1))
    lengths = axis_aligned_box_lengths(AxisAlignedBox(lower, upper))
    spacing = minimum((lengths[1] / nx, lengths[2] / ny, lengths[3] / nz))
    points = zeros(Float64, 3, (nx + 1) * (ny + 1) * (nz + 1))

    for k in 0:nz, j in 0:ny, i in 0:nx
        node = structured_box_node_id(i, j, k, nx, ny)
        x = xs[i + 1]
        y = ys[j + 1]
        z = zs[k + 1]
        if 0 < i < nx && 0 < j < ny && 0 < k < nz && jitter > 0.0
            scale = jitter * spacing
            x += scale * (2.0 * deterministic_unit_interval(seed, nx, ny, nz, i, j, k, 1) - 1.0)
            y += scale * (2.0 * deterministic_unit_interval(seed, nx, ny, nz, i, j, k, 2) - 1.0)
            z += scale * (2.0 * deterministic_unit_interval(seed, nx, ny, nz, i, j, k, 3) - 1.0)
        end
        points[:, node] .= (x, y, z)
    end
    return points
end

function structured_box_tets(nx::Int, ny::Int, nz::Int)
    tetrahedra = NTuple{4, Int}[]
    for k in 0:(nz - 1), j in 0:(ny - 1), i in 0:(nx - 1)
        v000 = structured_box_node_id(i, j, k, nx, ny)
        v100 = structured_box_node_id(i + 1, j, k, nx, ny)
        v010 = structured_box_node_id(i, j + 1, k, nx, ny)
        v110 = structured_box_node_id(i + 1, j + 1, k, nx, ny)
        v001 = structured_box_node_id(i, j, k + 1, nx, ny)
        v101 = structured_box_node_id(i + 1, j, k + 1, nx, ny)
        v011 = structured_box_node_id(i, j + 1, k + 1, nx, ny)
        v111 = structured_box_node_id(i + 1, j + 1, k + 1, nx, ny)
        append!(
            tetrahedra,
            (
                (v000, v100, v110, v111),
                (v000, v110, v010, v111),
                (v000, v010, v011, v111),
                (v000, v011, v001, v111),
                (v000, v001, v101, v111),
                (v000, v101, v100, v111),
            ),
        )
    end
    return reduce(hcat, collect.(tetrahedra))
end

function structured_box_mesh(
    nx::Int,
    ny::Int,
    nz::Int;
    lower::NTuple{3, Float64} = (0.0, 0.0, 0.0),
    upper::NTuple{3, Float64} = (1.0, 1.0, 1.0),
    boundary_id::Union{Integer, Function} = DEFAULT_CAVITY_BOUNDARY_ID,
    jitter::Float64 = 0.0,
    seed::Int = 0,
    boundary_tag_name::String = "boundary_id",
)
    points = structured_box_points(
        nx,
        ny,
        nz;
        lower = lower,
        upper = upper,
        jitter = jitter,
        seed = seed,
    )
    tets = structured_box_tets(nx, ny, nz)
    tris = build_tetrahedral_boundary_triangles(tets)
    box = AxisAlignedBox(lower, upper)
    tolerance = max(1.0e-10, 1.0e-10 * maximum(axis_aligned_box_lengths(box)))
    boundary_ids = if boundary_id isa Function
        [
            Int(boundary_id(points, @view(tris[:, triangle]); box = box, tolerance = tolerance))
            for triangle in axes(tris, 2)
        ]
    else
        fill(Int(boundary_id), size(tris, 2))
    end
    return raw_mesh_with_boundary_triangles(
        points,
        tets,
        tris,
        boundary_ids;
        boundary_tag_name = boundary_tag_name,
    )
end

function observed_convergence_rate(
    coarse_error::Float64,
    fine_error::Float64,
    coarse_h::Float64,
    fine_h::Float64,
)
    fine_error > 0.0 && coarse_error > 0.0 || return missing
    fine_h > 0.0 && coarse_h > 0.0 || return missing
    coarse_h != fine_h || return missing
    return log(coarse_error / fine_error) / log(coarse_h / fine_h)
end

function component_convergence_rate(
    coarse_error::Float64,
    fine_error::Float64,
    coarse_h::Float64,
    fine_h::Float64;
    floor::Float64 = DEFAULT_COMPONENT_RATE_ERROR_FLOOR,
)
    coarse_error > floor || return missing
    return observed_convergence_rate(coarse_error, fine_error, coarse_h, fine_h)
end

function convergence_rate_pass(
    rate::Union{Missing, Float64},
    expected::Float64;
    tolerance::Float64 = DEFAULT_MAXWELL_CONVERGENCE_RATE_TOLERANCE,
)
    return rate !== missing && rate >= expected - tolerance
end

function convergence_rate_pass(
    rate::Union{Missing, Float64},
    expected::Float64,
    tolerance::Float64,
)
    return convergence_rate_pass(rate, expected; tolerance = tolerance)
end

function formatted_optional_rate(rate::Union{Missing, Float64})
    return rate === missing ? "-" : @sprintf("%.3f", rate)
end

function write_convergence_table(
    path::AbstractString,
    results,
    columns::AbstractVector{ConvergenceTableColumn},
)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join((column.name for column in columns), ","))
        for result in results
            println(
                io,
                join((column.getter(result) for column in columns), ","),
            )
        end
    end
    return path
end
