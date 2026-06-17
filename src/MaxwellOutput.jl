function reference_vertex_node_ids(ref::ReferenceTet; tol::Float64 = 1e-12)
    reference_vertices = (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
    )
    ids = Vector{Int}(undef, 4)

    for vertex in 1:4
        rv, sv, tv = reference_vertices[vertex]
        matches = findall(
            node -> abs(ref.r[node] - rv) < tol &&
                    abs(ref.s[node] - sv) < tol &&
                    abs(ref.t[node] - tv) < tol,
            1:ref.Np,
        )
        length(matches) == 1 ||
            error("Could not identify reference tetrahedron vertex $vertex.")
        ids[vertex] = only(matches)
    end

    return ids
end

function vtk_lagrange_triangle_barycentric_indices(order::Int)
    order >= 0 || throw(ArgumentError("Lagrange order must be non-negative."))
    order == 0 && return NTuple{3, Int}[(0, 0, 0)]

    indices = NTuple{3, Int}[
        (0, 0, order),
        (order, 0, 0),
        (0, order, 0),
    ]
    for offset in 1:(order - 1)
        push!(indices, (offset, 0, order - offset))
    end
    for offset in 1:(order - 1)
        push!(indices, (order - offset, offset, 0))
    end
    for offset in 1:(order - 1)
        push!(indices, (0, order - offset, offset))
    end
    if order >= 3
        for inner in vtk_lagrange_triangle_barycentric_indices(order - 3)
            push!(indices, ntuple(i -> inner[i] + 1, 3))
        end
    end
    return indices
end

function vtk_lagrange_tetra_barycentric_indices(order::Int)
    order >= 0 || throw(ArgumentError("Lagrange order must be non-negative."))
    order == 0 && return NTuple{4, Int}[(0, 0, 0, 0)]

    indices = NTuple{4, Int}[
        (0, 0, 0, order),
        (order, 0, 0, 0),
        (0, order, 0, 0),
        (0, 0, order, 0),
    ]
    edge_vertices = (
        (0, 1),
        (1, 2),
        (2, 0),
        (0, 3),
        (1, 3),
        (2, 3),
    )
    barycentric_slot = (4, 1, 2, 3)
    for (first_vertex, second_vertex) in edge_vertices
        for offset in 1:(order - 1)
            values = zeros(Int, 4)
            values[barycentric_slot[first_vertex + 1]] = order - offset
            values[barycentric_slot[second_vertex + 1]] = offset
            push!(indices, Tuple(values))
        end
    end

    if order >= 3
        face_vertices = (
            (0, 1, 3),
            (2, 3, 1),
            (0, 3, 2),
            (0, 2, 1),
        )
        face_interior = [
            ntuple(i -> value[i] + 1, 3)
            for value in
                vtk_lagrange_triangle_barycentric_indices(order - 3)
        ]
        for (first_vertex, second_vertex, third_vertex) in face_vertices
            for local_index in face_interior
                values = zeros(Int, 4)
                values[barycentric_slot[first_vertex + 1]] = local_index[3]
                values[barycentric_slot[second_vertex + 1]] = local_index[1]
                values[barycentric_slot[third_vertex + 1]] = local_index[2]
                push!(indices, Tuple(values))
            end
        end
    end

    if order >= 4
        for inner in vtk_lagrange_tetra_barycentric_indices(order - 4)
            push!(indices, ntuple(i -> inner[i] + 1, 4))
        end
    end
    return indices
end

function vtk_lagrange_tetra_node_ids(ref::ReferenceTet)
    ref.N >= 1 ||
        throw(ArgumentError("ParaView Lagrange output requires order >= 1."))
    nodes_by_barycentric = Dict{NTuple{4, Int}, Int}()
    for node in 1:ref.Np
        lambda2 = round(Int, ref.N * (ref.r[node] + 1.0) / 2.0)
        lambda3 = round(Int, ref.N * (ref.s[node] + 1.0) / 2.0)
        lambda4 = round(Int, ref.N * (ref.t[node] + 1.0) / 2.0)
        lambda1 = ref.N - lambda2 - lambda3 - lambda4
        nodes_by_barycentric[(lambda2, lambda3, lambda4, lambda1)] = node
    end

    vtk_indices = vtk_lagrange_tetra_barycentric_indices(ref.N)
    length(vtk_indices) == ref.Np ||
        error("Incorrect VTK Lagrange node count for order $(ref.N).")
    all(haskey(nodes_by_barycentric, index) for index in vtk_indices) ||
        error("The DG and VTK Lagrange node sets do not match.")
    return [nodes_by_barycentric[index] for index in vtk_indices]
end

function write_parallel_maxwell_fields(
    output_basename::String,
    distributed_dg::DistributedDGDiscretization,
    U::MaxwellField;
    time::Float64,
    exact_electric,
    exact_magnetic,
)
    rank = MPI.Comm_rank(distributed_dg.comm)
    nranks = MPI.Comm_size(distributed_dg.comm)
    mesh = distributed_dg.dg.mesh
    distributed_mesh = distributed_dg.distributed_mesh
    owned = distributed_mesh.partition.owned
    ref = distributed_dg.dg.ref
    vtk_node_ids = vtk_lagrange_tetra_node_ids(ref)
    nowned = length(owned)
    nodes_per_element = ref.Np

    points = zeros(Float64, 3, nodes_per_element * nowned)
    electric = zeros(Float64, 3, nodes_per_element * nowned)
    magnetic = zeros(Float64, 3, nodes_per_element * nowned)
    exact_electric_values = similar(electric)
    exact_magnetic_values = similar(magnetic)
    cells = Vector{MeshCell}(undef, nowned)
    global_element_ids = Vector{Int}(undef, nowned)

    for (owned_index, local_elem) in enumerate(owned)
        first_node = nodes_per_element * (owned_index - 1) + 1
        cell_nodes = collect(first_node:(first_node + nodes_per_element - 1))
        cells[owned_index] =
            MeshCell(VTKCellTypes.VTK_LAGRANGE_TETRAHEDRON, cell_nodes)
        global_element_ids[owned_index] =
            distributed_mesh.elements.global_ids[local_elem]

        tet_nodes = @view mesh.tets[:, local_elem]
        for vtk_node in 1:nodes_per_element
            output_node = cell_nodes[vtk_node]
            field_node = vtk_node_ids[vtk_node]
            x, y, z = map_to_physical(
                mesh.points,
                tet_nodes,
                ref.r[field_node],
                ref.s[field_node],
                ref.t[field_node],
            )
            points[:, output_node] .= (x, y, z)
            electric[:, output_node] .= (
                U.Ex[field_node, local_elem],
                U.Ey[field_node, local_elem],
                U.Ez[field_node, local_elem],
            )
            magnetic[:, output_node] .= (
                U.Hx[field_node, local_elem],
                U.Hy[field_node, local_elem],
                U.Hz[field_node, local_elem],
            )
            exact_electric_values[:, output_node] .= exact_electric(x, y, z)
            exact_magnetic_values[:, output_node] .= exact_magnetic(x, y, z)
        end
    end

    electric_error = electric - exact_electric_values
    magnetic_error = magnetic - exact_magnetic_values
    electric_magnitude = vec(sqrt.(sum(abs2, electric; dims = 1)))
    magnetic_magnitude = vec(sqrt.(sum(abs2, magnetic; dims = 1)))

    return pvtk_grid(
        output_basename,
        points,
        cells;
        part = rank + 1,
        nparts = nranks,
        ismain = rank == 0,
        append = false,
        compress = false,
    ) do vtk
        vtk[
            "ElectricField",
            VTKPointData(),
            component_names = ("Ex", "Ey", "Ez"),
        ] = electric
        vtk[
            "MagneticField",
            VTKPointData(),
            component_names = ("Hx", "Hy", "Hz"),
        ] = magnetic
        vtk[
            "ExactElectricField",
            VTKPointData(),
            component_names = ("ExactEx", "ExactEy", "ExactEz"),
        ] = exact_electric_values
        vtk[
            "ExactMagneticField",
            VTKPointData(),
            component_names = ("ExactHx", "ExactHy", "ExactHz"),
        ] = exact_magnetic_values
        vtk[
            "ElectricFieldError",
            VTKPointData(),
            component_names = ("ErrorEx", "ErrorEy", "ErrorEz"),
        ] = electric_error
        vtk[
            "MagneticFieldError",
            VTKPointData(),
            component_names = ("ErrorHx", "ErrorHy", "ErrorHz"),
        ] = magnetic_error
        vtk["ElectricFieldMagnitude", VTKPointData()] = electric_magnitude
        vtk["MagneticFieldMagnitude", VTKPointData()] = magnetic_magnitude
        vtk["GlobalElementId", VTKCellData()] = global_element_ids
        vtk["OwnerRank", VTKCellData()] = fill(rank, nowned)
        vtk["PolynomialOrder", VTKCellData()] =
            fill(distributed_dg.dg.ref.N, nowned)
        vtk["TimeValue", VTKFieldData()] = time
    end
end

function xml_escape(value::AbstractString)
    escaped = replace(value, '&' => "&amp;")
    escaped = replace(escaped, '<' => "&lt;")
    escaped = replace(escaped, '>' => "&gt;")
    escaped = replace(escaped, '"' => "&quot;")
    return replace(escaped, '\'' => "&apos;")
end

function atomic_output_file(writer::Function, path::String)
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

function checkpoint_step_dir(root::String, step::Int)
    return joinpath(root, @sprintf("step%08d", step))
end

function write_latest_checkpoint(
    checkpoint_root::String,
    checkpoint_path::String,
    step::Int,
    time::Float64,
)
    path = joinpath(checkpoint_root, "latest.toml")
    atomic_output_file(path) do io
        println(io, "step = ", step)
        println(io, "time = ", time)
        println(
            io,
            "path = \"",
            replace(relpath(checkpoint_path, checkpoint_root), '"' => "\\\""),
            "\"",
        )
    end
    return path
end

function collectively_write_latest_checkpoint(
    comm::MPI.Comm,
    checkpoint_root::String,
    checkpoint_path::String,
    step::Int,
    time::Float64,
)
    rank = MPI.Comm_rank(comm)
    root_error = nothing
    if rank == 0
        try
            write_latest_checkpoint(
                checkpoint_root,
                checkpoint_path,
                step,
                time,
            )
        catch error
            root_error = sprint(showerror, error)
        end
    end
    root_error = MPI.bcast(root_error, comm; root = 0)
    root_error === nothing ||
        error("Latest-checkpoint pointer writing failed: $root_error")
    MPI.Barrier(comm)
    return nothing
end

function read_paraview_series(path::String)
    entries = NamedTuple{(:step, :time, :dataset), Tuple{Int, Float64, String}}[]
    isfile(path) || return entries

    open(path, "r") do io
        first = true
        for line in eachline(io)
            if first
                first = false
                continue
            end
            stripped = strip(line)
            isempty(stripped) && continue
            fields = split(stripped, ','; limit = 3)
            length(fields) == 3 ||
                error("Invalid ParaView series row in $path: $line")
            push!(
                entries,
                (
                    step = parse(Int, fields[1]),
                    time = parse(Float64, fields[2]),
                    dataset = fields[3],
                ),
            )
        end
    end
    return entries
end

function write_paraview_series(output_dir::String, entries)
    entries_by_step = Dict(entry.step => entry for entry in entries)
    sorted_entries =
        sort(collect(values(entries_by_step)); by = entry -> entry.step)
    csv_path = joinpath(output_dir, "paraview_series.csv")
    atomic_output_file(csv_path) do io
        println(io, "step,time,dataset")
        for entry in sorted_entries
            println(io, entry.step, ',', entry.time, ',', entry.dataset)
        end
    end

    pvd_path = joinpath(output_dir, "fields.pvd")
    atomic_output_file(pvd_path) do io
        println(io, "<?xml version=\"1.0\"?>")
        println(
            io,
            "<VTKFile type=\"Collection\" version=\"0.1\" " *
            "byte_order=\"LittleEndian\">",
        )
        println(io, "  <Collection>")
        for entry in sorted_entries
            println(
                io,
                "    <DataSet timestep=\"", entry.time,
                "\" group=\"\" part=\"0\" file=\"",
                xml_escape(entry.dataset),
                "\"/>",
            )
        end
        println(io, "  </Collection>")
        println(io, "</VTKFile>")
    end
    return sorted_entries
end

function collective_root_action(action::Function, comm::MPI.Comm, context::String)
    rank = MPI.Comm_rank(comm)
    local_error = nothing
    if rank == 0
        try
            action()
        catch error
            local_error = sprint(showerror, error)
        end
    end
    local_error = MPI.bcast(local_error, comm; root = 0)
    local_error === nothing || error("$context failed: $local_error")
    return nothing
end

function collective_rank_action(action::Function, comm::MPI.Comm, context::String)
    local_error = nothing
    try
        action()
    catch error
        local_error = sprint(showerror, error)
    end
    errors = MPI.gather(local_error, comm; root = 0)
    rank = MPI.Comm_rank(comm)
    root_error = nothing
    if rank == 0
        failing_rank = findfirst(!isnothing, errors)
        if failing_rank !== nothing
            root_error =
                "rank $(failing_rank - 1): $(errors[failing_rank])"
        end
    end
    root_error = MPI.bcast(root_error, comm; root = 0)
    root_error === nothing || error("$context failed: $root_error")
    return nothing
end

function write_maxwell_paraview_snapshot!(
    entries,
    output_dir::String,
    distributed_dg::DistributedDGDiscretization,
    U::MaxwellField,
    step::Int,
    time::Float64;
    exact_electric,
    exact_magnetic,
)
    rank = MPI.Comm_rank(distributed_dg.comm)
    fields_dir = joinpath(output_dir, "fields")
    collective_root_action(
        distributed_dg.comm,
        "ParaView output directory creation",
    ) do
        mkpath(fields_dir)
    end
    MPI.Barrier(distributed_dg.comm)

    basename = @sprintf("fields_step%08d", step)
    output_basename = joinpath(fields_dir, basename)
    collective_rank_action(
        distributed_dg.comm,
        "Parallel VTK snapshot writing at step $step",
    ) do
        write_parallel_maxwell_fields(
            output_basename,
            distributed_dg,
            U;
            time = time,
            exact_electric = exact_electric,
            exact_magnetic = exact_magnetic,
        )
    end
    MPI.Barrier(distributed_dg.comm)

    root_error = nothing
    if rank == 0
        try
            filter!(entry -> entry.step != step, entries)
            push!(
                entries,
                (
                    step = step,
                    time = time,
                    dataset = joinpath("fields", basename * ".pvtu"),
                ),
            )
            replacement = write_paraview_series(output_dir, entries)
            empty!(entries)
            append!(entries, replacement)
        catch error
            root_error = sprint(showerror, error)
        end
    end
    root_error = MPI.bcast(root_error, distributed_dg.comm; root = 0)
    root_error === nothing ||
        error("ParaView collection writing failed: $root_error")
    MPI.Barrier(distributed_dg.comm)
    return nothing
end

function write_maxwell_integration_points(
    output_dir::String,
    distributed_dg::DistributedDGDiscretization,
    U::MaxwellField,
    cubature_order::Int;
    epsilon::Float64,
    mu::Float64,
    exact_electric,
    exact_magnetic,
    exact_curl_electric,
    exact_curl_magnetic,
)
    rank = MPI.Comm_rank(distributed_dg.comm)
    rank_label = lpad(string(rank), 4, '0')
    output_path =
        joinpath(output_dir, "integration_points_rank$rank_label.csv")
    cubature_points, cubature_weights, number_cubature_points =
        get_JaskowiecSukumar_cubature(cubature_order)
    interpolation =
        reference_interpolation_matrix(distributed_dg.dg.ref, cubature_points)

    mesh = distributed_dg.dg.mesh
    distributed_mesh = distributed_dg.distributed_mesh
    mappings = distributed_dg.dg.mappings.tet_mappings
    physical_operators = distributed_dg.dg.physops.elements

    open(output_path, "w") do io
        println(
            io,
            "global_element,local_element,integration_point,r,s,t,x,y,z," *
            "physical_weight,Ex,Ey,Ez,exact_Ex,exact_Ey,exact_Ez," *
            "Hx,Hy,Hz,exact_Hx,exact_Hy,exact_Hz," *
            "electric_energy_density,magnetic_energy_density," *
            "total_energy_density,exact_total_energy_density," *
            "energy_density_error,optical_chirality_density," *
            "exact_optical_chirality_density," *
            "optical_chirality_density_error,electric_charge_density," *
            "magnetic_charge_density,exact_electric_charge_density," *
            "exact_magnetic_charge_density,momentum_x,momentum_y," *
            "momentum_z,exact_momentum_x,exact_momentum_y," *
            "exact_momentum_z,angular_momentum_x,angular_momentum_y," *
            "angular_momentum_z,exact_angular_momentum_x," *
            "exact_angular_momentum_y,exact_angular_momentum_z",
        )

        for elem in distributed_mesh.partition.owned
            global_elem = distributed_mesh.elements.global_ids[elem]
            tet_nodes = @view mesh.tets[:, elem]
            jacobian = mappings[elem].absdetJ
            operators = physical_operators[elem]

            @views begin
                Ex = U.Ex[:, elem]
                Ey = U.Ey[:, elem]
                Ez = U.Ez[:, elem]
                Hx = U.Hx[:, elem]
                Hy = U.Hy[:, elem]
                Hz = U.Hz[:, elem]
                divergence_electric =
                    operators.Dx * Ex +
                    operators.Dy * Ey +
                    operators.Dz * Ez
                divergence_magnetic =
                    operators.Dx * Hx +
                    operators.Dy * Hy +
                    operators.Dz * Hz
                curl_electric_x =
                    operators.Dy * Ez - operators.Dz * Ey
                curl_electric_y =
                    operators.Dz * Ex - operators.Dx * Ez
                curl_electric_z =
                    operators.Dx * Ey - operators.Dy * Ex
                curl_magnetic_x =
                    operators.Dy * Hz - operators.Dz * Hy
                curl_magnetic_y =
                    operators.Dz * Hx - operators.Dx * Hz
                curl_magnetic_z =
                    operators.Dx * Hy - operators.Dy * Hx

                for q in 1:number_cubature_points
                    r = cubature_points[q, 1]
                    s = cubature_points[q, 2]
                    t = cubature_points[q, 3]
                    x, y, z = map_to_physical(
                        mesh.points,
                        tet_nodes,
                        r,
                        s,
                        t,
                    )

                    exact_Ex, exact_Ey, exact_Ez =
                        exact_electric(x, y, z)
                    exact_Hx, exact_Hy, exact_Hz =
                        exact_magnetic(x, y, z)
                    row = view(interpolation, q, :)
                    numerical_Ex = dot(row, Ex)
                    numerical_Ey = dot(row, Ey)
                    numerical_Ez = dot(row, Ez)
                    numerical_Hx = dot(row, Hx)
                    numerical_Hy = dot(row, Hy)
                    numerical_Hz = dot(row, Hz)

                    electric_energy_density =
                        0.5 * epsilon *
                        (
                            numerical_Ex^2 +
                            numerical_Ey^2 +
                            numerical_Ez^2
                        )
                    magnetic_energy_density =
                        0.5 * mu *
                        (
                            numerical_Hx^2 +
                            numerical_Hy^2 +
                            numerical_Hz^2
                        )
                    total_energy_density =
                        electric_energy_density + magnetic_energy_density
                    exact_total_energy_density =
                        0.5 * epsilon *
                        (exact_Ex^2 + exact_Ey^2 + exact_Ez^2) +
                        0.5 * mu *
                        (exact_Hx^2 + exact_Hy^2 + exact_Hz^2)
                    numerical_optical_chirality =
                        optical_chirality_density(
                            (
                                numerical_Ex,
                                numerical_Ey,
                                numerical_Ez,
                            ),
                            (
                                dot(row, curl_electric_x),
                                dot(row, curl_electric_y),
                                dot(row, curl_electric_z),
                            ),
                            (
                                numerical_Hx,
                                numerical_Hy,
                                numerical_Hz,
                            ),
                            (
                                dot(row, curl_magnetic_x),
                                dot(row, curl_magnetic_y),
                                dot(row, curl_magnetic_z),
                            );
                            epsilon = epsilon,
                            mu = mu,
                        )
                    exact_optical_chirality =
                        optical_chirality_density(
                            (exact_Ex, exact_Ey, exact_Ez),
                            exact_curl_electric(x, y, z),
                            (exact_Hx, exact_Hy, exact_Hz),
                            exact_curl_magnetic(x, y, z);
                            epsilon = epsilon,
                            mu = mu,
                        )
                    electric_charge_density =
                        epsilon * dot(row, divergence_electric)
                    magnetic_charge_density =
                        mu * dot(row, divergence_magnetic)

                    momentum_scale = epsilon * mu
                    momentum_x =
                        momentum_scale *
                        (numerical_Ey * numerical_Hz -
                         numerical_Ez * numerical_Hy)
                    momentum_y =
                        momentum_scale *
                        (numerical_Ez * numerical_Hx -
                         numerical_Ex * numerical_Hz)
                    momentum_z =
                        momentum_scale *
                        (numerical_Ex * numerical_Hy -
                         numerical_Ey * numerical_Hx)
                    exact_momentum_x =
                        momentum_scale *
                        (exact_Ey * exact_Hz - exact_Ez * exact_Hy)
                    exact_momentum_y =
                        momentum_scale *
                        (exact_Ez * exact_Hx - exact_Ex * exact_Hz)
                    exact_momentum_z =
                        momentum_scale *
                        (exact_Ex * exact_Hy - exact_Ey * exact_Hx)

                    angular_momentum_x =
                        y * momentum_z - z * momentum_y
                    angular_momentum_y =
                        z * momentum_x - x * momentum_z
                    angular_momentum_z =
                        x * momentum_y - y * momentum_x
                    exact_angular_momentum_x =
                        y * exact_momentum_z - z * exact_momentum_y
                    exact_angular_momentum_y =
                        z * exact_momentum_x - x * exact_momentum_z
                    exact_angular_momentum_z =
                        x * exact_momentum_y - y * exact_momentum_x

                    values = (
                        global_elem,
                        elem,
                        q,
                        r,
                        s,
                        t,
                        x,
                        y,
                        z,
                        jacobian * cubature_weights[q],
                        numerical_Ex,
                        numerical_Ey,
                        numerical_Ez,
                        exact_Ex,
                        exact_Ey,
                        exact_Ez,
                        numerical_Hx,
                        numerical_Hy,
                        numerical_Hz,
                        exact_Hx,
                        exact_Hy,
                        exact_Hz,
                        electric_energy_density,
                        magnetic_energy_density,
                        total_energy_density,
                        exact_total_energy_density,
                        total_energy_density - exact_total_energy_density,
                        numerical_optical_chirality,
                        exact_optical_chirality,
                        numerical_optical_chirality -
                        exact_optical_chirality,
                        electric_charge_density,
                        magnetic_charge_density,
                        0.0,
                        0.0,
                        momentum_x,
                        momentum_y,
                        momentum_z,
                        exact_momentum_x,
                        exact_momentum_y,
                        exact_momentum_z,
                        angular_momentum_x,
                        angular_momentum_y,
                        angular_momentum_z,
                        exact_angular_momentum_x,
                        exact_angular_momentum_y,
                        exact_angular_momentum_z,
                    )
                    println(io, join(values, ','))
                end
            end
        end
    end

    return output_path
end


