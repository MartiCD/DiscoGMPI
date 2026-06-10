#!/usr/bin/env julia

# Distributed Maxwell cavity experiment using the Poisson-bracket formulation.
#
# From the DiscoGMPI repository root:
#   mpiexec -n 2 julia --project=. examples/distributed_poisson_bracket_maxwell.jl
#
# The default mesh ships with 2-rank and 4-rank METIS partitions. The legacy
# VTK mesh was generated as a periodic box, but the distributed solver does not
# yet support periodic boundaries. This experiment therefore derives every
# exterior tetrahedral face and treats it as a perfect electric conductor (PEC).

using MPI
using DiscoGMPI
using LinearAlgebra: dot
using WriteVTK:
    MeshCell,
    VTKCellTypes,
    VTKCellData,
    VTKFieldData,
    VTKPointData,
    pvtk_grid

include(
    joinpath(
        @__DIR__,
        "..",
        "src",
        "solver",
        "kernels",
        "JaskowiecSukumar.jl",
    ),
)

const PEC_BOUNDARY_ID = 10
const TET_FACE_NODE_IDS = (
    (2, 3, 4),
    (1, 4, 3),
    (1, 2, 4),
    (1, 3, 2),
)
const REF_TET_VERTEX_COORDS = (
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
)

struct ExperimentConfig
    mesh_path::String
    partition_path::String
    output_dir::String
    polynomial_order::Int
    esprk_order::Int
    final_time::Float64
    cfl::Float64
    epsilon::Float64
    mu::Float64
    energy_every::Int
    cubature_order::Int
end

struct MaxwellQuadratureDiagnostics
    cubature_order::Int
    electric_energy::Float64
    magnetic_energy::Float64
    total_energy::Float64
    exact_electric_energy::Float64
    exact_magnetic_energy::Float64
    exact_total_energy::Float64
    electric_l2::Float64
    exact_electric_l2::Float64
    electric_error_l2::Float64
    electric_relative_error::Float64
    magnetic_l2::Float64
    exact_magnetic_l2::Float64
    magnetic_error_l2::Float64
    magnetic_relative_error::Float64
    field_error_l2::Float64
    field_relative_error::Float64
    energy_density_l2::Float64
    exact_energy_density_l2::Float64
    energy_density_error_l2::Float64
    energy_density_relative_error::Float64
    electric_charge::Float64
    magnetic_charge::Float64
    exact_electric_charge::Float64
    exact_magnetic_charge::Float64
    linear_momentum_x::Float64
    linear_momentum_y::Float64
    linear_momentum_z::Float64
    exact_linear_momentum_x::Float64
    exact_linear_momentum_y::Float64
    exact_linear_momentum_z::Float64
    angular_momentum_x::Float64
    angular_momentum_y::Float64
    angular_momentum_z::Float64
    exact_angular_momentum_x::Float64
    exact_angular_momentum_y::Float64
    exact_angular_momentum_z::Float64
end

function usage(io::IO = stdout)
    println(io, """
Distributed Poisson-bracket Maxwell experiment

Usage:
  mpiexec -n <ranks> julia --project=. \\
    examples/distributed_poisson_bracket_maxwell.jl [options]

Options:
  --mesh PATH           Legacy tetrahedral VTK mesh of the unit cube.
                        Default: examples/meshes/tet_mesh.vtk
  --partition PATH      Zero-based METIS element partition.
                        Default: <mesh stem>.mesh.epart.<MPI ranks>
  --output-dir PATH     Output directory.
                        Default: output/distributed_poisson_bracket
  --order N             DG polynomial order (N >= 1). Default: 2
  --esprk-order N       ESPRK order in 1:6. Default: 4
  --final-time T        Simulation end time. Default: 0.25
  --cfl C               CFL used to estimate dt. Default: 0.05
  --epsilon VALUE       Electric permittivity. Default: 1.0
  --mu VALUE            Magnetic permeability. Default: 1.0
  --energy-every N      Write energy every N steps. Default: 1
  --cubature-order N    Jaskowiec-Sukumar volume cubature order.
                        Default: max(2, 2 * DG order + 4)
  --help                Show this message.

Outputs:
  energy.csv                       DG mass-matrix energy history.
  quadrature_diagnostics.csv       Energy, L2 errors, charges, and momenta.
  integration_points_rankNNNN.csv  Final integration-point values per rank.
  final_fields.pvtu                Parallel final E and H fields for ParaView.
  final_fields/*.vtu               One owned-cell VTK piece per MPI rank.
""")
end

function option_value(args::Vector{String}, i::Int, name::String)
    arg = args[i]
    prefix = name * "="

    if startswith(arg, prefix)
        return arg[(length(prefix) + 1):end], i
    end

    arg == name ||
        throw(ArgumentError("Unknown option '$arg'. Use --help for valid options."))
    i < length(args) ||
        throw(ArgumentError("Option '$name' requires a value."))

    return args[i + 1], i + 1
end

function parse_arguments(
    args::Vector{String},
    nranks::Int,
    repository_root::String,
)
    mesh_path = joinpath(repository_root, "examples", "meshes", "tet_mesh.vtk")
    partition_path = ""
    output_dir =
        joinpath(repository_root, "output", "distributed_poisson_bracket")
    polynomial_order = 2
    esprk_order = 4
    final_time = 0.25
    cfl = 0.05
    epsilon = 1.0
    mu = 1.0
    energy_every = 1
    cubature_order = 0

    i = 1
    while i <= length(args)
        arg = args[i]

        if arg == "--help" || arg == "-h"
            return nothing
        elseif startswith(arg, "--mesh")
            value, i = option_value(args, i, "--mesh")
            mesh_path = abspath(value)
        elseif startswith(arg, "--partition")
            value, i = option_value(args, i, "--partition")
            partition_path = abspath(value)
        elseif startswith(arg, "--output-dir")
            value, i = option_value(args, i, "--output-dir")
            output_dir = abspath(value)
        elseif startswith(arg, "--order")
            value, i = option_value(args, i, "--order")
            polynomial_order = parse(Int, value)
        elseif startswith(arg, "--esprk-order")
            value, i = option_value(args, i, "--esprk-order")
            esprk_order = parse(Int, value)
        elseif startswith(arg, "--final-time")
            value, i = option_value(args, i, "--final-time")
            final_time = parse(Float64, value)
        elseif startswith(arg, "--cfl")
            value, i = option_value(args, i, "--cfl")
            cfl = parse(Float64, value)
        elseif startswith(arg, "--epsilon")
            value, i = option_value(args, i, "--epsilon")
            epsilon = parse(Float64, value)
        elseif startswith(arg, "--mu")
            value, i = option_value(args, i, "--mu")
            mu = parse(Float64, value)
        elseif startswith(arg, "--energy-every")
            value, i = option_value(args, i, "--energy-every")
            energy_every = parse(Int, value)
        elseif startswith(arg, "--cubature-order")
            value, i = option_value(args, i, "--cubature-order")
            cubature_order = parse(Int, value)
        else
            throw(ArgumentError("Unknown option '$arg'. Use --help for valid options."))
        end

        i += 1
    end

    if isempty(partition_path)
        partition_path = splitext(mesh_path)[1] * ".mesh.epart.$nranks"
    end

    polynomial_order >= 1 ||
        throw(ArgumentError("--order must be at least 1."))
    1 <= esprk_order <= 6 ||
        throw(ArgumentError("--esprk-order must be in 1:6."))
    final_time > 0.0 ||
        throw(ArgumentError("--final-time must be positive."))
    cfl > 0.0 ||
        throw(ArgumentError("--cfl must be positive."))
    epsilon > 0.0 ||
        throw(ArgumentError("--epsilon must be positive."))
    mu > 0.0 ||
        throw(ArgumentError("--mu must be positive."))
    energy_every >= 1 ||
        throw(ArgumentError("--energy-every must be at least 1."))
    cubature_order == 0 || 2 <= cubature_order <= 20 ||
        throw(ArgumentError("--cubature-order must be in 2:20."))

    return ExperimentConfig(
        mesh_path,
        partition_path,
        output_dir,
        polynomial_order,
        esprk_order,
        final_time,
        cfl,
        epsilon,
        mu,
        energy_every,
        cubature_order,
    )
end

function sorted_face_key(nodes::NTuple{3, Int})
    values = sort(collect(nodes))
    return (values[1], values[2], values[3])
end

function build_boundary_tris(tets::Matrix{Int})
    counts = Dict{NTuple{3, Int}, Int}()
    oriented_nodes = Dict{NTuple{3, Int}, NTuple{3, Int}}()

    for elem in axes(tets, 2)
        tet = @view tets[:, elem]

        for ids in TET_FACE_NODE_IDS
            nodes = (tet[ids[1]], tet[ids[2]], tet[ids[3]])
            key = sorted_face_key(nodes)
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

    sort!(boundary_faces; by = sorted_face_key)
    return reduce(hcat, collect.(boundary_faces))
end

function load_pec_mesh(mesh_path::String)
    points, tets = read_mesh_file_tet_vtk(mesh_path)
    size(tets, 2) > 0 ||
        error("The mesh '$mesh_path' contains no tetrahedra.")

    tolerance = 1e-10
    bounds = [
        (minimum(@view points[dimension, :]),
         maximum(@view points[dimension, :]))
        for dimension in 1:3
    ]
    all(
        bound -> abs(bound[1]) <= tolerance &&
                 abs(bound[2] - 1.0) <= tolerance,
        bounds,
    ) ||
        error(
            "The analytical cavity mode requires a [0,1]^3 mesh; " *
            "coordinate bounds are $bounds.",
        )

    tris = build_boundary_tris(tets)
    ntets = size(tets, 2)
    ntris = size(tris, 2)
    tet_cell_ids = collect(1:ntets)
    tri_cell_ids = collect((ntets + 1):(ntets + ntris))
    boundary_ids = zeros(Int, ntets + ntris)
    boundary_ids[tri_cell_ids] .= PEC_BOUNDARY_ID

    mesh = RawVTUMesh(
        points,
        tets,
        tris,
        tet_cell_ids,
        tri_cell_ids,
        Dict{String, Any}("boundary_id" => boundary_ids),
    )
    check_mesh_consistency(mesh)
    return mesh
end

function exact_cavity_mode_functions(
    time::Float64;
    epsilon::Float64,
    mu::Float64,
)
    omega = sqrt(3.0) * pi / sqrt(epsilon * mu)
    electric_time_factor = cos(omega * time)
    magnetic_time_factor = sin(omega * time)
    magnetic_scale = pi / (mu * omega)

    electric = function (x, y, z)
        return (
            -cos(pi * x) * sin(pi * y) * sin(pi * z) *
            electric_time_factor,
            0.0,
            sin(pi * x) * sin(pi * y) * cos(pi * z) *
            electric_time_factor,
        )
    end

    magnetic = function (x, y, z)
        return (
            -magnetic_scale * sin(pi * x) * cos(pi * y) * cos(pi * z) *
            magnetic_time_factor,
            2.0 * magnetic_scale * cos(pi * x) * sin(pi * y) *
            cos(pi * z) * magnetic_time_factor,
            -magnetic_scale * cos(pi * x) * cos(pi * y) * sin(pi * z) *
            magnetic_time_factor,
        )
    end

    return electric, magnetic
end

function resolved_cubature_order(config::ExperimentConfig)
    order = config.cubature_order == 0 ?
            max(2, 2 * config.polynomial_order + 4) :
            config.cubature_order
    order <= 20 ||
        error(
            "DG order $(config.polynomial_order) requires cubature order " *
            "$order, but Jaskowiec-Sukumar rules are available only through 20.",
        )
    return order
end

function reference_interpolation_matrix(
    ref::ReferenceTet,
    cubature_points::Matrix{Float64},
)
    rq = collect(@view cubature_points[:, 1])
    sq = collect(@view cubature_points[:, 2])
    tq = collect(@view cubature_points[:, 3])
    modal_values =
        DiscoGMPI.orthonormal_vandermonde_tet(rq, sq, tq, ref.basis)
    return modal_values * ref.invV
end

function distributed_quadrature_diagnostics(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    time::Float64,
    cubature_order::Int;
    epsilon::Float64,
    mu::Float64,
)
    cubature_points, cubature_weights, number_cubature_points =
        get_JaskowiecSukumar_cubature(cubature_order)
    interpolation =
        reference_interpolation_matrix(distributed_dg.dg.ref, cubature_points)
    exact_electric, exact_magnetic = exact_cavity_mode_functions(
        time;
        epsilon = epsilon,
        mu = mu,
    )

    mesh = distributed_dg.dg.mesh
    mappings = distributed_dg.dg.mappings.tet_mappings
    physical_operators = distributed_dg.dg.physops.elements

    # Energies, field norms/errors, charges, and momenta.
    local_sums = zeros(Float64, 27)

    for elem in distributed_dg.distributed_mesh.partition.owned
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

            for q in 1:number_cubature_points
                r = cubature_points[q, 1]
                s = cubature_points[q, 2]
                t = cubature_points[q, 3]
                x, y, z = DiscoGMPI.map_to_physical(
                    mesh.points,
                    tet_nodes,
                    r,
                    s,
                    t,
                )

                exact_Ex, exact_Ey, exact_Ez = exact_electric(x, y, z)
                exact_Hx, exact_Hy, exact_Hz = exact_magnetic(x, y, z)

                row = view(interpolation, q, :)
                numerical_Ex = dot(row, Ex)
                numerical_Ey = dot(row, Ey)
                numerical_Ez = dot(row, Ez)
                numerical_Hx = dot(row, Hx)
                numerical_Hy = dot(row, Hy)
                numerical_Hz = dot(row, Hz)

                numerical_electric_squared =
                    numerical_Ex^2 + numerical_Ey^2 + numerical_Ez^2
                numerical_magnetic_squared =
                    numerical_Hx^2 + numerical_Hy^2 + numerical_Hz^2
                exact_electric_squared =
                    exact_Ex^2 + exact_Ey^2 + exact_Ez^2
                exact_magnetic_squared =
                    exact_Hx^2 + exact_Hy^2 + exact_Hz^2

                electric_error_squared =
                    (numerical_Ex - exact_Ex)^2 +
                    (numerical_Ey - exact_Ey)^2 +
                    (numerical_Ez - exact_Ez)^2
                magnetic_error_squared =
                    (numerical_Hx - exact_Hx)^2 +
                    (numerical_Hy - exact_Hy)^2 +
                    (numerical_Hz - exact_Hz)^2

                numerical_electric_energy_density =
                    0.5 * epsilon * numerical_electric_squared
                numerical_magnetic_energy_density =
                    0.5 * mu * numerical_magnetic_squared
                numerical_energy_density =
                    numerical_electric_energy_density +
                    numerical_magnetic_energy_density

                exact_electric_energy_density =
                    0.5 * epsilon * exact_electric_squared
                exact_magnetic_energy_density =
                    0.5 * mu * exact_magnetic_squared
                exact_energy_density =
                    exact_electric_energy_density +
                    exact_magnetic_energy_density

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

                angular_momentum_x = y * momentum_z - z * momentum_y
                angular_momentum_y = z * momentum_x - x * momentum_z
                angular_momentum_z = x * momentum_y - y * momentum_x
                exact_angular_momentum_x =
                    y * exact_momentum_z - z * exact_momentum_y
                exact_angular_momentum_y =
                    z * exact_momentum_x - x * exact_momentum_z
                exact_angular_momentum_z =
                    x * exact_momentum_y - y * exact_momentum_x

                physical_weight = jacobian * cubature_weights[q]

                local_sums[1] +=
                    physical_weight * numerical_electric_energy_density
                local_sums[2] +=
                    physical_weight * numerical_magnetic_energy_density
                local_sums[3] +=
                    physical_weight * exact_electric_energy_density
                local_sums[4] +=
                    physical_weight * exact_magnetic_energy_density
                local_sums[5] +=
                    physical_weight * numerical_electric_squared
                local_sums[6] +=
                    physical_weight * exact_electric_squared
                local_sums[7] +=
                    physical_weight * electric_error_squared
                local_sums[8] +=
                    physical_weight * numerical_magnetic_squared
                local_sums[9] +=
                    physical_weight * exact_magnetic_squared
                local_sums[10] +=
                    physical_weight * magnetic_error_squared
                local_sums[11] +=
                    physical_weight * numerical_energy_density^2
                local_sums[12] +=
                    physical_weight * exact_energy_density^2
                local_sums[13] +=
                    physical_weight *
                    (numerical_energy_density - exact_energy_density)^2
                local_sums[14] +=
                    physical_weight * electric_charge_density
                local_sums[15] +=
                    physical_weight * magnetic_charge_density
                local_sums[16] += physical_weight * momentum_x
                local_sums[17] += physical_weight * momentum_y
                local_sums[18] += physical_weight * momentum_z
                local_sums[19] += physical_weight * exact_momentum_x
                local_sums[20] += physical_weight * exact_momentum_y
                local_sums[21] += physical_weight * exact_momentum_z
                local_sums[22] += physical_weight * angular_momentum_x
                local_sums[23] += physical_weight * angular_momentum_y
                local_sums[24] += physical_weight * angular_momentum_z
                local_sums[25] +=
                    physical_weight * exact_angular_momentum_x
                local_sums[26] +=
                    physical_weight * exact_angular_momentum_y
                local_sums[27] +=
                    physical_weight * exact_angular_momentum_z
            end
        end
    end

    sums = MPI.Allreduce(local_sums, +, distributed_dg.comm)
    electric_l2 = sqrt(max(sums[5], 0.0))
    exact_electric_l2 = sqrt(max(sums[6], 0.0))
    electric_error_l2 = sqrt(max(sums[7], 0.0))
    magnetic_l2 = sqrt(max(sums[8], 0.0))
    exact_magnetic_l2 = sqrt(max(sums[9], 0.0))
    magnetic_error_l2 = sqrt(max(sums[10], 0.0))
    field_error_l2 =
        sqrt(max(sums[7] + sums[10], 0.0))
    exact_field_l2 =
        sqrt(max(sums[6] + sums[9], 0.0))
    energy_density_l2 = sqrt(max(sums[11], 0.0))
    exact_energy_density_l2 = sqrt(max(sums[12], 0.0))
    energy_density_error_l2 = sqrt(max(sums[13], 0.0))

    return MaxwellQuadratureDiagnostics(
        cubature_order,
        sums[1],
        sums[2],
        sums[1] + sums[2],
        sums[3],
        sums[4],
        sums[3] + sums[4],
        electric_l2,
        exact_electric_l2,
        electric_error_l2,
        electric_error_l2 / max(exact_electric_l2, eps(Float64)),
        magnetic_l2,
        exact_magnetic_l2,
        magnetic_error_l2,
        magnetic_error_l2 / max(exact_magnetic_l2, eps(Float64)),
        field_error_l2,
        field_error_l2 / max(exact_field_l2, eps(Float64)),
        energy_density_l2,
        exact_energy_density_l2,
        energy_density_error_l2,
        energy_density_error_l2 /
        max(exact_energy_density_l2, eps(Float64)),
        sums[14],
        sums[15],
        0.0,
        0.0,
        sums[16],
        sums[17],
        sums[18],
        sums[19],
        sums[20],
        sums[21],
        sums[22],
        sums[23],
        sums[24],
        sums[25],
        sums[26],
        sums[27],
    )
end

function write_quadrature_diagnostics_row(
    io::IO,
    step::Int,
    time::Float64,
    diagnostics::MaxwellQuadratureDiagnostics,
)
    energy_error =
        diagnostics.total_energy - diagnostics.exact_total_energy
    relative_energy_error =
        energy_error /
        max(abs(diagnostics.exact_total_energy), eps(Float64))

    values = (
        step,
        time,
        diagnostics.cubature_order,
        diagnostics.electric_energy,
        diagnostics.magnetic_energy,
        diagnostics.total_energy,
        diagnostics.exact_electric_energy,
        diagnostics.exact_magnetic_energy,
        diagnostics.exact_total_energy,
        energy_error,
        relative_energy_error,
        diagnostics.electric_l2,
        diagnostics.exact_electric_l2,
        diagnostics.electric_error_l2,
        diagnostics.electric_relative_error,
        diagnostics.magnetic_l2,
        diagnostics.exact_magnetic_l2,
        diagnostics.magnetic_error_l2,
        diagnostics.magnetic_relative_error,
        diagnostics.field_error_l2,
        diagnostics.field_relative_error,
        diagnostics.energy_density_l2,
        diagnostics.exact_energy_density_l2,
        diagnostics.energy_density_error_l2,
        diagnostics.energy_density_relative_error,
        diagnostics.electric_charge,
        diagnostics.magnetic_charge,
        diagnostics.exact_electric_charge,
        diagnostics.exact_magnetic_charge,
        diagnostics.linear_momentum_x,
        diagnostics.linear_momentum_y,
        diagnostics.linear_momentum_z,
        diagnostics.exact_linear_momentum_x,
        diagnostics.exact_linear_momentum_y,
        diagnostics.exact_linear_momentum_z,
        diagnostics.angular_momentum_x,
        diagnostics.angular_momentum_y,
        diagnostics.angular_momentum_z,
        diagnostics.exact_angular_momentum_x,
        diagnostics.exact_angular_momentum_y,
        diagnostics.exact_angular_momentum_z,
    )
    println(io, join(values, ','))
    flush(io)
    return nothing
end

function reference_vertex_node_ids(ref::ReferenceTet; tol::Float64 = 1e-12)
    ids = Vector{Int}(undef, 4)

    for vertex in 1:4
        rv, sv, tv = REF_TET_VERTEX_COORDS[vertex]
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

function write_energy_row(
    io::IO,
    step::Int,
    time::Float64,
    energy::DiscoGMPI.MaxwellEnergy,
    initial_total::Float64,
)
    relative_drift =
        (energy.total - initial_total) / max(initial_total, eps(Float64))
    values = (
        step,
        time,
        energy.electric,
        energy.magnetic,
        energy.total,
        relative_drift,
        energy.Ex,
        energy.Ey,
        energy.Ez,
        energy.Hx,
        energy.Hy,
        energy.Hz,
    )
    println(io, join(values, ','))
    flush(io)
    return relative_drift
end

function write_parallel_final_fields(
    output_basename::String,
    distributed_dg::DistributedDGDiscretization,
    U::MaxwellField;
    time::Float64,
)
    rank = MPI.Comm_rank(distributed_dg.comm)
    nranks = MPI.Comm_size(distributed_dg.comm)
    mesh = distributed_dg.dg.mesh
    distributed_mesh = distributed_dg.distributed_mesh
    owned = distributed_mesh.partition.owned
    vertex_ids = reference_vertex_node_ids(distributed_dg.dg.ref)
    nowned = length(owned)

    points = zeros(Float64, 3, 4 * nowned)
    electric = zeros(Float64, 3, 4 * nowned)
    magnetic = zeros(Float64, 3, 4 * nowned)
    cells = Vector{MeshCell}(undef, nowned)
    global_element_ids = Vector{Int}(undef, nowned)

    for (owned_index, local_elem) in enumerate(owned)
        cell_nodes = ntuple(i -> 4 * (owned_index - 1) + i, 4)
        cells[owned_index] =
            MeshCell(VTKCellTypes.VTK_TETRA, cell_nodes)
        global_element_ids[owned_index] =
            distributed_mesh.elements.global_ids[local_elem]

        for vertex in 1:4
            output_node = cell_nodes[vertex]
            mesh_node = mesh.tets[vertex, local_elem]
            field_node = vertex_ids[vertex]

            points[:, output_node] .= mesh.points[:, mesh_node]
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
        end
    end

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
        vtk["ElectricFieldMagnitude", VTKPointData()] = electric_magnitude
        vtk["MagneticFieldMagnitude", VTKPointData()] = magnetic_magnitude
        vtk["GlobalElementId", VTKCellData()] = global_element_ids
        vtk["OwnerRank", VTKCellData()] = fill(rank, nowned)
        vtk["PolynomialOrder", VTKCellData()] =
            fill(distributed_dg.dg.ref.N, nowned)
        vtk["TimeValue", VTKFieldData()] = time
    end
end

function write_final_integration_points(
    output_dir::String,
    distributed_dg::DistributedDGDiscretization,
    U::MaxwellField,
    time::Float64,
    cubature_order::Int;
    epsilon::Float64,
    mu::Float64,
)
    rank = MPI.Comm_rank(distributed_dg.comm)
    rank_label = lpad(string(rank), 4, '0')
    output_path =
        joinpath(output_dir, "integration_points_rank$rank_label.csv")
    cubature_points, cubature_weights, number_cubature_points =
        get_JaskowiecSukumar_cubature(cubature_order)
    interpolation =
        reference_interpolation_matrix(distributed_dg.dg.ref, cubature_points)
    exact_electric, exact_magnetic = exact_cavity_mode_functions(
        time;
        epsilon = epsilon,
        mu = mu,
    )

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
            "energy_density_error,electric_charge_density," *
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

                for q in 1:number_cubature_points
                    r = cubature_points[q, 1]
                    s = cubature_points[q, 2]
                    t = cubature_points[q, 3]
                    x, y, z = DiscoGMPI.map_to_physical(
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

function load_root_inputs(
    config::ExperimentConfig,
    rank::Int,
    nranks::Int,
    comm::MPI.Comm,
)
    mesh = nothing
    partition = nothing
    load_error = nothing

    if rank == 0
        try
            isfile(config.mesh_path) ||
                error("Mesh file not found: $(config.mesh_path)")
            isfile(config.partition_path) ||
                error(
                    "Partition file not found: $(config.partition_path). " *
                    "Use 2 or 4 MPI ranks with the shipped mesh, or pass " *
                    "--partition PATH.",
                )

            mesh = load_pec_mesh(config.mesh_path)
            partition = read_metis_epart(config.partition_path)
            length(partition) == size(mesh.tets, 2) ||
                error(
                    "Partition has $(length(partition)) entries but the mesh " *
                    "has $(size(mesh.tets, 2)) tetrahedra.",
                )
            all(part -> 0 <= part < nranks, partition) ||
                error("Partition entries must be zero-based ranks in 0:$(nranks - 1).")
            all(part -> any(==(part), partition), 0:(nranks - 1)) ||
                error("Every MPI rank must own at least one tetrahedron.")
        catch error
            load_error = sprint(showerror, error)
        end
    end

    load_error = MPI.bcast(load_error, comm; root = 0)
    load_error === nothing || error(load_error)
    return mesh, partition
end

function run_experiment(config::ExperimentConfig, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    root_mesh, root_partition =
        load_root_inputs(config, rank, nranks, comm)

    distributed_dg = build_distributed_dg_from_root(
        root_mesh,
        root_partition,
        config.polynomial_order;
        comm = comm,
    )

    cubature_order = resolved_cubature_order(config)
    electric, magnetic = exact_cavity_mode_functions(
        0.0;
        epsilon = config.epsilon,
        mu = config.mu,
    )
    U = interpolate_maxwell_field(distributed_dg, electric, magnetic)

    registry = MaxwellBoundaryRegistry(
        Dict(PEC_BOUNDARY_ID => DiscoGMPI.MaxwellBC_PEC),
    )
    formulation = PoissonBracketFormulation()
    scheme = explicit_partitioned_symplectic_rk_scheme(
        config.esprk_order;
        first_partition = :H,
    )
    workspace = MaxwellPartitionedRKWorkspace(U, scheme)

    local_dt, local_sizes = estimate_maxwell_dt(
        distributed_dg.dg.mesh,
        distributed_dg.dg.geometry,
        distributed_dg.dg.ref;
        CFL = config.cfl,
        ε = config.epsilon,
        μ = config.mu,
    )
    estimated_dt = MPI.Allreduce(local_dt, min, comm)
    global_hmin = MPI.Allreduce(local_sizes.hmin, min, comm)
    nsteps = max(1, ceil(Int, config.final_time / estimated_dt))
    dt = config.final_time / nsteps

    if rank == 0
        mkpath(config.output_dir)
    end
    MPI.Barrier(comm)

    energy_path = joinpath(config.output_dir, "energy.csv")
    quadrature_path =
        joinpath(config.output_dir, "quadrature_diagnostics.csv")
    energy_io = rank == 0 ? open(energy_path, "w") : nothing
    quadrature_io = rank == 0 ? open(quadrature_path, "w") : nothing
    initial_energy = distributed_maxwell_energy(
        U,
        distributed_dg;
        ε = config.epsilon,
        μ = config.mu,
    )
    initial_quadrature = distributed_quadrature_diagnostics(
        U,
        distributed_dg,
        0.0,
        cubature_order;
        epsilon = config.epsilon,
        mu = config.mu,
    )

    if rank == 0
        println(
            energy_io,
            "step,time,electric,magnetic,total,relative_drift," *
            "Ex,Ey,Ez,Hx,Hy,Hz",
        )
        println(
            quadrature_io,
            "step,time,cubature_order,electric_energy,magnetic_energy," *
            "total_energy,exact_electric_energy,exact_magnetic_energy," *
            "exact_total_energy,energy_error,relative_energy_error," *
            "electric_l2,exact_electric_l2,electric_error_l2," *
            "electric_relative_error,magnetic_l2,exact_magnetic_l2," *
            "magnetic_error_l2,magnetic_relative_error,field_error_l2," *
            "field_relative_error,energy_density_l2," *
            "exact_energy_density_l2,energy_density_error_l2," *
            "energy_density_relative_error,electric_charge," *
            "magnetic_charge,exact_electric_charge," *
            "exact_magnetic_charge,linear_momentum_x," *
            "linear_momentum_y,linear_momentum_z," *
            "exact_linear_momentum_x,exact_linear_momentum_y," *
            "exact_linear_momentum_z,angular_momentum_x," *
            "angular_momentum_y,angular_momentum_z," *
            "exact_angular_momentum_x,exact_angular_momentum_y," *
            "exact_angular_momentum_z",
        )
        write_energy_row(
            energy_io,
            0,
            0.0,
            initial_energy,
            initial_energy.total,
        )
        write_quadrature_diagnostics_row(
            quadrature_io,
            0,
            0.0,
            initial_quadrature,
        )

        println("Distributed Poisson-bracket Maxwell experiment")
        println("----------------------------------------------")
        println("MPI ranks:            ", nranks)
        println("mesh:                 ", config.mesh_path)
        println("partition:            ", config.partition_path)
        println("DG order:             ", config.polynomial_order)
        println("time integrator:      ", scheme.name)
        println("cubature order:       ", cubature_order)
        println("boundary condition:   PEC on all exterior faces")
        println("analytical solution:  unit-cube PEC eigenmode")
        println("electric charge:      integral of div(epsilon E)")
        println("magnetic charge:      integral of div(mu H)")
        println("linear momentum:      integral of epsilon*mu*(E x H)")
        println("angular momentum:     about coordinate origin")
        println("global hmin:          ", global_hmin)
        println("estimated dt:         ", estimated_dt)
        println("used dt:              ", dt)
        println("steps:                ", nsteps)
        println("initial energy:       ", initial_energy.total)
        println("output directory:     ", config.output_dir)
    end

    try
        for step in 1:nsteps
            distributed_partitioned_symplectic_rk_step!(
                U,
                workspace,
                scheme,
                dt,
                distributed_dg,
                registry,
                formulation;
                ε = config.epsilon,
                μ = config.mu,
            )

            if step % config.energy_every == 0 || step == nsteps
                energy = distributed_maxwell_energy(
                    U,
                    distributed_dg;
                    ε = config.epsilon,
                    μ = config.mu,
                )
                time = step * dt
                quadrature = distributed_quadrature_diagnostics(
                    U,
                    distributed_dg,
                    time,
                    cubature_order;
                    epsilon = config.epsilon,
                    mu = config.mu,
                )

                if rank == 0
                    relative_drift = write_energy_row(
                        energy_io,
                        step,
                        time,
                        energy,
                        initial_energy.total,
                    )
                    write_quadrature_diagnostics_row(
                        quadrature_io,
                        step,
                        time,
                        quadrature,
                    )
                    println(
                        "step ", step, "/", nsteps,
                        ", t = ", time,
                        ", energy = ", energy.total,
                        ", relative drift = ", relative_drift,
                        ", L2(E error) = ", quadrature.electric_error_l2,
                        ", L2(H error) = ", quadrature.magnetic_error_l2,
                        ", L2(w error) = ",
                        quadrature.energy_density_error_l2,
                        ", Qe = ", quadrature.electric_charge,
                        ", Qm = ", quadrature.magnetic_charge,
                    )
                end
            end
        end
    finally
        if rank == 0
            close(energy_io)
            close(quadrature_io)
        end
    end

    final_basename = joinpath(config.output_dir, "final_fields")
    write_parallel_final_fields(
        final_basename,
        distributed_dg,
        U;
        time = config.final_time,
    )
    write_final_integration_points(
        config.output_dir,
        distributed_dg,
        U,
        config.final_time,
        cubature_order;
        epsilon = config.epsilon,
        mu = config.mu,
    )
    MPI.Barrier(comm)

    if rank == 0
        println()
        println("Energy history:       ", energy_path)
        println("Quadrature diagnostics:", quadrature_path)
        println(
            "Integration points:   ",
            joinpath(config.output_dir, "integration_points_rankNNNN.csv"),
        )
        println("ParaView dataset:     ", final_basename * ".pvtu")
        println(
            "Plot diagnostics:     python3 examples/plot_maxwell_energy.py ",
            quadrature_path,
        )
    end

    return nothing
end

function main(args::Vector{String})
    initialized_here = !MPI.Initialized()
    initialized_here && MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    repository_root = normpath(joinpath(@__DIR__, ".."))

    try
        config = parse_arguments(args, nranks, repository_root)

        if config === nothing
            rank == 0 && usage()
            return nothing
        end

        run_experiment(config, comm)
    catch error
        if rank == 0
            println(stderr, "ERROR: ", sprint(showerror, error))
        end
        rethrow()
    finally
        if initialized_here && !MPI.Finalized()
            MPI.Finalize()
        end
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
