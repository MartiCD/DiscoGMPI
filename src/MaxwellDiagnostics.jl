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
    optical_chirality::Float64
    exact_optical_chirality::Float64
    optical_chirality_error::Float64
    optical_chirality_density_l2::Float64
    exact_optical_chirality_density_l2::Float64
    optical_chirality_density_error_l2::Float64
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

mutable struct DistributedMaxwellComponentL2Workspace
    cubature_order::Int
    cubature_points::Matrix{Float64}
    cubature_weights::Vector{Float64}
    interpolation::Matrix{Float64}
    local_sums::Vector{Float64}
end

function DistributedMaxwellComponentL2Workspace(
    distributed_dg::DistributedDGDiscretization,
    cubature_order::Int,
)
    cubature_points, cubature_weights, _ =
        get_JaskowiecSukumar_cubature(cubature_order)
    interpolation =
        reference_interpolation_matrix(distributed_dg.dg.ref, cubature_points)
    return DistributedMaxwellComponentL2Workspace(
        cubature_order,
        Matrix{Float64}(cubature_points),
        Vector{Float64}(cubature_weights),
        Matrix{Float64}(interpolation),
        zeros(Float64, 6),
    )
end

struct DistributedMeshQualityMetrics
    volume_min::Float64
    volume_max::Float64
    volume_total::Float64
    volume_ratio::Float64
    h_min::Float64
    h_max::Float64
    h_ratio::Float64
    edge_min::Float64
    edge_max::Float64
    edge_ratio::Float64
    mean_ratio_min::Float64
    mean_ratio_avg::Float64
end

function _tetrahedron_edge_metrics(
    points::AbstractMatrix{<:Real},
    tet_nodes,
)
    edge_min = Inf
    edge_max = 0.0
    edge_squared_sum = 0.0

    @inbounds for a in 1:3
        ia = tet_nodes[a]
        xa = points[1, ia]
        ya = points[2, ia]
        za = points[3, ia]

        for b in (a + 1):4
            ib = tet_nodes[b]
            dx = xa - points[1, ib]
            dy = ya - points[2, ib]
            dz = za - points[3, ib]
            squared = dx^2 + dy^2 + dz^2
            length = sqrt(squared)
            edge_min = min(edge_min, length)
            edge_max = max(edge_max, length)
            edge_squared_sum += squared
        end
    end

    return edge_min, edge_max, edge_squared_sum
end

function _tetrahedron_mean_ratio_quality(
    volume::Float64,
    edge_squared_sum::Float64,
)
    volume > 0.0 && edge_squared_sum > 0.0 || return 0.0
    return 12.0 * (3.0 * volume)^(2.0 / 3.0) / edge_squared_sum
end

function distributed_mesh_quality_metrics(
    distributed_dg::DistributedDGDiscretization,
)
    mesh = distributed_dg.dg.mesh
    mappings = distributed_dg.dg.mappings.tet_mappings

    local_volume_min = Inf
    local_volume_max = 0.0
    local_volume_total = 0.0
    local_h_min = Inf
    local_h_max = 0.0
    local_edge_min = Inf
    local_edge_max = 0.0
    local_mean_ratio_min = Inf
    local_mean_ratio_sum = 0.0
    local_count = 0

    for elem in distributed_dg.distributed_mesh.partition.owned
        tet_nodes = @view mesh.tets[:, elem]
        volume = (4.0 / 3.0) * mappings[elem].absdetJ
        edge_min, edge_max, edge_squared_sum =
            _tetrahedron_edge_metrics(mesh.points, tet_nodes)
        mean_ratio =
            _tetrahedron_mean_ratio_quality(volume, edge_squared_sum)

        local_volume_min = min(local_volume_min, volume)
        local_volume_max = max(local_volume_max, volume)
        local_volume_total += volume
        local_h_min = min(local_h_min, edge_max)
        local_h_max = max(local_h_max, edge_max)
        local_edge_min = min(local_edge_min, edge_min)
        local_edge_max = max(local_edge_max, edge_max)
        local_mean_ratio_min = min(local_mean_ratio_min, mean_ratio)
        local_mean_ratio_sum += mean_ratio
        local_count += 1
    end

    volume_min = MPI.Allreduce(local_volume_min, min, distributed_dg.comm)
    volume_max = MPI.Allreduce(local_volume_max, max, distributed_dg.comm)
    volume_total = MPI.Allreduce(local_volume_total, +, distributed_dg.comm)
    h_min = MPI.Allreduce(local_h_min, min, distributed_dg.comm)
    h_max = MPI.Allreduce(local_h_max, max, distributed_dg.comm)
    edge_min = MPI.Allreduce(local_edge_min, min, distributed_dg.comm)
    edge_max = MPI.Allreduce(local_edge_max, max, distributed_dg.comm)
    mean_ratio_min =
        MPI.Allreduce(local_mean_ratio_min, min, distributed_dg.comm)
    mean_ratio_sum =
        MPI.Allreduce(local_mean_ratio_sum, +, distributed_dg.comm)
    count = MPI.Allreduce(local_count, +, distributed_dg.comm)

    volume_ratio = volume_min > 0.0 ? volume_max / volume_min : Inf
    h_ratio = h_min > 0.0 ? h_max / h_min : Inf
    edge_ratio = edge_min > 0.0 ? edge_max / edge_min : Inf
    mean_ratio_avg = count > 0 ? mean_ratio_sum / count : 0.0

    return DistributedMeshQualityMetrics(
        volume_min,
        volume_max,
        volume_total,
        volume_ratio,
        h_min,
        h_max,
        h_ratio,
        edge_min,
        edge_max,
        edge_ratio,
        mean_ratio_min,
        mean_ratio_avg,
    )
end

function reference_interpolation_matrix(
    ref::ReferenceTet,
    cubature_points::Matrix{Float64},
)
    rq = collect(@view cubature_points[:, 1])
    sq = collect(@view cubature_points[:, 2])
    tq = collect(@view cubature_points[:, 3])
    modal_values = orthonormal_vandermonde_tet(rq, sq, tq, ref.basis)
    return modal_values * ref.invV
end

function distributed_maxwell_quadrature_diagnostics(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    cubature_order::Int;
    epsilon::Float64,
    mu::Float64,
    exact_electric,
    exact_magnetic,
    exact_curl_electric,
    exact_curl_magnetic,
)
    cubature_points, cubature_weights, number_cubature_points =
        get_JaskowiecSukumar_cubature(cubature_order)
    interpolation =
        reference_interpolation_matrix(distributed_dg.dg.ref, cubature_points)
    mesh = distributed_dg.dg.mesh
    mappings = distributed_dg.dg.mappings.tet_mappings
    physical_operators = distributed_dg.dg.physops.elements

    # Energies, field norms/errors, chirality, charges, and momenta.
    local_sums = zeros(Float64, 32)

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
            curl_electric_x = operators.Dy * Ez - operators.Dz * Ey
            curl_electric_y = operators.Dz * Ex - operators.Dx * Ez
            curl_electric_z = operators.Dx * Ey - operators.Dy * Ex
            curl_magnetic_x = operators.Dy * Hz - operators.Dz * Hy
            curl_magnetic_y = operators.Dz * Hx - operators.Dx * Hz
            curl_magnetic_z = operators.Dx * Hy - operators.Dy * Hx

            for q in 1:number_cubature_points
                r = cubature_points[q, 1]
                s = cubature_points[q, 2]
                t = cubature_points[q, 3]
                x, y, z = map_to_physical(mesh.points, tet_nodes, r, s, t)

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

                numerical_electric =
                    (numerical_Ex, numerical_Ey, numerical_Ez)
                numerical_magnetic =
                    (numerical_Hx, numerical_Hy, numerical_Hz)
                numerical_curl_electric = (
                    dot(row, curl_electric_x),
                    dot(row, curl_electric_y),
                    dot(row, curl_electric_z),
                )
                numerical_curl_magnetic = (
                    dot(row, curl_magnetic_x),
                    dot(row, curl_magnetic_y),
                    dot(row, curl_magnetic_z),
                )
                exact_electric_vector = (exact_Ex, exact_Ey, exact_Ez)
                exact_magnetic_vector = (exact_Hx, exact_Hy, exact_Hz)
                exact_curl_electric_vector =
                    exact_curl_electric(x, y, z)
                exact_curl_magnetic_vector =
                    exact_curl_magnetic(x, y, z)
                numerical_optical_chirality =
                    optical_chirality_density(
                        numerical_electric,
                        numerical_curl_electric,
                        numerical_magnetic,
                        numerical_curl_magnetic;
                        epsilon = epsilon,
                        mu = mu,
                    )
                exact_optical_chirality =
                    optical_chirality_density(
                        exact_electric_vector,
                        exact_curl_electric_vector,
                        exact_magnetic_vector,
                        exact_curl_magnetic_vector;
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
                    physical_weight * numerical_optical_chirality
                local_sums[15] +=
                    physical_weight * exact_optical_chirality
                local_sums[16] +=
                    physical_weight * numerical_optical_chirality^2
                local_sums[17] +=
                    physical_weight * exact_optical_chirality^2
                local_sums[18] +=
                    physical_weight *
                    (numerical_optical_chirality -
                     exact_optical_chirality)^2
                local_sums[19] +=
                    physical_weight * electric_charge_density
                local_sums[20] +=
                    physical_weight * magnetic_charge_density
                local_sums[21] += physical_weight * momentum_x
                local_sums[22] += physical_weight * momentum_y
                local_sums[23] += physical_weight * momentum_z
                local_sums[24] += physical_weight * exact_momentum_x
                local_sums[25] += physical_weight * exact_momentum_y
                local_sums[26] += physical_weight * exact_momentum_z
                local_sums[27] += physical_weight * angular_momentum_x
                local_sums[28] += physical_weight * angular_momentum_y
                local_sums[29] += physical_weight * angular_momentum_z
                local_sums[30] +=
                    physical_weight * exact_angular_momentum_x
                local_sums[31] +=
                    physical_weight * exact_angular_momentum_y
                local_sums[32] +=
                    physical_weight * exact_angular_momentum_z
            end
        end
    end

    sums = MPI.Allreduce(local_sums, +, distributed_dg.comm)
    return maxwell_quadrature_diagnostics_from_sums(cubature_order, sums)
end

function maxwell_quadrature_diagnostics_from_sums(
    cubature_order::Int,
    sums::AbstractVector{Float64},
)
    electric_l2 = sqrt(max(sums[5], 0.0))
    exact_electric_l2 = sqrt(max(sums[6], 0.0))
    electric_error_l2 = sqrt(max(sums[7], 0.0))
    magnetic_l2 = sqrt(max(sums[8], 0.0))
    exact_magnetic_l2 = sqrt(max(sums[9], 0.0))
    magnetic_error_l2 = sqrt(max(sums[10], 0.0))
    field_error_l2 = sqrt(max(sums[7] + sums[10], 0.0))
    exact_field_l2 = sqrt(max(sums[6] + sums[9], 0.0))
    energy_density_l2 = sqrt(max(sums[11], 0.0))
    exact_energy_density_l2 = sqrt(max(sums[12], 0.0))
    energy_density_error_l2 = sqrt(max(sums[13], 0.0))
    optical_chirality_density_l2 = sqrt(max(sums[16], 0.0))
    exact_optical_chirality_density_l2 = sqrt(max(sums[17], 0.0))
    optical_chirality_density_error_l2 = sqrt(max(sums[18], 0.0))

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
        sums[14] - sums[15],
        optical_chirality_density_l2,
        exact_optical_chirality_density_l2,
        optical_chirality_density_error_l2,
        sums[19],
        sums[20],
        0.0,
        0.0,
        sums[21],
        sums[22],
        sums[23],
        sums[24],
        sums[25],
        sums[26],
        sums[27],
        sums[28],
        sums[29],
        sums[30],
        sums[31],
        sums[32],
    )
end

function distributed_cavity_quadrature_diagnostics(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
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

    return distributed_maxwell_quadrature_diagnostics(
        U,
        distributed_dg,
        cubature_order;
        epsilon = epsilon,
        mu = mu,
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
        exact_curl_electric = exact_curl_electric,
        exact_curl_magnetic = exact_curl_magnetic,
    )
end

function distributed_periodic_quadrature_diagnostics(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
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

    return distributed_maxwell_quadrature_diagnostics(
        U,
        distributed_dg,
        cubature_order;
        epsilon = epsilon,
        mu = mu,
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
        exact_curl_electric = exact_curl_electric,
        exact_curl_magnetic = exact_curl_magnetic,
    )
end

function distributed_maxwell_linf_errors(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    cubature_order::Int;
    exact_electric,
    exact_magnetic,
)
    cubature_points, _, number_cubature_points =
        get_JaskowiecSukumar_cubature(cubature_order)
    interpolation =
        reference_interpolation_matrix(distributed_dg.dg.ref, cubature_points)
    mesh = distributed_dg.dg.mesh
    local_maxima = zeros(Float64, 3)

    for elem in distributed_dg.distributed_mesh.partition.owned
        tet_nodes = @view mesh.tets[:, elem]

        @views begin
            Ex = U.Ex[:, elem]
            Ey = U.Ey[:, elem]
            Ez = U.Ez[:, elem]
            Hx = U.Hx[:, elem]
            Hy = U.Hy[:, elem]
            Hz = U.Hz[:, elem]

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
                exact_Ex, exact_Ey, exact_Ez = exact_electric(x, y, z)
                exact_Hx, exact_Hy, exact_Hz = exact_magnetic(x, y, z)
                row = view(interpolation, q, :)
                electric_error_squared =
                    (dot(row, Ex) - exact_Ex)^2 +
                    (dot(row, Ey) - exact_Ey)^2 +
                    (dot(row, Ez) - exact_Ez)^2
                magnetic_error_squared =
                    (dot(row, Hx) - exact_Hx)^2 +
                    (dot(row, Hy) - exact_Hy)^2 +
                    (dot(row, Hz) - exact_Hz)^2
                local_maxima[1] =
                    max(local_maxima[1], sqrt(electric_error_squared))
                local_maxima[2] =
                    max(local_maxima[2], sqrt(magnetic_error_squared))
                local_maxima[3] = max(
                    local_maxima[3],
                    sqrt(electric_error_squared + magnetic_error_squared),
                )
            end
        end
    end

    maxima = MPI.Allreduce(local_maxima, max, distributed_dg.comm)
    return maxima[1], maxima[2], maxima[3]
end

function distributed_cavity_linf_errors(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    time::Float64,
    cubature_order::Int;
    epsilon::Float64,
    mu::Float64,
    boundary_condition::Symbol,
)
    exact_electric, exact_magnetic = exact_cavity_mode_functions(
        time;
        epsilon = epsilon,
        mu = mu,
        boundary_condition = boundary_condition,
    )
    return distributed_maxwell_linf_errors(
        U,
        distributed_dg,
        cubature_order;
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
    )
end

function distributed_periodic_linf_errors(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
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
    return distributed_maxwell_linf_errors(
        U,
        distributed_dg,
        cubature_order;
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
    )
end

function distributed_maxwell_component_l2_errors(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    cubature_order::Int;
    exact_electric,
    exact_magnetic,
)
    workspace = DistributedMaxwellComponentL2Workspace(
        distributed_dg,
        cubature_order,
    )
    return distributed_maxwell_component_l2_errors!(
        workspace,
        U,
        distributed_dg;
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
    )
end


function distributed_maxwell_component_l2_errors!(
    workspace::DistributedMaxwellComponentL2Workspace,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization;
    exact_electric,
    exact_magnetic,
)
    cubature_points = workspace.cubature_points
    cubature_weights = workspace.cubature_weights
    interpolation = workspace.interpolation
    number_cubature_points = size(cubature_points, 1)
    mesh = distributed_dg.dg.mesh
    mappings = distributed_dg.dg.mappings.tet_mappings
    local_sums = workspace.local_sums
    fill!(local_sums, 0.0)

    for elem in distributed_dg.distributed_mesh.partition.owned
        tet_nodes = @view mesh.tets[:, elem]
        jacobian = mappings[elem].absdetJ
        components = (
            @view(U.Ex[:, elem]),
            @view(U.Ey[:, elem]),
            @view(U.Ez[:, elem]),
            @view(U.Hx[:, elem]),
            @view(U.Hy[:, elem]),
            @view(U.Hz[:, elem]),
        )

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
            exact_values = (
                exact_electric(x, y, z)...,
                exact_magnetic(x, y, z)...,
            )
            row = view(interpolation, q, :)
            physical_weight = jacobian * cubature_weights[q]
            for component in 1:6
                error = dot(row, components[component]) -
                        exact_values[component]
                local_sums[component] += physical_weight * error^2
            end
        end
    end

    sums = MPI.Allreduce(local_sums, +, distributed_dg.comm)
    return ntuple(component -> sqrt(max(sums[component], 0.0)), 6)
end


function distributed_cavity_component_l2_errors(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    time::Float64,
    cubature_order::Int;
    epsilon::Float64,
    mu::Float64,
    boundary_condition::Symbol,
)
    exact_electric, exact_magnetic = exact_cavity_mode_functions(
        time;
        epsilon = epsilon,
        mu = mu,
        boundary_condition = boundary_condition,
    )
    return distributed_maxwell_component_l2_errors(
        U,
        distributed_dg,
        cubature_order;
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
    )
end

function distributed_periodic_component_l2_errors(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
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
    return distributed_maxwell_component_l2_errors(
        U,
        distributed_dg,
        cubature_order;
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
    )
end


function distributed_periodic_component_l2_errors!(
    workspace::DistributedMaxwellComponentL2Workspace,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    time::Float64;
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
    return distributed_maxwell_component_l2_errors!(
        workspace,
        U,
        distributed_dg;
        exact_electric = exact_electric,
        exact_magnetic = exact_magnetic,
    )
end

function write_energy_header(io::IO)
    println(
        io,
        "step,time,electric,magnetic,total,relative_drift," *
        "Ex,Ey,Ez,Hx,Hy,Hz",
    )
    return nothing
end

function write_energy_row(
    io::IO,
    step::Int,
    time::Float64,
    energy::MaxwellEnergy,
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

function write_quadrature_diagnostics_header(io::IO)
    println(
        io,
        "step,time,cubature_order,electric_energy,magnetic_energy," *
        "total_energy,exact_electric_energy,exact_magnetic_energy," *
        "exact_total_energy,energy_error,relative_energy_error," *
        "electric_l2,exact_electric_l2,electric_error_l2," *
        "electric_relative_error,magnetic_l2,exact_magnetic_l2," *
        "magnetic_error_l2,magnetic_relative_error,field_error_l2," *
        "field_relative_error,energy_density_l2," *
        "exact_energy_density_l2,energy_density_error_l2," *
        "energy_density_relative_error,optical_chirality," *
        "exact_optical_chirality,optical_chirality_error," *
        "optical_chirality_density_l2," *
        "exact_optical_chirality_density_l2," *
        "optical_chirality_density_error_l2,electric_charge," *
        "magnetic_charge,exact_electric_charge," *
        "exact_magnetic_charge,linear_momentum_x," *
        "linear_momentum_y,linear_momentum_z," *
        "exact_linear_momentum_x,exact_linear_momentum_y," *
        "exact_linear_momentum_z,angular_momentum_x," *
        "angular_momentum_y,angular_momentum_z," *
        "exact_angular_momentum_x,exact_angular_momentum_y," *
        "exact_angular_momentum_z",
    )
    return nothing
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
        diagnostics.optical_chirality,
        diagnostics.exact_optical_chirality,
        diagnostics.optical_chirality_error,
        diagnostics.optical_chirality_density_l2,
        diagnostics.exact_optical_chirality_density_l2,
        diagnostics.optical_chirality_density_error_l2,
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
