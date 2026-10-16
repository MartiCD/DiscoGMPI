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
    ex_error_l2::Float64
    ey_error_l2::Float64
    ez_error_l2::Float64
    hx_error_l2::Float64
    hy_error_l2::Float64
    hz_error_l2::Float64
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
    electric_divergence_l2::Float64
    magnetic_divergence_l2::Float64
    electric_divergence_linf::Float64
    magnetic_divergence_linf::Float64
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
    # Volume-integrated D=epsilon E and B=mu H, not divergence-based charges.
    integrated_d_x::Float64
    integrated_d_y::Float64
    integrated_d_z::Float64
    integrated_b_x::Float64
    integrated_b_y::Float64
    integrated_b_z::Float64
    # Directional contributions to the signed broken-divergence integrals.
    electric_charge_x::Float64
    electric_charge_y::Float64
    electric_charge_z::Float64
    magnetic_charge_x::Float64
    magnetic_charge_y::Float64
    magnetic_charge_z::Float64
    discog_charges::NTuple{16,Float64}
    integrated_d_sum::Float64
    integrated_b_sum::Float64
end

struct MaxwellLightweightDiagnostics
    optical_chirality::Float64
    linear_momentum::NTuple{3,Float64}
    angular_momentum::NTuple{3,Float64}
    integrated_d::NTuple{3,Float64}
    integrated_b::NTuple{3,Float64}
end

mutable struct DistributedMaxwellLightweightWorkspace
    cubature_points::Matrix{Float64}
    cubature_weights::Vector{Float64}
    interpolation::Matrix{Float64}
    work::NTuple{7,Vector{Float64}}
    nodal_values::Matrix{Float64}
    cubature_values::Matrix{Float64}
    local_sums::Vector{Float64}
    x::Matrix{Float64}
    y::Matrix{Float64}
    z::Matrix{Float64}
end

function DistributedMaxwellLightweightWorkspace(
    distributed_dg::DistributedDGDiscretization,
    cubature_order::Int,
)
    dg = distributed_dg.dg
    cubature_points, cubature_weights, number_cubature_points =
        get_JaskowiecSukumar_cubature(cubature_order)
    interpolation = reference_interpolation_matrix(dg.ref, cubature_points)
    number_elements = size(dg.mesh.tets, 2)
    x = zeros(Float64, number_cubature_points, number_elements)
    y = similar(x)
    z = similar(x)
    for elem in distributed_dg.distributed_mesh.partition.owned
        tet_nodes = @view dg.mesh.tets[:, elem]
        for q in 1:number_cubature_points
            x[q, elem], y[q, elem], z[q, elem] = map_to_physical(
                dg.mesh.points,
                tet_nodes,
                cubature_points[q, 1],
                cubature_points[q, 2],
                cubature_points[q, 3],
            )
        end
    end
    return DistributedMaxwellLightweightWorkspace(
        Matrix{Float64}(cubature_points),
        Vector{Float64}(cubature_weights),
        Matrix{Float64}(interpolation),
        ntuple(_ -> zeros(Float64, dg.ref.Np), 7),
        zeros(Float64, dg.ref.Np, 12),
        zeros(Float64, number_cubature_points, 12),
        zeros(Float64, 19),
        x,
        y,
        z,
    )
end

function distributed_maxwell_lightweight_diagnostics_and_energy!(
    workspace::DistributedMaxwellLightweightWorkspace,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    ;
    epsilon::Float64,
    mu::Float64,
)
    fill!(workspace.local_sums, 0.0)
    curl_ex, curl_ey, curl_ez, curl_hx, curl_hy, curl_hz, scratch =
        workspace.work
    nodal_values = workspace.nodal_values
    cubature_values = workspace.cubature_values
    dg = distributed_dg.dg

    for elem in distributed_dg.distributed_mesh.partition.owned
        op = dg.physops.elements[elem]
        jacobian = dg.mappings.tet_mappings[elem].absdetJ
        permeability = mu

        @views begin
            Ex = U.Ex[:, elem]
            Ey = U.Ey[:, elem]
            Ez = U.Ez[:, elem]
            Hx = U.Hx[:, elem]
            Hy = U.Hy[:, elem]
            Hz = U.Hz[:, elem]

            mul!(curl_ex, op.Dy, Ez)
            mul!(scratch, op.Dz, Ey)
            curl_ex .-= scratch
            mul!(curl_ey, op.Dz, Ex)
            mul!(scratch, op.Dx, Ez)
            curl_ey .-= scratch
            mul!(curl_ez, op.Dx, Ey)
            mul!(scratch, op.Dy, Ex)
            curl_ez .-= scratch

            mul!(curl_hx, op.Dy, Hz)
            mul!(scratch, op.Dz, Hy)
            curl_hx .-= scratch
            mul!(curl_hy, op.Dz, Hx)
            mul!(scratch, op.Dx, Hz)
            curl_hy .-= scratch
            mul!(curl_hz, op.Dx, Hy)
            mul!(scratch, op.Dy, Hx)
            curl_hz .-= scratch

            copyto!(view(nodal_values, :, 1), Ex)
            copyto!(view(nodal_values, :, 2), Ey)
            copyto!(view(nodal_values, :, 3), Ez)
            copyto!(view(nodal_values, :, 4), Hx)
            copyto!(view(nodal_values, :, 5), Hy)
            copyto!(view(nodal_values, :, 6), Hz)
            copyto!(view(nodal_values, :, 7), curl_ex)
            copyto!(view(nodal_values, :, 8), curl_ey)
            copyto!(view(nodal_values, :, 9), curl_ez)
            copyto!(view(nodal_values, :, 10), curl_hx)
            copyto!(view(nodal_values, :, 11), curl_hy)
            copyto!(view(nodal_values, :, 12), curl_hz)
            mul!(cubature_values, workspace.interpolation, nodal_values)

            for q in axes(workspace.cubature_points, 1)
                weight = jacobian * workspace.cubature_weights[q]
                ex = cubature_values[q, 1]
                ey = cubature_values[q, 2]
                ez = cubature_values[q, 3]
                hx = cubature_values[q, 4]
                hy = cubature_values[q, 5]
                hz = cubature_values[q, 6]
                numerical_curl_ex = cubature_values[q, 7]
                numerical_curl_ey = cubature_values[q, 8]
                numerical_curl_ez = cubature_values[q, 9]
                numerical_curl_hx = cubature_values[q, 10]
                numerical_curl_hy = cubature_values[q, 11]
                numerical_curl_hz = cubature_values[q, 12]
                px = epsilon * permeability * (ey * hz - ez * hy)
                py = epsilon * permeability * (ez * hx - ex * hz)
                pz = epsilon * permeability * (ex * hy - ey * hx)
                x = workspace.x[q, elem]
                y = workspace.y[q, elem]
                z = workspace.z[q, elem]

                workspace.local_sums[1] += 0.5 * weight * (
                    epsilon * (ex * numerical_curl_ex +
                               ey * numerical_curl_ey +
                               ez * numerical_curl_ez) +
                    permeability * (hx * numerical_curl_hx +
                                    hy * numerical_curl_hy +
                                    hz * numerical_curl_hz)
                )
                workspace.local_sums[2] += weight * px
                workspace.local_sums[3] += weight * py
                workspace.local_sums[4] += weight * pz
                workspace.local_sums[5] += weight * (y * pz - z * py)
                workspace.local_sums[6] += weight * (z * px - x * pz)
                workspace.local_sums[7] += weight * (x * py - y * px)
                workspace.local_sums[8] += weight * epsilon * ex
                workspace.local_sums[9] += weight * epsilon * ey
                workspace.local_sums[10] += weight * epsilon * ez
                workspace.local_sums[11] += weight * permeability * hx
                workspace.local_sums[12] += weight * permeability * hy
                workspace.local_sums[13] += weight * permeability * hz
                workspace.local_sums[14] += 0.5 * weight * epsilon * ex^2
                workspace.local_sums[15] += 0.5 * weight * epsilon * ey^2
                workspace.local_sums[16] += 0.5 * weight * epsilon * ez^2
                workspace.local_sums[17] += 0.5 * weight * permeability * hx^2
                workspace.local_sums[18] += 0.5 * weight * permeability * hy^2
                workspace.local_sums[19] += 0.5 * weight * permeability * hz^2
            end
        end
    end

    sums = MPI.Allreduce(workspace.local_sums, +, distributed_dg.comm)
    electric_energy = sum(@view sums[14:16])
    magnetic_energy = sum(@view sums[17:19])
    energy = MaxwellEnergy(
        electric_energy,
        magnetic_energy,
        electric_energy + magnetic_energy,
        sums[14],
        sums[15],
        sums[16],
        sums[17],
        sums[18],
        sums[19],
    )
    diagnostics = MaxwellLightweightDiagnostics(
        sums[1],
        Tuple(sums[2:4]),
        Tuple(sums[5:7]),
        Tuple(sums[8:10]),
        Tuple(sums[11:13]),
    )
    return (; energy, diagnostics)
end

function distributed_maxwell_lightweight_diagnostics!(
    workspace::DistributedMaxwellLightweightWorkspace,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization;
    epsilon::Float64,
    mu::Float64,
)
    result = distributed_maxwell_lightweight_diagnostics_and_energy!(
        workspace,
        U,
        distributed_dg;
        epsilon,
        mu,
    )
    return result.diagnostics
end

function distributed_periodic_exact_lightweight_diagnostics!(
    workspace::DistributedMaxwellLightweightWorkspace,
    distributed_dg::DistributedDGDiscretization,
    time::Float64;
    epsilon::Float64,
    mu::Float64,
    wave::PlaneWaveParameters,
)
    fill!(workspace.local_sums, 0.0)
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
    dg = distributed_dg.dg

    for elem in distributed_dg.distributed_mesh.partition.owned
        jacobian = dg.mappings.tet_mappings[elem].absdetJ
        for q in axes(workspace.cubature_points, 1)
            weight = jacobian * workspace.cubature_weights[q]
            x = workspace.x[q, elem]
            y = workspace.y[q, elem]
            z = workspace.z[q, elem]
            ex, ey, ez = exact_electric(x, y, z)
            hx, hy, hz = exact_magnetic(x, y, z)
            curl_ex, curl_ey, curl_ez = exact_curl_electric(x, y, z)
            curl_hx, curl_hy, curl_hz = exact_curl_magnetic(x, y, z)
            px = epsilon * mu * (ey * hz - ez * hy)
            py = epsilon * mu * (ez * hx - ex * hz)
            pz = epsilon * mu * (ex * hy - ey * hx)

            workspace.local_sums[1] += 0.5 * weight * (
                epsilon * (ex * curl_ex + ey * curl_ey + ez * curl_ez) +
                mu * (hx * curl_hx + hy * curl_hy + hz * curl_hz)
            )
            workspace.local_sums[2] += weight * px
            workspace.local_sums[3] += weight * py
            workspace.local_sums[4] += weight * pz
            workspace.local_sums[5] += weight * (y * pz - z * py)
            workspace.local_sums[6] += weight * (z * px - x * pz)
            workspace.local_sums[7] += weight * (x * py - y * px)
        end
    end

    sums = MPI.Allreduce(workspace.local_sums, +, distributed_dg.comm)
    zeros3 = (0.0, 0.0, 0.0)
    return MaxwellLightweightDiagnostics(
        sums[1],
        Tuple(sums[2:4]),
        Tuple(sums[5:7]),
        zeros3,
        zeros3,
    )
end

function lightweight_diagnostics(
    diagnostics::MaxwellQuadratureDiagnostics,
)
    return MaxwellLightweightDiagnostics(
        diagnostics.optical_chirality,
        (
            diagnostics.linear_momentum_x,
            diagnostics.linear_momentum_y,
            diagnostics.linear_momentum_z,
        ),
        (
            diagnostics.angular_momentum_x,
            diagnostics.angular_momentum_y,
            diagnostics.angular_momentum_z,
        ),
        (
            diagnostics.integrated_d_x,
            diagnostics.integrated_d_y,
            diagnostics.integrated_d_z,
        ),
        (
            diagnostics.integrated_b_x,
            diagnostics.integrated_b_y,
            diagnostics.integrated_b_z,
        ),
    )
end

function exact_lightweight_diagnostics(
    diagnostics::MaxwellQuadratureDiagnostics,
)
    zeros3 = (0.0, 0.0, 0.0)
    return MaxwellLightweightDiagnostics(
        diagnostics.exact_optical_chirality,
        (
            diagnostics.exact_linear_momentum_x,
            diagnostics.exact_linear_momentum_y,
            diagnostics.exact_linear_momentum_z,
        ),
        (
            diagnostics.exact_angular_momentum_x,
            diagnostics.exact_angular_momentum_y,
            diagnostics.exact_angular_momentum_z,
        ),
        zeros3,
        zeros3,
    )
end

struct MaxwellMagneticChargeRHSBreakdown
    full::Float64
    volume::Float64
    interior_faces::Float64
    periodic_faces::Float64
    boundary_faces::Float64
    reconstructed::Float64
    full_minus_reconstructed::Float64
end

mutable struct DistributedMaxwellComponentL2Workspace
    cubature_order::Int
    cubature_points::Matrix{Float64}
    cubature_weights::Vector{Float64}
    interpolation::Matrix{Float64}
    local_sums::Vector{Float64}
    owned_elements::Vector{Int}
    physical_x::Matrix{Float64}
    physical_y::Matrix{Float64}
    physical_z::Matrix{Float64}
    exact_spatial_a::Union{Nothing,Array{Float64,3}}
    exact_spatial_b::Union{Nothing,Array{Float64,3}}
end

function DistributedMaxwellComponentL2Workspace(
    distributed_dg::DistributedDGDiscretization,
    cubature_order::Int,
)
    cubature_points, cubature_weights, _ =
        get_JaskowiecSukumar_cubature(cubature_order)
    interpolation =
        reference_interpolation_matrix(distributed_dg.dg.ref, cubature_points)
    owned_elements = collect(distributed_dg.distributed_mesh.partition.owned)
    nq = size(cubature_points, 1)
    ne = length(owned_elements)
    physical_x = Matrix{Float64}(undef, nq, ne)
    physical_y = similar(physical_x)
    physical_z = similar(physical_x)
    mesh = distributed_dg.dg.mesh
    for (local_index, elem) in enumerate(owned_elements)
        tet_nodes = @view mesh.tets[:, elem]
        for q in 1:nq
            x, y, z = map_to_physical(mesh.points, tet_nodes,
                                      cubature_points[q, 1], cubature_points[q, 2],
                                      cubature_points[q, 3])
            physical_x[q, local_index] = x
            physical_y[q, local_index] = y
            physical_z[q, local_index] = z
        end
    end
    return DistributedMaxwellComponentL2Workspace(
        cubature_order,
        Matrix{Float64}(cubature_points),
        Vector{Float64}(cubature_weights),
        Matrix{Float64}(interpolation),
        zeros(Float64, 6),
        owned_elements,
        physical_x,
        physical_y,
        physical_z,
        nothing,
        nothing,
    )
end

function precompute_maxwell_exact_spatial_modes!(
    workspace::DistributedMaxwellComponentL2Workspace,
    mode_a,
    mode_b = (x, y, z) -> (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)
    nq, ne = size(workspace.physical_x)
    exact_a = Array{Float64}(undef, 6, nq, ne)
    exact_b = similar(exact_a)
    for local_index in 1:ne, q in 1:nq
        x = workspace.physical_x[q, local_index]
        y = workspace.physical_y[q, local_index]
        z = workspace.physical_z[q, local_index]
        a = mode_a(x, y, z)
        b = mode_b(x, y, z)
        length(a) == 6 && length(b) == 6 ||
            throw(ArgumentError("Exact spatial modes must return six components."))
        @inbounds for component in 1:6
            exact_a[component, q, local_index] = a[component]
            exact_b[component, q, local_index] = b[component]
        end
    end
    workspace.exact_spatial_a = exact_a
    workspace.exact_spatial_b = exact_b
    return workspace
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

function local_strong_magnetic_charge_from_components(
    Hx::AbstractMatrix{Float64},
    Hy::AbstractMatrix{Float64},
    Hz::AbstractMatrix{Float64},
    dg::DGDiscretization,
    materials::MaxwellElementMaterials,
    elements,
)
    validate_maxwell_materials(materials, size(Hx, 2))
    size(Hy) == size(Hx) ||
        throw(ArgumentError("Hy must have the same size as Hx."))
    size(Hz) == size(Hx) ||
        throw(ArgumentError("Hz must have the same size as Hx."))

    mass_ones = dg.ref.M * ones(Float64, dg.ref.Np)
    tmpx = zeros(Float64, dg.ref.Np)
    tmpy = similar(tmpx)
    tmpz = similar(tmpx)
    value = 0.0

    for elem in elements
        op = dg.physops.elements[elem]
        J = dg.mappings.tet_mappings[elem].absdetJ
        permeability = materials.permeability[elem]

        mul!(tmpx, op.Dx, @view Hx[:, elem])
        mul!(tmpy, op.Dy, @view Hy[:, elem])
        mul!(tmpz, op.Dz, @view Hz[:, elem])
        tmpx .+= tmpy .+ tmpz

        value += permeability * integrate_nodal_values(tmpx, mass_ones, J)
    end

    return value
end

function distributed_strong_magnetic_charge_from_components(
    Hx::AbstractMatrix{Float64},
    Hy::AbstractMatrix{Float64},
    Hz::AbstractMatrix{Float64},
    distributed_dg::DistributedDGDiscretization,
    materials::MaxwellElementMaterials,
)
    local_value = local_strong_magnetic_charge_from_components(
        Hx,
        Hy,
        Hz,
        distributed_dg.dg,
        materials,
        distributed_dg.distributed_mesh.partition.owned,
    )
    return MPI.Allreduce(local_value, +, distributed_dg.comm)
end

function distributed_strong_magnetic_charge(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    materials::MaxwellElementMaterials,
)
    return distributed_strong_magnetic_charge_from_components(
        U.Hx,
        U.Hy,
        U.Hz,
        distributed_dg,
        materials,
    )
end

function distributed_strong_magnetic_charge_derivative(
    rhs::MaxwellRHS,
    distributed_dg::DistributedDGDiscretization,
    materials::MaxwellElementMaterials,
)
    return distributed_strong_magnetic_charge_from_components(
        rhs.rhsHx,
        rhs.rhsHy,
        rhs.rhsHz,
        distributed_dg,
        materials,
    )
end

function distributed_magnetic_charge_rhs_breakdown(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    periodic::DistributedPeriodicMaxwellExchange,
    registry::MaxwellBoundaryRegistry,
    formulation::PoissonBracketFormulation,
    materials::MaxwellElementMaterials;
    boundary_data::Union{Nothing, MaxwellBoundaryData} = nothing,
)
    validate_maxwell_materials(materials, size(U.Ex, 2))
    dg = distributed_dg.dg

    full_rhs = similar_maxwell_rhs(U)
    maxwell_rhs_periodic!(
        full_rhs,
        U,
        distributed_dg,
        periodic,
        registry,
        formulation,
        materials,
    )
    full = distributed_strong_magnetic_charge_derivative(
        full_rhs,
        distributed_dg,
        materials,
    )

    exchange_maxwell_ghost_traces!(U, distributed_dg)
    exchange_distributed_periodic_traces!(U, distributed_dg, periodic)

    volume_rhs = similar_maxwell_rhs(U)
    fill_maxwell_rhs!(volume_rhs, 0.0)
    maxwell_volume_rhs!(
        volume_rhs,
        U,
        dg.ref,
        dg.physops,
        formulation,
        materials;
        reset = false,
    )
    volume = distributed_strong_magnetic_charge_derivative(
        volume_rhs,
        distributed_dg,
        materials,
    )

    interior_rhs = similar_maxwell_rhs(U)
    fill_maxwell_rhs!(interior_rhs, 0.0)
    add_material_interior_surface_rhs!(
        interior_rhs,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        formulation,
        materials,
    )
    interior_faces = distributed_strong_magnetic_charge_derivative(
        interior_rhs,
        distributed_dg,
        materials,
    )

    periodic_rhs = similar_maxwell_rhs(U)
    fill_maxwell_rhs!(periodic_rhs, 0.0)
    add_distributed_periodic_surface_rhs!(
        periodic_rhs,
        U,
        distributed_dg,
        periodic,
        formulation;
        materials = materials,
    )
    periodic_faces = distributed_strong_magnetic_charge_derivative(
        periodic_rhs,
        distributed_dg,
        materials,
    )

    boundary_rhs = similar_maxwell_rhs(U)
    fill_maxwell_rhs!(boundary_rhs, 0.0)
    add_material_boundary_surface_rhs!(
        boundary_rhs,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        registry,
        formulation,
        materials,
        mesh = dg.mesh,
        boundary_data = boundary_data,
    )
    boundary_faces = distributed_strong_magnetic_charge_derivative(
        boundary_rhs,
        distributed_dg,
        materials,
    )

    reconstructed = volume + interior_faces + periodic_faces + boundary_faces
    return MaxwellMagneticChargeRHSBreakdown(
        full,
        volume,
        interior_faces,
        periodic_faces,
        boundary_faces,
        reconstructed,
        full - reconstructed,
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

    # Energies, field norms/errors, chirality, charges, divergence constraints,
    # momenta, and volume-integrated field components.
    local_sums = zeros(Float64, 52)
    local_divergence_maxima = zeros(Float64, 2)

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
            directional_electric = (operators.Dx * Ex, operators.Dy * Ey, operators.Dz * Ez)
            directional_magnetic = (operators.Dx * Hx, operators.Dy * Hy, operators.Dz * Hz)
            divergence_electric =
                directional_electric[1] + directional_electric[2] + directional_electric[3]
            divergence_magnetic =
                directional_magnetic[1] + directional_magnetic[2] + directional_magnetic[3]
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

                ex_error_squared = (numerical_Ex - exact_Ex)^2
                ey_error_squared = (numerical_Ey - exact_Ey)^2
                ez_error_squared = (numerical_Ez - exact_Ez)^2
                hx_error_squared = (numerical_Hx - exact_Hx)^2
                hy_error_squared = (numerical_Hy - exact_Hy)^2
                hz_error_squared = (numerical_Hz - exact_Hz)^2
                electric_error_squared =
                    ex_error_squared + ey_error_squared + ez_error_squared
                magnetic_error_squared =
                    hx_error_squared + hy_error_squared + hz_error_squared

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
                abs_electric_charge_density = abs(electric_charge_density)
                abs_magnetic_charge_density = abs(magnetic_charge_density)
                local_divergence_maxima[1] =
                    max(local_divergence_maxima[1], abs_electric_charge_density)
                local_divergence_maxima[2] =
                    max(local_divergence_maxima[2], abs_magnetic_charge_density)

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
                local_sums[33] += physical_weight * ex_error_squared
                local_sums[34] += physical_weight * ey_error_squared
                local_sums[35] += physical_weight * ez_error_squared
                local_sums[36] += physical_weight * hx_error_squared
                local_sums[37] += physical_weight * hy_error_squared
                local_sums[38] += physical_weight * hz_error_squared
                local_sums[39] +=
                    physical_weight * electric_charge_density^2
                local_sums[40] +=
                    physical_weight * magnetic_charge_density^2
                local_sums[41] += physical_weight * epsilon * numerical_Ex
                local_sums[42] += physical_weight * epsilon * numerical_Ey
                local_sums[43] += physical_weight * epsilon * numerical_Ez
                local_sums[44] += physical_weight * mu * numerical_Hx
                local_sums[45] += physical_weight * mu * numerical_Hy
                local_sums[46] += physical_weight * mu * numerical_Hz
                for direction in 1:3
                    local_sums[46 + direction] +=
                        physical_weight * epsilon * dot(row, directional_electric[direction])
                    local_sums[49 + direction] +=
                        physical_weight * mu * dot(row, directional_magnetic[direction])
                end
            end
        end
    end

    sums = MPI.Allreduce(local_sums, +, distributed_dg.comm)
    legacy = distributed_discog_charges(U, distributed_dg, interpolation,
        cubature_weights; epsilon, mu)
    divergence_maxima =
        MPI.Allreduce(local_divergence_maxima, max, distributed_dg.comm)
    return maxwell_quadrature_diagnostics_from_sums(
        cubature_order,
        sums,
        divergence_maxima,
        legacy,
    )
end

function maxwell_quadrature_diagnostics_from_sums(
    cubature_order::Int,
    sums::AbstractVector{Float64},
    divergence_maxima = (0.0, 0.0),
    discog_charges = ntuple(_ -> NaN, 16),
)
    electric_l2 = sqrt(max(sums[5], 0.0))
    exact_electric_l2 = sqrt(max(sums[6], 0.0))
    electric_error_l2 = sqrt(max(sums[7], 0.0))
    magnetic_l2 = sqrt(max(sums[8], 0.0))
    exact_magnetic_l2 = sqrt(max(sums[9], 0.0))
    magnetic_error_l2 = sqrt(max(sums[10], 0.0))
    field_error_l2 = sqrt(max(sums[7] + sums[10], 0.0))
    exact_field_l2 = sqrt(max(sums[6] + sums[9], 0.0))
    ex_error_l2 = sqrt(max(sums[33], 0.0))
    ey_error_l2 = sqrt(max(sums[34], 0.0))
    ez_error_l2 = sqrt(max(sums[35], 0.0))
    hx_error_l2 = sqrt(max(sums[36], 0.0))
    hy_error_l2 = sqrt(max(sums[37], 0.0))
    hz_error_l2 = sqrt(max(sums[38], 0.0))
    energy_density_l2 = sqrt(max(sums[11], 0.0))
    exact_energy_density_l2 = sqrt(max(sums[12], 0.0))
    energy_density_error_l2 = sqrt(max(sums[13], 0.0))
    optical_chirality_density_l2 = sqrt(max(sums[16], 0.0))
    exact_optical_chirality_density_l2 = sqrt(max(sums[17], 0.0))
    optical_chirality_density_error_l2 = sqrt(max(sums[18], 0.0))
    electric_divergence_l2 =
        length(sums) >= 39 ? sqrt(max(sums[39], 0.0)) : 0.0
    magnetic_divergence_l2 =
        length(sums) >= 40 ? sqrt(max(sums[40], 0.0)) : 0.0
    electric_divergence_linf =
        length(divergence_maxima) >= 1 ? divergence_maxima[1] : 0.0
    magnetic_divergence_linf =
        length(divergence_maxima) >= 2 ? divergence_maxima[2] : 0.0

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
        ex_error_l2,
        ey_error_l2,
        ez_error_l2,
        hx_error_l2,
        hy_error_l2,
        hz_error_l2,
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
        electric_divergence_l2,
        magnetic_divergence_l2,
        electric_divergence_linf,
        magnetic_divergence_linf,
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
        # Legacy aggregate inputs did not measure these quantities.
        (length(sums) >= 46 ? Tuple(sums[41:46]) : ntuple(_ -> NaN, 6))...,
        (length(sums) >= 52 ? Tuple(sums[47:52]) : ntuple(_ -> NaN, 6))...,
        discog_charges,
        length(sums) >= 46 ? sum(sums[41:43]) : NaN,
        length(sums) >= 46 ? sum(sums[44:46]) : NaN,
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
    mappings = distributed_dg.dg.mappings.tet_mappings
    local_sums = workspace.local_sums
    fill!(local_sums, 0.0)

    for (local_index, elem) in enumerate(workspace.owned_elements)
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
            x = workspace.physical_x[q, local_index]
            y = workspace.physical_y[q, local_index]
            z = workspace.physical_z[q, local_index]
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

function distributed_maxwell_component_l2_errors_separable!(
    workspace::DistributedMaxwellComponentL2Workspace,
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    factors_a::NTuple{6,Float64},
    factors_b::NTuple{6,Float64},
)
    exact_a = workspace.exact_spatial_a
    exact_b = workspace.exact_spatial_b
    exact_a === nothing && throw(ArgumentError("Exact spatial modes are not initialized."))
    exact_b === nothing && throw(ArgumentError("Exact spatial modes are not initialized."))
    interpolation = workspace.interpolation
    weights = workspace.cubature_weights
    mappings = distributed_dg.dg.mappings.tet_mappings
    local_sums = workspace.local_sums
    fill!(local_sums, 0.0)
    for (local_index, elem) in enumerate(workspace.owned_elements)
        jacobian = mappings[elem].absdetJ
        components = (@view(U.Ex[:, elem]), @view(U.Ey[:, elem]), @view(U.Ez[:, elem]),
                      @view(U.Hx[:, elem]), @view(U.Hy[:, elem]), @view(U.Hz[:, elem]))
        for q in axes(interpolation, 1)
            row = @view interpolation[q, :]
            physical_weight = jacobian * weights[q]
            @inbounds for component in 1:6
                exact = factors_a[component] * exact_a[component, q, local_index] +
                        factors_b[component] * exact_b[component, q, local_index]
                error = dot(row, components[component]) - exact
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
        "field_relative_error,ex_error_l2,ey_error_l2,ez_error_l2," *
        "hx_error_l2,hy_error_l2,hz_error_l2,energy_density_l2," *
        "exact_energy_density_l2,energy_density_error_l2," *
        "energy_density_relative_error,optical_chirality," *
        "exact_optical_chirality,optical_chirality_error," *
        "optical_chirality_density_l2," *
        "exact_optical_chirality_density_l2," *
        "optical_chirality_density_error_l2,electric_charge," *
        "magnetic_charge,exact_electric_charge," *
        "exact_magnetic_charge,electric_divergence_l2," *
        "magnetic_divergence_l2,electric_divergence_linf," *
        "magnetic_divergence_linf,linear_momentum_x," *
        "linear_momentum_y,linear_momentum_z," *
        "exact_linear_momentum_x,exact_linear_momentum_y," *
        "exact_linear_momentum_z,angular_momentum_x," *
        "angular_momentum_y,angular_momentum_z," *
        "exact_angular_momentum_x,exact_angular_momentum_y," *
        "exact_angular_momentum_z,integrated_d_x,integrated_d_y," *
        "integrated_d_z,integrated_b_x,integrated_b_y,integrated_b_z," *
        "electric_charge_x,electric_charge_y,electric_charge_z," *
        "magnetic_charge_x,magnetic_charge_y,magnetic_charge_z," *
        join(DISCOG_CHARGE_COLUMNS, ',') * ",integrated_d_sum,integrated_b_sum",
    )
    return nothing
end

function write_lightweight_diagnostics_header(io::IO)
    println(
        io,
        "step,time,total_energy,exact_total_energy,optical_chirality," *
        "exact_optical_chirality,linear_momentum_x,linear_momentum_y," *
        "linear_momentum_z,exact_linear_momentum_x," *
        "exact_linear_momentum_y,exact_linear_momentum_z," *
        "angular_momentum_x,angular_momentum_y,angular_momentum_z," *
        "exact_angular_momentum_x,exact_angular_momentum_y," *
        "exact_angular_momentum_z,integrated_d_x,integrated_d_y," *
        "integrated_d_z,integrated_b_x,integrated_b_y,integrated_b_z",
    )
    return nothing
end

function write_lightweight_diagnostics_row(
    io::IO,
    step::Int,
    time::Float64,
    energy::MaxwellEnergy,
    diagnostics::MaxwellLightweightDiagnostics,
    exact_total_energy::Float64,
    exact_diagnostics::MaxwellLightweightDiagnostics,
)
    values = (
        step,
        time,
        energy.total,
        exact_total_energy,
        diagnostics.optical_chirality,
        exact_diagnostics.optical_chirality,
        diagnostics.linear_momentum...,
        exact_diagnostics.linear_momentum...,
        diagnostics.angular_momentum...,
        exact_diagnostics.angular_momentum...,
        diagnostics.integrated_d...,
        diagnostics.integrated_b...,
    )
    println(io, join(values, ','))
    flush(io)
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
        diagnostics.ex_error_l2,
        diagnostics.ey_error_l2,
        diagnostics.ez_error_l2,
        diagnostics.hx_error_l2,
        diagnostics.hy_error_l2,
        diagnostics.hz_error_l2,
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
        diagnostics.electric_divergence_l2,
        diagnostics.magnetic_divergence_l2,
        diagnostics.electric_divergence_linf,
        diagnostics.magnetic_divergence_linf,
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
        diagnostics.integrated_d_x,
        diagnostics.integrated_d_y,
        diagnostics.integrated_d_z,
        diagnostics.integrated_b_x,
        diagnostics.integrated_b_y,
        diagnostics.integrated_b_z,
        diagnostics.electric_charge_x,
        diagnostics.electric_charge_y,
        diagnostics.electric_charge_z,
        diagnostics.magnetic_charge_x,
        diagnostics.magnetic_charge_y,
        diagnostics.magnetic_charge_z,
        diagnostics.discog_charges...,
        diagnostics.integrated_d_sum,
        diagnostics.integrated_b_sum,
    )
    println(io, join(values, ','))
    flush(io)
    return nothing
end
