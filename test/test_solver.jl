using Test
using LinearAlgebra
using DiscoGMPI

@testset "DiscoGMPI loads" begin
    @test isdefined(DiscoGMPI, :RawVTUMesh)
    @test isdefined(DiscoGMPI, :ReferenceTet)
    @test isdefined(DiscoGMPI, :MaxwellField)
    @test isdefined(DiscoGMPI, :AbstractBackend)
    @test isdefined(DiscoGMPI, :SerialBackend)
    @test isdefined(DiscoGMPI, :ThreadedBackend)
    @test isdefined(DiscoGMPI, :DGDiscretization)
    @test isdefined(DiscoGMPI, :HesthavenWarburtonFormulation)
    @test isdefined(DiscoGMPI, :PoissonBracketFormulation)
    @test isdefined(DiscoGMPI, :MaxwellFlux_Alternating)
end

function single_tet_boundary_mesh()
    points = [
        -1.0  1.0 -1.0 -1.0
        -1.0 -1.0  1.0 -1.0
        -1.0 -1.0 -1.0  1.0
    ]

    tets = reshape([1, 2, 3, 4], 4, 1)

    tris = [
        2 1 1 1
        3 4 2 3
        4 3 4 2
    ]

    return RawVTUMesh(
        points,
        tets,
        tris,
        [1],
        collect(2:5),
        Dict{String, Any}("boundary_id" => [0, 10, 10, 10, 10]),
    )
end

function two_tet_boundary_mesh()
    points = [
        0.0 1.0 0.0 0.0 1.0
        0.0 0.0 1.0 0.0 1.0
        0.0 0.0 0.0 1.0 1.0
    ]

    tets = [
        1 2
        2 3
        3 4
        4 5
    ]

    tris = [
        1 1 1 3 2 2
        4 2 3 4 5 3
        3 4 2 5 4 5
    ]

    return RawVTUMesh(
        points,
        tets,
        tris,
        [1, 2],
        collect(3:8),
        Dict{String, Any}("boundary_id" => [0, 0, 10, 10, 10, 10, 10, 10]),
    )
end

function max_abs_rhs_difference(a::MaxwellRHS, b::MaxwellRHS)
    return maximum((
        maximum(abs.(a.rhsEx .- b.rhsEx)),
        maximum(abs.(a.rhsEy .- b.rhsEy)),
        maximum(abs.(a.rhsEz .- b.rhsEz)),
        maximum(abs.(a.rhsHx .- b.rhsHx)),
        maximum(abs.(a.rhsHy .- b.rhsHy)),
        maximum(abs.(a.rhsHz .- b.rhsHz)),
    ))
end

function max_abs_field_difference(a::MaxwellField, b::MaxwellField)
    return maximum((
        maximum(abs.(a.Ex .- b.Ex)),
        maximum(abs.(a.Ey .- b.Ey)),
        maximum(abs.(a.Ez .- b.Ez)),
        maximum(abs.(a.Hx .- b.Hx)),
        maximum(abs.(a.Hy .- b.Hy)),
        maximum(abs.(a.Hz .- b.Hz)),
    ))
end

function max_abs_rhs_offset(rhs::MaxwellRHS, value::Float64)
    return maximum((
        maximum(abs.(rhs.rhsEx .- value)),
        maximum(abs.(rhs.rhsEy .- value)),
        maximum(abs.(rhs.rhsEz .- value)),
        maximum(abs.(rhs.rhsHx .- value)),
        maximum(abs.(rhs.rhsHy .- value)),
        maximum(abs.(rhs.rhsHz .- value)),
    ))
end

function colors_are_element_disjoint(colors, face_elements)
    for color in colors
        seen = Set{Int}()

        for face_index in color
            for elem in face_elements(face_index)
                if elem in seen
                    return false
                end

                push!(seen, elem)
            end
        end
    end

    return true
end

struct SentinelBackend <: AbstractBackend end

sentinel_rhs_value(ε::Float64, μ::Float64) = 37.0 + ε + 2.0 * μ
sentinel_periodic_rhs_value(ε::Float64, μ::Float64) = 73.0 + 3.0 * ε + 5.0 * μ

function DiscoGMPI.maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization{SentinelBackend},
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    backend::SentinelBackend;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    DiscoGMPI.fill_maxwell_rhs!(rhs, sentinel_rhs_value(ε, μ))
    return rhs
end

function DiscoGMPI.maxwell_rhs_periodic!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    dg::DGDiscretization{SentinelBackend},
    periodic_faces,
    registry::MaxwellBoundaryRegistry,
    formulation::AbstractMaxwellDGFormulation,
    backend::SentinelBackend;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    DiscoGMPI.fill_maxwell_rhs!(rhs, sentinel_periodic_rhs_value(ε, μ))
    return rhs
end

function max_abs_electric_rhs(rhs::MaxwellRHS)
    return maximum((
        maximum(abs.(rhs.rhsEx)),
        maximum(abs.(rhs.rhsEy)),
        maximum(abs.(rhs.rhsEz)),
    ))
end

function max_abs_magnetic_rhs(rhs::MaxwellRHS)
    return maximum((
        maximum(abs.(rhs.rhsHx)),
        maximum(abs.(rhs.rhsHy)),
        maximum(abs.(rhs.rhsHz)),
    ))
end

function unit_tangent_to_normal(n::NTuple{3, Float64})
    trial = abs(n[1]) < 0.9 ? (1.0, 0.0, 0.0) : (0.0, 1.0, 0.0)
    tangent = (
        n[2] * trial[3] - n[3] * trial[2],
        n[3] * trial[1] - n[1] * trial[3],
        n[1] * trial[2] - n[2] * trial[1],
    )
    scale = sqrt(tangent[1]^2 + tangent[2]^2 + tangent[3]^2)
    return (tangent[1] / scale, tangent[2] / scale, tangent[3] / scale)
end

@testset "Physical weak derivative operators" begin
    mesh = two_tet_boundary_mesh()
    dg = DGDiscretization(mesh, 2)

    for op in dg.physops.elements
        @test maximum(abs.(op.weak.Sx .- dg.ref.M * op.Dx)) <= 1e-12
        @test maximum(abs.(op.weak.Sy .- dg.ref.M * op.Dy)) <= 1e-12
        @test maximum(abs.(op.weak.Sz .- dg.ref.M * op.Dz)) <= 1e-12

        @test op.weak.SxT == transpose(op.weak.Sx)
        @test op.weak.SyT == transpose(op.weak.Sy)
        @test op.weak.SzT == transpose(op.weak.Sz)
    end

    @test test_physical_stiffness_consistency(dg.ref, dg.physops) <= 1e-12
end

@testset "Poisson-bracket Maxwell volume operator" begin
    mesh = two_tet_boundary_mesh()
    dg = DGDiscretization(mesh, 2)

    nvals = dg.ref.Np * size(mesh.tets, 2)

    U = MaxwellField(
        reshape(collect(1.0:nvals), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(2.0:(nvals + 1.0)), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(3.0:(nvals + 2.0)), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(4.0:(nvals + 3.0)), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(5.0:(nvals + 4.0)), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(6.0:(nvals + 5.0)), dg.ref.Np, size(mesh.tets, 2)),
    )

    rhs_pb = DiscoGMPI.similar_maxwell_rhs(U)
    rhs_hw = DiscoGMPI.similar_maxwell_rhs(U)

    maxwell_volume_rhs!(
        rhs_pb,
        U,
        dg.ref,
        dg.physops,
        PoissonBracketFormulation();
        ε = 2.0,
        μ = 3.0,
    )

    maxwell_volume_rhs!(
        rhs_hw,
        U,
        dg.physops;
        ε = 2.0,
        μ = 3.0,
    )

    @test maximum(abs.(rhs_pb.rhsEx .- rhs_hw.rhsEx)) <= 1e-10
    @test maximum(abs.(rhs_pb.rhsEy .- rhs_hw.rhsEy)) <= 1e-10
    @test maximum(abs.(rhs_pb.rhsEz .- rhs_hw.rhsEz)) <= 1e-10
    @test max_abs_magnetic_rhs(rhs_pb) > 1e-8
    @test max_abs_rhs_difference(rhs_pb, rhs_hw) > 1e-8

    rate = maxwell_energy_rate(
        U,
        rhs_pb,
        dg.ref,
        dg.mappings;
        ε = 2.0,
        μ = 3.0,
    )

    @test abs(rate) <= 1e-8
end

@testset "Poisson-bracket Maxwell surface operator" begin
    mesh = two_tet_boundary_mesh()
    dg = DGDiscretization(mesh, 1)

    Efun = (x, y, z) -> (1.0, 0.0, 0.0)
    Hfun = (x, y, z) -> (0.0, 1.0, 0.0)

    U = interpolate_maxwell_field(mesh, dg.ref, Efun, Hfun)

    rhs_hw = DiscoGMPI.similar_maxwell_rhs(U)
    rhs_pb = DiscoGMPI.similar_maxwell_rhs(U)

    DiscoGMPI.fill_maxwell_rhs!(rhs_hw, 0.0)
    DiscoGMPI.fill_maxwell_rhs!(rhs_pb, 0.0)

    DiscoGMPI.maxwell_interior_surface_rhs!(
        rhs_hw,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces;
        flux_kind = MaxwellFlux_Central,
    )

    DiscoGMPI.maxwell_interior_surface_rhs!(
        rhs_pb,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        PoissonBracketFormulation(),
    )

    @test max_abs_electric_rhs(rhs_hw) <= 1e-12
    @test max_abs_magnetic_rhs(rhs_hw) <= 1e-12

    @test max_abs_electric_rhs(rhs_pb) <= 1e-12
    @test max_abs_magnetic_rhs(rhs_pb) > 1e-12

    nvals = dg.ref.Np * size(mesh.tets, 2)
    U_jump = MaxwellField(
        reshape(collect(1.0:nvals), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(2.0:(nvals + 1.0)), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(3.0:(nvals + 2.0)), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(4.0:(nvals + 3.0)), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(5.0:(nvals + 4.0)), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(6.0:(nvals + 5.0)), dg.ref.Np, size(mesh.tets, 2)),
    )
    rhs_pb_jump = DiscoGMPI.similar_maxwell_rhs(U_jump)

    DiscoGMPI.fill_maxwell_rhs!(rhs_pb_jump, 0.0)
    DiscoGMPI.maxwell_interior_surface_rhs!(
        rhs_pb_jump,
        U_jump,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        PoissonBracketFormulation(),
    )

    rate = maxwell_energy_rate(
        U_jump,
        rhs_pb_jump,
        dg.ref,
        dg.mappings,
    )

    @test abs(rate) <= 1e-10
end

@testset "Poisson-bracket PEC boundary magnetic flux" begin
    normals = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
        inv(sqrt(3.0)) .* (-1.0, 1.0, -1.0),
    )
    amplitudes = (1.0, -2.0, 0.5, 3.0)

    for normal in normals
        tangent = unit_tangent_to_normal(normal)
        minus = (
            Ex = [amplitude * tangent[1] for amplitude in amplitudes],
            Ey = [amplitude * tangent[2] for amplitude in amplitudes],
            Ez = [amplitude * tangent[3] for amplitude in amplitudes],
            Hx = [0.3 + 0.2 * q for q in eachindex(amplitudes)],
            Hy = [-0.4 + 0.1 * q for q in eachindex(amplitudes)],
            Hz = [0.7 - 0.3 * q for q in eachindex(amplitudes)],
        )
        normal_electric = normal[1] .* minus.Ex .+
                          normal[2] .* minus.Ey .+
                          normal[3] .* minus.Ez
        plus = DiscoGMPI.pec_boundary_plus_trace(minus, normal)
        flux = DiscoGMPI.maxwell_poisson_bracket_surface_flux_values(
            minus,
            plus,
            normal;
            ε = 2.0,
            μ = 3.0,
        )

        @test maximum(abs, normal_electric) <= 1e-14
        @test maximum(abs, flux.fluxHx) <= 1e-14
        @test maximum(abs, flux.fluxHy) <= 1e-14
        @test maximum(abs, flux.fluxHz) <= 1e-14
    end
end

@testset "Poisson-bracket alternating flux values" begin
    minus = (
        Ex = [1.0, 2.0],
        Ey = [3.0, 4.0],
        Ez = [5.0, 6.0],
        Hx = [7.0, 8.0],
        Hy = [9.0, 10.0],
        Hz = [11.0, 12.0],
    )
    plus = (
        Ex = [-1.0, -2.0],
        Ey = [0.5, 1.5],
        Ez = [2.5, 3.5],
        Hx = [-7.0, -8.0],
        Hy = [1.0, 2.0],
        Hz = [3.0, 4.0],
    )
    ε = 2.0
    μ = 4.0

    flux_plus_trace = DiscoGMPI.maxwell_poisson_bracket_surface_flux_values(
        minus,
        plus,
        (1.0, 0.0, 0.0);
        flux_kind = MaxwellFlux_Alternating,
        ε = ε,
        μ = μ,
    )
    @test flux_plus_trace.fluxEx == [0.0, 0.0]
    @test flux_plus_trace.fluxEy == [0.0, 0.0]
    @test flux_plus_trace.fluxEz == [0.0, 0.0]
    @test flux_plus_trace.fluxHx == [0.0, 0.0]
    @test flux_plus_trace.fluxHy ≈ plus.Ez ./ μ
    @test flux_plus_trace.fluxHz ≈ -plus.Ey ./ μ

    flux_minus_trace = DiscoGMPI.maxwell_poisson_bracket_surface_flux_values(
        minus,
        plus,
        (-1.0, 0.0, 0.0);
        flux_kind = MaxwellFlux_Alternating,
        ε = ε,
        μ = μ,
    )
    dHy = minus.Hy .- plus.Hy
    dHz = minus.Hz .- plus.Hz
    @test flux_minus_trace.fluxEx == [0.0, 0.0]
    @test flux_minus_trace.fluxEy ≈ -dHz ./ ε
    @test flux_minus_trace.fluxEz ≈ dHy ./ ε
    @test flux_minus_trace.fluxHx == [0.0, 0.0]
    @test flux_minus_trace.fluxHy ≈ -minus.Ez ./ μ
    @test flux_minus_trace.fluxHz ≈ minus.Ey ./ μ
end

@testset "Incident plane wave and inhomogeneous PEC traces" begin
    wave = IncidentPlaneWaveParameters()

    @test incident_plane_wave_number(wave) ≈ 2.0 * pi
    @test incident_plane_wave_angular_frequency(wave) ≈ 2.0 * pi
    @test incident_plane_wave_impedance(wave) ≈ 1.0
    @test collect(incident_electric_plane_wave(0.0, 0.0, 0.0, 0.0, wave)) ≈
          [1.0, 0.0, 0.0]
    @test collect(incident_magnetic_plane_wave(0.0, 0.0, 0.0, 0.0, wave)) ≈
          [0.0, 1.0, 0.0]
    @test maximum(
        abs,
        incident_electric_plane_wave(0.0, 0.0, 0.25, 0.0, wave),
    ) <= 1.0e-14
    @test_throws ArgumentError IncidentPlaneWaveParameters(
        propagation_direction = (0.0, 0.0, 1.0),
        polarization = (0.0, 0.0, 1.0),
    )

    mesh = single_tet_boundary_mesh()
    dg = DGDiscretization(mesh, 1)
    ff = dg.flux_faces.boundary[1]
    n = ff.normal
    nfp = length(ff.trace.nodes)
    minus = DiscoGMPI.MaxwellFaceWorkspace(nfp)
    plus = DiscoGMPI.MaxwellFaceWorkspace(nfp)
    fill!(minus.Ex, 0.0)
    fill!(minus.Ey, 0.0)
    fill!(minus.Ez, 0.0)
    fill!(minus.Hx, 0.0)
    fill!(minus.Hy, 0.0)
    fill!(minus.Hz, 0.0)

    data = incident_pec_boundary_data(
        10,
        (x, y, z, t) -> incident_electric_plane_wave(x, y, z, t, wave);
        time = 0.0,
    )
    set_boundary_data_time!(data, 0.0)
    @test boundary_data_time(data) == 0.0

    DiscoGMPI.maxwell_boundary_plus_trace!(
        plus,
        minus,
        mesh,
        dg.ref,
        ff.trace,
        n,
        MaxwellBC_PEC,
        data,
    )

    for q in 1:nfp
        local_node = ff.trace.nodes[q]
        x, y, z = DiscoGMPI.physical_point_on_element(
            mesh,
            dg.ref,
            ff.trace.elem,
            local_node,
        )
        inc = incident_electric_plane_wave(x, y, z, 0.0, wave)
        total_boundary_average = (
            0.5 * plus.Ex[q] + inc[1],
            0.5 * plus.Ey[q] + inc[2],
            0.5 * plus.Ez[q] + inc[3],
        )
        tangential_residual = (
            n[2] * total_boundary_average[3] -
            n[3] * total_boundary_average[2],
            n[3] * total_boundary_average[1] -
            n[1] * total_boundary_average[3],
            n[1] * total_boundary_average[2] -
            n[2] * total_boundary_average[1],
        )
        @test maximum(abs, tangential_residual) <= 1.0e-14
        @test plus.Hx[q] == minus.Hx[q]
        @test plus.Hy[q] == minus.Hy[q]
        @test plus.Hz[q] == minus.Hz[q]
    end

    zero_minus = (
        Ex = [0.0],
        Ey = [0.0],
        Ez = [0.0],
        Hx = [0.0],
        Hy = [0.0],
        Hz = [0.0],
    )
    inhomogeneous_plus = (
        Ex = [-2.0],
        Ey = [0.0],
        Ez = [0.0],
        Hx = [0.0],
        Hy = [0.0],
        Hz = [0.0],
    )
    centered_flux = DiscoGMPI.maxwell_poisson_bracket_surface_flux_values(
        zero_minus,
        inhomogeneous_plus,
        (0.0, 0.0, 1.0);
        flux_kind = MaxwellFlux_Central,
        ε = 1.0,
        μ = 1.0,
    )
    alternating_flux = DiscoGMPI.maxwell_poisson_bracket_surface_flux_values(
        zero_minus,
        inhomogeneous_plus,
        (0.0, 0.0, 1.0);
        flux_kind = MaxwellFlux_Alternating,
        ε = 1.0,
        μ = 1.0,
    )
    @test centered_flux.fluxEx == [0.0]
    @test centered_flux.fluxEy == [0.0]
    @test centered_flux.fluxEz == [0.0]
    @test centered_flux.fluxHx == [0.0]
    @test centered_flux.fluxHy ≈ [1.0]
    @test centered_flux.fluxHz == [0.0]
    @test alternating_flux.fluxEx == [0.0]
    @test alternating_flux.fluxEy == [0.0]
    @test alternating_flux.fluxEz == [0.0]
    @test alternating_flux.fluxHx == [0.0]
    @test alternating_flux.fluxHy ≈ [2.0]
    @test alternating_flux.fluxHz == [0.0]

    scattered_zero = interpolate_maxwell_field(
        mesh,
        dg.ref,
        (x, y, z) -> (0.0, 0.0, 0.0),
        (x, y, z) -> (0.0, 0.0, 0.0),
    )
    materials = homogeneous_maxwell_materials(size(scattered_zero.Ex, 2))
    registry = MaxwellBoundaryRegistry(Dict(10 => MaxwellBC_PEC))
    for flux_kind in (MaxwellFlux_Central, MaxwellFlux_Alternating)
        rhs = DiscoGMPI.similar_maxwell_rhs(scattered_zero)
        maxwell_rhs!(
            rhs,
            scattered_zero,
            dg,
            registry,
            PoissonBracketFormulation(flux_kind),
            materials;
            boundary_data = data,
        )
        @test max_abs_rhs_offset(rhs, 0.0) > 0.0
    end
end

@testset "Maxwell DG formulations" begin
    mesh = two_tet_boundary_mesh()
    dg = DGDiscretization(mesh, 1)
    dg_threaded = DGDiscretization(mesh, 1; backend = ThreadedBackend())

    @test dg isa DGDiscretization{SerialBackend}
    @test dg.backend isa SerialBackend
    @test dg_threaded isa DGDiscretization{ThreadedBackend}
    @test dg_threaded.backend isa ThreadedBackend
    @test colors_are_element_disjoint(
        dg.flux_faces.interior_colors,
        i -> (
            dg.flux_faces.interior[i].trace.minus_elem,
            dg.flux_faces.interior[i].trace.plus_elem,
        ),
    )
    @test colors_are_element_disjoint(
        dg.flux_faces.boundary_colors,
        i -> (dg.flux_faces.boundary[i].trace.elem,),
    )

    registry = MaxwellBoundaryRegistry(
        Dict(10 => DiscoGMPI.MaxwellBC_PEC),
    )

    Efun = (x, y, z) -> (x + 0.2 * y, y - 0.3 * z, z + 0.4 * x)
    Hfun = (x, y, z) -> (z - 0.1 * x, x + 0.5 * y, y - 0.6 * z)

    U = interpolate_maxwell_field(mesh, dg.ref, Efun, Hfun)

    rhs_old = DiscoGMPI.similar_maxwell_rhs(U)
    rhs_new = DiscoGMPI.similar_maxwell_rhs(U)
    rhs_closure = DiscoGMPI.similar_maxwell_rhs(U)
    rhs_threaded = DiscoGMPI.similar_maxwell_rhs(U)
    rhs_threaded_closure = DiscoGMPI.similar_maxwell_rhs(U)
    rhs_pb = DiscoGMPI.similar_maxwell_rhs(U)
    rhs_pb_closure = DiscoGMPI.similar_maxwell_rhs(U)
    rhs_pb_threaded = DiscoGMPI.similar_maxwell_rhs(U)
    rhs_pb_alternating = DiscoGMPI.similar_maxwell_rhs(U)

    maxwell_rhs!(
        rhs_old,
        U,
        dg.ref,
        dg.fops,
        dg.physops,
        dg.mappings,
        dg.flux_faces,
        registry;
        flux_kind = MaxwellFlux_Upwind,
    )

    formulation = HesthavenWarburtonFormulation(MaxwellFlux_Upwind)

    maxwell_rhs!(
        rhs_new,
        U,
        dg,
        registry,
        formulation,
    )

    rhs_function! = DiscoGMPI.make_maxwell_rhs_function(dg, registry, formulation)
    rhs_function!(rhs_closure, U)

    maxwell_rhs!(
        rhs_threaded,
        U,
        dg_threaded,
        registry,
        formulation,
    )

    threaded_rhs_function! = DiscoGMPI.make_maxwell_rhs_function(
        dg_threaded,
        registry,
        formulation,
    )
    threaded_rhs_function!(rhs_threaded_closure, U)

    @test max_abs_rhs_difference(rhs_new, rhs_old) == 0.0
    @test max_abs_rhs_difference(rhs_closure, rhs_old) == 0.0
    @test max_abs_rhs_difference(rhs_threaded, rhs_old) <= 1e-12
    @test max_abs_rhs_difference(rhs_threaded_closure, rhs_old) <= 1e-12

    pb_formulation = PoissonBracketFormulation()

    maxwell_rhs!(
        rhs_pb,
        U,
        dg,
        registry,
        pb_formulation,
    )

    pb_rhs_function! = DiscoGMPI.make_maxwell_rhs_function(dg, registry, pb_formulation)
    pb_rhs_function!(rhs_pb_closure, U)

    maxwell_rhs!(
        rhs_pb_threaded,
        U,
        dg_threaded,
        registry,
        pb_formulation,
    )

    @test max_abs_rhs_difference(rhs_pb_closure, rhs_pb) == 0.0
    @test max_abs_rhs_difference(rhs_pb_threaded, rhs_pb) <= 1e-12
    @test max_abs_rhs_difference(rhs_pb, rhs_new) > 1e-8

    maxwell_rhs!(
        rhs_pb_alternating,
        U,
        dg,
        registry,
        PoissonBracketFormulation(MaxwellFlux_Alternating),
    )
    @test isfinite(max_abs_rhs_offset(rhs_pb_alternating, 0.0))

    pmc_registry = MaxwellBoundaryRegistry(
        Dict(10 => DiscoGMPI.MaxwellBC_PMC),
    )
    @test_throws ArgumentError maxwell_rhs!(
        rhs_pb_alternating,
        U,
        dg,
        pmc_registry,
        PoissonBracketFormulation(MaxwellFlux_Alternating),
    )

    @test_throws ArgumentError maxwell_rhs!(
        rhs_new,
        U,
        dg,
        registry,
        PoissonBracketFormulation(MaxwellFlux_Upwind),
    )
    @test_throws ArgumentError maxwell_rhs!(
        rhs_threaded,
        U,
        dg_threaded,
        registry,
        PoissonBracketFormulation(MaxwellFlux_Upwind),
    )
end

@testset "Maxwell RHS factories preserve backend dispatch" begin
    mesh = single_tet_boundary_mesh()
    dg = DGDiscretization(mesh, 1; backend = SentinelBackend())
    registry = empty_maxwell_boundary_registry()
    formulation = PoissonBracketFormulation()

    zero_E = (x, y, z) -> (0.0, 0.0, 0.0)
    zero_H = (x, y, z) -> (0.0, 0.0, 0.0)
    U = interpolate_maxwell_field(mesh, dg.ref, zero_E, zero_H)

    ε = 2.0
    μ = 3.0
    rhs = DiscoGMPI.similar_maxwell_rhs(U)
    rhs_function! = DiscoGMPI.make_maxwell_rhs_function(
        dg,
        registry,
        formulation;
        ε = ε,
        μ = μ,
    )

    rhs_function!(rhs, U)
    @test max_abs_rhs_offset(rhs, sentinel_rhs_value(ε, μ)) == 0.0

    periodic_faces = DiscoGMPI.DGPeriodicFluxFaces(DiscoGMPI.PeriodicFluxFace[])
    rhs_periodic = DiscoGMPI.similar_maxwell_rhs(U)
    periodic_rhs_function! = DiscoGMPI.make_maxwell_periodic_rhs_function(
        dg,
        periodic_faces,
        registry,
        formulation;
        ε = ε,
        μ = μ,
    )

    periodic_rhs_function!(rhs_periodic, U)
    @test max_abs_rhs_offset(rhs_periodic, sentinel_periodic_rhs_value(ε, μ)) == 0.0
end

@testset "Poisson-bracket time marching matches threaded backend" begin
    mesh = two_tet_boundary_mesh()
    dg_serial = DGDiscretization(mesh, 1; backend = SerialBackend())
    dg_threaded = DGDiscretization(mesh, 1; backend = ThreadedBackend())
    registry = MaxwellBoundaryRegistry(
        Dict(10 => DiscoGMPI.MaxwellBC_PEC),
    )
    formulation = PoissonBracketFormulation()

    Efun = (x, y, z) -> (
        sin(x + 0.25 * y),
        cos(y - 0.5 * z),
        x * z + 0.1 * y,
    )
    Hfun = (x, y, z) -> (
        y * z - 0.2 * x,
        sin(z + x),
        cos(x - y),
    )

    U_serial = interpolate_maxwell_field(mesh, dg_serial.ref, Efun, Hfun)
    U_threaded = DiscoGMPI.similar_maxwell_field(U_serial)
    DiscoGMPI.copy_maxwell_field!(U_threaded, U_serial)

    run_maxwell_partitioned_symplectic_time_steps!(
        U_serial,
        dg_serial,
        registry,
        formulation;
        psrk_order = 2,
        first_partition = :H,
        dt = 0.01,
        nsteps = 2,
        energy_every = 1000,
    )

    run_maxwell_partitioned_symplectic_time_steps!(
        U_threaded,
        dg_threaded,
        registry,
        formulation;
        psrk_order = 2,
        first_partition = :H,
        dt = 0.01,
        nsteps = 2,
        energy_every = 1000,
    )

    @test max_abs_field_difference(U_threaded, U_serial) <= 1e-12
end

function oscillator_field(E::Float64, H::Float64)
    z = zeros(Float64, 1, 1)

    return MaxwellField(
        fill(E, 1, 1),
        copy(z),
        copy(z),
        fill(H, 1, 1),
        copy(z),
        copy(z),
    )
end

function oscillator_rhs!(rhs::MaxwellRHS, U::MaxwellField)
    fill!(rhs.rhsEx, 0.0)
    fill!(rhs.rhsEy, 0.0)
    fill!(rhs.rhsEz, 0.0)
    fill!(rhs.rhsHx, 0.0)
    fill!(rhs.rhsHy, 0.0)
    fill!(rhs.rhsHz, 0.0)

    rhs.rhsEx .= U.Hx
    rhs.rhsHx .= -U.Ex

    return rhs
end

@testset "Partitioned symplectic RK schemes" begin
    expected_stages = Dict(1 => 1, 2 => 2, 3 => 3, 4 => 6, 5 => 6, 6 => 11)

    for order in 1:6
        scheme = explicit_partitioned_symplectic_rk_scheme(order; first_partition = :H)

        @test scheme.order == order
        @test scheme.first_partition == :H
        @test DiscoGMPI.num_stages(scheme) == expected_stages[order]
        @test sum(scheme.first_weights) ≈ 1.0
        @test sum(scheme.second_weights) ≈ 1.0
    end

    s2 = explicit_partitioned_symplectic_rk_scheme(2; first_partition = :H)
    @test s2.first_weights == [0.0, 1.0]
    @test s2.second_weights == [0.5, 0.5]

    s6 = explicit_partitioned_symplectic_rk_scheme(6; first_partition = :E)
    @test s6.first_partition == :E
    @test DiscoGMPI.num_stages(s6) == 11

    @test_throws ErrorException explicit_partitioned_symplectic_rk_scheme(7)
    @test_throws ErrorException explicit_partitioned_symplectic_rk_scheme(2; first_partition = :bad)
end

@testset "Partitioned symplectic RK Maxwell stepping" begin
    U = oscillator_field(1.0, 0.0)
    scheme = explicit_partitioned_symplectic_rk_scheme(2; first_partition = :H)
    work = MaxwellPartitionedRKWorkspace(U, scheme)

    partitioned_symplectic_rk_step!(
        U,
        work,
        scheme,
        0.1,
        oscillator_rhs!,
    )

    @test U.Ex[1, 1] ≈ 0.995
    @test U.Hx[1, 1] ≈ -0.1
    @test U.Ey[1, 1] == 0.0
    @test U.Hy[1, 1] == 0.0

    U4 = oscillator_field(1.0, 0.0)
    s4 = explicit_partitioned_symplectic_rk_scheme(4; first_partition = :H)
    work4 = MaxwellPartitionedRKWorkspace(U4, s4)

    partitioned_symplectic_rk_step!(
        U4,
        work4,
        s4,
        0.1,
        oscillator_rhs!,
    )

    exact_E = cos(0.1)
    exact_H = -sin(0.1)
    @test abs(U4.Ex[1, 1] - exact_E) < 2e-6
    @test abs(U4.Hx[1, 1] - exact_H) < 2e-6

    UH = oscillator_field(1.0, 0.0)
    sH = explicit_partitioned_symplectic_rk_scheme(1; first_partition = :H)
    workH = MaxwellPartitionedRKWorkspace(UH, sH)

    partitioned_symplectic_rk_step!(
        UH,
        workH,
        sH,
        0.1,
        oscillator_rhs!,
    )

    @test UH.Hx[1, 1] ≈ -0.1
    @test UH.Ex[1, 1] ≈ 0.99
end

@testset "Poisson-bracket Maxwell time marching restrictions" begin
    mesh = single_tet_boundary_mesh()
    dg = DGDiscretization(mesh, 1)
    registry = empty_maxwell_boundary_registry()
    formulation = PoissonBracketFormulation()

    zero_E = (x, y, z) -> (0.0, 0.0, 0.0)
    zero_H = (x, y, z) -> (0.0, 0.0, 0.0)

    @test_throws ArgumentError run_maxwell_time_steps!(
        interpolate_maxwell_field(mesh, dg.ref, zero_E, zero_H),
        dg,
        registry,
        formulation;
        dt = 0.1,
        nsteps = 1,
    )

    @test_throws ArgumentError run_maxwell_partitioned_symplectic_time_steps!(
        interpolate_maxwell_field(mesh, dg.ref, zero_E, zero_H),
        dg,
        registry,
        formulation;
        psrk_order = 1,
        first_partition = :E,
        dt = 0.1,
        nsteps = 1,
    )

    U = interpolate_maxwell_field(mesh, dg.ref, zero_E, zero_H)

    run_maxwell_partitioned_symplectic_time_steps!(
        U,
        dg,
        registry,
        formulation;
        psrk_order = 2,
        first_partition = :H,
        dt = 0.1,
        nsteps = 1,
    )

    @test DiscoGMPI.max_abs_maxwell_field(U) <= 1e-14

    U6 = interpolate_maxwell_field(mesh, dg.ref, zero_E, zero_H)

    run_maxwell_partitioned_symplectic_time_steps!(
        U6,
        dg,
        registry,
        formulation;
        psrk_order = 6,
        first_partition = :H,
        dt = 0.1,
        nsteps = 1,
    )

    @test DiscoGMPI.max_abs_maxwell_field(U6) <= 1e-14
end

@testset "Spatial Maxwell materials" begin
    @test_throws ArgumentError MaxwellMaterial(0.0, 1.0)
    @test_throws ArgumentError MaxwellElementMaterials([1.0], [1.0, 2.0])

    mesh = two_tet_boundary_mesh()
    dg = DGDiscretization(mesh, 1)
    registry = empty_maxwell_boundary_registry()
    formulation = PoissonBracketFormulation()
    Efun = (x, y, z) -> (y + z, z + x, x + y)
    Hfun = (x, y, z) -> (y - z, z - x, x - y)
    U = interpolate_maxwell_field(mesh, dg.ref, Efun, Hfun)
    scalar_rhs = DiscoGMPI.similar_maxwell_rhs(U)
    material_rhs = DiscoGMPI.similar_maxwell_rhs(U)
    materials = homogeneous_maxwell_materials(2)

    maxwell_rhs!(
        scalar_rhs,
        U,
        dg,
        registry,
        formulation;
        ε = 1.0,
        μ = 1.0,
    )
    maxwell_rhs!(
        material_rhs,
        U,
        dg,
        registry,
        formulation,
        materials,
    )
    @test max_abs_rhs_difference(scalar_rhs, material_rhs) < 1e-12

    material_table = Dict(
        1 => MaxwellMaterial(2.0, 3.0),
        2 => MaxwellMaterial(4.0, 5.0),
    )
    heterogeneous = maxwell_element_materials([1, 2], material_table)
    zero = interpolate_maxwell_field(
        mesh,
        dg.ref,
        (x, y, z) -> (0.0, 0.0, 0.0),
        (x, y, z) -> (0.0, 0.0, 0.0),
    )
    zero_rhs = DiscoGMPI.similar_maxwell_rhs(zero)
    maxwell_rhs!(
        zero_rhs,
        zero,
        dg,
        registry,
        formulation,
        heterogeneous,
    )
    @test max_abs_rhs_offset(zero_rhs, 0.0) < 1e-14

    invariants = maxwell_invariants(U, dg, heterogeneous)
    @test invariants.energy.total > 0.0
    @test all(isfinite, invariants.linear_momentum)
    @test all(isfinite, invariants.angular_momentum)
end

@testset "PMC and absorbing Maxwell boundary states" begin
    minus = (
        Ex = [1.0],
        Ey = [2.0],
        Ez = [3.0],
        Hx = [4.0],
        Hy = [5.0],
        Hz = [6.0],
    )
    normal = (1.0, 0.0, 0.0)

    pmc = DiscoGMPI.maxwell_boundary_plus_trace(
        minus,
        normal,
        MaxwellBC_PMC,
    )
    @test pmc.Ex == minus.Ex
    @test pmc.Hx == [4.0]
    @test pmc.Hy == [-5.0]
    @test pmc.Hz == [-6.0]

    absorbing = DiscoGMPI.maxwell_boundary_plus_trace(
        minus,
        normal,
        MaxwellBC_Absorbing;
        ε = 1.0,
        μ = 4.0,
    )
    @test absorbing.Ex == [1.0]
    @test absorbing.Ey == [12.0]
    @test absorbing.Ez == [-10.0]
    @test absorbing.Hx == [4.0]
    @test absorbing.Hy == [-1.5]
    @test absorbing.Hz == [1.0]

    mesh = single_tet_boundary_mesh()
    dg = DGDiscretization(mesh, 1)
    zero = interpolate_maxwell_field(
        mesh,
        dg.ref,
        (x, y, z) -> (0.0, 0.0, 0.0),
        (x, y, z) -> (0.0, 0.0, 0.0),
    )
    for kind in (MaxwellBC_PMC, MaxwellBC_Absorbing)
        for formulation in (
            HesthavenWarburtonFormulation(),
            PoissonBracketFormulation(),
        )
            rhs = DiscoGMPI.similar_maxwell_rhs(zero)
            registry = MaxwellBoundaryRegistry(Dict(10 => kind))
            maxwell_rhs!(
                rhs,
                zero,
                dg,
                registry,
                formulation;
                ε = 2.0,
                μ = 3.0,
            )
            @test max_abs_rhs_offset(rhs, 0.0) < 1e-14
        end
    end
end

@testset "Abarbanel-Gottlieb-Hesthaven nonlinear PML" begin
    electric = (1.2, -0.7, 0.9)
    magnetic = (-0.4, 1.1, 0.3)
    sigma = (0.2, 0.5, 0.8)
    regularization = 1e-12
    source = nonlinear_pml_source(
        electric,
        magnetic,
        sigma;
        a = 0.5,
        regularization = regularization,
    )

    Ex, Ey, Ez = electric
    Hx, Hy, Hz = magnetic
    sigma_x, sigma_y, sigma_z = sigma
    cross_x = Ey * Hz - Ez * Hy
    cross_y = Ez * Hx - Ex * Hz
    cross_z = Ex * Hy - Ey * Hx
    denominator =
        0.5 * (
            Ex^2 + Ey^2 + Ez^2 +
            Hx^2 + Hy^2 + Hz^2
        ) + regularization

    appendix_electric = (
        (sigma_y * cross_y * Hz - sigma_z * cross_z * Hy) /
        denominator,
        (sigma_z * cross_z * Hx - sigma_x * cross_x * Hz) /
        denominator,
        (sigma_x * cross_x * Hy - sigma_y * cross_y * Hx) /
        denominator,
    )
    appendix_magnetic = (
        (-sigma_y * cross_y * Ez + sigma_z * cross_z * Ey) /
        denominator,
        (-sigma_z * cross_z * Ex + sigma_x * cross_x * Ez) /
        denominator,
        (-sigma_x * cross_x * Ey + sigma_y * cross_y * Ex) /
        denominator,
    )

    @test all(
        isapprox.(source.electric, appendix_electric; atol = 1e-14),
    )
    @test all(
        isapprox.(source.magnetic, appendix_magnetic; atol = 1e-14),
    )

    energy_rate =
        dot(electric, source.electric) +
        dot(magnetic, source.magnetic)
    expected_rate =
        -2.0 * (
            sigma_x * cross_x^2 +
            sigma_y * cross_y^2 +
            sigma_z * cross_z^2
        ) / denominator
    @test energy_rate ≈ expected_rate atol = 1e-14
    @test energy_rate < 0.0

    ε_pml = 2.0
    μ_pml = 5.0
    material_source = nonlinear_pml_source(
        electric,
        magnetic,
        sigma;
        ε = ε_pml,
        μ = μ_pml,
        a = 0.5,
        regularization = regularization,
    )
    material_energy_rate =
        ε_pml * dot(electric, material_source.electric) +
        μ_pml * dot(magnetic, material_source.magnetic)
    material_denominator =
        0.5 * (
            ε_pml * (Ex^2 + Ey^2 + Ez^2) +
            μ_pml * (Hx^2 + Hy^2 + Hz^2)
        ) + regularization
    material_expected_rate =
        -2.0 * (
            sigma_x * cross_x^2 +
            sigma_y * cross_y^2 +
            sigma_z * cross_z^2
        ) / material_denominator
    @test material_energy_rate ≈ material_expected_rate atol = 1e-14
    @test material_energy_rate < 0.0

    zero_source = nonlinear_pml_source(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        sigma,
    )
    @test zero_source.electric == (0.0, 0.0, 0.0)
    @test zero_source.magnetic == (0.0, 0.0, 0.0)

    amplitude = 2.0
    plane_wave_source = nonlinear_pml_source(
        (0.0, 0.0, amplitude),
        (0.0, -amplitude, 0.0),
        (3.0, 0.0, 0.0);
        regularization = regularization,
    )
    damping = 3.0 * amplitude^2 / (amplitude^2 + regularization)
    @test plane_wave_source.electric[3] ≈ -damping * amplitude
    @test plane_wave_source.magnetic[2] ≈ damping * amplitude

    material_plane_wave_source = nonlinear_pml_source(
        (0.0, 0.0, amplitude),
        (0.0, -amplitude, 0.0),
        (3.0, 0.0, 0.0);
        ε = ε_pml,
        μ = μ_pml,
        regularization = regularization,
    )
    material_damping =
        3.0 * amplitude^2 /
        (0.5 * (ε_pml + μ_pml) * amplitude^2 + regularization)
    @test material_plane_wave_source.electric[3] ≈
          -material_damping * amplitude / ε_pml
    @test material_plane_wave_source.magnetic[2] ≈
          material_damping * amplitude / μ_pml

    @test polynomial_pml_sigma(
        1.0,
        1.5,
        2.0;
        sigma_max = 8.0,
    ) == 0.0
    @test polynomial_pml_sigma(
        1.75,
        1.5,
        2.0;
        sigma_max = 8.0,
    ) ≈ 2.0
    @test polynomial_pml_sigma(
        2.5,
        1.5,
        2.0;
        sigma_max = 8.0,
    ) == 8.0
    @test polynomial_pml_sigma(
        0.25,
        0.5,
        0.0;
        sigma_max = 8.0,
    ) ≈ 2.0

    mesh = two_tet_boundary_mesh()
    dg = DGDiscretization(mesh, 1)
    U = interpolate_maxwell_field(
        mesh,
        dg.ref,
        (x, y, z) -> (y + z, z + x, x + y),
        (x, y, z) -> (y - z, z - x, x - y),
    )
    pml = build_maxwell_nonlinear_pml(dg)
    @test size(pml.sigma_x) == size(U.Ex)
    base_rhs = DiscoGMPI.similar_maxwell_rhs(U)
    pml_rhs = DiscoGMPI.similar_maxwell_rhs(U)
    registry = empty_maxwell_boundary_registry()
    formulation = HesthavenWarburtonFormulation()
    maxwell_rhs!(
        base_rhs,
        U,
        dg,
        registry,
        formulation,
    )
    maxwell_nonlinear_pml_rhs!(
        pml_rhs,
        U,
        dg,
        registry,
        formulation,
        pml,
    )
    @test max_abs_rhs_difference(base_rhs, pml_rhs) < 1e-14

    poisson_bracket = PoissonBracketFormulation()
    maxwell_rhs!(
        base_rhs,
        U,
        dg,
        registry,
        poisson_bracket,
    )
    maxwell_nonlinear_pml_rhs!(
        pml_rhs,
        U,
        dg,
        registry,
        poisson_bracket,
        pml,
    )
    @test max_abs_rhs_difference(base_rhs, pml_rhs) < 1e-14

    damped = interpolate_maxwell_field(
        mesh,
        dg.ref,
        (x, y, z) -> (0.0, 0.0, 2.0),
        (x, y, z) -> (0.0, -2.0, 0.0),
    )
    damped_pml = build_maxwell_nonlinear_pml(
        dg;
        sigma_x = (x, y, z) -> 10.0,
    )
    energy_before = maxwell_energy(damped, dg.ref, dg.mappings).total
    run_maxwell_nonlinear_pml_time_steps!(
        damped,
        dg,
        empty_maxwell_boundary_registry(),
        poisson_bracket,
        damped_pml;
        rk_order = 4,
        dt = 1e-4,
        nsteps = 1,
        energy_every = 1,
    )
    energy_after = maxwell_energy(damped, dg.ref, dg.mappings).total
    @test energy_after < energy_before

    material_damped = interpolate_maxwell_field(
        mesh,
        dg.ref,
        (x, y, z) -> (0.0, 0.0, 2.0),
        (x, y, z) -> (0.0, -2.0, 0.0),
    )
    material_rhs = DiscoGMPI.similar_maxwell_rhs(material_damped)
    material_source_rhs = DiscoGMPI.similar_maxwell_rhs(material_damped)
    DiscoGMPI.fill_maxwell_rhs!(material_source_rhs, 0.0)
    heterogeneous = MaxwellElementMaterials([2.0, 4.0], [5.0, 7.0])
    maxwell_nonlinear_pml_rhs!(
        material_rhs,
        material_damped,
        dg,
        empty_maxwell_boundary_registry(),
        PoissonBracketFormulation(MaxwellFlux_Alternating),
        heterogeneous,
        damped_pml,
    )
    @test isfinite(max_abs_rhs_difference(material_rhs, material_source_rhs))
    add_maxwell_nonlinear_pml_source!(
        material_source_rhs,
        material_damped,
        damped_pml,
        heterogeneous,
    )
    for elem in 1:2
        local_damping =
            10.0 * amplitude^2 /
            (
                0.5 * (
                    heterogeneous.epsilon[elem] +
                    heterogeneous.permeability[elem]
                ) * amplitude^2 +
                damped_pml.regularization
            )
        @test all(
            isapprox.(
                material_source_rhs.rhsEz[:, elem],
                -local_damping * amplitude /
                heterogeneous.epsilon[elem];
                atol = 1e-12,
            ),
        )
        @test all(
            isapprox.(
                material_source_rhs.rhsHy[:, elem],
                local_damping * amplitude /
                heterogeneous.permeability[elem];
                atol = 1e-12,
            ),
        )
    end

    esprk_damped = interpolate_maxwell_field(
        mesh,
        dg.ref,
        (x, y, z) -> (0.0, 0.0, 2.0),
        (x, y, z) -> (0.0, -2.0, 0.0),
    )
    esprk_scheme =
        explicit_partitioned_symplectic_rk_scheme(2; first_partition = :H)
    esprk_work = MaxwellPartitionedRKWorkspace(esprk_damped, esprk_scheme)
    esprk_energy_before =
        maxwell_energy(
            esprk_damped,
            dg.ref,
            dg.mappings;
            ε = ε_pml,
            μ = μ_pml,
        ).total
    maxwell_nonlinear_pml_partitioned_symplectic_rk_step!(
        esprk_damped,
        esprk_work,
        esprk_scheme,
        1e-4,
        dg,
        empty_maxwell_boundary_registry(),
        PoissonBracketFormulation(MaxwellFlux_Alternating),
        damped_pml;
        ε = ε_pml,
        μ = μ_pml,
    )
    esprk_energy_after =
        maxwell_energy(
            esprk_damped,
            dg.ref,
            dg.mappings;
            ε = ε_pml,
            μ = μ_pml,
        ).total
    @test esprk_energy_after < esprk_energy_before

    @test_throws ArgumentError MaxwellNonlinearPML(
        zeros(1, 1),
        zeros(1, 2),
        zeros(1, 1),
    )
    @test_throws ArgumentError MaxwellNonlinearPML(
        fill(-1.0, 1, 1),
        zeros(1, 1),
        zeros(1, 1),
    )
end
