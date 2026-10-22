using Test
using LinearAlgebra
using DiscoGMPI

@testset "FFD phasor normalization" begin
    response = 0.7 - 1.9im
    for amplitude in (2.0, -2.0, 1.0e-250, -1.0e-250), phase in (0.0, 0.71, -2.3)
        incident = amplitude * cis(phase)
        internal = incident * response
        canonical = canonical_ffd_field(internal, incident)
        @test canonical ≈ conj(response) rtol = 2.0e-15
        @test 4.0 * pi * abs2(canonical) ≈
              4.0 * pi * abs2(response) rtol = 3.0e-15
    end
    @test canonical_ffd_field(0.0, 1.0) === 0.0 - 0.0im
    for amplitude in (0.0, 0.0im, Inf, NaN, 1.0 + Inf * im)
        @test_throws ArgumentError canonical_ffd_field(response, amplitude)
    end
    @test_throws ArgumentError canonical_ffd_field(Inf, 1.0)
    @test_throws ArgumentError canonical_ffd_field(NaN * im, 1.0)
    @test_throws ArgumentError canonical_ffd_field(1.0e300, 1.0e-300)
end

@testset "FFD incident coordinates and polarization" begin
    default = ffd_incident_coordinates(IncidentPlaneWaveParameters())
    @test default.theta_inc == 180.0
    @test default.phi_inc == 180.0
    @test default.pol == FFD_V
    @test default.incident_amplitude == 1.0 + 0.0im

    for theta in (0.0, 0.43, 1.2, pi), phi in (0.0, 0.37, 3.9)
        sin_theta = theta == 0.0 || theta == pi ? 0.0 : sin(theta)
        s = (sin_theta * cos(phi), sin_theta * sin(phi), cos(theta))
        e_theta = (cos(theta) * cos(phi), cos(theta) * sin(phi), -sin_theta)
        e_phi = (-sin(phi), cos(phi), 0.0)
        for (pol, basis) in ((FFD_V, e_theta), (FFD_H, e_phi)), sign in (-1.0, 1.0)
            wave = IncidentPlaneWaveParameters(;
                propagation_direction = map(-, s),
                polarization = map(x -> sign * x, basis),
                amplitude = -2.3,
                phase_shift = 0.6,
            )
            result = ffd_incident_coordinates(wave)
            @test result.theta_inc ≈ rad2deg(theta) atol = 3.0e-14
            if 0.0 < theta < pi
                @test result.phi_inc ≈ rad2deg(phi) atol = 3.0e-14
                @test result.pol == pol
                @test result.incident_amplitude ≈ -2.3 * sign * cis(0.6)
            else
                @test result.pol == FFD_V
                canonical_basis = (
                    cosd(result.theta_inc) * cosd(result.phi_inc),
                    cosd(result.theta_inc) * sind(result.phi_inc),
                    -sind(result.theta_inc),
                )
                @test dot(collect(canonical_basis), collect(wave.polarization)) ≈ 1.0
                @test result.incident_amplitude ≈ -2.3 * cis(0.6)
            end
        end
    end

    # Exact -z propagation is the north incident pole; its basis follows +y.
    north = ffd_incident_coordinates(IncidentPlaneWaveParameters(;
        propagation_direction = (0.0, 0.0, -1.0),
        polarization = (0.0, 1.0, 0.0),
    ))
    @test north.theta_inc == 0.0
    @test north.phi_inc == 90.0
    @test north.pol == FFD_V

    # Retain the azimuth of a small nonzero tilt even if z rounds to exactly 1.
    for tilt in (1.0e-6, 1.0e-12, 1.0e-15)
        wave = IncidentPlaneWaveParameters(;
            propagation_direction = (tilt, 0.0, 1.0),
            polarization = (0.0, 1.0, 0.0),
        )
        result = ffd_incident_coordinates(wave)
        @test result.phi_inc == 180.0
        @test result.pol == FFD_H
        @test result.incident_amplitude ≈ -1.0
    end

    slanted = IncidentPlaneWaveParameters(;
        propagation_direction = (1.0, 0.0, 0.0),
        polarization = (0.0, 1.0, 1.0),
    )
    @test_throws ArgumentError ffd_incident_coordinates(slanted)
    @test_throws ArgumentError ffd_incident_coordinates(
        IncidentPlaneWaveParameters(; amplitude = 0.0),
    )
    @test_throws ArgumentError ffd_incident_coordinates(
        IncidentPlaneWaveParameters(; amplitude = NaN),
    )
    @test_throws ArgumentError ffd_incident_coordinates(
        IncidentPlaneWaveParameters(; phase_shift = Inf),
    )
    for tolerance in (-1.0, 1.0, NaN, Inf)
        @test_throws ArgumentError ffd_incident_coordinates(
            IncidentPlaneWaveParameters(); atol = tolerance,
        )
    end
    invalid_wave = IncidentPlaneWaveParameters(
        (0.0, 0.0, 2.0), (1.0, 0.0, 0.0), 1.0, 1.0, 1.0, 1.0, 0.0,
    )
    @test_throws ArgumentError ffd_incident_coordinates(invalid_wave)
end

@testset "FFD coherent translation of surface currents" begin
    # Translate fixed prescribed currents without re-exciting them. The incident
    # amplitude remains fixed; moving a scatterer under a globally referenced
    # incident plane wave would additionally introduce the incident phase.
    positions = [[0.1, -0.2, 0.3], [-0.4, 0.2, 0.1], [0.3, 0.5, -0.1]]
    currents = [ComplexF64[0.2 + 0.3im, -0.7im, 1.0],
                ComplexF64[-0.1im, 0.6, 0.9 + 0.1im],
                ComplexF64[0.5, -0.2 + 0.4im, 0.7im]]
    weights = [0.3, 0.7, 0.4]
    shift = [0.27, -0.11, 0.38]
    k = 2.9
    impedance = 1.7
    incident = -2.0 * cis(0.41)
    theta, phi = 0.82, 1.19
    rhat = [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)]
    e_theta = [cos(theta) * cos(phi), cos(theta) * sin(phi), -sin(theta)]
    e_phi = [-sin(phi), cos(phi), 0.0]
    function projected_far_field(displacement)
        integral = zeros(ComplexF64, 3)
        for (point, current, weight) in zip(positions, currents, weights)
            integral .+= weight * cis(-k * dot(rhat, point + displacement)) .* current
        end
        field = (im * k * impedance / (4.0 * pi)) .* (rhat * dot(rhat, integral) - integral)
        return (dot(e_theta, field), dot(e_phi, field))
    end
    original = projected_far_field(zeros(3))
    translated = projected_far_field(shift)
    internal_phase = cis(-k * dot(rhat, shift))
    ffd_phase = cis(k * dot(rhat, shift))
    @test abs(ffd_phase - internal_phase) > 0.1
    for component in 1:2
        @test abs(original[component]) > 1.0e-3
        @test translated[component] ≈ original[component] * internal_phase rtol = 2.0e-15
        @test canonical_ffd_field(translated[component], incident) ≈
              canonical_ffd_field(original[component], incident) * ffd_phase rtol = 2.0e-15
    end
end
