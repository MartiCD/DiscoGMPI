struct PlaneWaveParameters
    wave_number::Float64
    angular_frequency::Float64
    magnetic_amplitude::Float64
    x_origin::Float64
end

struct IncidentPlaneWaveParameters
    propagation_direction::NTuple{3, Float64}
    polarization::NTuple{3, Float64}
    wavelength::Float64
    amplitude::Float64
    epsilon::Float64
    mu::Float64
    phase_shift::Float64
end

function normalize_vector3(
    vector::NTuple{3, <:Real},
    name::AbstractString,
)
    norm_value = sqrt(
        Float64(vector[1])^2 +
        Float64(vector[2])^2 +
        Float64(vector[3])^2,
    )
    norm_value > 0.0 ||
        throw(ArgumentError("$name must be nonzero."))
    return (
        Float64(vector[1]) / norm_value,
        Float64(vector[2]) / norm_value,
        Float64(vector[3]) / norm_value,
    )
end

function IncidentPlaneWaveParameters(;
    propagation_direction::NTuple{3, <:Real} = (0.0, 0.0, 1.0),
    polarization::NTuple{3, <:Real} = (1.0, 0.0, 0.0),
    wavelength::Real = 1.0,
    amplitude::Real = 1.0,
    epsilon::Real = 1.0,
    mu::Real = 1.0,
    phase_shift::Real = 0.0,
)
    wavelength > 0.0 ||
        throw(ArgumentError("Incident wavelength must be positive."))
    epsilon > 0.0 ||
        throw(ArgumentError("Incident permittivity must be positive."))
    mu > 0.0 ||
        throw(ArgumentError("Incident permeability must be positive."))

    direction = normalize_vector3(propagation_direction, "Propagation direction")
    pol = normalize_vector3(polarization, "Polarization")
    alignment =
        direction[1] * pol[1] + direction[2] * pol[2] + direction[3] * pol[3]
    abs(alignment) <= 1.0e-12 ||
        throw(
            ArgumentError(
                "Incident polarization must be perpendicular to the " *
                "propagation direction; dot product is $alignment.",
            ),
        )

    return IncidentPlaneWaveParameters(
        direction,
        pol,
        Float64(wavelength),
        Float64(amplitude),
        Float64(epsilon),
        Float64(mu),
        Float64(phase_shift),
    )
end

incident_plane_wave_number(wave::IncidentPlaneWaveParameters) =
    2.0 * pi / wave.wavelength

incident_plane_wave_speed(wave::IncidentPlaneWaveParameters) =
    1.0 / sqrt(wave.epsilon * wave.mu)

incident_plane_wave_angular_frequency(wave::IncidentPlaneWaveParameters) =
    incident_plane_wave_speed(wave) * incident_plane_wave_number(wave)

incident_plane_wave_impedance(wave::IncidentPlaneWaveParameters) =
    sqrt(wave.mu / wave.epsilon)

function incident_plane_wave_phase(
    wave::IncidentPlaneWaveParameters,
    x::Real,
    y::Real,
    z::Real,
    time::Real,
)
    k = incident_plane_wave_number(wave)
    ω = incident_plane_wave_angular_frequency(wave)
    coordinate =
        wave.propagation_direction[1] * Float64(x) +
        wave.propagation_direction[2] * Float64(y) +
        wave.propagation_direction[3] * Float64(z)
    return k * coordinate - ω * Float64(time) + wave.phase_shift
end

function incident_electric_plane_wave(
    x::Real,
    y::Real,
    z::Real,
    time::Real,
    wave::IncidentPlaneWaveParameters = IncidentPlaneWaveParameters(),
)
    value =
        wave.amplitude *
        cos(incident_plane_wave_phase(wave, x, y, z, time))
    return (
        value * wave.polarization[1],
        value * wave.polarization[2],
        value * wave.polarization[3],
    )
end

function incident_magnetic_plane_wave(
    x::Real,
    y::Real,
    z::Real,
    time::Real,
    wave::IncidentPlaneWaveParameters = IncidentPlaneWaveParameters(),
)
    value =
        wave.amplitude *
        cos(incident_plane_wave_phase(wave, x, y, z, time)) /
        incident_plane_wave_impedance(wave)
    direction = wave.propagation_direction
    polarization = wave.polarization
    magnetic_direction = (
        direction[2] * polarization[3] - direction[3] * polarization[2],
        direction[3] * polarization[1] - direction[1] * polarization[3],
        direction[1] * polarization[2] - direction[2] * polarization[1],
    )
    return (
        value * magnetic_direction[1],
        value * magnetic_direction[2],
        value * magnetic_direction[3],
    )
end

function exact_incident_plane_wave_functions(
    time::Float64;
    wave::IncidentPlaneWaveParameters = IncidentPlaneWaveParameters(),
)
    electric = (x, y, z) ->
        incident_electric_plane_wave(x, y, z, time, wave)
    magnetic = (x, y, z) ->
        incident_magnetic_plane_wave(x, y, z, time, wave)
    return electric, magnetic
end

function exact_cavity_mode_functions(
    time::Float64;
    epsilon::Float64,
    mu::Float64,
    boundary_condition::Symbol = :pec,
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

    if boundary_condition == :pec
        return electric, magnetic
    elseif boundary_condition == :pmc
        impedance = sqrt(mu / epsilon)
        dual_electric = function (x, y, z)
            Hx, Hy, Hz = magnetic(x, y, z)
            return (impedance * Hx, impedance * Hy, impedance * Hz)
        end
        dual_magnetic = function (x, y, z)
            Ex, Ey, Ez = electric(x, y, z)
            return (-Ex / impedance, -Ey / impedance, -Ez / impedance)
        end
        return dual_electric, dual_magnetic
    end

    throw(ArgumentError("Unsupported cavity boundary condition $boundary_condition."))
end

function exact_cavity_mode_curl_functions(
    time::Float64;
    epsilon::Float64,
    mu::Float64,
    boundary_condition::Symbol = :pec,
)
    omega = sqrt(3.0) * pi / sqrt(epsilon * mu)
    electric_time_factor = cos(omega * time)
    magnetic_time_factor = sin(omega * time)
    magnetic_scale = pi / (mu * omega)

    curl_electric = function (x, y, z)
        return (
            pi * sin(pi * x) * cos(pi * y) * cos(pi * z) *
            electric_time_factor,
            -2.0 * pi * cos(pi * x) * sin(pi * y) * cos(pi * z) *
            electric_time_factor,
            pi * cos(pi * x) * cos(pi * y) * sin(pi * z) *
            electric_time_factor,
        )
    end

    curl_magnetic = function (x, y, z)
        scale = 3.0 * pi * magnetic_scale
        return (
            scale * cos(pi * x) * sin(pi * y) * sin(pi * z) *
            magnetic_time_factor,
            0.0,
            -scale * sin(pi * x) * sin(pi * y) * cos(pi * z) *
            magnetic_time_factor,
        )
    end

    if boundary_condition == :pec
        return curl_electric, curl_magnetic
    elseif boundary_condition == :pmc
        impedance = sqrt(mu / epsilon)
        dual_curl_electric = (x, y, z) -> begin
            cHx, cHy, cHz = curl_magnetic(x, y, z)
            (impedance * cHx, impedance * cHy, impedance * cHz)
        end
        dual_curl_magnetic = (x, y, z) -> begin
            cEx, cEy, cEz = curl_electric(x, y, z)
            (-cEx / impedance, -cEy / impedance, -cEz / impedance)
        end
        return dual_curl_electric, dual_curl_magnetic
    end

    throw(ArgumentError("Unsupported cavity boundary condition $boundary_condition."))
end

function exact_periodic_wave_functions(
    time::Float64;
    epsilon::Float64,
    mu::Float64,
    wave::PlaneWaveParameters,
)
    electric_scale =
        -wave.wave_number * wave.magnetic_amplitude /
        (wave.angular_frequency * epsilon)
    phase = (x, time_value) ->
        wave.wave_number * (x - wave.x_origin) -
        wave.angular_frequency * time_value

    electric = (x, y, z) ->
        (0.0, 0.0, electric_scale * sin(phase(x, time)))
    magnetic = (x, y, z) ->
        (0.0, wave.magnetic_amplitude * sin(phase(x, time)), 0.0)

    return electric, magnetic
end

function exact_periodic_wave_curl_functions(
    time::Float64;
    epsilon::Float64,
    mu::Float64,
    wave::PlaneWaveParameters,
)
    electric_scale =
        -wave.wave_number * wave.magnetic_amplitude /
        (wave.angular_frequency * epsilon)
    phase = (x, time_value) ->
        wave.wave_number * (x - wave.x_origin) -
        wave.angular_frequency * time_value

    curl_electric = (x, y, z) ->
        (
            0.0,
            -electric_scale * wave.wave_number * cos(phase(x, time)),
            0.0,
        )
    curl_magnetic = (x, y, z) ->
        (
            0.0,
            0.0,
            wave.wave_number * wave.magnetic_amplitude * cos(phase(x, time)),
        )

    return curl_electric, curl_magnetic
end

function optical_chirality_density(
    electric::NTuple{3, Float64},
    curl_electric::NTuple{3, Float64},
    magnetic::NTuple{3, Float64},
    curl_magnetic::NTuple{3, Float64};
    epsilon::Float64,
    mu::Float64,
)
    return 0.5 * (
        epsilon * dot(electric, curl_electric) +
        mu * dot(magnetic, curl_magnetic)
    )
end
