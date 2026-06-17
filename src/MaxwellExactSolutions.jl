struct PlaneWaveParameters
    wave_number::Float64
    angular_frequency::Float64
    magnetic_amplitude::Float64
    x_origin::Float64
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
