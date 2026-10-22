"""
    canonical_ffd_field(value, incident_amplitude)

Normalize a completed DiscoGMPI far-field phasor by the signed, complex incident
amplitude and convert it to the FFD convention. DiscoGMPI uses `exp(-im*omega*t)`;
FFD uses `exp(+im*omega*t)` and the outgoing factor `exp(-im*k*r)/r`.
Consequently the result is `conj(value / incident_amplitude)`. For SI input its
unit is metres, and `4*pi*(abs2(Etheta) + abs2(Ephi))` is the RCS in square metres.

The incident amplitude must be finite and nonzero. No epsilon floor is applied:
arbitrarily small, representable source amplitudes receive the same linear
normalization as a unit source.
"""
function canonical_ffd_field(value::Number, incident_amplitude::Number)
    field = ComplexF64(value)
    amplitude = ComplexF64(incident_amplitude)
    isfinite(field) || throw(ArgumentError("The far-field phasor must be finite."))
    isfinite(amplitude) && !iszero(amplitude) || throw(
        ArgumentError("The complex incident amplitude must be finite and nonzero."),
    )
    result = conj(field / amplitude)
    isfinite(result) || throw(
        ArgumentError("The normalized far-field phasor is not representable as ComplexF64."),
    )
    return result
end

"""
    ffd_incident_coordinates(wave::IncidentPlaneWaveParameters; atol=1e-10)

Return `(theta_inc, phi_inc, pol, incident_amplitude)` for an incident plane wave.
Angles are in degrees, `pol` is `FFD_V` or `FFD_H`, and `incident_amplitude`
includes the sign of the projection onto the canonical incident polarization
and the source phase. Pass that amplitude to [`canonical_ffd_field`](@ref).

FFD defines the emitter direction as the negative of the propagation direction.
Away from its poles, the incident polarization must align with either the
spherical theta basis (V) or phi basis (H), to absolute tolerance `atol`.
Slanted polarization cannot be represented by an FFD v1 V/H token and raises
an `ArgumentError`.

Exactly at a pole, choose V and the azimuth that aligns the theta basis positively
with the supplied polarization. A nonzero transverse propagation component,
however small, defines an azimuth and is not rounded to a pole. In particular,
propagation along +z with polarization +x maps to `(180, 180, FFD_V)`.
"""
function ffd_incident_coordinates(
    wave::IncidentPlaneWaveParameters;
    atol::Real = 1.0e-10,
)
    tolerance = Float64(atol)
    isfinite(tolerance) && 0.0 <= tolerance < 1.0 || throw(
        ArgumentError("The FFD polarization tolerance must be finite and in [0, 1)."),
    )
    all(isfinite, wave.propagation_direction) || throw(
        ArgumentError("The incident propagation direction must be finite."),
    )
    all(isfinite, wave.polarization) || throw(
        ArgumentError("The incident polarization must be finite."),
    )
    isfinite(wave.amplitude) && !iszero(wave.amplitude) || throw(
        ArgumentError("FFD output requires a finite, nonzero incident amplitude."),
    )
    isfinite(wave.phase_shift) || throw(
        ArgumentError("The incident phase shift must be finite."),
    )

    direction_norm = hypot(wave.propagation_direction...)
    polarization_norm = hypot(wave.polarization...)
    abs(direction_norm - 1.0) <= tolerance || throw(
        ArgumentError("The incident propagation direction must have unit length."),
    )
    abs(polarization_norm - 1.0) <= tolerance || throw(
        ArgumentError("The incident polarization must have unit length."),
    )
    sx, sy, sz = map(x -> -x / direction_norm, wave.propagation_direction)
    px, py, pz = map(x -> x / polarization_norm, wave.polarization)
    abs(sx * px + sy * py + sz * pz) <= tolerance || throw(
        ArgumentError("The incident polarization must be transverse to propagation."),
    )

    transverse = hypot(sx, sy)
    # atan(y, x) retains a small tilt for which acos(sz) rounds to 0 or pi.
    theta_inc = rad2deg(atan(transverse, sz))
    if iszero(transverse)
        # e_theta = sign(sz) * (cos(phi), sin(phi), 0) at either pole.
        phi = mod(atan(sz * py, sz * px), 2.0 * pi)
        pol = FFD_V
        component = hypot(px, py)
    else
        cos_phi = sx / transverse
        sin_phi = sy / transverse
        phi = mod(atan(sy, sx), 2.0 * pi)
        p_theta = sz * cos_phi * px + sz * sin_phi * py - transverse * pz
        p_phi = -sin_phi * px + cos_phi * py
        if abs(abs(p_theta) - 1.0) <= tolerance && abs(p_phi) <= tolerance
            pol = FFD_V
            component = p_theta
        elseif abs(p_theta) <= tolerance && abs(abs(p_phi) - 1.0) <= tolerance
            pol = FFD_H
            component = p_phi
        else
            throw(
                ArgumentError(
                    "FFD v1 represents only pure incident V or H polarization; " *
                    "the spherical components are p_theta=$p_theta and p_phi=$p_phi.",
                ),
            )
        end
    end

    incident_amplitude = wave.amplitude * component * cis(wave.phase_shift)
    isfinite(incident_amplitude) && !iszero(incident_amplitude) || throw(
        ArgumentError("The canonical incident amplitude must be finite and nonzero."),
    )
    phi_inc = mod(rad2deg(phi), 360.0)
    # Canonicalize signed zero for deterministic FFD output.
    phi_inc = iszero(phi_inc) ? 0.0 : phi_inc
    return (; theta_inc, phi_inc, pol, incident_amplitude)
end
