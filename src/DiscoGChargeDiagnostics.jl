"""
DiscoG-compatible element diagnostics. These deliberately retain the legacy
contractions, not a corrected divergence definition. I interpolates nodes to
quadrature points, weights include the physical Jacobian, G contains physical
basis derivatives at quadrature points, and C contains physical stiffness
matrices (including the Jacobian, WITHOUT a mass inverse).
"""
function compute_electric_charge_directional(E, I, weights, G; epsilon=1.0)
    return ntuple(d -> -epsilon * dot(weights, (I * E[d]) .* vec(sum(G[d]; dims=2))), 3)
end

function compute_electric_charge(E, I, weights, G; epsilon=1.0)
    return sum(compute_electric_charge_directional(E, I, weights, G; epsilon))
end

function _discog_stiffness_charge_directional(U, I, weights, C, coefficient)
    return ntuple(d -> coefficient * dot(weights, I * (C[d] * U[d])), 3)
end

"""
Legacy magnetic diagnostic using Cp=Cwp at the DiscoG ODEs.jl call site.
The original routine ignores its permeability keyword; this compatibility
routine intentionally does too. It must NOT be interpreted as integral div(B).
"""
function compute_magnetic_charge_directional(H, I, weights, Cp; mu=1.0)
    return _discog_stiffness_charge_directional(H, I, weights, Cp, 1.0)
end

function compute_magnetic_charge(H, I, weights, Cp; mu=1.0)
    q = Cp[1] * H[1] + Cp[2] * H[2] + Cp[3] * H[3]
    return dot(weights, I * q)
end

function compute_electric_charge_strong_directional(E, I, weights, Csp; epsilon=1.0)
    return _discog_stiffness_charge_directional(E, I, weights, Csp, epsilon)
end

function compute_magnetic_charge_strong_directional(H, I, weights, Csp; mu=1.0)
    return _discog_stiffness_charge_directional(H, I, weights, Csp, mu)
end

function compute_electric_charge_strong(E, I, weights, Csp; epsilon=1.0)
    q = Csp[1] * E[1] + Csp[2] * E[2] + Csp[3] * E[3]
    return epsilon * dot(weights, I * q)
end

function compute_magnetic_charge_strong(H, I, weights, Csp; mu=1.0)
    q = Csp[1] * H[1] + Csp[2] * H[2] + Csp[3] * H[3]
    return mu * dot(weights, I * q)
end

const DISCOG_CHARGE_COLUMNS = Tuple(
    "discog_" * field * "_charge" * variant * suffix
    for field in ("electric", "magnetic")
    for variant in ("", "_strong")
    for suffix in ("", "_x", "_y", "_z")
)

function distributed_discog_charges(U, ddg, I, weights; epsilon, mu)
    values = zeros(16)
    for e in ddg.distributed_mesh.partition.owned
        op = ddg.dg.physops.elements[e]
        jac = ddg.dg.mappings.tet_mappings[e].absdetJ
        w = jac .* weights
        G = (I * op.Dx, I * op.Dy, I * op.Dz)
        Csp = (jac .* op.weak.Sx, jac .* op.weak.Sy, jac .* op.weak.Sz)
        Cwp = map(transpose, Csp)
        E = (view(U.Ex, :, e), view(U.Ey, :, e), view(U.Ez, :, e))
        H = (view(U.Hx, :, e), view(U.Hy, :, e), view(U.Hz, :, e))
        values .+= (
            compute_electric_charge(E, I, w, G; epsilon),
            compute_electric_charge_directional(E, I, w, G; epsilon)...,
            compute_electric_charge_strong(E, I, w, Csp; epsilon),
            compute_electric_charge_strong_directional(E, I, w, Csp; epsilon)...,
            compute_magnetic_charge(H, I, w, Cwp; mu),
            compute_magnetic_charge_directional(H, I, w, Cwp; mu)...,
            compute_magnetic_charge_strong(H, I, w, Csp; mu),
            compute_magnetic_charge_strong_directional(H, I, w, Csp; mu)...,
        )
    end
    return Tuple(MPI.Allreduce(values, +, ddg.comm))
end

export compute_electric_charge, compute_magnetic_charge,
    compute_electric_charge_strong, compute_magnetic_charge_strong,
    compute_electric_charge_directional, compute_magnetic_charge_directional,
    compute_electric_charge_strong_directional, compute_magnetic_charge_strong_directional
