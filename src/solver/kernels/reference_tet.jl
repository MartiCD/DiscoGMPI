# -------------------------------------------------------------------------
# Reference tetrahedron nodal basis
# -------------------------------------------------------------------------


function num_tet_nodes(N::Int)
    if N < 0
        error("Polynomial order N must be nonnegative.")
    end

    return (N + 1) * (N + 2) * (N + 3) ÷ 6
end

function equispaced_tet_nodes(N::Int)
    Np = num_tet_nodes(N)

    r = Vector{Float64}(undef, Np)
    s = Vector{Float64}(undef, Np)
    t = Vector{Float64}(undef, Np)

    if N == 0
        # Centroid of the reference tetrahedron.
        r[1] = -0.5
        s[1] = -0.5
        t[1] = -0.5
        return r, s, t
    end

    sk = 1

    for i in 0:N
        for j in 0:(N - i)
            for k in 0:(N - i - j)
                l = N - i - j - k

                # Barycentric coordinates:
                # λ₁ = i/N, λ₂ = j/N, λ₃ = k/N, λ₄ = l/N
                λ1 = i / N
                λ2 = j / N
                λ3 = k / N
                λ4 = l / N

                r[sk] = -1.0 + 2.0 * λ2
                s[sk] = -1.0 + 2.0 * λ3
                t[sk] = -1.0 + 2.0 * λ4

                sk += 1
            end
        end
    end

    return r, s, t
end

const WARP_BLEND_TET_ALPHA = [
    0.0,
    0.0,
    0.0,
    0.1002,
    1.1332,
    1.5608,
    1.3413,
    1.2577,
    1.1603,
    1.10153,
    0.6080,
    0.4523,
    0.8856,
    0.8717,
    0.9655,
]

function canonical_tet_nodal_set(nodal_set::Symbol)
    if nodal_set in (:equispaced, :equi, :uniform)
        return :equispaced
    elseif nodal_set in (:warp_blend, :warpblend, :hesthaven_warburton)
        return :warp_blend
    else
        throw(
            ArgumentError(
                "Unknown tetrahedral nodal set $nodal_set. " *
                "Use :equispaced or :warp_blend.",
            ),
        )
    end
end

function warp_blend_tet_alpha(N::Int)
    N <= 0 && return 0.0
    N <= length(WARP_BLEND_TET_ALPHA) && return WARP_BLEND_TET_ALPHA[N]
    return 1.0
end

function legendre_p_and_derivative(N::Int, x::Float64)
    if N < 0
        throw(ArgumentError("Legendre degree must be nonnegative."))
    elseif N == 0
        return 1.0, 0.0
    elseif N == 1
        return x, 1.0
    end

    pnm2 = 1.0
    pnm1 = x

    for n in 2:N
        pn = ((2n - 1) * x * pnm1 - (n - 1) * pnm2) / n
        pnm2, pnm1 = pnm1, pn
    end

    dPn = N * (pnm2 - x * pnm1) / (1.0 - x^2)
    return pnm1, dPn
end

function gauss_lobatto_legendre_nodes(N::Int)
    if N < 0
        throw(ArgumentError("Gauss-Lobatto-Legendre order must be nonnegative."))
    elseif N == 0
        return [0.0]
    elseif N == 1
        return [-1.0, 1.0]
    end

    nodes = Vector{Float64}(undef, N + 1)
    nodes[1] = -1.0
    nodes[end] = 1.0

    for i in 1:(N - 1)
        x = -cospi(i / N)

        for _ in 1:50
            Pn, dPn = legendre_p_and_derivative(N, x)
            ddPn = (2.0 * x * dPn - N * (N + 1) * Pn) / (1.0 - x^2)
            dx = dPn / ddPn
            x -= dx

            abs(dx) <= 64eps(Float64) * max(1.0, abs(x)) && break
        end

        nodes[i + 1] = x
    end

    sort!(nodes)
    return nodes
end

function hesthaven_warburton_equi_tet_nodes(N::Int)
    Np = num_tet_nodes(N)

    r = Vector{Float64}(undef, Np)
    s = Vector{Float64}(undef, Np)
    t = Vector{Float64}(undef, Np)

    if N == 0
        r[1] = -0.5
        s[1] = -0.5
        t[1] = -0.5
        return r, s, t
    end

    sk = 1
    for n in 1:(N + 1)
        for m in 1:(N + 2 - n)
            for q in 1:(N + 3 - n - m)
                r[sk] = -1.0 + 2.0 * (q - 1) / N
                s[sk] = -1.0 + 2.0 * (m - 1) / N
                t[sk] = -1.0 + 2.0 * (n - 1) / N
                sk += 1
            end
        end
    end

    return r, s, t
end

function edge_warp_factor(N::Int, xnodes::Vector{Float64}, xout::Vector{Float64})
    warp = zeros(Float64, length(xout))
    N <= 0 && return warp

    xeq = [-1.0 + 2.0 * (N + 1 - i) / N for i in 1:(N + 1)]

    for i in 1:(N + 1)
        d = fill(xnodes[i] - xeq[i], length(xout))

        for j in 2:N
            if i != j
                d .*= (xout .- xeq[j]) ./ (xeq[i] - xeq[j])
            end
        end

        if i != 1
            d .*= -1.0 / (xeq[i] - xeq[1])
        end

        if i != (N + 1)
            d .*= 1.0 / (xeq[i] - xeq[end])
        end

        warp .+= d
    end

    return warp
end

function eval_shift_2d(N::Int, alpha::Float64, L1, L2, L3)
    gaussX = -gauss_lobatto_legendre_nodes(N)

    blend1 = L2 .* L3
    blend2 = L1 .* L3
    blend3 = L1 .* L2

    warpfactor1 = 4.0 .* edge_warp_factor(N, gaussX, L3 .- L2)
    warpfactor2 = 4.0 .* edge_warp_factor(N, gaussX, L1 .- L3)
    warpfactor3 = 4.0 .* edge_warp_factor(N, gaussX, L2 .- L1)

    warp1 = blend1 .* warpfactor1 .* (1.0 .+ (alpha .* L1).^2)
    warp2 = blend2 .* warpfactor2 .* (1.0 .+ (alpha .* L2).^2)
    warp3 = blend3 .* warpfactor3 .* (1.0 .+ (alpha .* L3).^2)

    dx =
        warp1 .+
        cos(2.0 * pi / 3.0) .* warp2 .+
        cos(4.0 * pi / 3.0) .* warp3
    dy =
        sin(2.0 * pi / 3.0) .* warp2 .+
        sin(4.0 * pi / 3.0) .* warp3

    return dx, dy
end

function normalize_tuple3(v::NTuple{3, Float64})
    nrm = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
    nrm > 0.0 || throw(ArgumentError("Cannot normalize a zero vector."))
    return (v[1] / nrm, v[2] / nrm, v[3] / nrm)
end

function equilateral_to_reference_tet(
    X::Vector{Float64},
    Y::Vector{Float64},
    Z::Vector{Float64},
)
    v1 = (-1.0, -1.0 / sqrt(3.0), -1.0 / sqrt(6.0))
    v2 = (1.0, -1.0 / sqrt(3.0), -1.0 / sqrt(6.0))
    v3 = (0.0, 2.0 / sqrt(3.0), -1.0 / sqrt(6.0))
    v4 = (0.0, 0.0, 3.0 / sqrt(6.0))

    A = [
        0.5 * (v2[1] - v1[1]) 0.5 * (v3[1] - v1[1]) 0.5 * (v4[1] - v1[1])
        0.5 * (v2[2] - v1[2]) 0.5 * (v3[2] - v1[2]) 0.5 * (v4[2] - v1[2])
        0.5 * (v2[3] - v1[3]) 0.5 * (v3[3] - v1[3]) 0.5 * (v4[3] - v1[3])
    ]

    offset = 0.5 .* (
        (
            v2[1] + v3[1] + v4[1] - v1[1],
            v2[2] + v3[2] + v4[2] - v1[2],
            v2[3] + v3[3] + v4[3] - v1[3],
        )
    )

    Np = length(X)
    r = Vector{Float64}(undef, Np)
    s = Vector{Float64}(undef, Np)
    t = Vector{Float64}(undef, Np)

    for n in 1:Np
        rst = A \ [X[n] - offset[1], Y[n] - offset[2], Z[n] - offset[3]]
        r[n] = rst[1]
        s[n] = rst[2]
        t[n] = rst[3]
    end

    return r, s, t
end

function warp_blend_tet_nodes(N::Int)
    if N < 0
        error("Polynomial order N must be nonnegative.")
    elseif N == 0
        return equispaced_tet_nodes(N)
    end

    alpha = warp_blend_tet_alpha(N)
    r, s, t = hesthaven_warburton_equi_tet_nodes(N)

    L1 = (1.0 .+ t) ./ 2.0
    L2 = (1.0 .+ s) ./ 2.0
    L3 = .-(1.0 .+ r .+ s .+ t) ./ 2.0
    L4 = (1.0 .+ r) ./ 2.0

    v1 = (-1.0, -1.0 / sqrt(3.0), -1.0 / sqrt(6.0))
    v2 = (1.0, -1.0 / sqrt(3.0), -1.0 / sqrt(6.0))
    v3 = (0.0, 2.0 / sqrt(3.0), -1.0 / sqrt(6.0))
    v4 = (0.0, 0.0, 3.0 / sqrt(6.0))

    Np = length(r)
    X = Vector{Float64}(undef, Np)
    Y = Vector{Float64}(undef, Np)
    Z = Vector{Float64}(undef, Np)

    for n in 1:Np
        X[n] = L3[n] * v1[1] + L4[n] * v2[1] + L2[n] * v3[1] + L1[n] * v4[1]
        Y[n] = L3[n] * v1[2] + L4[n] * v2[2] + L2[n] * v3[2] + L1[n] * v4[2]
        Z[n] = L3[n] * v1[3] + L4[n] * v2[3] + L2[n] * v3[3] + L1[n] * v4[3]
    end

    t1 = [
        normalize_tuple3((v2[1] - v1[1], v2[2] - v1[2], v2[3] - v1[3])),
        normalize_tuple3((v2[1] - v1[1], v2[2] - v1[2], v2[3] - v1[3])),
        normalize_tuple3((v3[1] - v2[1], v3[2] - v2[2], v3[3] - v2[3])),
        normalize_tuple3((v3[1] - v1[1], v3[2] - v1[2], v3[3] - v1[3])),
    ]

    t2 = [
        normalize_tuple3((v3[1] - 0.5 * (v1[1] + v2[1]), v3[2] - 0.5 * (v1[2] + v2[2]), v3[3] - 0.5 * (v1[3] + v2[3]))),
        normalize_tuple3((v4[1] - 0.5 * (v1[1] + v2[1]), v4[2] - 0.5 * (v1[2] + v2[2]), v4[3] - 0.5 * (v1[3] + v2[3]))),
        normalize_tuple3((v4[1] - 0.5 * (v2[1] + v3[1]), v4[2] - 0.5 * (v2[2] + v3[2]), v4[3] - 0.5 * (v2[3] + v3[3]))),
        normalize_tuple3((v4[1] - 0.5 * (v1[1] + v3[1]), v4[2] - 0.5 * (v1[2] + v3[2]), v4[3] - 0.5 * (v1[3] + v3[3]))),
    ]

    shiftX = zeros(Float64, Np)
    shiftY = zeros(Float64, Np)
    shiftZ = zeros(Float64, Np)
    tol = 1e-10

    for face in 1:4
        if face == 1
            La, Lb, Lc, Ld = L1, L2, L3, L4
        elseif face == 2
            La, Lb, Lc, Ld = L2, L1, L3, L4
        elseif face == 3
            La, Lb, Lc, Ld = L3, L1, L4, L2
        else
            La, Lb, Lc, Ld = L4, L1, L3, L2
        end

        warp1, warp2 = eval_shift_2d(N, alpha, Lb, Lc, Ld)

        blend = Lb .* Lc .* Ld
        denom = (Lb .+ 0.5 .* La) .* (Lc .+ 0.5 .* La) .* (Ld .+ 0.5 .* La)

        for n in 1:Np
            if denom[n] > tol
                blend[n] =
                    (1.0 + (alpha * La[n])^2) * blend[n] / denom[n]
            end
        end

        for n in 1:Np
            bx = blend[n] * warp1[n]
            by = blend[n] * warp2[n]
            shiftX[n] += bx * t1[face][1] + by * t2[face][1]
            shiftY[n] += bx * t1[face][2] + by * t2[face][2]
            shiftZ[n] += bx * t1[face][3] + by * t2[face][3]
        end

        for n in 1:Np
            nonzero_face_coords =
                (Lb[n] > tol ? 1 : 0) +
                (Lc[n] > tol ? 1 : 0) +
                (Ld[n] > tol ? 1 : 0)

            if La[n] < tol && nonzero_face_coords < 3
                shiftX[n] = warp1[n] * t1[face][1] + warp2[n] * t2[face][1]
                shiftY[n] = warp1[n] * t1[face][2] + warp2[n] * t2[face][2]
                shiftZ[n] = warp1[n] * t1[face][3] + warp2[n] * t2[face][3]
            end
        end
    end

    X .+= shiftX
    Y .+= shiftY
    Z .+= shiftZ

    return equilateral_to_reference_tet(X, Y, Z)
end

function tetrahedral_nodes(N::Int; nodal_set::Symbol = :equispaced)
    family = canonical_tet_nodal_set(nodal_set)

    if family == :equispaced
        return equispaced_tet_nodes(N)
    elseif family == :warp_blend
        return warp_blend_tet_nodes(N)
    end

    error("Unhandled tetrahedral nodal set $nodal_set.")
end

function reference_tet_nodes_are_equispaced(ref::ReferenceTet; tol::Float64 = 1e-10)
    if ref.N == 0
        return ref.Np == 1 &&
               abs(ref.r[1] + 0.5) <= tol &&
               abs(ref.s[1] + 0.5) <= tol &&
               abs(ref.t[1] + 0.5) <= tol
    end

    seen = Set{NTuple{4, Int}}()

    for node in 1:ref.Np
        lambda2_float = ref.N * (ref.r[node] + 1.0) / 2.0
        lambda3_float = ref.N * (ref.s[node] + 1.0) / 2.0
        lambda4_float = ref.N * (ref.t[node] + 1.0) / 2.0

        lambda2 = round(Int, lambda2_float)
        lambda3 = round(Int, lambda3_float)
        lambda4 = round(Int, lambda4_float)
        lambda1 = ref.N - lambda2 - lambda3 - lambda4

        if abs(lambda2_float - lambda2) > tol ||
           abs(lambda3_float - lambda3) > tol ||
           abs(lambda4_float - lambda4) > tol ||
           lambda1 < 0 ||
           lambda2 < 0 ||
           lambda3 < 0 ||
           lambda4 < 0
            return false
        end

        push!(seen, (lambda1, lambda2, lambda3, lambda4))
    end

    return length(seen) == ref.Np
end

function monomial_exponents_3d(N::Int)
    exps = NTuple{3, Int}[]

    for total_degree in 0:N
        for a in 0:total_degree
            for b in 0:(total_degree - a)
                c = total_degree - a - b
                push!(exps, (a, b, c))
            end
        end
    end

    return exps
end


function eval_monomial(r::Float64, s::Float64, t::Float64, exp::NTuple{3, Int})
    a, b, c = exp
    return r^a * s^b * t^c
end


function eval_dmonomial_dr(r::Float64, s::Float64, t::Float64, exp::NTuple{3, Int})
    a, b, c = exp

    if a == 0
        return 0.0
    else
        return a * r^(a - 1) * s^b * t^c
    end
end


function eval_dmonomial_ds(r::Float64, s::Float64, t::Float64, exp::NTuple{3, Int})
    a, b, c = exp

    if b == 0
        return 0.0
    else
        return b * r^a * s^(b - 1) * t^c
    end
end


function eval_dmonomial_dt(r::Float64, s::Float64, t::Float64, exp::NTuple{3, Int})
    a, b, c = exp

    if c == 0
        return 0.0
    else
        return c * r^a * s^b * t^(c - 1)
    end
end

function vandermonde_tet(r::Vector{Float64}, s::Vector{Float64}, t::Vector{Float64}, exponents)
    Np = length(r)
    Nm = length(exponents)

    V = Matrix{Float64}(undef, Np, Nm)

    for i in 1:Np
        for j in 1:Nm
            V[i, j] = eval_monomial(r[i], s[i], t[i], exponents[j])
        end
    end

    return V
end


function grad_vandermonde_tet(r::Vector{Float64}, s::Vector{Float64}, t::Vector{Float64}, exponents)
    Np = length(r)
    Nm = length(exponents)

    Vr = Matrix{Float64}(undef, Np, Nm)
    Vs = Matrix{Float64}(undef, Np, Nm)
    Vt = Matrix{Float64}(undef, Np, Nm)

    for i in 1:Np
        for j in 1:Nm
            exp = exponents[j]

            Vr[i, j] = eval_dmonomial_dr(r[i], s[i], t[i], exp)
            Vs[i, j] = eval_dmonomial_ds(r[i], s[i], t[i], exp)
            Vt[i, j] = eval_dmonomial_dt(r[i], s[i], t[i], exp)
        end
    end

    return Vr, Vs, Vt
end

function factorial_float(n::Int)
    return Float64(factorial(big(n)))
end


function unit_simplex_monomial_integral(i::Int, j::Int, k::Int)
    # ∫ u^i v^j w^k du dv dw over u,v,w >= 0, u+v+w <= 1
    return factorial_float(i) * factorial_float(j) * factorial_float(k) /
           factorial_float(i + j + k + 3)
end


function ref_tet_monomial_integral(a::Int, b::Int, c::Int)
    # Integral over the reference tet:
    # r = -1 + 2u
    # s = -1 + 2v
    # t = -1 + 2w
    # dV_ref = 8 du dv dw

    val = 0.0

    for i in 0:a
        coeff_r = binomial(a, i) * (-1.0)^(a - i) * 2.0^i

        for j in 0:b
            coeff_s = binomial(b, j) * (-1.0)^(b - j) * 2.0^j

            for k in 0:c
                coeff_t = binomial(c, k) * (-1.0)^(c - k) * 2.0^k

                val += coeff_r * coeff_s * coeff_t *
                       unit_simplex_monomial_integral(i, j, k)
            end
        end
    end

    return 8.0 * val
end

function build_monomial_gram_matrix(exponents)
    Np = length(exponents)

    G = zeros(Float64, Np, Np)

    for i in 1:Np
        ei = exponents[i]

        for j in 1:Np
            ej = exponents[j]

            exp_prod = (
                ei[1] + ej[1],
                ei[2] + ej[2],
                ei[3] + ej[3],
            )

            G[i, j] = ref_tet_monomial_integral(
                exp_prod[1],
                exp_prod[2],
                exp_prod[3],
            )
        end
    end

    return G
end

function orthonormal_vandermonde_tet(
    r::Vector{Float64},
    s::Vector{Float64},
    t::Vector{Float64},
    basis::OrthonormalTetBasis,
)
    Vmono = vandermonde_tet(r, s, t, basis.exponents)

    return Vmono * basis.modal_coeffs
end


function orthonormal_grad_vandermonde_tet(
    r::Vector{Float64},
    s::Vector{Float64},
    t::Vector{Float64},
    basis::OrthonormalTetBasis,
)
    Vrmono, Vsmono, Vtmono = grad_vandermonde_tet(
        r,
        s,
        t,
        basis.exponents,
    )

    Vr = Vrmono * basis.modal_coeffs
    Vs = Vsmono * basis.modal_coeffs
    Vt = Vtmono * basis.modal_coeffs

    return Vr, Vs, Vt
end

function build_orthonormal_tet_basis(N::Int)
    Np = num_tet_nodes(N)
    exponents = monomial_exponents_3d(N)

    if length(exponents) != Np
        error("Number of monomials does not match number of tetrahedral nodes.")
    end

    G = build_monomial_gram_matrix(exponents)

    F = cholesky(Symmetric(G))
    U = F.U

    modal_coeffs = inv(U)

    # Check orthonormality.
    Icheck = modal_coeffs' * G * modal_coeffs
    err = norm(Icheck - I, Inf)

    if err > 1e-10
        @warn "Orthonormal modal basis check is not very accurate" err
    end

    return OrthonormalTetBasis(
        N,
        Np,
        exponents,
        modal_coeffs,
        G,
    )
end

function build_reference_mass_matrix(exponents, invV::Matrix{Float64})
    Np = length(exponents)

    M = zeros(Float64, Np, Np)

    for i in 1:Np
        for j in 1:Np
            val = 0.0

            for a_idx in 1:Np
                ea = exponents[a_idx]
                ca = invV[a_idx, i]

                for b_idx in 1:Np
                    eb = exponents[b_idx]
                    cb = invV[b_idx, j]

                    exp_prod = (
                        ea[1] + eb[1],
                        ea[2] + eb[2],
                        ea[3] + eb[3],
                    )

                    val += ca * cb *
                           ref_tet_monomial_integral(
                               exp_prod[1],
                               exp_prod[2],
                               exp_prod[3],
                           )
                end
            end

            M[i, j] = val
        end
    end

    return M
end


function derivative_coeff(exp::NTuple{3, Int}, direction::Symbol)
    a, b, c = exp

    if direction == :r
        if a == 0
            return 0.0, (0, 0, 0)
        else
            return Float64(a), (a - 1, b, c)
        end

    elseif direction == :s
        if b == 0
            return 0.0, (0, 0, 0)
        else
            return Float64(b), (a, b - 1, c)
        end

    elseif direction == :t
        if c == 0
            return 0.0, (0, 0, 0)
        else
            return Float64(c), (a, b, c - 1)
        end

    else
        error("Unknown direction $direction. Use :r, :s, or :t.")
    end
end


function build_reference_stiffness_matrix(
    exponents,
    invV::Matrix{Float64},
    direction::Symbol,
)
    Np = length(exponents)

    S = zeros(Float64, Np, Np)

    for i in 1:Np
        for j in 1:Np
            val = 0.0

            # Basis function ℓᵢ
            for a_idx in 1:Np
                ea = exponents[a_idx]
                ca = invV[a_idx, i]

                # Derivative of basis function ℓⱼ
                for b_idx in 1:Np
                    eb = exponents[b_idx]
                    cb = invV[b_idx, j]

                    dcoeff, dexp = derivative_coeff(eb, direction)

                    if dcoeff == 0.0
                        continue
                    end

                    exp_prod = (
                        ea[1] + dexp[1],
                        ea[2] + dexp[2],
                        ea[3] + dexp[3],
                    )

                    val += ca * cb * dcoeff *
                           ref_tet_monomial_integral(
                               exp_prod[1],
                               exp_prod[2],
                               exp_prod[3],
                           )
                end
            end

            S[i, j] = val
        end
    end

    return S
end

function build_reference_tet(N::Int; nodal_set::Symbol = :equispaced)
    Np = num_tet_nodes(N)

    r, s, t = tetrahedral_nodes(N; nodal_set = nodal_set)

    # exponents = monomial_exponents_3d(N)
    basis = build_orthonormal_tet_basis(N)

    # if length(exponents) != Np
    #     error("Number of monomials does not match number of nodes.")
    # end

    # V = vandermonde_tet(r, s, t, exponents)
    V = orthonormal_vandermonde_tet(r, s, t, basis)
    invV = inv(V)

    Vr, Vs, Vt = orthonormal_grad_vandermonde_tet(r, s, t, basis)
    # Vr, Vs, Vt = grad_vandermonde_tet(r, s, t, exponents)

    Dr = Vr * invV
    Ds = Vs * invV
    Dt = Vt * invV

    # Since the modal basis is orthonormal:
    #
    # nodal basis ℓ = modal_basis * inv(V)
    #
    # M = inv(V)' * inv(V)
    M = invV' * invV

    # Nodal stiffness matrices:
    #
    # Sᵣᵢⱼ = ∫ ℓᵢ ∂ℓⱼ/∂r dV
    #       = M * Dr
    Sr = M * Dr
    Ss = M * Ds
    St = M * Dt

    # M = build_reference_mass_matrix(exponents, invV)

    # Sr = build_reference_stiffness_matrix(exponents, invV, :r)
    # Ss = build_reference_stiffness_matrix(exponents, invV, :s)
    # St = build_reference_stiffness_matrix(exponents, invV, :t)

    # return ReferenceTet(
    #     N,
    #     Np,
    #     r,
    #     s,
    #     t,
    #     exponents,
    #     V,
    #     invV,
    #     Dr,
    #     Ds,
    #     Dt,
    #     M,
    #     Sr,
    #     Ss,
    #     St,
    # )
    return ReferenceTet(
        N,
        Np,
        r,
        s,
        t,
        # basis.exponents,
        basis,
        V,
        invV,
        Dr,
        Ds,
        Dt,
        M,
        Sr,
        Ss,
        St,
    )
end

function print_reference_tet_summary(ref::ReferenceTet)
    println("Reference tetrahedron")
    println("---------------------")
    println("Polynomial order N:        ", ref.N)
    println("Number of nodes Np:        ", ref.Np)
    println("Reference volume:          ", ref_tet_monomial_integral(0, 0, 0))
    println("Condition number of V:     ", cond(ref.V))

    println()
    println("Mass matrix")
    println("-----------")
    println("size(M):                   ", size(ref.M))
    println("min diagonal(M):           ", minimum(diag(ref.M)))
    println("max diagonal(M):           ", maximum(diag(ref.M)))
    println("symmetry error ||M-M'||:   ", norm(ref.M - ref.M'))

    println()
    println("Differentiation checks")
    println("----------------------")

    ones_vec = ones(ref.Np)

    err_dr_const = norm(ref.Dr * ones_vec)
    err_ds_const = norm(ref.Ds * ones_vec)
    err_dt_const = norm(ref.Dt * ones_vec)

    println("||Dr * 1||:                ", err_dr_const)
    println("||Ds * 1||:                ", err_ds_const)
    println("||Dt * 1||:                ", err_dt_const)

    # Check derivatives of coordinate functions.
    err_dr_r = norm(ref.Dr * ref.r .- 1.0)
    err_ds_r = norm(ref.Ds * ref.r)
    err_dt_r = norm(ref.Dt * ref.r)

    err_dr_s = norm(ref.Dr * ref.s)
    err_ds_s = norm(ref.Ds * ref.s .- 1.0)
    err_dt_s = norm(ref.Dt * ref.s)

    err_dr_t = norm(ref.Dr * ref.t)
    err_ds_t = norm(ref.Ds * ref.t)
    err_dt_t = norm(ref.Dt * ref.t .- 1.0)

    println()
    println("Coordinate derivative checks")
    println("----------------------------")
    println("||Dr*r - 1||:              ", err_dr_r)
    println("||Ds*r||:                  ", err_ds_r)
    println("||Dt*r||:                  ", err_dt_r)
    println("||Dr*s||:                  ", err_dr_s)
    println("||Ds*s - 1||:              ", err_ds_s)
    println("||Dt*s||:                  ", err_dt_s)
    println("||Dr*t||:                  ", err_dr_t)
    println("||Ds*t||:                  ", err_ds_t)
    println("||Dt*t - 1||:              ", err_dt_t)

    println()
    println("Stiffness matrix sizes")
    println("----------------------")
    println("size(Sr):                  ", size(ref.Sr))
    println("size(Ss):                  ", size(ref.Ss))
    println("size(St):                  ", size(ref.St))

    return nothing
end

function print_orthonormal_basis_summary(N::Int)
    basis = build_orthonormal_tet_basis(N)

    C = basis.modal_coeffs
    G = basis.gram

    Icheck = C' * G * C

    println("Orthonormal tetrahedral modal basis")
    println("-----------------------------------")
    println("Polynomial order N:              ", N)
    println("Number of basis functions:       ", basis.Np)
    println("||C' G C - I||∞:                 ", norm(Icheck - I, Inf))
    println("condition number of monomial G:  ", cond(G))

    return nothing
end
