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

function build_reference_tet(N::Int)
    Np = num_tet_nodes(N)

    r, s, t = equispaced_tet_nodes(N)

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
