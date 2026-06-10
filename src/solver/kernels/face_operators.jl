# -------------------------------------------------------------------------
# Face Stuff 
# -------------------------------------------------------------------------

function reference_face_nodes(ref::ReferenceTet; tol = 1e-10)
    ids = Vector{Int}[]

    push!(
        ids,
        findall(i -> abs(ref.r[i] + ref.s[i] + ref.t[i] + 1.0) < tol, 1:ref.Np),
    )

    push!(
        ids,
        findall(i -> abs(ref.r[i] + 1.0) < tol, 1:ref.Np),
    )

    push!(
        ids,
        findall(i -> abs(ref.s[i] + 1.0) < tol, 1:ref.Np),
    )

    push!(
        ids,
        findall(i -> abs(ref.t[i] + 1.0) < tol, 1:ref.Np),
    )

    return (
        ids[1],
        ids[2],
        ids[3],
        ids[4],
    )
end

function build_debug_face_mass(ref::ReferenceTet, face_node_ids::Vector{Int})
    Nfp = length(face_node_ids)

    Mf = zeros(Float64, Nfp, Nfp)

    # Reference face area scale.
    # Face 1 is the slanted face through nodes 2,3,4.
    # Faces 2,3,4 are coordinate planes.
    #
    # For now distribute area equally to face nodes.
    area = 1.0
    for i in 1:Nfp
        Mf[i, i] = area / Nfp
    end

    return Mf
end

function build_face_mass_quadrature(
    ref::ReferenceTet,
    face_id::Int,
    triq::TriangleQuadrature,
)
    rq, sq, tq, wq = reference_face_quadrature(face_id, triq)

    Lq = eval_nodal_basis_at_points(
        rq,
        sq,
        tq,
        ref.basis,
        ref.invV,
    )

    # Full Np × Np embedded face matrix:
    # Mf_ij = ∫_face ℓ_i ℓ_j dS
    Mf_full = Lq' * Diagonal(wq) * Lq

    # Local face-node-only block, useful for diagnostics.
    ids = reference_face_nodes(ref)[face_id]
    Mf_face = Mf_full[ids, ids]

    return Mf_face, Mf_full
end

function gauss_legendre_1d(n::Int)
    if n <= 0
        error("Number of Gauss-Legendre points must be positive.")
    end

    β = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:(n - 1)]

    T = SymTridiagonal(zeros(Float64, n), β)

    eig = eigen(T)

    x = eig.values
    w = 2.0 .* eig.vectors[1, :].^2

    return x, w
end

function triangle_quadrature_gauss(degree::Int)
    if degree < 0
        error("Quadrature degree must be nonnegative.")
    end

    # Tensor-product Gauss rule.
    # n points integrates 1D polynomials up to degree 2n-1.
    #
    # The Duffy transform introduces an extra factor (1-u),
    # so this conservative choice is safe for polynomial degree `degree`.
    n = ceil(Int, (degree + 2) / 2)

    x, wx = gauss_legendre_1d(n)
    y, wy = gauss_legendre_1d(n)

    # Map [-1, 1] to [0, 1].
    u = 0.5 .* (x .+ 1.0)
    v = 0.5 .* (y .+ 1.0)

    wu = 0.5 .* wx
    wv = 0.5 .* wy

    rq = Float64[]
    sq = Float64[]
    wq = Float64[]

    for i in 1:n
        for j in 1:n
            a = u[i]
            b = (1.0 - u[i]) * v[j]

            weight = wu[i] * wv[j] * (1.0 - u[i])

            push!(rq, a)
            push!(sq, b)
            push!(wq, weight)
        end
    end

    return TriangleQuadrature(rq, sq, wq, degree)
end

# function triangle_quadrature_wandzura()
# end 


function reference_face_area(face_id::Int)
    vids = REF_TET_FACE_VERTEX_IDS[face_id]

    x1 = REF_TET_VERTEX_COORDS[vids[1]]
    x2 = REF_TET_VERTEX_COORDS[vids[2]]
    x3 = REF_TET_VERTEX_COORDS[vids[3]]

    a = vsub(x2, x1)
    b = vsub(x3, x1)

    return 0.5 * norm3(cross3(a, b))
end

function map_triangle_to_reference_face(face_id::Int, a::Float64, b::Float64)
    vids = REF_TET_FACE_VERTEX_IDS[face_id]

    x1 = REF_TET_VERTEX_COORDS[vids[1]]
    x2 = REF_TET_VERTEX_COORDS[vids[2]]
    x3 = REF_TET_VERTEX_COORDS[vids[3]]

    p = (
        x1[1] + a * (x2[1] - x1[1]) + b * (x3[1] - x1[1]),
        x1[2] + a * (x2[2] - x1[2]) + b * (x3[2] - x1[2]),
        x1[3] + a * (x2[3] - x1[3]) + b * (x3[3] - x1[3]),
    )

    return p
end


function reference_face_quadrature(face_id::Int, triq::TriangleQuadrature)
    nq = length(triq.wq)

    rq = Vector{Float64}(undef, nq)
    sq = Vector{Float64}(undef, nq)
    tq = Vector{Float64}(undef, nq)

    # Unit triangle has area 1/2.
    # Its affine image has area Aface.
    # The scaling factor is Aface / (1/2) = 2Aface.
    area_scale = 2.0 * reference_face_area(face_id)

    wq = area_scale .* triq.wq

    for q in 1:nq
        p = map_triangle_to_reference_face(face_id, triq.rq[q], triq.sq[q])

        rq[q] = p[1]
        sq[q] = p[2]
        tq[q] = p[3]
    end

    return rq, sq, tq, wq
end

function eval_orthonormal_modal_basis_at_point(
    r::Float64,
    s::Float64,
    t::Float64,
    basis::OrthonormalTetBasis,
)
    Np = basis.Np

    mono = Vector{Float64}(undef, Np)

    for j in 1:Np
        mono[j] = eval_monomial(r, s, t, basis.exponents[j])
    end

    return mono' * basis.modal_coeffs
end

function eval_nodal_basis_at_points(
    rq::Vector{Float64},
    sq::Vector{Float64},
    tq::Vector{Float64},
    basis::OrthonormalTetBasis,
    invV::Matrix{Float64},
)
    nq = length(rq)
    Np = basis.Np

    Vq = Matrix{Float64}(undef, nq, Np)

    for q in 1:nq
        mono = Vector{Float64}(undef, Np)

        for j in 1:Np
            mono[j] = eval_monomial(rq[q], sq[q], tq[q], basis.exponents[j])
        end

        Vq[q, :] .= (mono' * basis.modal_coeffs)
    end

    # Nodal basis matrix:
    # rows = quadrature points
    # cols = nodal basis functions
    return Vq * invV
end

# Old version for debugging
# function build_reference_face_operators(ref::ReferenceTet)
#     face_nodes = reference_face_nodes(ref)

#     face_mass_vec = Matrix{Float64}[]

#     Emat = zeros(Float64, ref.Np, ref.Np)

#     for f in 1:4
#         ids = face_nodes[f]

#         expected_nfp = (ref.N + 1) * (ref.N + 2) ÷ 2

#         if length(ids) != expected_nfp
#             error(
#                 "Face $f has $(length(ids)) nodes, expected $expected_nfp. " *
#                 "Check reference-node generation."
#             )
#         end

#         Mf = build_debug_face_mass(ref, ids)
#         push!(face_mass_vec, Mf)

#         for a in 1:length(ids)
#             ia = ids[a]

#             for b in 1:length(ids)
#                 ib = ids[b]
#                 Emat[ia, ib] += Mf[a, b]
#             end
#         end
#     end

#     LIFT = ref.M \ Emat

#     return ReferenceTetFaceOperators(
#         face_nodes,
#         (
#             face_mass_vec[1],
#             face_mass_vec[2],
#             face_mass_vec[3],
#             face_mass_vec[4],
#         ),
#         Emat,
#         LIFT,
#     )
# end

function build_reference_face_operators_quadrature(ref::ReferenceTet)
    face_nodes = reference_face_nodes(ref)

    # Need exactness for products of degree-N basis functions.
    triq = triangle_quadrature_gauss(2 * ref.N)
    # triq = triangle_quadrature_wandzura(2 * ref.N)

    face_mass_vec = Matrix{Float64}[]

    Emat = zeros(Float64, ref.Np, ref.Np)

    for f in 1:4
        ids = face_nodes[f]

        expected_nfp = (ref.N + 1) * (ref.N + 2) ÷ 2

        if length(ids) != expected_nfp
            error(
                "Face $f has $(length(ids)) nodes, expected $expected_nfp. " *
                "Check reference-node generation."
            )
        end

        Mf_face, Mf_full = build_face_mass_quadrature(ref, f, triq)

        push!(face_mass_vec, Mf_face)

        Emat .+= Mf_full
    end

    LIFT = ref.M \ Emat

    return ReferenceTetFaceOperators(
        face_nodes,
        (
            face_mass_vec[1],
            face_mass_vec[2],
            face_mass_vec[3],
            face_mass_vec[4],
        ),
        Emat,
        LIFT,
    )
end

# Old version
# function print_reference_face_operator_summary(ref::ReferenceTet, fops::ReferenceTetFaceOperators)
#     println("Reference face operators")
#     println("------------------------")

#     expected_nfp = (ref.N + 1) * (ref.N + 2) ÷ 2

#     println("Polynomial order N:       ", ref.N)
#     println("Nodes per face Nfp:       ", expected_nfp)

#     for f in 1:4
#         println("Face $f node count:        ", length(fops.face_nodes[f]))
#     end

#     println()
#     println("Operator sizes")
#     println("--------------")
#     println("size(Emat):              ", size(fops.Emat))
#     println("size(LIFT):              ", size(fops.LIFT))

#     println()
#     println("Operator norms")
#     println("--------------")
#     println("||Emat||:                ", norm(fops.Emat))
#     println("||LIFT||:                ", norm(fops.LIFT))

#     return nothing
# end

# function print_reference_face_operator_summary(ref::ReferenceTet, fops::ReferenceTetFaceOperators)
#     println("Reference face operators")
#     println("------------------------")

#     expected_nfp = (ref.N + 1) * (ref.N + 2) ÷ 2

#     println("Polynomial order N:       ", ref.N)
#     println("Nodes per face Nfp:       ", expected_nfp)

#     for f in 1:4
#         println("Face $f node count:        ", length(fops.face_nodes[f]))
#     end

#     println()
#     println("Reference face areas from face mass")
#     println("-----------------------------------")

#     ones_vec = ones(ref.Np)

#     for f in 1:4
#         triq = triangle_quadrature_gauss(2 * ref.N)
#         _, _, _, wq = reference_face_quadrature(f, triq)
#         println("Face $f area:              ", sum(wq))
#     end

#     println()
#     println("Operator sizes")
#     println("--------------")
#     println("size(Emat):              ", size(fops.Emat))
#     println("size(LIFT):              ", size(fops.LIFT))

#     println()
#     println("Operator checks")
#     println("---------------")
#     println("||Emat - Emat'||:        ", norm(fops.Emat - fops.Emat'))
#     println("||LIFT||:                ", norm(fops.LIFT))

#     surface_area_total = dot(ones_vec, fops.Emat * ones_vec)

#     println("Total reference surface area from Emat: ", surface_area_total)
#     println("Expected total reference surface area:  ", 6.0 + 2.0 * sqrt(3.0))

#     return nothing
# end

# -------------------------------------------------------------------------
# FACE STUFF Hesthaven & Warburton approach
# -------------------------------------------------------------------------

function num_tri_nodes(N::Int)
    if N < 0
        error("Polynomial order N must be nonnegative.")
    end

    return (N + 1) * (N + 2) ÷ 2
end

function equispaced_tri_nodes(N::Int)
    Np = num_tri_nodes(N)

    a = Vector{Float64}(undef, Np)
    b = Vector{Float64}(undef, Np)

    if N == 0
        a[1] = -1.0 / 3.0
        b[1] = -1.0 / 3.0
        return a, b
    end

    sk = 1

    for i in 0:N
        for j in 0:(N - i)
            k = N - i - j

            λ1 = i / N
            λ2 = j / N
            λ3 = k / N

            a[sk] = -1.0 + 2.0 * λ2
            b[sk] = -1.0 + 2.0 * λ3

            sk += 1
        end
    end

    return a, b
end

function monomial_exponents_2d(N::Int)
    exps = NTuple{2, Int}[]

    for total_degree in 0:N
        for i in 0:total_degree
            j = total_degree - i
            push!(exps, (i, j))
        end
    end

    return exps
end


function eval_monomial_2d(a::Float64, b::Float64, exp::NTuple{2, Int})
    i, j = exp
    return a^i * b^j
end


function eval_dmonomial_da(a::Float64, b::Float64, exp::NTuple{2, Int})
    i, j = exp

    if i == 0
        return 0.0
    else
        return i * a^(i - 1) * b^j
    end
end


function eval_dmonomial_db(a::Float64, b::Float64, exp::NTuple{2, Int})
    i, j = exp

    if j == 0
        return 0.0
    else
        return j * a^i * b^(j - 1)
    end
end

function unit_triangle_monomial_integral(i::Int, j::Int)
    # ∫ u^i v^j du dv over u,v >= 0, u+v <= 1
    return factorial_float(i) * factorial_float(j) /
           factorial_float(i + j + 2)
end


function ref_tri_monomial_integral(i::Int, j::Int)
    # ∫ a^i b^j dA over reference triangle
    # a = -1 + 2u
    # b = -1 + 2v
    # dA = 4 du dv

    val = 0.0

    for p in 0:i
        coeff_a = binomial(i, p) * (-1.0)^(i - p) * 2.0^p

        for q in 0:j
            coeff_b = binomial(j, q) * (-1.0)^(j - q) * 2.0^q

            val += coeff_a * coeff_b *
                   unit_triangle_monomial_integral(p, q)
        end
    end

    return 4.0 * val
end


function build_tri_monomial_gram_matrix(exponents)
    Np = length(exponents)

    G = zeros(Float64, Np, Np)

    for i in 1:Np
        ei = exponents[i]

        for j in 1:Np
            ej = exponents[j]

            G[i, j] = ref_tri_monomial_integral(
                ei[1] + ej[1],
                ei[2] + ej[2],
            )
        end
    end

    return G
end


function build_orthonormal_tri_basis(N::Int)
    Np = num_tri_nodes(N)
    exponents = monomial_exponents_2d(N)

    if length(exponents) != Np
        error("Number of triangle monomials does not match number of nodes.")
    end

    G = build_tri_monomial_gram_matrix(exponents)

    F = cholesky(Symmetric(G))
    U = F.U

    modal_coeffs = inv(U)

    Icheck = modal_coeffs' * G * modal_coeffs
    err = norm(Icheck - I, Inf)

    if err > 1e-10
        @warn "Triangle orthonormal modal basis check is not very accurate" err
    end

    return OrthonormalTriBasis(
        N,
        Np,
        exponents,
        modal_coeffs,
        G,
    )
end

function vandermonde_tri(a::Vector{Float64}, b::Vector{Float64}, basis::OrthonormalTriBasis)
    Np = length(a)

    Vmono = Matrix{Float64}(undef, Np, basis.Np)

    for n in 1:Np
        for m in 1:basis.Np
            Vmono[n, m] = eval_monomial_2d(a[n], b[n], basis.exponents[m])
        end
    end

    return Vmono * basis.modal_coeffs
end


function grad_vandermonde_tri(a::Vector{Float64}, b::Vector{Float64}, basis::OrthonormalTriBasis)
    Np = length(a)

    Va_mono = Matrix{Float64}(undef, Np, basis.Np)
    Vb_mono = Matrix{Float64}(undef, Np, basis.Np)

    for n in 1:Np
        for m in 1:basis.Np
            exp = basis.exponents[m]

            Va_mono[n, m] = eval_dmonomial_da(a[n], b[n], exp)
            Vb_mono[n, m] = eval_dmonomial_db(a[n], b[n], exp)
        end
    end

    Va = Va_mono * basis.modal_coeffs
    Vb = Vb_mono * basis.modal_coeffs

    return Va, Vb
end

function build_reference_tri(N::Int)
    Np = num_tri_nodes(N)

    a, b = equispaced_tri_nodes(N)

    basis = build_orthonormal_tri_basis(N)

    V = vandermonde_tri(a, b, basis)
    invV = inv(V)

    Va, Vb = grad_vandermonde_tri(a, b, basis)

    Da = Va * invV
    Db = Vb * invV

    # Since modal basis is orthonormal:
    #
    # ℓ = modal_basis * invV
    # M = invV' * invV
    M = invV' * invV

    return ReferenceTri(
        N,
        Np,
        a,
        b,
        basis.exponents,
        V,
        invV,
        Da,
        Db,
        M,
    )
end

function print_reference_tri_summary(tri::ReferenceTri)
    println("Reference triangle")
    println("------------------")
    println("Polynomial order N:        ", tri.N)
    println("Number of nodes Np:        ", tri.Np)
    println("Reference area:            ", ref_tri_monomial_integral(0, 0))
    println("Condition number of V:     ", cond(tri.V))

    println()
    println("Mass matrix")
    println("-----------")
    println("size(M):                   ", size(tri.M))
    println("min diagonal(M):           ", minimum(diag(tri.M)))
    println("max diagonal(M):           ", maximum(diag(tri.M)))
    println("symmetry error ||M-M'||:   ", norm(tri.M - tri.M'))

    ones_vec = ones(tri.Np)

    println()
    println("Differentiation checks")
    println("----------------------")
    println("||Da * 1||:                ", norm(tri.Da * ones_vec))
    println("||Db * 1||:                ", norm(tri.Db * ones_vec))

    println()
    println("Coordinate derivative checks")
    println("----------------------------")
    println("||Da*a - 1||:              ", norm(tri.Da * tri.a .- 1.0))
    println("||Db*a||:                  ", norm(tri.Db * tri.a))
    println("||Da*b||:                  ", norm(tri.Da * tri.b))
    println("||Db*b - 1||:              ", norm(tri.Db * tri.b .- 1.0))

    area_from_mass = dot(ones_vec, tri.M * ones_vec)

    println()
    println("Area from mass matrix")
    println("---------------------")
    println("1' M 1:                    ", area_from_mass)
    println("expected area:             ", 2.0)

    return nothing
end

function build_reference_face_mass_from_tri(ref::ReferenceTet, tri::ReferenceTri, face_id::Int)
    scale = reference_face_area(face_id) / 2.0
    return scale * tri.M
end

function tet_face_to_tri_coords(face_id::Int, r::Float64, s::Float64, t::Float64)
    if face_id == 1
        # Face through tet vertices 2,3,4.
        # Triangle vertices:
        # a,b = (-1,-1) -> tet vertex 2: ( 1,-1,-1)
        # a,b = ( 1,-1) -> tet vertex 3: (-1, 1,-1)
        # a,b = (-1, 1) -> tet vertex 4: (-1,-1, 1)
        #
        # Barycentric on this face:
        # μ1 = (r + 1)/2  associated with tet vertex 2
        # μ2 = (s + 1)/2  associated with tet vertex 3
        # μ3 = (t + 1)/2  associated with tet vertex 4
        #
        # a = -1 + 2μ2 = s
        # b = -1 + 2μ3 = t
        return s, t

    elseif face_id == 2
        # r = -1, face vertices 1,4,3
        # choose:
        # tri vertex 1 -> tet vertex 1
        # tri vertex 2 -> tet vertex 4
        # tri vertex 3 -> tet vertex 3
        #
        # a follows tet vertex 4 -> t
        # b follows tet vertex 3 -> s
        return t, s

    elseif face_id == 3
        # s = -1, face vertices 1,2,4
        # tri vertex 1 -> tet vertex 1
        # tri vertex 2 -> tet vertex 2
        # tri vertex 3 -> tet vertex 4
        #
        # a follows tet vertex 2 -> r
        # b follows tet vertex 4 -> t
        return r, t

    elseif face_id == 4
        # t = -1, face vertices 1,3,2
        # tri vertex 1 -> tet vertex 1
        # tri vertex 2 -> tet vertex 3
        # tri vertex 3 -> tet vertex 2
        #
        # a follows tet vertex 3 -> s
        # b follows tet vertex 2 -> r
        return s, r

    else
        error("Invalid face_id $face_id")
    end
end

function ordered_reference_face_nodes(ref::ReferenceTet, tri::ReferenceTri; tol = 1e-10)
    face_nodes = reference_face_nodes(ref)

    ordered = Vector{Int}[]

    for f in 1:4
        ids = face_nodes[f]

        used = falses(length(ids))
        ordered_ids = Vector{Int}(undef, tri.Np)

        for q in 1:tri.Np
            target_a = tri.a[q]
            target_b = tri.b[q]

            found = false

            for local_i in 1:length(ids)
                if used[local_i]
                    continue
                end

                node_id = ids[local_i]

                a, b = tet_face_to_tri_coords(
                    f,
                    ref.r[node_id],
                    ref.s[node_id],
                    ref.t[node_id],
                )

                if abs(a - target_a) < tol && abs(b - target_b) < tol
                    ordered_ids[q] = node_id
                    used[local_i] = true
                    found = true
                    break
                end
            end

            if !found
                error(
                    "Could not match triangle node $q on face $f " *
                    "with target coordinates ($target_a, $target_b)."
                )
            end
        end

        push!(ordered, ordered_ids)
    end

    return (
        ordered[1],
        ordered[2],
        ordered[3],
        ordered[4],
    )
end

function build_reference_face_operators(ref::ReferenceTet)
    tri = build_reference_tri(ref.N)

    face_nodes = ordered_reference_face_nodes(ref, tri)

    face_mass_vec = Matrix{Float64}[]

    Emat = zeros(Float64, ref.Np, ref.Np)

    for f in 1:4
        ids = face_nodes[f]

        Mf = build_reference_face_mass_from_tri(ref, tri, f)

        push!(face_mass_vec, Mf)

        for a in 1:tri.Np
            ia = ids[a]

            for b in 1:tri.Np
                ib = ids[b]

                Emat[ia, ib] += Mf[a, b]
            end
        end
    end

    LIFT = ref.M \ Emat

    return ReferenceTetFaceOperators(
        face_nodes,
        (
            face_mass_vec[1],
            face_mass_vec[2],
            face_mass_vec[3],
            face_mass_vec[4],
        ),
        Emat,
        LIFT,
    )
end

function print_reference_face_operator_summary(ref::ReferenceTet, fops::ReferenceTetFaceOperators)
    println("Reference face operators")
    println("------------------------")

    expected_nfp = (ref.N + 1) * (ref.N + 2) ÷ 2

    println("Polynomial order N:       ", ref.N)
    println("Nodes per face Nfp:       ", expected_nfp)

    for f in 1:4
        println("Face $f node count:        ", length(fops.face_nodes[f]))
    end

    println()
    println("Face mass checks")
    println("----------------")

    for f in 1:4
        ones_face = ones(expected_nfp)
        area_from_mass = dot(ones_face, fops.face_mass[f] * ones_face)
        expected_area = reference_face_area(f)

        println("Face $f:")
        println("  area from Mf:            ", area_from_mass)
        println("  expected area:           ", expected_area)
        println("  ||Mf - Mf'||:            ", norm(fops.face_mass[f] - fops.face_mass[f]'))
    end

    println()
    println("Operator sizes")
    println("--------------")
    println("size(Emat):              ", size(fops.Emat))
    println("size(LIFT):              ", size(fops.LIFT))

    println()
    println("Operator checks")
    println("---------------")
    println("||Emat - Emat'||:        ", norm(fops.Emat - fops.Emat'))
    println("||LIFT||:                ", norm(fops.LIFT))

    ones_vol = ones(ref.Np)
    surface_area_total = dot(ones_vol, fops.Emat * ones_vol)

    println()
    println("Total surface area")
    println("------------------")
    println("from Emat:               ", surface_area_total)
    println("expected:                ", 6.0 + 2.0 * sqrt(3.0))

    return nothing
end
