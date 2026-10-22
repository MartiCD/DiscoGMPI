"""Reference data for E in P_p and H in P_(p+1), on affine tetrahedra.

`S_EH[a][i,j] = ∫ l_E[i] ∂a l_H[j]` has shape Np_E × Np_H.
The opposite weak coupling uses its transpose (with the curl signs), never
an independently interpolated strong derivative. `S_HE` is retained to
verify integration by parts, not to substitute for that transpose.
"""
struct MixedReferenceTet
    ref_E::ReferenceTet
    ref_H::ReferenceTet
    volume_points::Matrix{Float64}
    volume_weights::Vector{Float64}
    volume_degree::Int
    face_quadrature::TriangleQuadrature
    V_E::Matrix{Float64}
    V_H::Matrix{Float64}
    M_E::Matrix{Float64}
    M_H::Matrix{Float64}
    S_EH::NTuple{3,Matrix{Float64}}
    S_HE::NTuple{3,Matrix{Float64}}
    face_E::NTuple{4,Matrix{Float64}}
    face_H::NTuple{4,Matrix{Float64}}
    face_weights::NTuple{4,Vector{Float64}}
    lift_E::NTuple{4,Matrix{Float64}}
    lift_H::NTuple{4,Matrix{Float64}}
end

"""One directed face; all quadrature rows refer to the same physical points."""
struct MixedFace
    element::Int
    neighbor::Int                 # zero means physical PEC boundary
    local_face::Int
    neighbor_face::Int
    normal::NTuple{3,Float64}
    E::Matrix{Float64}
    H::Matrix{Float64}
    Eplus::Matrix{Float64}
    Hplus::Matrix{Float64}
    lift_E::Matrix{Float64}
    lift_H::Matrix{Float64}
    boundary_id::Int
    points::Matrix{Float64}
end

struct MixedDGDiscretization{B<:AbstractBackend}
    mesh::RawVTUMesh
    reference::MixedReferenceTet
    mappings::DGReferenceMapping
    materials::MaxwellElementMaterials
    volume_E::Vector{NTuple{3,Matrix{Float64}}}
    volume_H::Vector{NTuple{3,Matrix{Float64}}}
    faces::Vector{Vector{MixedFace}}
    backend::B
end

"""Build the bracket DG operator with (p,p+1) spaces and PEC/periodic faces.

`periodic` contains PeriodicBoundarySpec entries. Unpaired physical boundaries
must be PEC. E/H face data are sampled at one common quadrature grid.
"""
function MixedDGDiscretization(mesh::RawVTUMesh, p::Int;
        p_H::Int=p+1, nodal_set::Symbol=:equispaced,
        materials::MaxwellElementMaterials=MaxwellElementMaterials(
            ones(size(mesh.tets,2)),ones(size(mesh.tets,2))),
        periodic=PeriodicBoundarySpec[],
        backend::AbstractBackend=SerialBackend(), boundary_tag_name="boundary_id")
    neighbors,boundary_tags = mixed_connectivity(mesh;periodic,boundary_tag_name)
    return assemble_mixed_dg(mesh,p,neighbors,boundary_tags;p_H,nodal_set,materials,backend)
end

function mixed_connectivity(mesh;periodic=PeriodicBoundarySpec[],boundary_tag_name="boundary_id")
    topology = build_dg_topology(mesh; boundary_tag_name)
    boundary_tags = Dict((f.elem,f.local_face)=>f.boundary_id for f in topology.boundary_faces)
    neighbors = Dict{Tuple{Int,Int},Tuple{Int,Int,NTuple{3,Float64}}}()
    for f in topology.interior_faces
        neighbors[(f.left_elem,f.left_local_face)] = (f.right_elem,f.right_local_face,(0.,0.,0.))
        neighbors[(f.right_elem,f.right_local_face)] = (f.left_elem,f.left_local_face,(0.,0.,0.))
    end
    # Match periodic triangles by translated vertices, not degree-dependent nodes.
    paired = Set{Tuple{Int,Int}}()
    facepoints(e,f) = mesh.points[:,mesh.tets[collect(TET_FACES[f]),e]]
    for spec in periodic
        minus = filter(f -> f.boundary_id == spec.minus_boundary_id,topology.boundary_faces)
        plus = filter(f -> f.boundary_id == spec.plus_boundary_id,topology.boundary_faces)
        length(minus)==length(plus) || throw(ArgumentError("Periodic face counts differ"))
        shift = collect(spec.plus_to_minus_shift)
        for f in minus
            X = facepoints(f.elem,f.local_face)
            candidates = filter(plus) do g
                Y = facepoints(g.elem,g.local_face) .+ shift
                all(i -> minimum(norm(X[:,i]-Y[:,j]) for j in 1:3)<1e-10,1:3)
            end
            length(candidates)==1 || throw(ArgumentError("Periodic face match is not unique"))
            g = only(candidates)
            for key in ((f.elem,f.local_face),(g.elem,g.local_face))
                key in paired && throw(ArgumentError("Periodic face paired twice"))
                push!(paired,key)
            end
            neighbors[(f.elem,f.local_face)] = (g.elem,g.local_face,Tuple(-shift))
            neighbors[(g.elem,g.local_face)] = (f.elem,f.local_face,Tuple(shift))
        end
    end
    return neighbors,boundary_tags
end

function assemble_mixed_dg(mesh,p,neighbors,boundary_tags;
                          p_H=p+1,nodal_set=:equispaced,materials,backend=SerialBackend())
    ref = build_mixed_reference_tet(p; p_H, nodal_set)
    mappings = build_reference_mappings(mesh)
    ne = size(mesh.tets,2)
    validate_maxwell_materials(materials,ne)
    facepoints(e,f) = mesh.points[:,mesh.tets[collect(TET_FACES[f]),e]]
    ME,MH = cholesky(Symmetric(ref.M_E)),cholesky(Symmetric(ref.M_H))
    VE = NTuple{3,Matrix{Float64}}[]
    VH = NTuple{3,Matrix{Float64}}[]
    faces = [MixedFace[] for _ in 1:ne]
    for e in 1:ne
        mapping = mappings.tet_mappings[e]
        S = ntuple(a -> sum(mapping.invJ[b,a]*ref.S_EH[b] for b in 1:3),3)
        push!(VE,ntuple(a -> (ME \ S[a])/materials.epsilon[e],3))
        # Curl block transposition reverses its component signs; using the same
        # scalar transpose here yields Hdot = -M_H^-1 C_EH' E.
        push!(VH,ntuple(a -> (MH \ S[a]')/materials.permeability[e],3))
        for f in 1:4
            neighbor,nface,shift = get(neighbors,(e,f),(0,0,(0.,0.,0.)))
            rq,sq,tq,wq = reference_face_quadrature(f,ref.face_quadrature)
            pts = hcat(rq,sq,tq)
            physical_points = Matrix{Float64}(undef,length(wq),3)
            for q in eachindex(wq)
                physical_points[q,:] .= map_to_physical(mesh.points,view(mesh.tets,:,e),rq[q],sq[q],tq[q])
            end
            E,H = ref.face_E[f],ref.face_H[f]
            vertices = facepoints(e,f)
            n = cross(vertices[:,2]-vertices[:,1],vertices[:,3]-vertices[:,1])
            area = norm(n)/2
            n ./= norm(n)
            dot(n,mesh.points[:,mesh.tets[f,e]]-vertices[:,1])>0 && (n .*= -1)
            Eplus,Hplus = zeros(size(E)),zeros(size(H))
            if neighbor != 0
                neighborpoints = similar(pts)
                for q in axes(pts,1)
                    x = collect(map_to_physical(mesh.points,view(mesh.tets,:,e),rq[q],sq[q],tq[q])) .+ collect(shift)
                    origin = mesh.points[:,mesh.tets[1,neighbor]]
                    neighborpoints[q,:] = mappings.tet_mappings[neighbor].invJ*(x-origin) .- 1
                end
                Eplus = mixed_basis(ref.ref_E,neighborpoints)
                Hplus = mixed_basis(ref.ref_H,neighborpoints)
            end
            scale = area/reference_face_area(f)/mapping.absdetJ
            LE = ref.lift_E[f] * (scale/materials.epsilon[e])
            LH = ref.lift_H[f] * (scale/materials.permeability[e])
            push!(faces[e],MixedFace(e,neighbor,f,nface,Tuple(n),E,H,Eplus,Hplus,LE,LH,
                get(boundary_tags,(e,f),0),physical_points))
        end
    end
    return MixedDGDiscretization(mesh,ref,mappings,materials,VE,VH,faces,backend)
end

mixed_components(U::MaxwellField) = (U.Ex,U.Ey,U.Ez,U.Hx,U.Hy,U.Hz)
mixed_components(R::MaxwellRHS) = (R.rhsEx,R.rhsEy,R.rhsEz,R.rhsHx,R.rhsHy,R.rhsHz)

function MaxwellField(dg::MixedDGDiscretization)
    ne = size(dg.mesh.tets,2)
    return MaxwellField(ntuple(c -> zeros(c<=3 ? dg.reference.ref_E.Np : dg.reference.ref_H.Np,ne),6)...)
end

function validate_mixed_field(U, dg::MixedDGDiscretization)
    ne = size(dg.mesh.tets,2)
    for (c,A) in enumerate(mixed_components(U))
        size(A)==(c<=3 ? dg.reference.ref_E.Np : dg.reference.ref_H.Np,ne) ||
            throw(DimensionMismatch("Component $c has incompatible mixed-order dimensions"))
    end
    return U
end

function interpolate_maxwell_field(dg::MixedDGDiscretization, electric, magnetic)
    U = MaxwellField(dg)
    for (ref,fun,components) in ((dg.reference.ref_E,electric,(U.Ex,U.Ey,U.Ez)),
                                (dg.reference.ref_H,magnetic,(U.Hx,U.Hy,U.Hz)))
        for e in axes(dg.mesh.tets,2), i in 1:ref.Np
            x = map_to_physical(dg.mesh.points,view(dg.mesh.tets,:,e),ref.r[i],ref.s[i],ref.t[i])
            value = fun(x...)
            for c in 1:3
                components[c][i,e] = value[c]
            end
        end
    end
    return U
end

struct MixedRHSWorkspace
    minus::Matrix{Float64}
    plus::Matrix{Float64}
    flux::Matrix{Float64}
end
MixedRHSWorkspace(dg::MixedDGDiscretization) = MixedRHSWorkspace(
    ntuple(_ -> zeros(length(dg.reference.face_quadrature.wq),6),3)...)

function mixed_element_rhs!(R,U,dg,e,ws,flux_kind,remote=nothing,boundary_electric=nothing,time=0.)
    fields,residuals = mixed_components(U),mixed_components(R)
    # Cyclic curl terms; H uses the scalar transpose matrices with the same
    # component curl signs, giving the negative transpose of the full curl.
    for c in 1:3
        j,k = mod1(c+1,3),mod1(c+2,3)
        mul!(view(residuals[c],:,e),dg.volume_E[e][j],view(fields[k+3],:,e))
        mul!(view(residuals[c],:,e),dg.volume_E[e][k],view(fields[j+3],:,e),-1.,1.)
        mul!(view(residuals[c+3],:,e),dg.volume_H[e][j],view(fields[k],:,e))
        mul!(view(residuals[c+3],:,e),dg.volume_H[e][k],view(fields[j],:,e),-1.,1.)
    end
    for face in dg.faces[e]
        for c in 1:6
            mul!(view(ws.minus,:,c),c<=3 ? face.E : face.H,view(fields[c],:,e))
            if remote !== nothing && haskey(remote,(e,face.local_face))
                copyto!(view(ws.plus,:,c),view(remote[(e,face.local_face)],:,c))
            elseif face.neighbor != 0
                mul!(view(ws.plus,:,c),c<=3 ? face.Eplus : face.Hplus,
                     view(fields[c],:,face.neighbor))
            end
        end
        if face.neighbor == 0
            # Eq. (21): homogeneous PEC has no magnetic-jump correction
            # in Ampere's equation and zero prescribed electric trace.
            if boundary_electric !== nothing
                for q in axes(ws.flux,1)
                    Etrace = boundary_electric(face.boundary_id,face.points[q,1],face.points[q,2],face.points[q,3],time)
                    fH = cross_n_components(face.normal,Etrace...)
                    for c in 1:3
                        ws.flux[q,c+3] = -fH[c]
                    end
                end
                for c in 4:6
                    mul!(view(residuals[c],:,e),face.lift_H,view(ws.flux,:,c),1.,1.)
                end
            end
            continue
        end
        alpha = flux_kind==MaxwellFlux_Central ? 0.5 : 1-poisson_bracket_alternating_theta(face.normal)
        for q in axes(ws.minus,1)
            h = ntuple(c -> ws.minus[q,c+3]-ws.plus[q,c+3],3)
            Ehat = ntuple(c -> alpha*ws.minus[q,c]+(1-alpha)*ws.plus[q,c],3)
            fe = cross_n_components(face.normal,h...)
            fh = cross_n_components(face.normal,Ehat...)
            for c in 1:3
                ws.flux[q,c] = -alpha*fe[c]
                ws.flux[q,c+3] = -fh[c]
            end
        end
        for c in 1:6
            mul!(view(residuals[c],:,e),c<=3 ? face.lift_E : face.lift_H,
                 view(ws.flux,:,c),1.,1.)
        end
    end
    return nothing
end

struct MixedTraceSend
    element::Int
    E::Matrix{Float64}
    H::Matrix{Float64}
end

struct MixedTraceExchange
    neighbors::Vector{Int}
    sends::Dict{Int,Vector{MixedTraceSend}}
    receives::Dict{Int,Vector{Tuple{Int,Int}}}
    send_buffers::Dict{Int,Matrix{Float64}}
    recv_buffers::Dict{Int,Matrix{Float64}}
    traces::Dict{Tuple{Int,Int},Matrix{Float64}}
    # Reused slots retain MPI buffers until Waitall completes each exchange.
    requests::MPI.MultiRequest
end

struct DistributedMixedDG{B<:AbstractBackend}
    dg::MixedDGDiscretization{B}
    owned::Vector{Int}
    global_ids::Vector{Int}
    exchange::MixedTraceExchange
    comm::MPI.Comm
end

"""Distribute mesh/face metadata from root; assemble mixed operators on each rank.

Root owns global mesh topology during setup, not global high-order operators.
Evolution uses neighbor-only nonblocking messages and no global state replication.
Periodic neighbor cells are included in the halo just like ordinary neighbors.
"""
function build_distributed_mixed_dg_from_root(mesh,parts,p::Int;
        comm::MPI.Comm=MPI.COMM_WORLD,root::Int=0,p_H::Int=p+1,
        nodal_set::Symbol=:equispaced,materials=nothing,periodic=PeriodicBoundarySpec[],
        backend::AbstractBackend=SerialBackend(),boundary_tag_name="boundary_id")
    rank,nranks = MPI.Comm_rank(comm),MPI.Comm_size(comm)
    root_configuration = MPI.bcast((p,p_H,nodal_set),comm;root)
    collective_io_error((p,p_H,nodal_set)==root_configuration ? nothing :
        "Mixed polynomial degrees/nodal set differ across ranks",comm,"Mixed configuration validation")
    packets = nothing
    setup_error = nothing
    if rank==root
        try
            length(parts)==size(mesh.tets,2) || throw(ArgumentError("Partition length mismatch"))
            all(r -> 0<=r<nranks,parts) || throw(ArgumentError("Invalid partition rank"))
            neighbors,tags = mixed_connectivity(mesh;periodic,boundary_tag_name)
            global_materials = materials===nothing ? MaxwellElementMaterials(ones(length(parts)),ones(length(parts))) : materials
            validate_maxwell_materials(global_materials,length(parts))
            packets = [mixed_rank_packet(mesh,neighbors,tags,global_materials,parts,r) for r in 0:nranks-1]
        catch err
            setup_error = sprint(showerror,err)
        end
    end
    setup_error = MPI.bcast(setup_error,comm;root)
    setup_error===nothing || error("Mixed DG setup failed: $setup_error")
    packet = if rank==root
        for r in 0:nranks-1
            r==root || MPI.send(packets[r+1],comm;dest=r,tag=29001)
        end
        packets[root+1]
    else
        MPI.recv(comm;source=root,tag=29001)
    end
    lm,owned,gids,local_neighbors,tags,local_materials,send_descriptors,receives = packet
    dg = nothing
    setup_error = nothing
    try
        dg = assemble_mixed_dg(lm,p,local_neighbors,tags;p_H,nodal_set,materials=local_materials,backend)
    catch err
        setup_error = sprint(showerror,err)
    end
    collective_io_error(setup_error,comm,"Mixed local operator assembly")
    sends = Dict(r=>[MixedTraceSend(e,dg.faces[receiver][f].Eplus,dg.faces[receiver][f].Hplus)
                    for (e,receiver,f) in descriptors] for (r,descriptors) in send_descriptors)
    for e in length(owned)+1:length(gids)
        empty!(dg.faces[e])
    end
    neighbors = sort!(collect(union(keys(sends),keys(receives))))
    nq = length(dg.reference.face_quadrature.wq)
    sb = Dict(r=>zeros(nq,6length(sends[r])) for r in neighbors)
    rb = Dict(r=>zeros(nq,6length(receives[r])) for r in neighbors)
    traces = Dict(key=>zeros(nq,6) for r in neighbors for key in receives[r])
    exchange = MixedTraceExchange(neighbors,sends,receives,sb,rb,traces,MPI.MultiRequest(2length(neighbors)))
    return DistributedMixedDG(dg,owned,gids,exchange,comm)
end

function mixed_rank_packet(gm,neighbors,boundary_tags,materials,parts,rank)
    owned_global = findall(==(rank),parts)
    isempty(owned_global) && throw(ArgumentError("Every rank must own at least one element"))
    ghost = sort!(unique([h for ((e,f),(h,hf,shift)) in neighbors
                         if parts[e]==rank && parts[h]!=rank]))
    gids = vcat(owned_global,ghost)
    local_ids = Dict(g=>i for (i,g) in enumerate(gids))
    vertices = sort!(unique(vec(gm.tets[:,gids])))
    vertex_ids = Dict(g=>i for (i,g) in enumerate(vertices))
    tets = map(g->vertex_ids[g],gm.tets[:,gids])
    # Face connectivity is supplied separately, including periodic halos.
    lm = RawVTUMesh(gm.points[:,vertices],tets,zeros(Int,3,0),gids,Int[],Dict{String,Any}())
    local_neighbors = Dict((local_ids[g],f)=>(local_ids[h],hf,shift)
        for ((g,f),(h,hf,shift)) in neighbors if haskey(local_ids,g) && haskey(local_ids,h))
    tags = Dict((local_ids[g],f)=>tag for ((g,f),tag) in boundary_tags if haskey(local_ids,g))
    local_materials = MaxwellElementMaterials(materials.epsilon[gids],materials.permeability[gids])
    sends = Dict{Int,Vector{NTuple{3,Int}}}()
    receives = Dict{Int,Vector{Tuple{Int,Int}}}()
    # The globally sorted (receiver element, receiver face) traversal gives
    # identical send/receive ordering, including repeated periodic neighbors.
    for (g,f) in sort!(collect(keys(neighbors)))
        h,hf,shift = neighbors[(g,f)]
        parts[g]==parts[h] && continue
        if parts[g]==rank
            push!(get!(receives,parts[h],Tuple{Int,Int}[]),(local_ids[g],f))
        elseif parts[h]==rank
            push!(get!(sends,parts[g],NTuple{3,Int}[]),(local_ids[h],local_ids[g],f))
        end
    end
    return lm,collect(1:length(owned_global)),gids,local_neighbors,tags,local_materials,sends,receives
end

MaxwellField(ddg::DistributedMixedDG) = MaxwellField(ddg.dg)
interpolate_maxwell_field(ddg::DistributedMixedDG,E,H) = interpolate_maxwell_field(ddg.dg,E,H)

function begin_mixed_trace_exchange!(U,ddg::DistributedMixedDG)
    ex = ddg.exchange
    fields = mixed_components(U)
    for (j,r) in enumerate(ex.neighbors)
        sb = ex.send_buffers[r]
        for (i,f) in enumerate(ex.sends[r]), c in 1:6
            mul!(view(sb,:,6(i-1)+c),c<=3 ? f.E : f.H,view(fields[c],:,f.element))
        end
        MPI.Irecv!(ex.recv_buffers[r],ddg.comm,ex.requests[2j-1];source=r,tag=29002)
        MPI.Isend(sb,ddg.comm,ex.requests[2j];dest=r,tag=29002)
    end
    return nothing
end

function finish_mixed_trace_exchange!(ddg::DistributedMixedDG)
    ex = ddg.exchange
    isempty(ex.requests) || MPI.Waitall(ex.requests)
    for r in ex.neighbors, (i,key) in enumerate(ex.receives[r])
        copyto!(ex.traces[key],view(ex.recv_buffers[r],:,6(i-1)+1:6i))
    end
    return nothing
end

function make_maxwell_rhs_function(ddg::DistributedMixedDG;
                                  flux_kind::MaxwellFluxKind=MaxwellFlux_Central,
                                  boundary_electric=nothing)
    flux_kind in (MaxwellFlux_Central,MaxwellFlux_Alternating) ||
        throw(ArgumentError("Mixed-order bracket supports centered and alternating fluxes"))
    work = [MixedRHSWorkspace(ddg.dg) for _ in 1:Threads.nthreads()]
    # Element-interior work whose faces are all locally owned overlaps exchange.
    remote_elements = Set(first(k) for k in keys(ddg.exchange.traces))
    local_elements = filter(e -> !(e in remote_elements),ddg.owned)
    boundary_elements = filter(e -> e in remote_elements,ddg.owned)
    function evaluate!(R,U,elements,time)
        if ddg.dg.backend isa ThreadedBackend
            Threads.@threads :static for i in eachindex(elements)
                mixed_element_rhs!(R,U,ddg.dg,elements[i],work[Threads.threadid()],flux_kind,ddg.exchange.traces,boundary_electric,time)
            end
        else
            for e in elements
                mixed_element_rhs!(R,U,ddg.dg,e,work[1],flux_kind,ddg.exchange.traces,boundary_electric,time)
            end
        end
    end
    return function (R::MaxwellRHS,U::MaxwellField,time::Float64=0.)
        validate_mixed_field(U,ddg.dg)
        validate_mixed_field(R,ddg.dg)
        fill_maxwell_rhs!(R,0.)
        begin_mixed_trace_exchange!(U,ddg)
        evaluate!(R,U,local_elements,time)
        finish_mixed_trace_exchange!(ddg)
        evaluate!(R,U,boundary_elements,time)
        return R
    end
end

function make_maxwell_rhs_function(dg::MixedDGDiscretization;
                                  flux_kind::MaxwellFluxKind=MaxwellFlux_Central,
                                  boundary_electric=nothing)
    flux_kind in (MaxwellFlux_Central,MaxwellFlux_Alternating) ||
        throw(ArgumentError("Mixed-order bracket supports centered and alternating fluxes"))
    work = [MixedRHSWorkspace(dg) for _ in 1:Threads.nthreads()]
    return function (R::MaxwellRHS,U::MaxwellField,time::Float64=0.)
        validate_mixed_field(U,dg)
        validate_mixed_field(R,dg)
        if dg.backend isa ThreadedBackend
            Threads.@threads :static for e in axes(dg.mesh.tets,2)
                mixed_element_rhs!(R,U,dg,e,work[Threads.threadid()],flux_kind,nothing,boundary_electric,time)
            end
        else
            for e in axes(dg.mesh.tets,2)
                mixed_element_rhs!(R,U,dg,e,work[1],flux_kind,nothing,boundary_electric,time)
            end
        end
        return R
    end
end

function mixed_esprk_scheme(p::Int; first_partition::Symbol=:H)
    1<=p<=4 || throw(ArgumentError("ESPRK p+2 requires 1 <= p <= 4; available temporal orders end at 6"))
    return explicit_partitioned_symplectic_rk_scheme(p+2; first_partition)
end

"""Positive tensor Gauss/Duffy quadrature on the reference tetrahedron.

Under (u,v,w) ↦ (u,(1-u)v,(1-u)(1-v)w), a total-degree d
polynomial times the Jacobian has degrees at most (d+2,d+1,d).
Using ceil((d+3)/2) Gauss points in each coordinate is therefore exact.
"""
function mixed_tetrahedron_quadrature(degree::Int)
    degree >= 0 || throw(ArgumentError("Quadrature degree must be nonnegative"))
    n = cld(degree + 3, 2)
    x, wg = gauss_legendre_1d(n)
    u = (x .+ 1) ./ 2
    wg ./= 2
    points = Matrix{Float64}(undef, n^3, 3)
    weights = Vector{Float64}(undef, n^3)
    q = 0
    for i in 1:n, j in 1:n, k in 1:n
        q += 1
        points[q,1] = 2u[i] - 1
        points[q,2] = 2(1-u[i])*u[j] - 1
        points[q,3] = 2(1-u[i])*(1-u[j])*u[k] - 1
        weights[q] = 8wg[i]*wg[j]*wg[k]*(1-u[i])^2*(1-u[j])
    end
    return points, weights
end

function mixed_basis(ref::ReferenceTet, points::Matrix{Float64})
    return eval_nodal_basis_at_points(points[:,1], points[:,2], points[:,3],
                                     ref.basis, ref.invV)
end

function mixed_basis_gradient(ref::ReferenceTet, points::Matrix{Float64})
    grads = orthonormal_grad_vandermonde_tet(points[:,1], points[:,2],
                                            points[:,3], ref.basis)
    return map(g -> g * ref.invV, grads)
end

function build_mixed_reference_tet(p::Int; p_H::Int = p+1,
                                  nodal_set::Symbol = :equispaced)
    p >= 1 || throw(ArgumentError("Mixed-order Maxwell requires p_E >= 1"))
    p_H == p+1 || throw(ArgumentError("Mixed-order Maxwell requires p_H = p_E + 1"))
    E = build_reference_tet(p; nodal_set)
    H = build_reference_tet(p_H; nodal_set)
    # Mixed derivatives: degree 2p; mixed face products: degree 2p+1.
    # H mass/energy needs 2p+2. Use that degree in the volume rule.
    points, weights = mixed_tetrahedron_quadrature(2p+2)
    VE, VH = mixed_basis(E, points), mixed_basis(H, points)
    GE, GH = mixed_basis_gradient(E, points), mixed_basis_gradient(H, points)
    W = Diagonal(weights)
    ME, MH = VE' * W * VE, VH' * W * VH
    SEH = ntuple(a -> VE' * W * GH[a], 3)
    SHE = ntuple(a -> VH' * W * GE[a], 3)
    tq = triangle_quadrature_gauss(2p+1)
    fq = ntuple(f -> reference_face_quadrature(f, tq), 4)
    FE = ntuple(f -> mixed_basis(E, hcat(fq[f][1:3]...)), 4)
    FH = ntuple(f -> mixed_basis(H, hcat(fq[f][1:3]...)), 4)
    fw = ntuple(f -> fq[f][4], 4)
    LE = ntuple(f -> cholesky(Symmetric(ME)) \ (FE[f]' * Diagonal(fw[f])), 4)
    LH = ntuple(f -> cholesky(Symmetric(MH)) \ (FH[f]' * Diagonal(fw[f])), 4)
    return MixedReferenceTet(E,H,points,weights,2p+2,tq,VE,VH,ME,MH,
                             SEH,SHE,FE,FH,fw,LE,LH)
end
