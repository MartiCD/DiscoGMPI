# Included inside MaxwellInvariantAudit. This serial oracle includes ALL directed
# interior and periodic faces. No distributed BigFloat storage is passed to MPI.
function directed_faces(ddg,periodic)
    MPI.Comm_size(ddg.comm)==1 || throw(ArgumentError("Precision oracle is serial only."))
    faces=NamedTuple[]
    for f in ddg.dg.flux_faces.interior
        t=f.trace; perm=t.plus_to_minus_perm
        push!(faces,(e=t.minus_elem,f=t.minus_face,nodes=t.minus_nodes,
            other=t.plus_elem,onodes=t.plus_nodes[perm],normal=f.normal,area=f.area))
        push!(faces,(e=t.plus_elem,f=t.plus_face,nodes=t.plus_nodes,
            other=t.minus_elem,onodes=t.minus_nodes[invperm(perm)],
            normal=map(-,f.normal),area=f.area))
    end
    for f in periodic.faces
        push!(faces,(e=f.local_elem,f=f.local_face,nodes=f.local_nodes,
            other=f.partner_local_elem,onodes=f.partner_local_nodes[f.partner_to_local_perm],
            normal=f.normal,area=f.area))
    end
    length(faces)==4size(ddg.dg.mesh.tets,2) || error("Oracle requires a fully periodic box.")
    faces
end

"""Exact simplex moments and nodal polynomial algebra at the working precision.

Variant C supports equispaced p=1:4 on affine straight tetrahedra. Nodes are
reconstructed as integer barycentric tuples / p, not promoted rounded nodes.
Polynomial integrals use factorial moments instead of Float64 cubature tables.
Geometry uses the exact binary values of input vertices; topology/permutations
remain those of production. Thus C does not validate the mesh parser/topology.
"""
function high_reference(ref,::Type{T}) where T
    p=ref.N
    1<=p<=4 || throw(ArgumentError("High-precision reconstruction supports p=1:4."))
    lf=hcat(-(ref.r+ref.s+ref.t.+1)/2,(ref.r.+1)/2,(ref.s.+1)/2,(ref.t.+1)/2)
    ints=round.(Int,p*lf)
    maximum(abs,lf-ints/p)<1e-12 || throw(ArgumentError("Variant C requires equispaced nodes."))
    lam=T.(ints)/T(p)
    exps=[(a,b,c,p-a-b-c) for a in 0:p for b in 0:p-a for c in 0:p-a-b]
    n=length(exps); V=zeros(T,n,n)
    dV=[zeros(T,n,n) for _ in 1:3]
    gl=T.([-1 1 0 0; -1 0 1 0; -1 0 0 1])/2
    for i in 1:n,(j,a) in enumerate(exps)
        V[i,j]=prod(lam[i,v]^a[v] for v in 1:4)
        for d in 1:3,v in 1:4
            a[v]==0 && continue
            dV[d][i,j]+=gl[d,v]*a[v]*prod(lam[i,k]^(a[k]-(k==v)) for k in 1:4)
        end
    end
    invV=inv(V)
    fact(n)=T(factorial(big(n)))
    G=[T(8)*prod(fact(a[k]+b[k]) for k in 1:4)/fact(2p+3) for a in exps,b in exps]
    M=transpose(invV)*G*invV
    # Face matrices divided by reference face area; no sqrt(3) cancellation.
    face=[transpose(invV)*[a[f]+b[f]==0 ?
        T(2)*prod(fact(a[k]+b[k]) for k in 1:4 if k!=f)/fact(2p+2) :
        zero(T) for a in exps,b in exps]*invV for f in 1:4]
    (;M,D=map(x->x*invV,dV),face,lam,gl)
end

function precision_operators(ddg,periodic,ws,::Type{T};rebuild=false) where T
    dg=ddg.dg; ref=dg.ref; ne=size(dg.mesh.tets,2)
    hr=rebuild ? high_reference(ref,T) : nothing
    M=rebuild ? hr.M : T.(ref.M)
    blocks=NamedTuple[]; lifts=NamedTuple[]
    for e in 1:ne
        if rebuild
            x=T.(dg.mesh.points[:,dg.mesh.tets[:,e]])
            J=hcat(x[:,2]-x[:,1],x[:,3]-x[:,1],x[:,4]-x[:,1])/2
            g=inv(J); jac=abs(det(J))
            D=ntuple(d->sum(g[k,d]*hr.D[k] for k in 1:3),3)
            S=map(d->M*d,D)
            # Recover exact P1 hat vertex values from the conforming workspace.
            vids=[argmax(hr.lam[:,v]) for v in 1:4]
            G=transpose(g)*hr.gl*T.(ws.phi[e][vids,:])
        else
            op=dg.physops.elements[e]
            D=map(d->T.(d),(op.Dx,op.Dy,op.Dz))
            S=map(d->T.(d),physical_weak_derivative_matrices(op))
            jac=T(dg.mappings.tet_mappings[e].absdetJ)
            G=T.(ws.grad[e])
        end
        w=jac*(M*ones(T,ref.Np))
        Q=hcat([transpose(d)*w for d in D]...)
        W=[-w*transpose(G[d,:]) for d in 1:3]
        push!(blocks,(D=D,S=S,jac=jac,Q=Q,W=W))
    end
    for f in directed_faces(ddg,periodic)
        if rebuild
            verts=DG.REF_TET_FACE_VERTEX_IDS[f.f]
            x=T.(dg.mesh.points[:,dg.mesh.tets[:,f.e]])
            raw=cross(x[:,verts[2]]-x[:,verts[1]],x[:,verts[3]]-x[:,verts[1]])
            dot(raw,x[:,f.f]-x[:,verts[1]])>0 && (raw=-raw)
            area=norm(raw)/2; normal=raw/norm(raw)
            L=(M \ hr.face[f.f][:,f.nodes])*(area/blocks[f.e].jac)
            theta=sum(raw)>0 ? one(T) : sum(raw)<0 ? zero(T) : T(1)/2
        else
            normal=T.(collect(f.normal))
            # Preserve the precomputed lifting coefficient errors in variant B.
            L=T.(dg.fops.face_lift[f.f]) *
                (T(f.area)/T(DG.reference_face_area(f.f))/blocks[f.e].jac)
            theta=T(DG.poisson_bracket_alternating_theta(f.normal))
        end
        push!(lifts,(;f...,normal=normal,L=L,theta=theta))
    end
    (;M,blocks,faces=lifts)
end

"""Same split weak/strong Poisson-bracket operator, in generic arithmetic.
Compare its Float64 action with maxwell_rhs_periodic! before interpreting B/C.
Restricted to epsilon=mu=1; not a generic replacement for production.
"""
function precision_rhs(op,U,flux)
    T=eltype(U); R=zeros(T,size(U))
    curl(S,F) = hcat(S[2]*F[:,3]-S[3]*F[:,2],
        S[3]*F[:,1]-S[1]*F[:,3],S[1]*F[:,2]-S[2]*F[:,1])
    for (e,b) in enumerate(op.blocks)
        R[:,e,1:3]=op.M \ curl(b.S,U[:,e,4:6])
        R[:,e,4:6]=op.M \ curl(map(transpose,b.S),U[:,e,1:3])
    end
    for f in op.faces
        a=U[f.nodes,f.e,:]; b=U[f.onodes,f.other,:]
        theta=flux==MaxwellFlux_Central ? T(1)/2 : f.theta
        alpha=1-theta
        jumpH=a[:,4:6]-b[:,4:6]; Ehat=alpha*a[:,1:3]+theta*b[:,1:3]
        fe=zeros(T,length(f.nodes),3);fh=similar(fe)
        for i in axes(fe,1)
            fe[i,:]=-alpha*cross(f.normal,jumpH[i,:])
            fh[i,:]=-cross(f.normal,Ehat[i,:])
        end
        R[:,f.e,1:3] .+= f.L*fe
        R[:,f.e,4:6] .+= f.L*fh
    end
    R
end

function precision_residuals(op,R)
    T=eltype(R); nt=size(op.blocks[1].W[1],2)
    q=zeros(T,2); qs=zeros(T,2); w=zeros(T,2,nt); s=zeros(T,2,nt)
    for (e,b) in enumerate(op.blocks), f in 1:2,d in 1:3
        v=R[:,e,3(f-1)+d]
        q[f]+=dot(b.Q[:,d],v); qs[f]+=sum(abs,b.Q[:,d].*v)
        for j in 1:nt
            w[f,j]+=dot(b.W[d][:,j],v)
            s[f,j]+=sum(abs,b.W[d][:,j].*v)
        end
    end
    normalize(x,a)=a==0 ? (x==0 ? zero(T) : T(Inf)) : abs(x)/a
    (;charge=q,charge_scale=qs,charge_normalized=normalize.(q,qs),
        weak=w,weak_scale=s,weak_normalized=normalize.(w,s))
end

function coefficient_array(U)
    cat(components(U)...;dims=3)
end
