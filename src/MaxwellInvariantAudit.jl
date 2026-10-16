"""Opt-in, allocation-tolerant laboratory for periodic Maxwell invariants.

Production RHS and time stepping are not modified. All volume sums use owned
elements; a unique global-element/face key selects each skeleton face once.
"""
module MaxwellInvariantAudit
using ..DiscoGMPI
using LinearAlgebra, MPI
const DG = DiscoGMPI

components(U::MaxwellField) = (U.Ex,U.Ey,U.Ez,U.Hx,U.Hy,U.Hz)
components(R::MaxwellRHS) = (R.rhsEx,R.rhsEy,R.rhsEz,R.rhsHx,R.rhsHy,R.rhsHz)
asfield(R::MaxwellRHS) = MaxwellField(components(R)...)

"""Cached periodic continuous P1 vertex hats (a subspace of P_(p+1)).

The constant is their sum when all hats are selected. Tests are NOT independent
element polynomials. Periodic vertex classes use normalized coordinates at 1e-10
resolution; only axis-aligned, translationally periodic affine boxes are supported.
"""
struct AuditWorkspace
    phi::Vector{Matrix{Float64}}
    grad::Vector{Matrix{Float64}}
    linear::Vector{Array{Float64,3}}
    interpolation::Matrix{Float64}
    weights::Vector{Float64}
    keys::Vector{NTuple{3,Int}}
    total_tests::Int
end

function periodic_vertex_key(x,box)
    ntuple(d -> mod(round(Int, 1e10*(x[d]-box.lower[d]) /
        (box.upper[d]-box.lower[d])),10^10),3)
end

function AuditWorkspace(ddg, global_mesh; max_tests=32)
    ref=ddg.dg.ref
    ref.N >= 1 || throw(ArgumentError("Audit face/P1 tests require p >= 1."))
    max_tests >= 1 || throw(ArgumentError("max_tests must be positive."))
    box=distributed_axis_aligned_box(ddg)
    keys=MPI.bcast(MPI.Comm_rank(ddg.comm)==0 ?
        sort!(unique([periodic_vertex_key(global_mesh.points[:,i],box)
            for i in axes(global_mesh.points,2)])) : nothing,ddg.comm;root=0)
    total=length(keys); keys=keys[1:min(max_tests,total)]
    lookup=Dict(k=>i for (i,k) in enumerate(keys))
    lambda=hcat(-(ref.r+ref.s+ref.t.+1)/2,(ref.r.+1)/2,
        (ref.s.+1)/2,(ref.t.+1)/2)
    gr=[-0.5 0.5 0 0; -0.5 0 0.5 0; -0.5 0 0 0.5]
    phi=Matrix{Float64}[]; grad=Matrix{Float64}[]; linear=Array{Float64,3}[]
    for (e,mapping) in enumerate(ddg.dg.mappings.tet_mappings)
        P=zeros(ref.Np,length(keys)); G=zeros(3,length(keys))
        for v in 1:4
            node=ddg.dg.mesh.tets[v,e]
            key=periodic_vertex_key(ddg.dg.mesh.points[:,node],box)
            j=get(lookup,key,0); j==0 && continue
            P[:,j] .+= lambda[:,v]
            G[:,j] .+= transpose(mapping.invJ)*gr[:,v]
        end
        # Coefficients of the signed broken charge and W(phi), without material.
        L=zeros(ref.Np,3,1+length(keys)); op=ddg.dg.physops.elements[e]
        w=mapping.absdetJ*(ref.M*ones(ref.Np))
        for (d,D) in enumerate((op.Dx,op.Dy,op.Dz))
            L[:,d,1]=transpose(D)*w
            for j in eachindex(keys)
                L[:,d,1+j]=-G[d,j]*w
            end
        end
        push!(phi,P);push!(grad,G);push!(linear,L)
    end
    q,w,_=get_JaskowiecSukumar_cubature(max(2,2ref.N+4))
    AuditWorkspace(phi,grad,linear,reference_interpolation_matrix(ref,q),w,keys,total)
end

"""Frozen Float64 linear functionals, evaluated at T precision.

BigFloat results use decimal-string gather/sum/broadcast, NOT MPI arithmetic on
BigFloat memory. This separates diagnostic arithmetic from coefficient errors.
"""
function linear_values(U,ddg,materials,ws; T=Float64)
    v=zeros(T,2,1+length(ws.keys)); C=components(U)
    for e in ddg.distributed_mesh.partition.owned, f in 1:2
        a=T(f==1 ? materials.epsilon[e] : materials.permeability[e])
        for j in axes(v,2),d in 1:3,i in axes(C[1],1)
            v[f,j] += a*T(ws.linear[e][i,d,j])*T(C[3(f-1)+d][i,e])
        end
    end
    if T==Float64
        return MPI.Allreduce(v,+,ddg.comm)
    end
    strings=MPI.gather(string.(v),ddg.comm;root=0)
    result = MPI.Comm_rank(ddg.comm)==0 ?
        string.(sum(parse.(T,s) for s in strings)) : nothing
    return parse.(T,MPI.bcast(result,ddg.comm;root=0))
end

"""Broken divergence, normal jumps and independent weak reconstruction.

Maximum is sampled at volume cubature points, not a certified L-infinity norm.
Jump norms are unscaled L2(skeleton) norms, interior and periodic separately.
"""
function constraint_diagnostics(U,ddg,periodic,materials,ws)
    exchange_maxwell_ghost_traces!(U,ddg)
    exchange_distributed_periodic_traces!(U,ddg,periodic)
    C=components(U); dg=ddg.dg; nt=length(ws.keys)
    Q=zeros(2); L=zeros(2); mx=zeros(2); W=zeros(2,nt); V=zeros(2,nt)
    F=zeros(2,nt); jump=zeros(2,2); face_charge=zeros(2,2); facecount=zeros(Int,2)
    owned=Set(ddg.distributed_mesh.partition.owned)
    ids=ddg.distributed_mesh.elements.global_ids
    for e in owned
        op=dg.physops.elements[e]; J=dg.mappings.tet_mappings[e].absdetJ
        mass=J*dg.ref.M; w=mass*ones(dg.ref.Np)
        for f in 1:2
            a=f==1 ? materials.epsilon[e] : materials.permeability[e]
            c=ntuple(d->C[3(f-1)+d][:,e],3)
            div=a*(op.Dx*c[1]+op.Dy*c[2]+op.Dz*c[3])
            Q[f]+=dot(w,div); L[f]+=dot(div,mass*div)
            mx[f]=max(mx[f],maximum(abs,ws.interpolation*div))
            V[f,:] .+= transpose(ws.phi[e])*(mass*div)
            for j in 1:nt,d in 1:3
                W[f,j]-=a*ws.grad[e][d,j]*dot(w,c[d])
            end
        end
    end
    function addface(e,face,nodes,n,area,minus,plus,aplus,kind)
        fm=dg.fops.face_mass[face]*(area/DG.reference_face_area(face))
        facecount[kind]+=1
        for f in 1:2
            a=f==1 ? materials.epsilon[e] : materials.permeability[e]
            jn=sum(n[d]*(a*minus[3(f-1)+d]-aplus[f]*plus[3(f-1)+d]) for d in 1:3)
            jump[f,kind]+=dot(jn,fm*jn)
            face_charge[f,kind]+=sum(fm*jn)
            F[f,:] .+= transpose(ws.phi[e][nodes,:])*(fm*jn)
        end
    end
    for ff in dg.flux_faces.interior
        t=ff.trace; e=t.minus_elem; other=t.plus_elem
        # Select owner of the smaller global element, irrespective of trace orientation.
        chosen=ids[e]<ids[other] ? e : other
        chosen in owned || continue
        a=map(c->c[t.minus_nodes,e],C)
        b=map(c->c[t.plus_nodes[t.plus_to_minus_perm],other],C)
        addface(e,t.minus_face,t.minus_nodes,ff.normal,ff.area,a,b,
            (materials.epsilon[other],materials.permeability[other]),1)
    end
    for f in periodic.faces
        (ids[f.local_elem],f.local_face)<(f.partner_global_elem,f.partner_face) || continue
        a=map(c->c[f.local_nodes,f.local_elem],C)
        b=DG.distributed_periodic_plus_trace(U,periodic,f,MPI.Comm_rank(ddg.comm))
        addface(f.local_elem,f.local_face,f.local_nodes,f.normal,f.area,a,Tuple(b),
            (f.partner_epsilon,f.partner_permeability),2)
    end
    Q=MPI.Allreduce(Q,+,ddg.comm); L=MPI.Allreduce(L,+,ddg.comm)
    mx=MPI.Allreduce(mx,max,ddg.comm); W=MPI.Allreduce(W,+,ddg.comm)
    V=MPI.Allreduce(V,+,ddg.comm); F=MPI.Allreduce(F,+,ddg.comm)
    jump=MPI.Allreduce(jump,+,ddg.comm); facecount=MPI.Allreduce(facecount,+,ddg.comm)
    face_charge=MPI.Allreduce(face_charge,+,ddg.comm)
    (;charge=Q,div_l2=sqrt.(max.(L,0)),div_max_sampled=mx,weak=W,
        weak_volume=V,weak_face=F,weak_reconstruction_error=W-V+F,
        jump_l2=sqrt.(max.(jump,0)),face_charge,facecount)
end

"""Quadrature momentum used by production quadrature_diagnostics.csv and its
semidiscrete directional derivative. No norm-of-momentum substitution is made.
"""
function momentum_diagnostics(U,R,ddg,materials,ws)
    c=components(U);r=components(R); sums=zeros(7)
    for e in ddg.distributed_mesh.partition.owned
        a=materials.epsilon[e];b=materials.permeability[e]
        v=map(x->ws.interpolation*x[:,e],c)
        rv=map(x->ws.interpolation*x[:,e],r)
        J=ddg.dg.mappings.tet_mappings[e].absdetJ
        for q in eachindex(ws.weights)
            E=[v[d][q] for d in 1:3]; H=[v[3+d][q] for d in 1:3]
            RE=[rv[d][q] for d in 1:3]; RH=[rv[3+d][q] for d in 1:3]
            w=J*ws.weights[q]
            sums[1:3] .+= w*a*b*cross(E,H)
            sums[4:6] .+= w*a*b*(cross(RE,H)+cross(E,RH))
            sums[7]+=w*(a*dot(E,E)+b*dot(H,H))/2
        end
    end
    sums=MPI.Allreduce(sums,+,ddg.comm)
    (;momentum=sums[1:3],momentum_rate=sums[4:6],energy=sums[7])
end

"""Instrument the same partition updates as production without modifying them.
With observer=nothing, delegate directly to the original integrator.
Observer receives (stage, partition, actual_coefficient, U, R, phase).
The RHS is evaluated once for each nonzero partition weight, not once per step.
"""
function audited_step!(U,work,scheme,dt,rhs!;observer=nothing)
    observer===nothing && return partitioned_symplectic_rk_step!(U,work,scheme,dt,rhs!)
    DG.validate_partitioned_symplectic_rk_scheme(scheme)
    first=DG.normalize_maxwell_partition(scheme.first_partition)
    second=DG.complementary_maxwell_partition(first)
    for i in 1:DG.num_stages(scheme)
        for (part,weight) in ((first,scheme.first_weights[i]),(second,scheme.second_weights[i]))
            weight==0 && continue
            rhs!(work.rhs,U)
            observer(i,part,dt*weight,U,work.rhs,:before)
            DG.add_scaled_partition_rhs_to_field!(U,work.rhs,part,dt*weight)
            observer(i,part,dt*weight,U,work.rhs,:after)
        end
    end
    U
end

function update_counts(scheme)
    a=count(!iszero,scheme.first_weights);b=count(!iszero,scheme.second_weights)
    scheme.first_partition==:H ? (electric=b,magnetic=a,rhs=a+b) : (electric=a,magnetic=b,rhs=a+b)
end

function build_case(mesh,p,comm;flux=MaxwellFlux_Central,max_tests=32)
    rank=MPI.Comm_rank(comm); nranks=MPI.Comm_size(comm)
    part=rank==0 ? balanced_spatial_partition(mesh,nranks) : nothing
    ddg=build_distributed_dg_from_root(rank==0 ? mesh : nothing,part,p;comm)
    materials=homogeneous_maxwell_materials(size(ddg.dg.mesh.tets,2))
    box=distributed_axis_aligned_box(ddg)
    periodic=build_distributed_periodic_maxwell_exchange(ddg,periodic_box_specs(box);materials)
    registry=MaxwellBoundaryRegistry(Dict(i=>MaxwellBC_None for i in 1:6))
    formulation=PoissonBracketFormulation(flux)
    # Match the periodic production driver exactly (scalar-material overload).
    # The separate element-material flux wrapper currently omits flux_kind.
    rhs! = make_distributed_periodic_maxwell_rhs_function(ddg,periodic,registry,formulation;ε=1.0,μ=1.0)
    wave=plane_wave_for_periodic_box(box;wave_number=2pi)
    ws=AuditWorkspace(ddg,mesh;max_tests)
    (;ddg,materials,periodic,registry,formulation,rhs!,wave,ws)
end

function initial_field(case;electric=:interpolate,magnetic=:curl_potential)
    (;ddg,materials,periodic,wave)=case
    E,H=exact_periodic_wave_functions(0.0;epsilon=1.0,mu=1.0,wave)
    U=interpolate_maxwell_field(ddg.dg.mesh,ddg.dg.ref,E,H)
    if magnetic==:curl_potential
        initialize_magnetic_from_potential!(U,ddg,materials,
            (x,y,z,t)->periodic_magnetic_vector_potential(x,y,z,t;mu=1.0,wave);periodic)
    elseif magnetic!=:interpolate
        throw(ArgumentError("Unknown magnetic initialization."))
    end
    if electric==:curl_potential
        initialize_electric_from_potential!(U,ddg,materials,
            (x,y,z,t)->periodic_electric_vector_potential(x,y,z,t;epsilon=1.0,wave);periodic)
    elseif electric!=:interpolate
        throw(ArgumentError("Unknown electric initialization."))
    end
    U
end

include("MaxwellInvariantPrecision.jl")
end
