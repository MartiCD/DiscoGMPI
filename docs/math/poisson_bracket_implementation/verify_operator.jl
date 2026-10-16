using DiscoGMPI, LinearAlgebra, MPI, Printf
const D = DiscoGMPI
MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const rows = NamedTuple[]

function report(name, value; tolerance=NaN, enforce=false)
    pass = !isfinite(tolerance) || value <= tolerance
    push!(rows, (;name, value=Float64(value), tolerance, pass))
    rank == 0 && @printf("%-58s %.8e tol=%.3e %s\n", name, value, tolerance,
        isfinite(tolerance) ? (pass ? "PASS" : "FAIL") : "MEASURED")
    enforce && !pass && error("Audit check failed: $name")
end
crossmat(n) = [0.0 -n[3] n[2]; n[3] 0.0 -n[1]; -n[2] n[1] 0.0]
curlmat(x,y,z) = [zero(x) -z y; z zero(x) -x; -y x zero(x)]
components(U::MaxwellField) = (U.Ex,U.Ey,U.Ez,U.Hx,U.Hy,U.Hz)
components(U::MaxwellRHS) = (U.rhsEx,U.rhsEy,U.rhsEz,U.rhsHx,U.rhsHy,U.rhsHz)
flat(U) = vcat(vec.(components(U))...)
field(np,ne) = MaxwellField((zeros(np,ne) for _ in 1:6)...)
idx(c,e,np,ne) = ((c-1)*np*ne+(e-1)*np+1):((c-1)*np*ne+e*np)

function reference_checks()
    normals=((1,1,1)./sqrt(3),(-1.,0.,0.),(0.,-1.,0.),(0.,0.,-1.))
    for p in 1:4, family in (:equispaced,:warp_blend)
        ref=build_reference_tet(p;nodal_set=family)
        fo=build_reference_face_operators(ref;nodal_set=family)
        label="ref_p$(p)_$(family)"
        tol=2000eps(Float64)*max(1.0,cond(ref.V)^2)
        report(label*"_constant_derivative", maximum(norm(A*ones(ref.Np),Inf)
            for A in (ref.Dr,ref.Ds,ref.Dt));tolerance=tol,enforce=true)
        ds=(ref.Dr,ref.Ds,ref.Dt); xs=(ref.r,ref.s,ref.t)
        report(label*"_coordinate_derivative",maximum(norm(ds[i]*xs[j].-(i==j),Inf)
            for i in 1:3,j in 1:3);tolerance=tol,enforce=true)
        report(label*"_mass_symmetry",opnorm(ref.M-ref.M',Inf);tolerance=tol,enforce=true)
        report(label*"_mass_min_eigenvalue",eigmin(Symmetric(ref.M)))
        @assert isposdef(Symmetric(ref.M))
        boundaries=[zeros(ref.Np,ref.Np) for _ in 1:3]
        for f in 1:4
            ids=fo.face_nodes[f]
            for a in 1:3
                boundaries[a][ids,ids] .+= normals[f][a]*fo.face_mass[f]
            end
        end
        ss=(ref.Sr,ref.Ss,ref.St)
        report(label*"_ibp",maximum(opnorm(ss[a]+ss[a]'-boundaries[a],Inf) for a in 1:3);
            tolerance=tol,enforce=true)
        q=D.triangle_quadrature_gauss(2p)
        face_error=0.0
        for f in 1:4
            rq,sq,tq,wq=D.reference_face_quadrature(f,q)
            L=D.orthonormal_vandermonde_tet(rq,sq,tq,ref.basis)*ref.invV
            B=L'*Diagonal(wq)*L
            face_error=max(face_error,opnorm(B[fo.face_nodes[f],fo.face_nodes[f]]-fo.face_mass[f],Inf))
        end
        report(label*"_face_quadrature",face_error;tolerance=tol,enforce=true)
        if p==1 && family==:equispaced
            failed=false
            try
                D.build_face_mass_quadrature(ref,1,q)
            catch err
                err isa DimensionMismatch || rethrow()
                failed=true
            end
            report("legacy_face_quadrature_DimensionMismatch",Float64(failed))
        end
        for sign in (-1.,1.)
            J=[sign*.7 .15 .1; 0. .9 .2; .1 -.1 .4]
            dx=D.physical_derivative_matrices(ref,inv(J))
            xyz=J*hcat(ref.r,ref.s,ref.t)' .+ [1.,2.,-1.]
            report(label*"_affine_sign$(Int(sign))",maximum(norm(dx[i]*xyz[j,:].-(i==j),Inf)
                for i in 1:3,j in 1:3);tolerance=5tol,enforce=true)
        end
    end
end

function quadrature_checks()
    for order in (2,4,8)
        xyz,w,_=get_JaskowiecSukumar_cubature(order)
        uvw=(xyz.+1)./2
        err=0.0
        for total in 0:order, a in 0:total, b in 0:(total-a)
            c=total-a-b
            exact=8*D.unit_simplex_monomial_integral(a,b,c)
            err=max(err,abs(dot(w,uvw[:,1].^a .* uvw[:,2].^b .* uvw[:,3].^c)-exact))
        end
        report("JS_order$(order)_unit_monomials",err;tolerance=2e-13,enforce=true)
    end
end

# Independent coefficient assembly: no production flux or lifting routines.
function independent_matrices(dg, periodic, flux, bc)
    np=dg.ref.Np; ne=size(dg.mesh.tets,2); m=np*ne
    M=zeros(6m,6m); A=zeros(6m,6m); Db=zeros(m,3m)
    integral=zeros(m)
    for e in 1:ne
        op=dg.physops.elements[e]; j=op.mass_scale
        ss=D.physical_weak_derivative_matrices(op)
        K=j*curlmat(ss...)
        ie=vcat((idx(c,e,np,ne) for c in 1:3)...)
        ih=vcat((idx(c,e,np,ne) for c in 4:6)...)
        A[ie,ih] .+= K; A[ih,ie] .-= K'
        for c in 1:6
            ii=idx(c,e,np,ne); M[ii,ii]=j*dg.ref.M
        end
        ee=idx(1,e,np,ne)
        for c in 1:3
            Db[ee,idx(c,e,np,ne)]=(op.Dx,op.Dy,op.Dz)[c]
        end
        integral[ee]=j*(dg.ref.M*ones(np))
    end
    function face!(e,face,nodes,partner,pnodes,n,area)
        R=zeros(length(nodes),np); Rp=similar(R);fill!(Rp,0)
        for i in eachindex(nodes); R[i,nodes[i]]=1; Rp[i,pnodes[i]]=1; end
        W=area/D.reference_face_area(face)*dg.fops.face_mass[face]
        theta=flux==MaxwellFlux_Central ? 0.5 : (sum(n)>0 ? 1. : sum(n)<0 ? 0. : 0.5)
        alpha=1-theta; N=crossmat(n)
        ie=vcat((idx(c,e,np,ne) for c in 1:3)...); ih=ie.+3m
        pe=vcat((idx(c,partner,np,ne) for c in 1:3)...); ph=pe.+3m
        self=kron(N,R'*W*R); other=kron(N,R'*W*Rp)
        A[ie,ih] .-= alpha*self; A[ie,ph] .+= alpha*other
        A[ih,ie] .-= alpha*self; A[ih,pe] .-= theta*other
    end
    for ff in vcat(dg.flux_faces.interior, periodic===nothing ? [] : periodic.faces)
        tr=ff.trace; perm=tr.plus_to_minus_perm
        face!(tr.minus_elem,tr.minus_face,tr.minus_nodes,tr.plus_elem,tr.plus_nodes[perm],ff.normal,ff.area)
        face!(tr.plus_elem,tr.plus_face,tr.plus_nodes,tr.minus_elem,tr.minus_nodes[invperm(perm)],.-ff.normal,ff.area)
    end
    if periodic===nothing && bc != MaxwellBC_PEC
        for ff in dg.flux_faces.boundary
            tr=ff.trace; ids=tr.nodes; B=zeros(np,np)
            B[ids,ids]=ff.area/D.reference_face_area(tr.face)*dg.fops.face_mass[tr.face]
            ie=vcat((idx(c,tr.elem,np,ne) for c in 1:3)...); ih=ie.+3m
            N=crossmat(ff.normal); T=Matrix{Float64}(I,3,3)-collect(ff.normal)*collect(ff.normal)'
            if bc==MaxwellBC_PMC
                A[ie,ih] .-= kron(N,B); A[ih,ie] .-= kron(N,B)
            elseif bc==MaxwellBC_Absorbing
                A[ie,ih] .-= 0.5kron(N,B); A[ih,ie] .-= 0.5kron(N,B)
                A[ie,ie] .-= 0.5kron(T,B); A[ih,ih] .-= 0.5kron(T,B)
            end
        end
    end
    return M,A,Db,integral
end

function serial_matrix(dg,pf,reg,form;materials=nothing)
    np=dg.ref.Np;ne=size(dg.mesh.tets,2);U=field(np,ne);rhs=D.similar_maxwell_rhs(U)
    n=6np*ne; L=zeros(n,n)
    for j in 1:n
        c=div(j-1,np*ne)+1; i=mod(j-1,np*ne)+1
        components(U)[c][i]=1
        if pf===nothing
            materials===nothing ? maxwell_rhs!(rhs,U,dg,reg,form) : maxwell_rhs!(rhs,U,dg,reg,form,materials)
        else
            maxwell_rhs_periodic!(rhs,U,dg,pf,reg,form)
        end
        L[:,j]=flat(rhs); components(U)[c][i]=0
    end
    return L
end

function operator_checks(mesh; prefix="", cases=("periodic_centered","periodic_alternating","pec_centered","pec_alternating","pmc_centered","absorbing_centered"))
    for p in (1,2), name in cases
        dg=DGDiscretization(mesh,p)
        flux=occursin("alternating",name) ? MaxwellFlux_Alternating : MaxwellFlux_Central
        bc=startswith(name,"pmc") ? MaxwellBC_PMC : startswith(name,"absorbing") ? MaxwellBC_Absorbing : MaxwellBC_PEC
        pf=startswith(name,"periodic") ? build_periodic_flux_faces(mesh,dg.ref,dg.flux_faces,default_unit_box_periodic_specs()) : nothing
        reg=MaxwellBoundaryRegistry(Dict(i=>(pf===nothing ? bc : MaxwellBC_None) for i in 1:6))
        M,A,Db,w=independent_matrices(dg,pf,flux,bc)
        L=serial_matrix(dg,pf,reg,PoissonBracketFormulation(flux)); actual=M*L
        scale=max(1.,opnorm(A,Inf)); label=prefix*"p$(p)_$(name)"
        report(label*"_reconstruction",opnorm(actual-A,Inf)/scale;tolerance=3e-12,enforce=true)
        if bc!=MaxwellBC_Absorbing
            report(label*"_skew",opnorm(actual+actual',Inf)/scale;tolerance=3e-12,enforce=true)
        else
            report(label*"_symmetric_max_eigenvalue",eigmax(Symmetric((actual+actual')/2));tolerance=3e-12,enforce=true)
        end
        m=3dg.ref.Np*size(mesh.tets,2)
        LH=L[m+1:end,1:m]; charge=w'*Db
        report(label*"_charge_null_scaled",norm(charge*LH,Inf)/max(1.,norm(charge,1)*opnorm(LH,Inf)))
        report(label*"_broken_divcurl_inf",opnorm(Db*LH,Inf))
        u=sin.(0.31.*(1:size(L,1)))
        report(label*"_energy_production",dot(u,actual*u))
    end
end

function heterogeneous_check(mesh)
    dg=DGDiscretization(mesh,1);ne=size(mesh.tets,2);np=dg.ref.Np
    mats=MaxwellElementMaterials([1.0+0.2*e for e in 1:ne],[0.7+0.1*e for e in 1:ne])
    reg=MaxwellBoundaryRegistry(Dict(i=>MaxwellBC_PEC for i in 1:6))
    M,A,_,_=independent_matrices(dg,nothing,MaxwellFlux_Central,MaxwellBC_PEC)
    for c in 1:6,e in 1:ne
        ii=idx(c,e,np,ne);M[ii,ii].*=c<=3 ? mats.epsilon[e] : mats.permeability[e]
    end
    L=serial_matrix(dg,nothing,reg,PoissonBracketFormulation();materials=mats)
    report("heterogeneous_pec_reconstruction",opnorm(M*L-A,Inf)/max(1.,opnorm(A,Inf));tolerance=3e-12,enforce=true)
end

function mpi_checks(mesh)
    nprocs=MPI.Comm_size(comm)
    part=balanced_spatial_partition(mesh,nprocs)
    for p in (1,2,3), family in (:equispaced,:warp_blend)
        ddg=build_distributed_dg(mesh,part,p;comm,nodal_set=family)
        dg=DGDiscretization(mesh,p;nodal_set=family)
        pf=build_periodic_flux_faces(mesh,dg.ref,dg.flux_faces,default_unit_box_periodic_specs())
        mats=homogeneous_maxwell_materials(size(ddg.dg.mesh.tets,2))
        pe=build_distributed_periodic_maxwell_exchange(ddg;materials=mats)
        U=field(dg.ref.Np,size(mesh.tets,2))
        for (c,a) in enumerate(components(U)), e in axes(a,2), i in axes(a,1)
            a[i,e]=sin(0.2*c+0.7*e+0.3*i)+cos(0.1*c*e*i)
        end
        localU=localize_maxwell_field(U,ddg);rhs=D.similar_maxwell_rhs(U);r=D.similar_maxwell_rhs(localU)
        reg=MaxwellBoundaryRegistry(Dict(i=>MaxwellBC_None for i in 1:6))
        for flux in (MaxwellFlux_Central,MaxwellFlux_Alternating)
            form=PoissonBracketFormulation(flux)
            maxwell_rhs_periodic!(rhs,U,dg,pf,reg,form)
            for a in components(localU); a[:,ddg.distributed_mesh.partition.ghosts].=1e6; end
            maxwell_rhs_periodic!(r,localU,ddg,pe,reg,form)
            err=0.
            for e in ddg.distributed_mesh.partition.owned, (a,b) in zip(components(r),components(rhs))
                g=ddg.distributed_mesh.elements.global_ids[e]
                err=max(err,norm(a[:,e]-b[:,g],Inf))
            end
            report("mpi$(nprocs)_p$(p)_$(family)_$(flux)_serial",MPI.Allreduce(err,max,comm);
                tolerance=2e-10,enforce=true)
        end
        ca=D.similar_maxwell_rhs(localU);ma=D.similar_maxwell_rhs(localU)
        maxwell_rhs_periodic!(ca,localU,ddg,pe,reg,PoissonBracketFormulation(MaxwellFlux_Central))
        maxwell_rhs_periodic!(ma,localU,ddg,pe,reg,PoissonBracketFormulation(MaxwellFlux_Alternating),mats)
        report("mpi$(nprocs)_p$(p)_$(family)_material_alt_minus_center",norm(flat(ma)-flat(ca),Inf);
            tolerance=2e-10,enforce=true)
        maxwell_rhs_periodic!(r,localU,ddg,pe,reg,PoissonBracketFormulation(MaxwellFlux_Alternating))
        report("mpi$(nprocs)_p$(p)_$(family)_material_alt_minus_alt",norm(flat(ma)-flat(r),Inf))
    end
end

try
    # Reference checks above also use a skew affine map with either determinant sign.
    reference_checks(); quadrature_checks()
    mesh=structured_box_mesh(1,1,1;boundary_id=periodic_box_boundary_id)
    operator_checks(mesh)
    mesh2=structured_box_mesh(2,1,1;boundary_id=periodic_box_boundary_id)
    operator_checks(mesh2;prefix="nx2_",cases=("periodic_centered","periodic_alternating"))
    heterogeneous_check(mesh2)
    mpi_checks(mesh2)
    if rank==0 && !isempty(ARGS)
        open(ARGS[1],"w") do io
            println(io,"check,value,tolerance,pass")
            for r in rows; println(io,"$(r.name),$(r.value),$(r.tolerance),$(r.pass)"); end
        end
    end
finally
    MPI.Finalize()
end
