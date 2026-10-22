"""Reusable quadrature and scratch storage for mixed-field diagnostics.

The default degree 4(p_H)+2 resolves polynomial energy-density products.
Supplying points_weights and basis allows sharing an error-sampling cache.
"""
function mixed_diagnostic_workspace(dg::MixedDGDiscretization;
        degree=4dg.reference.ref_H.N+2, points_weights=nothing, basis=nothing)
    points,w = points_weights===nothing ? mixed_tetrahedron_quadrature(degree) : points_weights
    V = basis===nothing ? (mixed_basis(dg.reference.ref_E,points),mixed_basis(dg.reference.ref_H,points)) : basis
    G = (mixed_basis_gradient(dg.reference.ref_E,points),mixed_basis_gradient(dg.reference.ref_H,points))
    return (; points,w,V,G,values=zeros(length(w),6),
        gradients=zeros(length(w),3,6),reference_gradients=zeros(length(w),3),
        sums=zeros(24),global_sums=zeros(24))
end

# A workspace belongs to one discretization/quadrature and must not be shared
# between concurrent calls. Returned sums alias it when explicitly supplied.
function mixed_diagnostics(U,dg::MixedDGDiscretization;
        elements=axes(U.Ex,2),time=0.,exact_electric=nothing,exact_magnetic=nothing,
        degree=4dg.reference.ref_H.N+2,workspace=nothing)
    validate_mixed_field(U,dg)
    ws = workspace===nothing ? mixed_diagnostic_workspace(dg;degree) : workspace
    points,w,V,G = ws.points,ws.w,ws.V,ws.G
    values,gradients,sums = ws.values,ws.gradients,ws.sums
    fill!(sums,0.)
    fields = mixed_components(U)
    for e in elements
        jac = dg.mappings.tet_mappings[e]
        for c in 1:6
            space = c<=3 ? 1 : 2
            coefficients = view(fields[c],:,e)
            mul!(view(values,:,c),V[space],coefficients)
            for b in 1:3
                mul!(view(ws.reference_gradients,:,b),G[space][b],coefficients)
            end
            for a in 1:3, q in eachindex(w)
                gradients[q,a,c] =
                    jac.invJ[1,a]*ws.reference_gradients[q,1] +
                    jac.invJ[2,a]*ws.reference_gradients[q,2] +
                    jac.invJ[3,a]*ws.reference_gradients[q,3]
            end
        end
        eps,mu = dg.materials.epsilon[e],dg.materials.permeability[e]
        for q in eachindex(w)
            weight = w[q]*jac.absdetJ
            x = map_to_physical(dg.mesh.points,view(dg.mesh.tets,:,e),points[q,1],points[q,2],points[q,3])
            E,H = ntuple(c->values[q,c],3),ntuple(c->values[q,c+3],3)
            sums[1] += weight*eps*sum(abs2,E)/2
            sums[2] += weight*mu*sum(abs2,H)/2
            if exact_electric!==nothing
                ex = exact_electric(x...,time)
                sums[3] += weight*sum(abs2(E[c]-ex[c]) for c in 1:3)
                sums[5] += weight*sum(abs2,ex)
            end
            if exact_magnetic!==nothing
                ex = exact_magnetic(x...,time)
                sums[4] += weight*sum(abs2(H[c]-ex[c]) for c in 1:3)
                sums[6] += weight*sum(abs2,ex)
            end
            divE = eps*sum(gradients[q,c,c] for c in 1:3)
            divH = mu*sum(gradients[q,c,c+3] for c in 1:3)
            sums[7] += weight*divE
            sums[8] += weight*divH
            sums[9] += weight*divE^2
            sums[10] += weight*divH^2
            momentum = eps*mu .* cross3(E,H)
            angular = cross3(x,momentum)
            for c in 1:3
                sums[10+c] += weight*eps*E[c]
                sums[13+c] += weight*mu*H[c]
                sums[16+c] += weight*momentum[c]
                sums[19+c] += weight*angular[c]
            end
            curlE = ntuple(c->gradients[q,mod1(c+1,3),mod1(c+2,3)]-gradients[q,mod1(c+2,3),mod1(c+1,3)],3)
            curlH = ntuple(c->gradients[q,mod1(c+1,3),mod1(c+2,3)+3]-gradients[q,mod1(c+2,3),mod1(c+1,3)+3],3)
            sums[23] += weight*(eps*dot3(E,curlE)+mu*dot3(H,curlH))/2
            sums[24] += weight
        end
    end
    return sums
end

"""Wrap a mixed RHS with a quadrature-projected volume source.

`source(x,y,z,t,E,H,epsilon,mu)` returns `(electric=(...),magnetic=(...))`
in field/time units. A prescribed current uses electric=-J/epsilon. Sources
may couple both fields; integrate such systems with ordinary explicit RK,
since the two shear flows assumed by ESPRK no longer apply in general.
"""
function with_mixed_source(rhs!,discretization,source;degree=nothing)
    dg = discretization isa DistributedMixedDG ? discretization.dg : discretization
    elements = discretization isa DistributedMixedDG ? discretization.owned : collect(axes(dg.mesh.tets,2))
    degree = degree===nothing ? 3dg.reference.ref_H.N+2 : degree
    points,weights = mixed_tetrahedron_quadrature(degree)
    VE,VH = mixed_basis(dg.reference.ref_E,points),mixed_basis(dg.reference.ref_H,points)
    LE = cholesky(Symmetric(dg.reference.M_E)) \ (VE'*Diagonal(weights))
    LH = cholesky(Symmetric(dg.reference.M_H)) \ (VH'*Diagonal(weights))
    physical = [zeros(length(weights),3) for _ in axes(dg.mesh.tets,2)]
    for e in elements,q in eachindex(weights)
        physical[e][q,:] .= map_to_physical(dg.mesh.points,view(dg.mesh.tets,:,e),points[q,:]...)
    end
    work = [(zeros(length(weights),6),zeros(length(weights),6)) for _ in 1:Threads.nthreads()]
    function add_element!(R,U,e,time,ws)
        values,sources = ws
        fields,residuals = mixed_components(U),mixed_components(R)
        for c in 1:6
            mul!(view(values,:,c),c<=3 ? VE : VH,view(fields[c],:,e))
        end
        for q in eachindex(weights)
            E,H = ntuple(c->values[q,c],3),ntuple(c->values[q,c+3],3)
            s = source(physical[e][q,1],physical[e][q,2],physical[e][q,3],time,E,H,
                dg.materials.epsilon[e],dg.materials.permeability[e])
            for c in 1:3
                sources[q,c],sources[q,c+3] = s.electric[c],s.magnetic[c]
            end
        end
        for c in 1:6
            mul!(view(residuals[c],:,e),c<=3 ? LE : LH,view(sources,:,c),1.,1.)
        end
    end
    return function (R,U,time::Float64=0.)
        rhs!(R,U,time)
        if dg.backend isa ThreadedBackend
            Threads.@threads :static for i in eachindex(elements)
                add_element!(R,U,elements[i],time,work[Threads.threadid()])
            end
        else
            for e in elements
                add_element!(R,U,e,time,work[1])
            end
        end
        return R
    end
end

"""Mixed nonlinear PML: evaluate both fields and sigma at shared volume points."""
function with_mixed_nonlinear_pml(rhs!,dg,sigma; a=0.5,regularization=1e-12,degree=nothing)
    0<a<1 || throw(ArgumentError("PML a must lie in (0,1)"))
    regularization>0 || throw(ArgumentError("PML regularization must be positive"))
    source = (x,y,z,t,E,H,eps,mu) -> nonlinear_pml_source(E,H,sigma(x,y,z);
        ε=eps,μ=mu,a,regularization)
    return with_mixed_source(rhs!,dg,source;degree)
end

"""Ordinary explicit RK with correct stage times for driven/dissipative systems."""
function mixed_rk_step!(U,work::MaxwellRKWorkspace,scheme::ExplicitRKScheme,
                        time::Float64,dt::Float64,rhs!)
    length(work.K)==num_stages(scheme) || throw(DimensionMismatch("RK stage count mismatch"))
    copy_maxwell_field!(work.U0,U)
    for i in 1:num_stages(scheme)
        build_rk_stage_field!(work.Ustage,work.U0,work.K,scheme,i,dt)
        rhs!(work.K[i],work.Ustage,time+scheme.c[i]*dt)
    end
    copy_maxwell_field!(U,work.U0)
    for i in 1:num_stages(scheme)
        add_scaled_rhs_to_field!(U,work.K[i],dt*scheme.b[i])
    end
    return U
end

function mixed_diagnostics(U,ddg::DistributedMixedDG;workspace=nothing,kwargs...)
    local_sums = mixed_diagnostics(U,ddg.dg;elements=ddg.owned,workspace,kwargs...)
    result = workspace===nothing ? similar(local_sums) : workspace.global_sums
    MPI.Allreduce!(local_sums,result,+,ddg.comm)
    return result
end

function mixed_diagnostic_values(sums)
    return (electric_energy=sums[1],magnetic_energy=sums[2],energy=sums[1]+sums[2],
        error_E=sqrt(sums[3]),error_H=sqrt(sums[4]),
        exact_norm_E=sqrt(sums[5]),exact_norm_H=sqrt(sums[6]),
        charge_E=sums[7],charge_H=sums[8],divergence_E=sqrt(sums[9]),divergence_H=sqrt(sums[10]),
        integrated_D=Tuple(sums[11:13]),integrated_B=Tuple(sums[14:16]),
        momentum=Tuple(sums[17:19]),angular_momentum=Tuple(sums[20:22]),chirality=sums[23],volume=sums[24])
end

"""L2-project analytic initial data independently into the electric and magnetic spaces."""
function project_maxwell_field(dg::MixedDGDiscretization,electric,magnetic;
        degree=2dg.reference.ref_H.N+6,elements=axes(dg.mesh.tets,2))
    U = MaxwellField(dg)
    points,weights = mixed_tetrahedron_quadrature(degree)
    VE,VH = mixed_basis(dg.reference.ref_E,points),mixed_basis(dg.reference.ref_H,points)
    LE = cholesky(Symmetric(dg.reference.M_E)) \ (VE'*Diagonal(weights))
    LH = cholesky(Symmetric(dg.reference.M_H)) \ (VH'*Diagonal(weights))
    values = zeros(length(weights),6)
    for e in elements
        for q in eachindex(weights)
            x = map_to_physical(dg.mesh.points,view(dg.mesh.tets,:,e),points[q,:]...)
            E,H = electric(x...),magnetic(x...)
            for c in 1:3
                values[q,c],values[q,c+3] = E[c],H[c]
            end
        end
        for c in 1:6
            mul!(view(mixed_components(U)[c],:,e),c<=3 ? LE : LH,view(values,:,c))
        end
    end
    return U
end
project_maxwell_field(ddg::DistributedMixedDG,E,H;kwargs...) =
    project_maxwell_field(ddg.dg,E,H;elements=ddg.owned,kwargs...)

"""Geometry/degree timestep estimate; CFL is a tunable safety factor.

h is the minimum altitude. The default linear degree factor is p_H; the
compatibility degree_rule=:quadratic uses (p_H+1)^2. This is a heuristic,
not a certified spectral bound for every ESPRK composition.
"""
function mixed_cfl_dt(dg::MixedDGDiscretization;CFL=0.1,elements=axes(dg.mesh.tets,2),degree_rule=:linear)
    isfinite(CFL) && CFL>0 || throw(ArgumentError("CFL must be finite and positive"))
    degree_rule in (:quadratic, :linear) || throw(ArgumentError("Unknown CFL degree rule"))
    pH = dg.reference.ref_H.N
    factor = degree_rule == :quadratic ? (pH+1)^2 : pH
    dt = Inf
    for e in elements
        volume = dg.mappings.tet_mappings[e].absdetJ*REF_TET_VOLUME
        area = maximum(1:4) do f
            X = dg.mesh.points[:,dg.mesh.tets[collect(TET_FACES[f]),e]]
            norm(cross(X[:,2]-X[:,1],X[:,3]-X[:,1]))/2
        end
        c = inv(sqrt(dg.materials.epsilon[e]*dg.materials.permeability[e]))
        dt = min(dt,CFL*(3volume/area)/(factor*c))
    end
    return dt
end
mixed_cfl_dt(ddg::DistributedMixedDG;kwargs...) = MPI.Allreduce(
    mixed_cfl_dt(ddg.dg;elements=ddg.owned,kwargs...),min,ddg.comm)

function initialize_electric_from_potential!(U,dg::MixedDGDiscretization,A;
        time=0.,elements=axes(U.Ex,2))
    ref = dg.reference.ref_E
    return initialize_electric_from_potential!(U,dg.mesh,ref,dg.mappings,dg.materials,A,
        CurlInitializationWorkspace(ref);time,elements)
end
function initialize_magnetic_from_potential!(U,dg::MixedDGDiscretization,A;
        time=0.,elements=axes(U.Hx,2))
    ref = dg.reference.ref_H
    return initialize_magnetic_from_potential!(U,dg.mesh,ref,dg.mappings,dg.materials,A,
        CurlInitializationWorkspace(ref);time,elements)
end
initialize_electric_from_potential!(U,ddg::DistributedMixedDG,A;kwargs...) =
    initialize_electric_from_potential!(U,ddg.dg,A;elements=ddg.owned,kwargs...)
initialize_magnetic_from_potential!(U,ddg::DistributedMixedDG,A;kwargs...) =
    initialize_magnetic_from_potential!(U,ddg.dg,A;elements=ddg.owned,kwargs...)

function mixed_metadata(ddg::DistributedMixedDG)
    ref = ddg.dg.reference
    return Dict{String,Any}("format"=>"DiscoGMPI mixed-order Maxwell", "version"=>1,
        "p_E"=>ref.ref_E.N,"p_H"=>ref.ref_H.N,"Np_E"=>ref.ref_E.Np,"Np_H"=>ref.ref_H.Np,
        "mpi_ranks"=>MPI.Comm_size(ddg.comm))
end

function mixed_checkpoint_signature(ddg)
    # Detect changed geometry, reference nodes, materials, and local ownership.
    io = IOBuffer()
    serialize(io,(ddg.global_ids,ddg.owned,ddg.dg.mesh.points,ddg.dg.mesh.tets,
        ddg.dg.materials,ddg.dg.reference.ref_E.r,ddg.dg.reference.ref_E.s,
        ddg.dg.reference.ref_E.t,ddg.dg.reference.ref_H.r,ddg.dg.reference.ref_H.s,
        ddg.dg.reference.ref_H.t,
        [[(f.element,f.neighbor,f.local_face,f.neighbor_face,f.boundary_id) for f in faces]
            for faces in ddg.dg.faces]))
    return bytes2hex(sha256(take!(io)))
end

function write_distributed_checkpoint(dir::AbstractString,U::MaxwellField,ddg::DistributedMixedDG;
        step::Int,time::Float64,dt::Float64,metadata::AbstractDict=Dict{String,Any}())
    rank = MPI.Comm_rank(ddg.comm)
    err = nothing
    checksum = ""
    try
        validate_mixed_field(U,ddg.dg)
        mkpath(dir)
        payload = (schema=mixed_metadata(ddg),signature=mixed_checkpoint_signature(ddg),
            global_ids=ddg.global_ids[ddg.owned],
            components=map(c->c[:,ddg.owned],mixed_components(U)),step=step,time=time,dt=dt,
            metadata=Dict{String,Any}(metadata))
        atomic_serialize(joinpath(dir,"mixed_rank$(rank_file_label(rank)).bin"),payload)
        checksum = bytes2hex(sha256(read(joinpath(dir,"mixed_rank$(rank_file_label(rank)).bin"))))
    catch e
        err = sprint(showerror,e)
    end
    collective_io_error(err,ddg.comm,"Mixed checkpoint write")
    checksums = MPI.gather(checksum,ddg.comm;root=0)
    err = nothing
    if rank==0
        try
            manifest = merge(mixed_metadata(ddg),Dict("step"=>step,"time"=>time,"dt"=>dt,"checksums"=>checksums))
            atomic_toml(joinpath(dir,"mixed_manifest.toml"),manifest)
        catch e
            err = sprint(showerror,e)
        end
    end
    collective_io_error(err,ddg.comm,"Mixed checkpoint manifest")
    return joinpath(dir,"mixed_manifest.toml")
end

function load_distributed_checkpoint(dir::AbstractString,ddg::DistributedMixedDG)
    result = nothing
    err = nothing
    try
        manifest = TOML.parsefile(joinpath(dir,"mixed_manifest.toml"))
        for (k,v) in mixed_metadata(ddg)
            get(manifest,k,nothing)==v || error("Incompatible mixed checkpoint: $k")
        end
        rank = MPI.Comm_rank(ddg.comm)
        path = joinpath(dir,"mixed_rank$(rank_file_label(rank)).bin")
        bytes2hex(sha256(read(path)))==manifest["checksums"][rank+1] || error("Checkpoint checksum mismatch")
        payload = deserialize(path)
        payload.schema==mixed_metadata(ddg) || error("Checkpoint piece schema mismatch")
        payload.signature==mixed_checkpoint_signature(ddg) || error("Checkpoint mesh, basis, material, or partition mismatch")
        payload.global_ids==ddg.global_ids[ddg.owned] || error("Checkpoint ownership mismatch")
        for key in (:step,:time,:dt)
            getproperty(payload,key)==manifest[string(key)] || error("Checkpoint manifest/piece mismatch: $key")
        end
        U = MaxwellField(ddg)
        length(payload.components)==6 || error("Checkpoint must contain six components")
        for (dest,src) in zip(mixed_components(U),payload.components)
            size(src)==(size(dest,1),length(ddg.owned)) || error("Checkpoint component shape mismatch")
            dest[:,ddg.owned] = src
        end
        result = (U,DistributedCheckpointState(payload.step,payload.time,payload.dt,payload.metadata))
    catch e
        err = sprint(showerror,e)
    end
    collective_io_error(err,ddg.comm,"Mixed checkpoint load")
    return result
end

"""Write both fields on an equispaced p_H visualization grid.

Both source spaces are evaluated independently. Output interpolation never
modifies the state or enters the evolution equations.
"""
function write_parallel_maxwell_fields(path::String,ddg::DistributedMixedDG,U::MaxwellField;
        time::Float64,exact_electric=nothing,exact_magnetic=nothing)
    dg = ddg.dg
    ref = build_reference_tet(dg.reference.ref_H.N)
    ids = vtk_lagrange_tetra_node_ids(ref)
    points_ref = hcat(ref.r[ids],ref.s[ids],ref.t[ids])
    VE,VH = mixed_basis(dg.reference.ref_E,points_ref),mixed_basis(dg.reference.ref_H,points_ref)
    n = ref.Np
    coords = zeros(3,n*length(ddg.owned))
    fields = zeros(6,n*length(ddg.owned))
    exactE = exact_electric===nothing ? nothing : zeros(3,n*length(ddg.owned))
    exactH = exact_magnetic===nothing ? nothing : zeros(3,n*length(ddg.owned))
    cells = MeshCell[]
    for (i,e) in enumerate(ddg.owned)
        range = (i-1)*n+1:i*n
        push!(cells,MeshCell(VTKCellTypes.VTK_LAGRANGE_TETRAHEDRON,collect(range)))
        for q in 1:n
            coords[:,range[q]] .= map_to_physical(dg.mesh.points,view(dg.mesh.tets,:,e),points_ref[q,:]...)
            exactE===nothing || (exactE[:,range[q]] .= exact_electric(coords[:,range[q]]...))
            exactH===nothing || (exactH[:,range[q]] .= exact_magnetic(coords[:,range[q]]...))
        end
        for c in 1:6
            fields[c,range] = (c<=3 ? VE : VH)*view(mixed_components(U)[c],:,e)
        end
    end
    rank = MPI.Comm_rank(ddg.comm)
    return pvtk_grid(path,coords,cells;part=rank+1,nparts=MPI.Comm_size(ddg.comm),ismain=rank==0) do vtk
        vtk["ElectricField",VTKPointData()] = fields[1:3,:]
        vtk["MagneticField",VTKPointData()] = fields[4:6,:]
        if exactE!==nothing
            vtk["ExactElectricField",VTKPointData()] = exactE
            vtk["ElectricFieldError",VTKPointData()] = fields[1:3,:]-exactE
        end
        if exactH!==nothing
            vtk["ExactMagneticField",VTKPointData()] = exactH
            vtk["MagneticFieldError",VTKPointData()] = fields[4:6,:]-exactH
        end
        vtk["p_E",VTKCellData()] = fill(dg.reference.ref_E.N,length(cells))
        vtk["p_H",VTKCellData()] = fill(dg.reference.ref_H.N,length(cells))
        vtk["GlobalElementId",VTKCellData()] = ddg.global_ids[ddg.owned]
        vtk["TimeValue",VTKFieldData()] = time
    end
end

function write_maxwell_integration_points(dir::String,ddg::DistributedMixedDG,U::MaxwellField,
        degree::Int;time=0.)
    dg = ddg.dg
    points,w = mixed_tetrahedron_quadrature(degree)
    VE,VH = mixed_basis(dg.reference.ref_E,points),mixed_basis(dg.reference.ref_H,points)
    rank = MPI.Comm_rank(ddg.comm)
    path = joinpath(dir,"mixed_integration_points_rank$(rank_file_label(rank)).csv")
    err = nothing
    try
        mkpath(dir)
        atomic_text(path) do io
            println(io,"global_element,p_E,p_H,time,x,y,z,physical_weight,Ex,Ey,Ez,Hx,Hy,Hz")
            for e in ddg.owned
                values = [(c<=3 ? VE : VH)*view(mixed_components(U)[c],:,e) for c in 1:6]
                for q in eachindex(w)
                    x = map_to_physical(dg.mesh.points,view(dg.mesh.tets,:,e),points[q,:]...)
                    row = (ddg.global_ids[e],dg.reference.ref_E.N,dg.reference.ref_H.N,time,
                        x...,w[q]*dg.mappings.tet_mappings[e].absdetJ,(v[q] for v in values)...)
                    println(io,join(row,','))
                end
            end
        end
    catch e
        err = sprint(showerror,e)
    end
    collective_io_error(err,ddg.comm,"Mixed integration-point output")
    return path
end
