#!/usr/bin/env julia
using DiscoGMPI, MPI, LinearAlgebra, TOML, Dates, SHA
const Audit = DiscoGMPI.MaxwellInvariantAudit

function audit_options(args)
    cfg=Dict("cfls"=>"0.025,0.05,0.1","orders"=>"2","final-time"=>"0.02",
        "steps"=>"0","sample-every"=>"1","stage-every"=>"0","max-tests"=>"32",
        "magnetic-modes"=>"curl-potential,interpolate","electric-modes"=>"interpolate",
        "flux"=>"centered","meshes"=>"examples/meshes/periodic_box_structured_nx4_ny2_nz2.vtk",
        "nx-targets"=>"","output-dir"=>"output/invariant_audit/"*Dates.format(now(),"yyyymmdd_HHMMSSsss"))
    for a in args
        if a=="--help"
            println("Periodic invariant audit (Float64; opt-in instrumentation)\nOptions --name=value:")
            foreach(k->println("  --",k,"=",cfg[k]),sort(collect(keys(cfg))))
            println("--steps>0 fixes the step count instead of final time. --nx-targets overrides meshes.\n",
                "--stage-every=0 disables per-update diagnostics; positive N samples every Nth step.\n",
                "Every output directory must be new. Use --final-time=1000 explicitly for long runs.")
            return nothing
        end
        startswith(a,"--") && occursin('=',a) || error("Expected --name=value: $a")
        k,v=split(a[3:end],'=';limit=2);haskey(cfg,k) || error("Unknown option $k")
        cfg[k]=v
    end
    cfg
end

csvline(io,values)=println(io,join(string.(values),','))
function audit_provenance(cfg,comm)
    root=MPI.bcast(abspath(cfg["output-dir"]),comm;root=0)
    cfg["output-dir"]=root
    collective_root_action(comm,"Create isolated audit output") do
        ispath(root) && error("Refusing to overwrite existing directory $root")
        mkpath(root)
        command_output(cmd)=try readchomp(cmd) catch; "unavailable" end
        meta=Dict("options"=>cfg,"julia_version"=>string(VERSION),"mpi_library"=>MPI.Get_library_version(),
            "ranks"=>MPI.Comm_size(comm),"precision_bits"=>53,"diagnostic_precision_bits"=>256,
            "git_revision"=>command_output(`git rev-parse HEAD`),"git_status"=>command_output(`git status --short`),
            "command"=>join(vcat(Base.julia_cmd().exec,PROGRAM_FILE,ARGS)," "),
            "created_utc"=>string(now(UTC)),"working_directory"=>pwd(),
            "manifest_sha256"=>isfile("Manifest.toml") ? bytes2hex(sha256(read("Manifest.toml"))) : "unavailable",
            "source_sha256"=>Dict(joinpath(dir,f)=>bytes2hex(sha256(read(joinpath(dir,f))))
                for (dir,_,files) in walkdir("src") for f in files if endswith(f,".jl")))
        open(joinpath(root,"campaign.toml"),"w") do io; TOML.print(io,meta);end
    end
    root
end

function state_hash(U,ddg)
    owned=ddg.distributed_mesh.partition.owned
    entries=[(ddg.distributed_mesh.elements.global_ids[e],
        reduce(vcat,[c[:,e] for c in Audit.components(U)])) for e in owned]
    gathered=MPI.gather(entries,ddg.comm;root=0)
    value=if MPI.Comm_rank(ddg.comm)==0
        ordered=sort(reduce(vcat,gathered);by=first)
        bytes2hex(sha256(reinterpret(UInt8,reduce(vcat,last.(ordered)))))
    else
        ""
    end
    MPI.bcast(value,ddg.comm;root=0)
end

function run_audit_case(case,U0,cfl,cfg,path)
    (;ddg,materials,periodic,ws,wave,rhs!)=case
    comm=ddg.comm; rank=MPI.Comm_rank(comm); p=ddg.dg.ref.N
    final=parse(Float64,cfg["final-time"]); fixed=parse(Int,cfg["steps"])
    every=parse(Int,cfg["sample-every"]); stages=parse(Int,cfg["stage-every"])
    isfinite(cfl) && 0<cfl<=1 || error("Audit CFL must lie in (0,1]; no silent clipping.")
    isfinite(final) && final>0 && fixed>=0 && every>0 && stages>=0 || error("Invalid time/sampling options.")
    localdt,_=estimate_maxwell_dt(ddg.dg.mesh,ddg.dg.geometry,ddg.dg.ref;CFL=cfl,ε=1.0,μ=1.0)
    dt=MPI.Allreduce(localdt,min,comm)
    nsteps=fixed>0 ? fixed : ceil(Int,final/dt)
    fixed>0 && (final=nsteps*dt)
    U=deepcopy(U0)
    @assert all(a==b for (a,b) in zip(Audit.components(U),Audit.components(U0)))
    scheme=explicit_partitioned_symplectic_rk_scheme(p+1;first_partition=:H)
    counts=Audit.update_counts(scheme); work=MaxwellPartitionedRKWorkspace(U,scheme)
    R=DiscoGMPI.similar_maxwell_rhs(U)
    initial=Audit.constraint_diagnostics(U,ddg,periodic,materials,ws)
    initial_linear=Audit.linear_values(U,ddg,materials,ws)
    digest=state_hash(U,ddg)
    ne=MPI.Allreduce(length(ddg.distributed_mesh.partition.owned),+,comm)
    box=distributed_axis_aligned_box(ddg)
    errors=distributed_periodic_quadrature_diagnostics(U,ddg,0.0,max(2,2p+4);epsilon=1.0,mu=1.0,wave)
    collective_root_action(comm,"Write audit case metadata") do
        mkpath(path)
        open(joinpath(path,"metadata.toml"),"w") do io
            TOML.print(io,Dict("dt_limit"=>dt,"final_time"=>final,"steps"=>nsteps,"cfl"=>cfl,
                "order"=>p,"esprk_order"=>p+1,"first_partition"=>"H",
                "tetrahedra"=>ne,"domain_lower"=>collect(box.lower),"domain_upper"=>collect(box.upper),
                "epsilon"=>1.0,"mu"=>1.0,"wave_number"=>wave.wave_number,
                "angular_frequency"=>wave.angular_frequency,"flux"=>cfg["flux"],"nodal_set"=>"equispaced",
                "first_weights"=>scheme.first_weights,"second_weights"=>scheme.second_weights,
                "initial_sha256"=>digest,"initial_electric_error_l2"=>errors.electric_error_l2,
                "initial_magnetic_error_l2"=>errors.magnetic_error_l2,"initial_energy"=>errors.total_energy,
                "weak_space"=>"continuous periodic P1 vertex hats; subset of P_(p+1)",
                "weak_tests_selected"=>length(ws.keys),"weak_tests_total"=>ws.total_tests,
                "weak_vertex_keys"=>[collect(k) for k in ws.keys],
                "electric_updates_per_step"=>counts.electric,"magnetic_updates_per_step"=>counts.magnetic,
                "integration_rhs_per_step"=>counts.rhs,"face_counts"=>initial.facecount))
        end
    end
    ios=rank==0 ? [open(joinpath(path,f),"w") for f in ("history.csv","weak.csv","updates.csv")] : [devnull,devnull,devnull]
    hist,weak,updates=ios
    csvline(hist,("step","time","dt","shortened","electric_updates","magnetic_updates","integration_rhs","diagnostic_rhs",
        "qe","qh","dqe","dqh","abs_dqe","abs_dqh","div_e_l2","div_h_l2","div_e_max_sampled","div_h_max_sampled",
        "jump_e_interior_l2","jump_h_interior_l2","jump_e_periodic_l2","jump_h_periodic_l2",
        "rhs_qe","rhs_qh","rhs_div_e_l2","rhs_div_h_l2","px","py","pz","rhs_px","rhs_py","rhs_pz","energy",
        "face_qe_interior","face_qh_interior","face_qe_periodic","face_qh_periodic"))
    csvline(weak,("step","time","field","test","w","drift","volume","face","reconstruction_error","rhs_w","rhs_gradient_pairing"))
    csvline(updates,("step","step_time","stage","partition","coefficient","partition_time_before","partition_time_after","field","functional",
        "delta64","predicted64","defect64","delta256","predicted256","defect256","diagnostic_delta_difference"))
    samples=0
    function sample(step,t,actualdt)
        rhs!(R,U); samples+=1
        d=Audit.constraint_diagnostics(U,ddg,periodic,materials,ws)
        rd=Audit.constraint_diagnostics(Audit.asfield(R),ddg,periodic,materials,ws)
        m=Audit.momentum_diagnostics(U,R,ddg,materials,ws)
        drift=d.charge-initial.charge
        csvline(hist,(step,t,actualdt,step>0 && actualdt<dt,step*counts.electric,step*counts.magnetic,
            step*counts.rhs,samples,d.charge...,drift...,abs.(drift)...,d.div_l2...,d.div_max_sampled...,
            d.jump_l2[:,1]...,d.jump_l2[:,2]...,rd.charge...,rd.div_l2...,m.momentum...,m.momentum_rate...,m.energy,
            d.face_charge[:,1]...,d.face_charge[:,2]...))
        for f in 1:2,j in eachindex(ws.keys)
            csvline(weak,(step,t,f,j,d.weak[f,j],d.weak[f,j]-initial.weak[f,j],
                d.weak_volume[f,j],d.weak_face[f,j],d.weak_reconstruction_error[f,j],rd.weak[f,j],-rd.weak[f,j]))
        end
    end
    before64=nothing;before256=nothing;rate64=nothing;rate256=nothing;pm=nothing
    step=0;time=0.0
    function observer(stage,part,a,U,R,phase)
        f=part==:E ? 1 : 2
        weights=part==scheme.first_partition ? scheme.first_weights : scheme.second_weights
        clock_before=time+(a/weights[stage])*sum(weights[1:stage-1];init=0.0)
        clock_after=clock_before+a
        if phase==:before
            before64=Audit.linear_values(U,ddg,materials,ws)
            rate64=Audit.linear_values(R,ddg,materials,ws)
            before256=setprecision(256) do; Audit.linear_values(U,ddg,materials,ws;T=BigFloat);end
            rate256=setprecision(256) do; Audit.linear_values(R,ddg,materials,ws;T=BigFloat);end
            active=deepcopy(R)
            for c in (part==:E ? Audit.components(active)[4:6] : Audit.components(active)[1:3]);fill!(c,0);end
            pm=Audit.momentum_diagnostics(U,active,ddg,materials,ws)
        else
            after=Audit.linear_values(U,ddg,materials,ws)
            high=setprecision(256) do; Audit.linear_values(U,ddg,materials,ws;T=BigFloat);end
            setprecision(256) do
                for j in axes(after,2)
                    delta=after[f,j]-before64[f,j];pred=a*rate64[f,j]
                    dh=high[f,j]-before256[f,j];ph=BigFloat(a)*rate256[f,j]
                    csvline(updates,(step,time,stage,part,a,clock_before,clock_after,f,j==1 ? "charge" : "weak_$(j-1)",
                        delta,pred,delta-pred,dh,ph,dh-ph,BigFloat(delta)-dh))
                end
            end
            afterP=Audit.momentum_diagnostics(U,R,ddg,materials,ws).momentum
            for d in 1:3
                delta=afterP[d]-pm.momentum[d]; pred=a*pm.momentum_rate[d]
                csvline(updates,(step,time,stage,part,a,clock_before,clock_after,0,"momentum_$d",delta,pred,delta-pred,"","","",""))
            end
        end
    end
    started=time_ns()
    try
        sample(0,0.0,0.0)
        for s in 1:nsteps
            step=s; actualdt=(fixed==0 && s==nsteps) ? final-time : dt
            actualdt>0 || error("Nonpositive final step.")
            Audit.audited_step!(U,work,scheme,actualdt,rhs!;
                observer=stages>0 && s%stages==0 ? observer : nothing)
            time = s==nsteps ? final : s*dt
            (s%every==0 || s==nsteps) && sample(s,time,actualdt)
        end
    finally
        rank==0 && foreach(close,ios)
    end
    elapsed=(time_ns()-started)/1e9
    collective_root_action(comm,"Finish audit metadata") do
        open(joinpath(path,"completed.toml"),"w") do io
            TOML.print(io,Dict("elapsed_seconds"=>elapsed,"final_time"=>time,"steps"=>step,
                "diagnostic_rhs_evaluations"=>samples,"integration_rhs_evaluations"=>step*counts.rhs,
                "electric_updates"=>step*counts.electric,"magnetic_updates"=>step*counts.magnetic,
                "initial_sha256"=>digest,"initial_state_reused_exactly"=>true))
        end
        println(path," completed: ",step," steps, ",elapsed," s")
    end
    U
end

function audit_main(args=ARGS)
    cfg=audit_options(args);cfg===nothing && return
    initialized=MPI.Initialized();initialized || MPI.Init()
    try
        comm=MPI.COMM_WORLD; rank=MPI.Comm_rank(comm)
        root=audit_provenance(cfg,comm)
        specs=isempty(cfg["nx-targets"]) ? split(cfg["meshes"],',') : split(cfg["nx-targets"],',')
        for (m,spec) in enumerate(specs),p in parse.(Int,split(cfg["orders"],','))
            p in 1:4 || error("Audit campaign supports p=1:4.")
            mesh=nothing
            collective_root_action(comm,"Read audit mesh") do
                mesh=if isempty(cfg["nx-targets"])
                    load_periodic_box_mesh(spec)
                else
                    nx=parse(Int,spec);iseven(nx) && nx>=2 || error("Nx must be even and >=2")
                    structured_box_mesh(nx,nx÷2,nx÷2;lower=(0.,0.,0.),upper=(2.,1.,1.),boundary_id=periodic_box_boundary_id)
                end
            end
            case=Audit.build_case(mesh,p,comm;flux=parse_maxwell_flux_kind(cfg["flux"]),max_tests=parse(Int,cfg["max-tests"]))
            collective_root_action(comm,"Mesh provenance") do
                open(joinpath(root,"mesh_$(m)_p$(p).toml"),"w") do io
                    TOML.print(io,Dict("specification"=>spec,"tetrahedra"=>size(mesh.tets,2),
                        "input_sha256"=>isempty(cfg["nx-targets"]) ? bytes2hex(sha256(read(spec))) : "generated structured mesh",
                        "points_sha256"=>bytes2hex(sha256(reinterpret(UInt8,vec(mesh.points)))),
                        "tets_sha256"=>bytes2hex(sha256(reinterpret(UInt8,vec(mesh.tets))))))
                end
            end
            for em in split(cfg["electric-modes"],','),hm in split(cfg["magnetic-modes"],',')
                U0=Audit.initial_field(case;electric=parse_electric_initialization(em),magnetic=parse_magnetic_initialization(hm))
                for cfl in parse.(Float64,split(cfg["cfls"],','))
                    path=joinpath(root,"m$(m)_p$(p)_E$(em)_H$(hm)_cfl$(cfl)")
                    ispath(path) && error("Duplicate case: $path")
                    run_audit_case(case,U0,cfl,cfg,path)
                end
            end
        end
    finally
        initialized || MPI.Finalize()
    end
end
abspath(PROGRAM_FILE)==abspath(@__FILE__) && audit_main()
