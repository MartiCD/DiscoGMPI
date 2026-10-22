#!/usr/bin/env julia
# Six-quantity periodic h/p experiment. The separate weak-charge audit is not
# interchangeable with this volume-quadrature history.
using DiscoGMPI, MPI, LinearAlgebra, TOML, SHA, UUIDs
include(joinpath(@__DIR__,"mixed_invariant_checkpoint.jl"))

function mixed_invariant_row(step,t,d,initial,p)
    row = Pair{String,Any}[
        "step"=>step,"time"=>t,"p_E"=>p,"p_H"=>p+1,"esprk_order"=>p+2,
        "total_energy"=>d.energy,"exact_total_energy"=>1.,
        "optical_chirality"=>d.chirality,"exact_optical_chirality"=>0.,
        "energy_drift"=>d.energy-initial.energy,
        "chirality_drift"=>d.chirality-initial.chirality,
        "divergence_charge_e"=>d.charge_E,"divergence_charge_h"=>d.charge_H,
        "div_e_l2"=>d.divergence_E,"div_h_l2"=>d.divergence_H]
    # Analytic integrals for E=(0,0,-sin(pi*x-pi*t)), H=(0,sin(...),0)
    # on [0,2]×[0,1]×[0,1], epsilon=mu=1.
    for (name,value,exact,baseline) in (
        ("linear_momentum",d.momentum,(1.,0.,0.),initial.momentum),
        ("angular_momentum",d.angular_momentum,(0.,.5,-.5),initial.angular_momentum),
        ("integrated_d",d.integrated_D,(0.,0.,0.),initial.integrated_D),
        ("integrated_b",d.integrated_B,(0.,0.,0.),initial.integrated_B))
        for (i,axis) in enumerate(("x","y","z"))
            push!(row,"$(name)_$axis"=>value[i],"exact_$(name)_$axis"=>exact[i],
                "$(name)_$(axis)_drift"=>value[i]-baseline[i])
        end
    end
    return row
end

function run_mixed_invariants(config;restart="auto",output_override=nothing,
        sample_every=nothing,checkpoint_every=nothing,heartbeat=nothing,
        max_wall_seconds=Inf,segment_steps=0,benchmark_steps=0,warmup_steps=20,
        benchmark_id="",finalize_mpi=true)
    process_started=time()
    MPI.Initialized() || MPI.Init()
    comm=MPI.COMM_WORLD
    rank=MPI.Comm_rank(comm)
    BLAS.set_num_threads(1)
    try
        c=nothing; mesh=nothing; parts=nothing
        collective_root_action(comm,"Validate mixed invariant case") do
            c=TOML.parsefile(config)
            sample_every===nothing || (c["sample_every"]=sample_every)
            checkpoint_every===nothing || (c["checkpoint_every"]=checkpoint_every)
            max_wall_seconds>0 && min(segment_steps,benchmark_steps,warmup_steps)>=0 || error("Invalid segment limit")
            c["discretization"]=="mixed" || error("Not a mixed case")
            p=c["p_E"]
            1<=p<=4 && c["p_H"]==p+1 && c["esprk_order"]==p+2 || error("Invalid orders")
            c["flux"] in ("centered","alternating") || error("Invalid flux")
            all(c[k]>0 for k in ("final_time","cfl","sample_every","print_every")) || error("Invalid time/sampling")
            c["checkpoint_every"]>=0 || error("Invalid checkpoint interval")
            c["cubature_degree"]>=2*(p+1)+1 || error("Insufficient invariant quadrature")
            for k in ("electric_initialization","magnetic_initialization")
                c[k] in ("interpolate","curl-potential") || error("Invalid initialization")
            end
            bytes2hex(sha256(read(c["mesh"])))==c["mesh_sha256"] || error("Mesh hash mismatch")
            mesh=load_periodic_box_mesh(c["mesh"])
            size(mesh.tets,2)==c["nelements"] || error("Element count mismatch")
            # The analytic invariant values below require this exact box.
            for a in 1:3
                isapprox(minimum(mesh.points[a,:]),0.;atol=1e-12) || error("Wrong lower bound")
                isapprox(maximum(mesh.points[a,:]),a==1 ? 2. : 1.;atol=1e-12) || error("Wrong upper bound")
            end
            parts=balanced_spatial_partition(mesh,MPI.Comm_size(comm))
        end
        c=MPI.bcast(c,comm;root=0)
        output=abspath(output_override===nothing ? c["output"] : output_override)
        done=false
        collective_root_action(comm,"Reserve invariant output") do
            done=isfile(joinpath(output,"completed.toml"))
            if done
                old=TOML.parsefile(joinpath(output,"config","run_metadata.toml"))
                all(get(old,k,nothing)==c[k] for k in MIXED_INV_KEYS) || error("Completed output belongs to another case")
                println("Already completed: $output")
            end
            mkpath(joinpath(output,"diagnostics"))
            mkpath(joinpath(output,"config"))
        end
        MPI.bcast(done,comm;root=0) && return
        p=c["p_E"]
        ddg=build_distributed_mixed_dg_from_root(mesh,parts,p;comm,
            periodic=DiscoGMPI.box_periodic_specs(2.,1.,1.),nodal_set=:equispaced)
        wave=PlaneWaveParameters(Float64(pi),Float64(pi),1.,0.)
        E,H=exact_periodic_wave_functions(0.;epsilon=1.,mu=1.,wave)
        U=interpolate_maxwell_field(ddg,E,H)
        if c["electric_initialization"]=="curl-potential"
            A=(x,y,z,t)->DiscoGMPI.periodic_electric_vector_potential(x,y,z,t;epsilon=1.,wave)
            initialize_electric_from_potential!(U,ddg,A)
        end
        if c["magnetic_initialization"]=="curl-potential"
            A=(x,y,z,t)->DiscoGMPI.periodic_magnetic_vector_potential(x,y,z,t;mu=1.,wave)
            initialize_magnetic_from_potential!(U,ddg,A)
        end
        ws=DiscoGMPI.mixed_diagnostic_workspace(ddg.dg;degree=c["cubature_degree"])
        initial=mixed_diagnostic_values(mixed_diagnostics(U,ddg;workspace=ws))
        dt_limit=mixed_cfl_dt(ddg;CFL=c["cfl"],degree_rule=Symbol(c["cfl_degree_rule"]))
        nsteps=ceil(Int,c["final_time"]/dt_limit)
        dt=c["final_time"]/nsteps
        first_step=0
        header=join(first.(mixed_invariant_row(0,0.,initial,initial,p)),',')
        segments=String[]; ends=Int[]; prior_seconds=0.; previous_slot=""
        restart=restart===nothing ? "none" : restart
        candidates=restart=="auto" ? invariant_checkpoint_candidates(output,comm) :
            restart=="none" ? String[] : [abspath(restart)]
        restart=="none" && !isempty(invariant_checkpoint_candidates(output,comm)) &&
            error("Committed checkpoints exist; use auto or a new output directory")
        collective_root_action(comm,"Check invariant continuation") do
            if isempty(candidates) && (isfile(joinpath(output,"diagnostics","invariant_diagnostics.csv")) ||
                    (isdir(joinpath(output,"segments")) && !isempty(readdir(joinpath(output,"segments")))))
                error("History exists without a usable checkpoint; choose a new output directory")
            end
        end
        restored=false
        for path in candidates
            try
                U,state,initial,segments,ends,prior_seconds=invariant_restore(path,ddg,c,dt,nsteps,initial,header)
                first_step=state.step
                previous_slot=dirname(path)==joinpath(output,"checkpoints") ? basename(path) : ""
                rank==0 && println("Resumed invariant checkpoint: $path step=$first_step")
                restored=true
                break
            catch err
                rank==0 && println(stderr,"Rejected invariant checkpoint $path: ",sprint(showerror,err))
            end
        end
        isempty(candidates) || restored || error("No valid invariant checkpoint; refusing a fresh start")
        scheme=DiscoGMPI.mixed_esprk_scheme(p;first_partition=:H)
        work=MaxwellPartitionedRKWorkspace(U,scheme)
        flux=c["flux"]=="centered" ? MaxwellFlux_Central : MaxwellFlux_Alternating
        rhs! = DiscoGMPI.make_maxwell_rhs_function(ddg;flux_kind=flux)
        metadata=merge(c,Dict("dt"=>dt,"dt_limit"=>dt_limit,"nsteps"=>nsteps,
            "actual_ranks"=>MPI.Comm_size(comm),"julia_version"=>string(VERSION),
            "mpi_library"=>MPI.Get_library_version(),"output"=>output,
            "restart"=>restart,"first_step"=>first_step,
            "slurm_job_id"=>get(ENV,"SLURM_JOB_ID","local")))
        collective_root_action(comm,"Write invariant metadata") do
            metadata["source_sha256"]=Dict(joinpath(dir,f)=>bytes2hex(sha256(read(joinpath(dir,f))))
                for (dir,_,files) in walkdir("src") for f in files if endswith(f,".jl"))
            metadata["runner_sha256"]=bytes2hex(sha256(read(@__FILE__)))
            DiscoGMPI.atomic_toml(joinpath(output,"config","run_metadata.toml"),metadata)
        end
        # Every attempt has its own file, so a crash suffix cannot replace the
        # committed history prefix. Completion streams segments into one CSV.
        segment=MPI.bcast(rank==0 ? joinpath(output,"segments",string(uuid4())*".csv") : nothing,comm;root=0)
        push!(segments,segment); push!(ends,first_step)
        io=nothing
        collective_root_action(comm,"Open invariant history") do
            mkpath(dirname(segment))
            io=open(segment,"w")
            println(io,header)
            invariant_heartbeat(heartbeat,first_step,"integration")
        end
        started=time()
        measure_started=0.; measure_first=first_step; measured_steps=0; measured_seconds=0.
        stop_step=segment_steps>0 ? min(nsteps,first_step+segment_steps) : nsteps
        benchmark_steps>0 && (stop_step=min(stop_step,first_step+warmup_steps+benchmark_steps))
        stop_reason="complete"
        try
            for step in first_step:nsteps
                step>first_step && DiscoGMPI.partitioned_symplectic_rk_step!(U,work,scheme,dt,rhs!)
                t=step==nsteps ? c["final_time"] : step*dt
                # One small control broadcast per <=100 steps; diagnostics keep
                # their configured cadence. The batch script reserves time for I/O.
                wallstop=false
                if step%min(c["sample_every"],100)==0 || step==first_step
                    wallstop=MPI.bcast(rank==0 && time()-process_started>=max_wall_seconds,comm;root=0)
                end
                stopping=step==stop_step || wallstop
                saving=(c["checkpoint_every"]>0 && step>first_step && step%c["checkpoint_every"]==0) || stopping
                if step==first_step || step==nsteps || step%c["sample_every"]==0 || saving
                    d=mixed_diagnostic_values(mixed_diagnostics(U,ddg;workspace=ws))
                    all(isfinite,(d.energy,d.chirality,d.momentum...,d.angular_momentum...,
                        d.integrated_D...,d.integrated_B...,d.charge_E,d.charge_H,
                        d.divergence_E,d.divergence_H)) || error("Nonfinite invariant diagnostic")
                    collective_root_action(comm,"Write invariant sample") do
                        println(io,join(last.(mixed_invariant_row(step,t,d,initial,p)),','))
                        flush(io)
                        invariant_heartbeat(heartbeat,step,"integration")
                    end
                end
                if saving
                    ends[end]=step
                    collective_root_action(comm,"Announce invariant checkpoint") do
                        invariant_heartbeat(heartbeat,step,"checkpoint")
                    end
                    previous_slot=invariant_save(output,previous_slot,U,ddg,c,step,dt,initial,
                        segments,ends,prior_seconds+time()-started)
                end
                if step==first_step+warmup_steps
                    # Includes RHS, sampled diagnostics, collectives and regular
                    # checkpoints after compilation/warmup. Use the slowest rank.
                    MPI.Barrier(comm)
                    measure_started=time()
                    measure_first=step
                elseif measure_started>0
                    measured_steps=step-measure_first
                    measured_seconds=time()-measure_started
                end
                if step==first_step || stopping || step%c["print_every"]==0
                    collective_root_action(comm,"Report invariant progress") do
                        println("step=$step/$nsteps t=$t elapsed_seconds=$(prior_seconds+time()-started)")
                        flush(stdout)
                    end
                end
                if stopping
                    stop_reason=step==nsteps ? "complete" : wallstop ? "walltime" : benchmark_steps>0 ? "benchmark" : "segment"
                    break
                end
            end
        finally
            rank==0 && io!==nothing && close(io)
        end
        measured_seconds=MPI.Allreduce(measured_seconds,max,comm)
        collective_root_action(comm,"Finalize invariant segment") do
            segment_metadata=Dict("reason"=>stop_reason,"step"=>last(ends),
                "elapsed_seconds"=>time()-started,"integration_seconds"=>prior_seconds+time()-started)
            DiscoGMPI.atomic_toml(joinpath(output,"segment_status.toml"),segment_metadata)
            if benchmark_steps>0 && measured_steps>0 && measured_seconds>0
                seconds_per_step=measured_seconds/measured_steps
                DiscoGMPI.atomic_toml(joinpath(output,"benchmark.toml"),merge(metadata,
                    Dict("benchmark_id"=>benchmark_id,"measured_steps"=>measured_steps,
                        "measured_seconds"=>measured_seconds,"seconds_per_step"=>seconds_per_step,
                        "estimated_total_seconds"=>seconds_per_step*nsteps,
                        "estimated_remaining_seconds"=>seconds_per_step*(nsteps-last(ends)),
                        "warmup_steps"=>warmup_steps,"segment_reason"=>stop_reason)))
                # Flat standard-library CSV lets MN5 Python 3.10 read the
                # estimate without installing a TOML parser or any packages.
                isempty(benchmark_id) || all(isxdigit,benchmark_id) || error("Invalid benchmark identifier")
                report=joinpath(output,"benchmark.csv")
                open(report*".tmp","w") do reportio
                    println(reportio,"benchmark_id,actual_ranks,sample_every,checkpoint_every,nsteps,measured_steps,measured_seconds,seconds_per_step,estimated_total_seconds,warmup_steps")
                    println(reportio,join((benchmark_id,MPI.Comm_size(comm),c["sample_every"],c["checkpoint_every"],nsteps,
                        measured_steps,measured_seconds,seconds_per_step,seconds_per_step*nsteps,warmup_steps),','))
                end
                mv(report*".tmp",report;force=true)
                println("BENCHMARK seconds_per_step=$seconds_per_step projected_days=$(seconds_per_step*nsteps/86400)")
            end
            if stop_reason=="complete"
                history=joinpath(output,"diagnostics","invariant_diagnostics.csv")
                tmp=history*".tmp"
                invariant_heartbeat(heartbeat,nsteps,"merge-history")
                invariant_merge_history(segments,ends,header,dt;destination=tmp)==nsteps || error("Incomplete final history")
                mv(tmp,history;force=true)
                DiscoGMPI.atomic_toml(joinpath(output,"completed.toml"),
                    Dict("final_time"=>c["final_time"],"steps"=>nsteps,
                        "elapsed_seconds"=>prior_seconds+time()-started))
            end
            invariant_heartbeat(heartbeat,last(ends),stop_reason)
        end
    finally
        finalize_mpi && MPI.Finalize()
    end
end

function mixed_invariant_main(args=ARGS)
    isempty(args) && error("Usage: run_mixed_conserved_quantities.jl CASE.toml [--restart=auto|none|DIR ...]")
    opts=Dict{String,String}()
    for arg in args[2:end]
        k,v=split(arg,'=';limit=2)
        k in ("--restart","--output-dir","--sample-every","--checkpoint-every","--heartbeat",
            "--max-wall-seconds","--segment-steps","--benchmark-steps","--warmup-steps","--benchmark-id") || error("Unknown option $k")
        opts[k]=v
    end
    optional_int(key)=haskey(opts,key) ? parse(Int,opts[key]) : nothing
    run_mixed_invariants(args[1];restart=get(opts,"--restart","auto"),
        output_override=get(opts,"--output-dir",nothing),sample_every=optional_int("--sample-every"),
        checkpoint_every=optional_int("--checkpoint-every"),heartbeat=get(opts,"--heartbeat",nothing),
        max_wall_seconds=parse(Float64,get(opts,"--max-wall-seconds","Inf")),
        segment_steps=parse(Int,get(opts,"--segment-steps","0")),
        benchmark_steps=parse(Int,get(opts,"--benchmark-steps","0")),
        warmup_steps=parse(Int,get(opts,"--warmup-steps","20")),benchmark_id=get(opts,"--benchmark-id",""))
end

if abspath(PROGRAM_FILE)==@__FILE__
    mixed_invariant_main()
end
