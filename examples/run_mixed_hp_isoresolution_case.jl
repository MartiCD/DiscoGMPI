#!/usr/bin/env julia
# Run one prepared mixed hp case, sampling spatial errors at every time step.
using MPI, TOML, SHA, LinearAlgebra, DiscoGMPI
include(joinpath(@__DIR__, "mixed_hp_checkpoint.jl"))

const MIXED_HP_QUANTITIES = ("electric", "magnetic", "total", "ex", "ey", "ez", "hx", "hy", "hz")

function mixed_hp_exact(mode, t)
    mode == "plane_wave" && return exact_periodic_wave_functions(t;
        epsilon=1., mu=1., wave=PlaneWaveParameters(Float64(pi), Float64(pi), 1., 0.))
    mode == "pec_cavity" && return exact_cavity_mode_functions(t; epsilon=1., mu=1.)
    error("Unknown physical case $mode")
end

# Cache owned geometry and harmonic exact fields, never global/ghost geometry.
function mixed_hp_error_workspace(ddg, degree, mode)
    points, weights = DiscoGMPI.mixed_tetrahedron_quadrature(degree)
    omega = mode == "plane_wave" ? Float64(pi) : sqrt(3.)*pi
    E0, H0 = mixed_hp_exact(mode, 0.)
    E1, H1 = mixed_hp_exact(mode, pi/(2omega))
    exact_cos = zeros(length(weights), 6, length(ddg.owned))
    exact_sin = similar(exact_cos)
    jacobians = zeros(length(ddg.owned))
    for (i,e) in enumerate(ddg.owned)
        jacobians[i] = ddg.dg.mappings.tet_mappings[e].absdetJ
        for q in eachindex(weights)
            x = DiscoGMPI.map_to_physical(ddg.dg.mesh.points, view(ddg.dg.mesh.tets,:,e),
                points[q,1], points[q,2], points[q,3])
            exact_cos[q,:,i] .= (E0(x...)..., H0(x...)...)
            exact_sin[q,:,i] .= (E1(x...)..., H1(x...)...)
        end
    end
    return (points=points, weights=weights,
        exact_cos=exact_cos, exact_sin=exact_sin,
        jacobians=jacobians, omega=omega, mode=mode,
        VE=DiscoGMPI.mixed_basis(ddg.dg.reference.ref_E, points),
        VH=DiscoGMPI.mixed_basis(ddg.dg.reference.ref_H, points),
        values=zeros(length(weights), 6), sums=zeros(6), global_sums=zeros(6))
end

function mixed_hp_errors!(ws, U, ddg, mode, t)
    mode == ws.mode || error("Error workspace physical case mismatch")
    st, ct = sincos(ws.omega*t)
    fill!(ws.sums, 0.)
    dg = ddg.dg
    fields = DiscoGMPI.mixed_components(U)
    for (i,e) in enumerate(ddg.owned)
        for c in 1:6
            mul!(view(ws.values, :, c), c<=3 ? ws.VE : ws.VH, view(fields[c], :, e))
        end
        for c in 1:6
            subtotal = 0.
            @inbounds @simd for q in eachindex(ws.weights)
                exact = ws.exact_cos[q,c,i]*ct + ws.exact_sin[q,c,i]*st
                subtotal += ws.weights[q]*(ws.values[q,c]-exact)^2
            end
            ws.sums[c] += ws.jacobians[i]*subtotal
        end
    end
    squares = ws.global_sums
    MPI.Allreduce!(ws.sums, squares, +, ddg.comm)
    all(x->isfinite(x) && x>=0, squares) || error("Non-finite mixed field errors")
    electric = squares[1]+squares[2]+squares[3]
    magnetic = squares[4]+squares[5]+squares[6]
    return (sqrt(electric), sqrt(magnetic), sqrt(electric+magnetic),
        ntuple(c->sqrt(squares[c]),6)...)
end

function mixed_hp_write_csv(path, row)
    mkpath(dirname(path))
    # Quote strings so absolute mesh paths containing commas remain valid CSV.
    encode(x) = x isa AbstractString ? "\""*replace(x, "\""=>"\"\"")*"\"" : string(x)
    open(path*".tmp", "w") do io
        println(io, join(first.(row), ','))
        println(io, join(encode.(last.(row)), ','))
    end
    mv(path*".tmp", path; force=true)
end

function run_mixed_hp_case(path; initial_only=false, checkpoint_every=1000,
        checkpoint_dir=nothing, restart="auto", heartbeat=nothing, stop_after_step=nothing,
        finalize_mpi=true)
    checkpoint_every>=0 || error("checkpoint_every must be nonnegative")
    stop_after_step===nothing || stop_after_step>0 || error("stop_after_step must be positive")
    initial_only && restart!="auto" && restart!="none" && error("Initial-only jobs cannot restart")
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    BLAS.set_num_threads(1)
    started = time()
    try
        c, mesh, parts = nothing, nothing, nothing
        setup_error = nothing
        if rank == 0
            try
                c = TOML.parsefile(path)
                p = c["p_E"]
                c["discretization"] == "mixed" && 1<=p<=4 && c["order"]==p &&
                    c["p_H"]==p+1 && c["esprk_order"]==p+2 || error("Invalid mixed degree metadata")
                c["Np_E"]==binomial(p+3,3) && c["Np_H"]==binomial(p+4,3) || error("Invalid nodal counts")
                c["flux"] in ("centered", "alternating") || error("Unsupported flux")
                c["mode"] in ("plane_wave", "pec_cavity") || error("Unsupported physical case")
                c["nsteps"]>0 && isfinite(c["final_time"]) && c["final_time"]>0 || error("Invalid time grid")
                c["cubature_degree"]>=2*(p+1) || error("Insufficient error cubature degree")
                open(c["mesh"]) do io
                    bytes2hex(sha256(io))==c["mesh_sha256"] || error("Mesh SHA256 mismatch")
                end
                mesh = c["mode"]=="plane_wave" ? load_periodic_box_mesh(c["mesh"]) :
                    load_box_boundary_mesh(c["mesh"]; expected_lower=(0.,0.,0.), expected_upper=(1.,1.,1.))
                size(mesh.tets,2)==c["nelements"] || error("Mesh element count mismatch")
                parts = balanced_spatial_partition(mesh, nranks)
                println("Running ", c["name"], ", ranks=", nranks, ", steps=", c["nsteps"])
            catch err
                setup_error = sprint(showerror, err)
            end
        end
        DiscoGMPI.collective_io_error(setup_error, comm, "Mixed hp preparation")
        c = MPI.bcast(c, comm; root=0)
        checkpoint_root = checkpoint_dir===nothing ?
            joinpath(dirname(dirname(c["output"])),"checkpoints",c["name"]) : abspath(checkpoint_dir)
        periodic = c["mode"]=="plane_wave" ? DiscoGMPI.box_periodic_specs(2.,1.,1.) : DiscoGMPI.PeriodicBoundarySpec[]
        backend = Threads.nthreads()>1 ? ThreadedBackend() : SerialBackend()
        ddg = build_distributed_mixed_dg_from_root(mesh, parts, c["p_E"];
            comm, periodic, backend, nodal_set=DiscoGMPI.parse_convergence_nodal_set(c["nodal_set"]))
        dt = c["final_time"]/c["nsteps"]
        limit = mixed_cfl_dt(ddg; CFL=c["cfl"], degree_rule=Symbol(get(c,"cfl_degree_rule","linear")))
        dt <= limit*(1+1e-12) || error("Prepared dt=$dt exceeds mixed CFL limit=$limit")
        E, H = mixed_hp_exact(c["mode"], 0.)
        U = interpolate_maxwell_field(ddg, E, H)
        ws = mixed_hp_error_workspace(ddg, c["cubature_degree"], c["mode"])
        if rank == 0
            println("Error cache on rank 0: ", round(Base.summarysize(ws)/2.0^20;digits=2),
                " MiB; sampling every step; CFL degree rule=", get(c,"cfl_degree_rule","linear"))
            flush(stdout)
        end
        initial = mixed_hp_errors!(ws, U, ddg, c["mode"], 0.)
        final = initial
        maxima = initial
        diagnostics_ws = DiscoGMPI.mixed_diagnostic_workspace(ddg.dg;
            points_weights=(ws.points,ws.weights),basis=(ws.VE,ws.VH))
        before = mixed_diagnostic_values(mixed_diagnostics(U, ddg;workspace=diagnostics_ws))
        first_step, previous_slot, previous_seconds = 0, "", 0.
        restart_used = ""
        if !initial_only
            candidates = restart in ("auto","none") ?
                mixed_hp_checkpoint_candidates(checkpoint_root,comm) : [abspath(restart)]
            restart=="none" && !isempty(candidates) &&
                error("Existing checkpoints; use --restart=auto or a new --checkpoint-dir")
            if restart!="none" && !isempty(candidates)
                failures = String[]
                for candidate in candidates
                    try
                        U, state = mixed_hp_restore(candidate,ddg,c,dt)
                        initial = Tuple(state.metadata["initial_errors"])
                        final = Tuple(state.metadata["final_errors"])
                        maxima = Tuple(state.metadata["max_errors"])
                        before = (; (Symbol(k)=>v for (k,v) in state.metadata["before"])...)
                        first_step = state.step
                        previous_seconds = state.metadata["integration_seconds"]
                        previous_slot = dirname(candidate)==checkpoint_root ? basename(candidate) : ""
                        restart_used = candidate
                        break
                    catch err
                        push!(failures,sprint(showerror,err))
                        rank==0 && println(stderr,"Rejected checkpoint ",candidate,": ",last(failures))
                    end
                end
                isempty(restart_used) && error("No compatible valid hp checkpoint: "*join(failures,"; "))
                rank==0 && println("Restarted from ",restart_used," step=",first_step,
                    "/",c["nsteps"],"; retained ",first_step+1," error samples")
            end
        end
        # Only rank zero writes the heartbeat; the external supervisor does not
        # depend on Julia threads or MPI making progress.
        DiscoGMPI.collective_root_action(comm,"Publish hp start progress") do
            mixed_hp_heartbeat(heartbeat,first_step,"integration")
        end
        scheme = mixed_esprk_scheme(c["p_E"]; first_partition=:H)
        flux = c["flux"]=="centered" ? MaxwellFlux_Central : MaxwellFlux_Alternating
        rhs! = DiscoGMPI.make_maxwell_rhs_function(ddg; flux_kind=flux)
        work = MaxwellPartitionedRKWorkspace(U, scheme)
        integration_start = time()
        if !initial_only
            for step in first_step+1:c["nsteps"]
                DiscoGMPI.partitioned_symplectic_rk_step!(U, work, scheme, dt, rhs!)
                final = mixed_hp_errors!(ws, U, ddg, c["mode"], step*dt)
                maxima = DiscoGMPI.maxwell_error_maxima(maxima, final)
                if step == first_step+1 || step % 100 == 0 || step == c["nsteps"]
                    DiscoGMPI.collective_root_action(comm,"Publish hp progress") do
                        mixed_hp_heartbeat(heartbeat,step,"integration")
                        println("step=", step, "/", c["nsteps"], " t=", step*dt,
                            " integration_seconds=", round(previous_seconds+time()-integration_start; digits=2))
                        flush(stdout)
                    end
                end
                stopping = stop_after_step!==nothing && step>=stop_after_step && step<c["nsteps"]
                if stopping || (checkpoint_every>0 &&
                        (step%checkpoint_every==0 || step==c["nsteps"]))
                    rank==0 && mixed_hp_heartbeat(heartbeat,step,"checkpoint")
                    previous_slot = mixed_hp_save_checkpoint(checkpoint_root,previous_slot,
                        U,ddg,c,step,dt,initial,final,maxima,before,
                        previous_seconds+time()-integration_start)
                    rank==0 && mixed_hp_heartbeat(heartbeat,step,"integration")
                end
                if stopping
                    rank==0 && println("Stopped after checkpoint at step=",step,
                        "; no completed result written. Resubmit to resume.")
                    return
                end
            end
        end
        integration_seconds = MPI.Allreduce(previous_seconds+time()-integration_start, max, comm)
        rank==0 && mixed_hp_heartbeat(heartbeat,c["nsteps"],"finalizing")
        after = mixed_diagnostic_values(mixed_diagnostics(U, ddg;workspace=diagnostics_ws))
        wall = MPI.Allreduce(time()-started, max, comm)
        io_error = nothing
        if rank == 0
            try
                row = Pair{String,Any}[
                    "mode"=>c["mode"], "order"=>c["p_E"], "mesh_level"=>c["level"],
                    "mesh_sha256"=>c["mesh_sha256"], "nodal_set"=>replace(c["nodal_set"], '-'=>'_'),
                    "mesh_family"=>c["mesh_family"], "flux"=>c["flux"],
                    "boundary_condition"=>(c["mode"]=="plane_wave" ? "periodic" : "pec"),
                    "error_mode"=>"max-time", "error_every"=>1,
                    "error_samples"=>(initial_only ? 1 : c["nsteps"]+1),
                    "final_time"=>c["final_time"], "nsteps"=>c["nsteps"], "dt"=>dt,
                    "nelements"=>c["nelements"], "volume_total"=>before.volume,
                    "h_max"=>c["h_max"], "mean_ratio_min"=>c["mean_ratio_min"],
                    "mean_ratio_avg"=>c["mean_ratio_avg"], "actual_ranks"=>nranks,
                    "initial_energy"=>before.energy, "final_energy"=>after.energy,
                    "relative_energy_drift"=>(after.energy-before.energy)/before.energy]
                for key in ("p_E", "p_H", "esprk_order", "Np_E", "Np_H", "electric_dofs", "magnetic_dofs", "field_dofs", "cubature_degree")
                    push!(row, key=>c[key])
                end
                for (i, name) in enumerate(MIXED_HP_QUANTITIES)
                    push!(row, "initial_l2_space_$(name)_error"=>initial[i])
                    if !initial_only
                        push!(row, "final_l2_space_$(name)_error"=>final[i])
                        push!(row, "linf_time_l2_space_$(name)_error"=>maxima[i])
                    end
                end
                output = c["output"]
                if initial_only
                    output = joinpath(dirname(dirname(output)), "initial_results",
                        "$(c["mode"])_$(c["nodal_set"])_$(c["mesh_family"])_p$(c["p_E"])_m$(c["level"]).csv")
                    # Match the manager's initial-result key spelling.
                    row[findfirst(x->first(x)=="nodal_set", row)] = "nodal_set"=>c["nodal_set"]
                end
                metadata = merge(c, Dict("actual_ranks"=>nranks, "dt"=>dt,
                    "initial_only"=>initial_only, "code_id"=>get(ENV,"MIXED_CODE_ID","manual-snapshot"),
                    "julia_version"=>string(VERSION), "slurm_job_id"=>get(ENV,"SLURM_JOB_ID","local"),
                    "rk_stages_per_step"=>length(scheme.first_weights),
                    "rhs_partition_evaluations"=>(initial_only ? 0 : c["nsteps"]*(count(!iszero,scheme.first_weights)+count(!iszero,scheme.second_weights))),
                    "case_wall_seconds"=>wall, "case_process_hours"=>wall*nranks/3600,
                    "integration_and_error_seconds"=>integration_seconds,
                    "restart_checkpoint"=>restart_used,"restart_step"=>first_step,
                    "checkpoint_every"=>checkpoint_every,
                    "relative_energy_drift"=>(after.energy-before.energy)/before.energy))
                for (stage, d) in (("initial", before), ("final", after))
                    for key in (:divergence_E, :divergence_H, :charge_E, :charge_H)
                        value = getproperty(d,key)
                        isfinite(value) || error("Non-finite validity diagnostic")
                        metadata["$(stage)_$(key)"] = value
                    end
                end
                mkpath(dirname(output))
                DiscoGMPI.atomic_toml(replace(output,r"\.csv$"=>"_metadata.toml"), metadata)
                mixed_hp_write_csv(output,row)
                println("Wrote ", output)
            catch err
                io_error = sprint(showerror,err)
            end
        end
        DiscoGMPI.collective_io_error(io_error, comm, "Mixed hp results")
        rank==0 && mixed_hp_heartbeat(heartbeat,c["nsteps"],"completed")
    finally
        finalize_mpi && MPI.Finalize()
    end
end

function mixed_hp_main(args=ARGS)
    isempty(args) && error("Usage: run_mixed_hp_isoresolution_case.jl CASE.toml [--initial-only] [--restart=auto|none|DIR] [--checkpoint-every=N] [--checkpoint-dir=DIR] [--heartbeat=FILE] [--stop-after-step=N]")
    opts = Dict{String,String}()
    initial_only = false
    for arg in args[2:end]
        if arg=="--initial-only"
            initial_only = true
        else
            parts = split(arg,'=';limit=2)
            length(parts)==2 && parts[1] in ("--restart","--checkpoint-every",
                "--checkpoint-dir","--heartbeat","--stop-after-step") || error("Unknown argument $arg")
            opts[parts[1]]=parts[2]
        end
    end
    run_mixed_hp_case(args[1]; initial_only,
        restart=get(opts,"--restart","auto"),
        checkpoint_every=parse(Int,get(opts,"--checkpoint-every","1000")),
        checkpoint_dir=get(opts,"--checkpoint-dir",nothing),heartbeat=get(opts,"--heartbeat",nothing),
        stop_after_step=haskey(opts,"--stop-after-step") ? parse(Int,opts["--stop-after-step"]) : nothing)
end

if abspath(PROGRAM_FILE)==@__FILE__
    mixed_hp_main()
end
