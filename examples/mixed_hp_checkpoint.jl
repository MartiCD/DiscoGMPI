# Checkpoints belong to the experiment as well as the fields: retaining the
# prefix maxima is required to report max-time errors after a restart.
const MIXED_HP_RESTART_KEYS = ("name", "mode", "order", "p_E", "p_H", "esprk_order",
    "mesh_sha256", "nodal_set", "mesh_family", "flux", "level", "cubature_degree",
    "final_time", "nsteps", "nelements", "Np_E", "Np_H")

function mixed_hp_heartbeat(path, step, phase)
    path === nothing && return
    mkpath(dirname(path))
    DiscoGMPI.atomic_toml(path, Dict("step"=>step,"phase"=>phase,"unix_time"=>time()))
end

function mixed_hp_checkpoint_candidates(root, comm)
    candidates = String[]
    DiscoGMPI.collective_root_action(comm, "Discover hp checkpoints") do
        entries = Tuple{Int,String}[]
        for slot in ("slot-a", "slot-b")
            path = joinpath(root, slot)
            manifest = joinpath(path, "mixed_manifest.toml")
            if isfile(manifest)
                # A corrupt committed manifest must not silently start t=0.
                step = try
                    Int(TOML.parsefile(manifest)["step"])
                catch
                    -1
                end
                push!(entries,(step,path))
            end
        end
        sort!(entries; by=first, rev=true)
        isempty(entries) && isfile(joinpath(root,"latest.toml")) &&
            error("Latest checkpoint pointer exists, but both committed manifests are missing")
        candidates = last.(entries)
    end
    return MPI.bcast(candidates,comm;root=0)
end

function mixed_hp_restore(dir, ddg, c, dt)
    U, state = load_distributed_checkpoint(dir, ddg)
    saved = state.metadata
    DiscoGMPI.collective_root_action(ddg.comm, "Validate hp restart history") do
        get(saved,"hp_checkpoint_version",nothing)==1 || error("Not an hp error-history checkpoint")
        for key in MIXED_HP_RESTART_KEYS
            get(saved["case"],key,nothing)==c[key] || error("Restart case mismatch: $key")
        end
        state.dt==dt || error("Restart timestep changed")
        0<=state.step<=c["nsteps"] && state.time==state.step*dt || error("Invalid checkpoint time grid")
        saved["error_samples"]==state.step+1 || error("Missing every-step error history")
        for key in ("initial_errors","final_errors","max_errors")
            error_values = saved[key]
            length(error_values)==9 && all(x->isfinite(x) && x>=0,error_values) || error("Invalid $key")
        end
        all(saved["max_errors"] .>= saved["initial_errors"]) &&
            all(saved["max_errors"] .>= saved["final_errors"]) || error("Invalid prefix maxima")
        all(isfinite,values(saved["before"])) || error("Invalid initial diagnostics")
        isfinite(saved["integration_seconds"]) && saved["integration_seconds"]>=0 ||
            error("Invalid checkpoint timing")
    end
    return U, state
end

function mixed_hp_save_checkpoint(root, previous_slot, U, ddg, c, step, dt,
        initial, final, maxima, before, elapsed)
    slot = previous_slot=="slot-a" ? "slot-b" : "slot-a"
    path = joinpath(root,slot)
    DiscoGMPI.collective_root_action(ddg.comm, "Invalidate inactive hp checkpoint") do
        mkpath(path)
        # Invalidate only the inactive generation. The previous valid slot is
        # never modified until this generation has committed successfully.
        manifest = joinpath(path,"mixed_manifest.toml")
        isfile(manifest) && rm(manifest)
    end
    metadata = Dict{String,Any}("hp_checkpoint_version"=>1,"case"=>c,
        "initial_errors"=>collect(initial),"final_errors"=>collect(final),
        "max_errors"=>collect(maxima),"error_samples"=>step+1,
        "before"=>Dict(string(k)=>getproperty(before,k) for k in
            (:volume,:energy,:divergence_E,:divergence_H,:charge_E,:charge_H)),
        "integration_seconds"=>elapsed)
    write_distributed_checkpoint(path,U,ddg;step,time=step*dt,dt,metadata)
    DiscoGMPI.collective_root_action(ddg.comm, "Publish latest hp checkpoint") do
        DiscoGMPI.atomic_toml(joinpath(root,"latest.toml"),
            Dict("slot"=>slot,"step"=>step,"time"=>step*dt))
        println("Checkpoint committed: ",path," step=",step)
        flush(stdout)
    end
    return slot
end
