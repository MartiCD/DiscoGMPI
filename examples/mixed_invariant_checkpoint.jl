# Experiment state accompanying the distributed field checkpoints. History
# segments are immutable once closed; a checkpoint records their committed end.
const MIXED_INV_KEYS = ("name","flux","p_E","p_H","esprk_order","mesh_sha256",
    "final_time","electric_initialization","magnetic_initialization","cubature_degree")

function invariant_heartbeat(path, step, phase)
    path===nothing && return
    mkpath(dirname(path))
    DiscoGMPI.atomic_toml(path,Dict("step"=>step,"phase"=>phase,"unix_time"=>time()))
end

function invariant_checkpoint_candidates(output, comm)
    paths=String[]
    collective_root_action(comm,"Discover invariant checkpoints") do
        root=joinpath(output,"checkpoints")
        entries=Tuple{Int,String}[]
        if isdir(root)
            for entry in readdir(root;join=true)
                (basename(entry) in ("slot-a","slot-b") || startswith(basename(entry),"step")) || continue
                manifest=joinpath(entry,"mixed_manifest.toml")
                isfile(manifest) || continue
                step=try Int(TOML.parsefile(manifest)["step"]) catch; -1 end
                push!(entries,(step,entry))
            end
        end
        sort!(entries;by=first,rev=true)
        paths=last.(entries)
        isempty(paths) && isfile(joinpath(root,"latest.toml")) &&
            error("Checkpoint pointer exists without a committed manifest")
    end
    MPI.bcast(paths,comm;root=0)
end

invariant_initial_dict(d)=Dict(string(k)=>(getproperty(d,k) isa Tuple ?
    collect(getproperty(d,k)) : getproperty(d,k)) for k in keys(d))

function invariant_initial_values(d)
    pairs=Pair{Symbol,Any}[]
    for k in sort!(collect(keys(d)))
        v=d[k]
        push!(pairs,Symbol(k)=>(v isa AbstractVector ? Tuple(v) : v))
    end
    (;pairs...)
end

# Stream histories rather than loading millions of rows. Duplicate checkpoint
# boundary samples are retained from the older segment. A crash suffix after a
# committed end is ignored. The t=0 row and continuous time ordering are checked.
function invariant_merge_history(segments, ends, header, dt; destination=nothing)
    length(segments)==length(ends)>0 || error("Missing invariant history lineage")
    out=destination===nothing ? nothing : open(destination,"w")
    laststep=-1
    previous_end=-1
    try
        out===nothing || println(out,header)
        for (path,limit) in zip(segments,ends)
            limit>=previous_end || error("History segment ends are not ordered")
            previous_end=limit
            localstep=-1
            open(path) do io
                readline(io)==header || error("History column mismatch: $path")
                for line in eachline(io)
                    fields=split(line,',')
                    # Only a suffix beyond the committed checkpoint may be incomplete.
                    step=tryparse(Int,first(fields))
                    step===nothing && error("Invalid committed history row in $path")
                    step>limit && break
                    length(fields)==length(split(header,',')) || error("Truncated history: $path")
                    step>localstep || error("Non-monotone history: $path")
                    localstep=step
                    vals=tryparse.(Float64,fields)
                    all(v->v!==nothing && isfinite(v),vals) || error("Nonfinite history: $path")
                    isapprox(vals[2],step*dt;atol=1e-12,rtol=1e-12) || error("History time grid changed")
                    if step>laststep
                        laststep<0 && step!=0 && error("History does not begin at t=0")
                        out===nothing || println(out,line)
                        laststep=step
                    end
                    step==limit && break
                end
            end
            localstep>=0 || error("Empty history segment: $path")
            # Older runners need not sample the checkpoint itself. New saves do.
            laststep<=limit || error("History extends beyond checkpoint")
        end
    finally
        out===nothing || close(out)
    end
    return laststep
end

function invariant_restore(path,ddg,c,dt,nsteps,initial,header)
    U,state=load_distributed_checkpoint(path,ddg)
    saved=state.metadata
    segments=String[]; ends=Int[]; elapsed=0.; baseline=initial
    collective_root_action(ddg.comm,"Validate invariant checkpoint and history") do
        for k in MIXED_INV_KEYS
            get(saved,k,nothing)==c[k] || error("Restart mismatch: $k")
        end
        isapprox(state.dt,dt;rtol=1e-14) || error("Restart timestep changed")
        0<=state.step<=nsteps && isapprox(state.time,state.step*dt) || error("Invalid restart step")
        if get(saved,"invariant_checkpoint_version",0)==1
            segments=String.(saved["history_segments"])
            ends=Int.(saved["history_ends"])
            last(ends)==state.step || error("Checkpoint history end mismatch")
            baseline=invariant_initial_values(saved["initial_diagnostics"])
            elapsed=Float64(saved["integration_seconds"])
            isfinite(elapsed) && elapsed>=0 || error("Invalid accumulated timing")
        else
            # Existing campaigns stored flat case metadata and one flushed CSV.
            # Keep that file as an immutable prefix for future continuations.
            oldoutput=dirname(dirname(path))
            history=joinpath(oldoutput,"diagnostics","invariant_diagnostics.csv")
            isfile(history) || error("Legacy checkpoint history missing: $history")
            segments=[history]; ends=[state.step]
        end
        lastsample=invariant_merge_history(segments,ends,header,dt)
        get(saved,"invariant_checkpoint_version",0)==1 && lastsample!=state.step &&
            error("Committed checkpoint sample is missing")
    end
    baseline=MPI.bcast(baseline,ddg.comm;root=0)
    segments=MPI.bcast(segments,ddg.comm;root=0)
    ends=MPI.bcast(ends,ddg.comm;root=0)
    elapsed=MPI.bcast(elapsed,ddg.comm;root=0)
    return U,state,baseline,segments,ends,elapsed
end

function invariant_save(output,previous,U,ddg,c,step,dt,initial,segments,ends,elapsed)
    slot=previous=="slot-a" ? "slot-b" : "slot-a"
    path=joinpath(output,"checkpoints",slot)
    collective_root_action(ddg.comm,"Prepare inactive invariant checkpoint") do
        mkpath(path)
        manifest=joinpath(path,"mixed_manifest.toml")
        isfile(manifest) && rm(manifest)
    end
    metadata=merge(c,Dict("invariant_checkpoint_version"=>1,
        "initial_diagnostics"=>invariant_initial_dict(initial),
        "history_segments"=>segments,"history_ends"=>ends,"integration_seconds"=>elapsed))
    write_distributed_checkpoint(path,U,ddg;step,time=step*dt,dt,metadata)
    collective_root_action(ddg.comm,"Commit invariant checkpoint") do
        DiscoGMPI.atomic_toml(joinpath(output,"checkpoints","latest.toml"),
            Dict("slot"=>slot,"step"=>step,"time"=>step*dt))
        println("Checkpoint committed: $path step=$step")
        flush(stdout)
    end
    return slot
end
