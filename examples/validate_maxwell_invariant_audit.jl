#!/usr/bin/env julia
# One serial orchestrator; launches matching MPI.jl executables, never T=1000.
using MPI, TOML, Dates

function main(args=ARGS)
    output="output/invariant_audit_validation/"*Dates.format(now(),"yyyymmdd_HHMMSSsss")
    ranks=[1,2,4]
    for a in args
        if startswith(a,"--output-dir="); output=split(a,'=';limit=2)[2]
        elseif startswith(a,"--ranks="); ranks=parse.(Int,split(split(a,'=';limit=2)[2],','))
        elseif a=="--help"
            println("julia --project=. examples/validate_maxwell_invariant_audit.jl --ranks=1,2,4 --output-dir=NEW_PATH");return
        else;error("Unknown option $a")
        end
    end
    all(>(0),ranks) || error("Ranks must be positive.")
    ispath(output) && error("Refusing overwrite: $output")
    mkpath(output)
    # Avoid MPI/BLAS oversubscription; no system launcher/library mismatch.
    ENV["OPENBLAS_NUM_THREADS"]="1";ENV["JULIA_NUM_THREADS"]="1";ENV["OMP_NUM_THREADS"]="1"
    julia=`$(Base.julia_cmd()) --project=. --compiled-modules=no`
    commands=Pair{String,Cmd}[]
    for n in ranks
        push!(commands,"mpi_$n"=>`$(MPI.mpiexec()) -n $n $julia test/test_mpi_maxwell_invariant_audit.jl`)
    end
    push!(commands,"precision_tests"=>`$julia test/test_maxwell_invariant_precision.jl`)
    push!(commands,"precision_central"=>`$julia examples/audit_maxwell_precision.jl --output-dir=$(joinpath(output,"precision_central"))`)
    push!(commands,"precision_alternating"=>`$julia examples/audit_maxwell_precision.jl --flux=alternating --output-dir=$(joinpath(output,"precision_alternating"))`)
    n=min(2,maximum(ranks))
    push!(commands,"short_campaign"=>`$(MPI.mpiexec()) -n $n $julia examples/audit_periodic_maxwell_invariants.jl --final-time=0.005 --stage-every=2 --output-dir=$(joinpath(output,"short"))`)
    push!(commands,"fixed_steps"=>`$(MPI.mpiexec()) -n $n $julia examples/audit_periodic_maxwell_invariants.jl --steps=2 --stage-every=1 --magnetic-modes=curl-potential --output-dir=$(joinpath(output,"fixed"))`)
    push!(commands,"plots"=>`python3 examples/plot_maxwell_invariant_audit.py $(joinpath(output,"short"))`)
    push!(commands,"output_contracts"=>`python3 test/test_invariant_audit_outputs.py $output`)
    records=Dict{String,Any}[]
    for (name,cmd) in commands
        start=time()
        process=open(joinpath(output,name*".log"),"w") do io
            run(pipeline(ignorestatus(cmd);stdout=io,stderr=io))
        end
        push!(records,Dict("name"=>name,"command"=>string(Cmd(cmd.exec)),"argv"=>cmd.exec,
            "passed"=>success(process),"elapsed_seconds"=>time()-start))
        open(joinpath(output,"summary.toml"),"w") do io
            TOML.print(io,Dict("checks"=>records,"passed_so_far"=>all(r["passed"] for r in records),
                "finished"=>length(records)==length(commands)))
        end
        println(name,": ",success(process) ? "PASS" : "FAIL (see log)");flush(stdout)
    end
    all(r["passed"] for r in records) || error("Invariant audit suite failed; see $output/summary.toml")
end
abspath(PROGRAM_FILE)==abspath(@__FILE__) && main()
