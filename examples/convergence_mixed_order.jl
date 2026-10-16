using DiscoGMPI, MPI, Printf, LinearAlgebra

"""Periodic mixed-order convergence study. Outputs CSV on rank zero.

Usage: mpiexec -n 2 julia --project=. examples/convergence_mixed_order.jl
       /tmp/mixed.csv 1,2,3,4 2,3,4 0.05 0.1
Arguments: output, electric degrees, box subdivisions, final time, CFL,
initialization (interpolate/project), flux (both/centered/alternating).
Use `iso:1,2,3` for the paper's directional isoresolution family:
nx = 12*2^(level-1)/p, ny=nz=2^(level-1), box=(1,1/4,1/4).
At a fixed level p_E*nx is matched; p_H*nx and actual DOF counts differ.
Use the same arguments with half CFL to check temporal contamination.
"""
function run_mixed_convergence(args=ARGS)
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank,nranks = MPI.Comm_rank(comm),MPI.Comm_size(comm)
    output = length(args)>=1 ? args[1] : "/tmp/mixed-convergence.csv"
    degrees = length(args)>=2 ? parse.(Int,split(args[2],',')) : [1,2,3,4]
    level_arg = length(args)>=3 ? args[3] : "2,3,4"
    isoresolution = startswith(level_arg,"iso:")
    levels = parse.(Int,split(replace(level_arg,"iso:"=>""),','))
    final_time = length(args)>=4 ? parse(Float64,args[4]) : 0.05
    cfl = length(args)>=5 ? parse(Float64,args[5]) : 0.1
    initialization = length(args)>=6 ? args[6] : "interpolate"
    flux_label = length(args)>=7 ? args[7] : "both"
    fluxes = flux_label=="both" ? (MaxwellFlux_Central,MaxwellFlux_Alternating) :
             flux_label=="centered" ? (MaxwellFlux_Central,) :
             flux_label=="alternating" ? (MaxwellFlux_Alternating,) :
             throw(ArgumentError("Flux must be both, centered, or alternating"))
    !isempty(degrees) && all(p->1<=p<=4,degrees) || error("Electric degrees must be in 1:4")
    !isempty(levels) && all(>(0),levels) && issorted(levels) && allunique(levels) ||
        error("Levels must be distinct, increasing positive integers")
    initialization in ("interpolate","project") || error("Initialization must be interpolate or project")
    isfinite(final_time) && final_time>0 || error("Final time must be finite and positive")
    isfinite(cfl) && cfl>0 || error("CFL must be finite and positive")
    BLAS.set_num_threads(1)
    io_error = nothing
    if rank==0
        try
            ispath(output) && error("Output already exists: $output; choose a new run path")
            mkpath(dirname(abspath(output)))
            open(output,"w") do io
                println(io,"p_E,p_H,esprk_order,flux,n,h,elements,dofs_E,dofs_H,ranks,dt,steps,error_E,error_H,rate_E,rate_H,energy_drift,seconds,study,level,resolution_E,resolution_H,error_norm,final_time,initialization")
            end
            DiscoGMPI.atomic_toml(output*".toml",Dict(
                "schema_version"=>1,"p_E"=>degrees,"p_H"=>degrees.+1,"esprk_orders"=>degrees.+2,
                "levels"=>levels,"isoresolution"=>isoresolution,"flux"=>flux_label,
                "initialization"=>initialization,"final_time"=>final_time,"CFL"=>cfl,
                "mpi_ranks"=>nranks,"julia_threads"=>Threads.nthreads(),
                "julia_version"=>string(VERSION),"slurm_job_id"=>get(ENV,"SLURM_JOB_ID","local"),
                "error_norm"=>"final_time_l2_space","nodal_set"=>"equispaced"))
        catch err
            io_error = sprint(showerror,err)
        end
    end
    DiscoGMPI.collective_io_error(io_error,comm,"Mixed campaign output initialization")
    exactE(x,y,z,t) = (0.,sin(2pi*(x-t)),0.)
    exactH(x,y,z,t) = (0.,0.,sin(2pi*(x-t)))
    for p in degrees, flux in fluxes
        scheme = mixed_esprk_scheme(p)
        previous = nothing
        for level in levels
            n = isoresolution ? div(12*2^(level-1),p) : level
            ny = isoresolution ? 2^(level-1) : n
            side = isoresolution ? 0.25 : 1.
            mesh = rank==0 ? structured_box_mesh(n,ny,ny;lower=(0.,0.,0.),upper=(1.,side,side),boundary_id=periodic_box_boundary_id) : nothing
            # Deterministic contiguous ownership; no state is replicated during evolution.
            parts = rank==0 ? [min(nranks-1,fld((i-1)*nranks,size(mesh.tets,2))) for i in axes(mesh.tets,2)] : nothing
            backend = Threads.nthreads()>1 ? ThreadedBackend() : SerialBackend()
            ddg = build_distributed_mixed_dg_from_root(mesh,parts,p;comm,backend,periodic=DiscoGMPI.box_periodic_specs(1.,side,side))
            init = initialization=="project" ? project_maxwell_field : interpolate_maxwell_field
            U = init(ddg,(x,y,z)->exactE(x,y,z,0.),(x,y,z)->exactH(x,y,z,0.))
            rhs! = DiscoGMPI.make_maxwell_rhs_function(ddg;flux_kind=flux)
            work = MaxwellPartitionedRKWorkspace(U,scheme)
            initial = mixed_diagnostic_values(mixed_diagnostics(U,ddg)).energy
            steps = ceil(Int,final_time/mixed_cfl_dt(ddg;CFL=cfl))
            dt = final_time/steps
            started = time_ns()
            for _ in 1:steps
                DiscoGMPI.partitioned_symplectic_rk_step!(U,work,scheme,dt,rhs!)
            end
            elapsed = MPI.Allreduce((time_ns()-started)*1e-9,max,comm)
            diag = mixed_diagnostic_values(mixed_diagnostics(U,ddg;time=final_time,exact_electric=exactE,exact_magnetic=exactH))
            h = 1/n
            rE,rH = previous===nothing ? (NaN,NaN) :
                (log(previous[2]/diag.error_E)/log(previous[1]/h),log(previous[3]/diag.error_H)/log(previous[1]/h))
            previous = (h,diag.error_E,diag.error_H)
            io_error = nothing
            if rank==0
                row = (p,p+1,p+2,DiscoGMPI.maxwell_flux_kind_label(flux),n,h,6n*ny^2,
                    18n*ny^2*binomial(p+3,3),18n*ny^2*binomial(p+4,3),nranks,dt,steps,
                    diag.error_E,diag.error_H,rE,rH,(diag.energy-initial)/initial,elapsed,
                    isoresolution ? "directional_isoresolution" : "cubic_h",level,p*n,(p+1)*n,"final_time_l2_space",final_time,initialization)
                try
                    open(output,"a") do io
                        println(io,join(row,','))
                    end
                    println(join(row,','))
                    flush(stdout)
                catch err
                    io_error = sprint(showerror,err)
                end
            end
            DiscoGMPI.collective_io_error(io_error,comm,"Mixed campaign result writing")
        end
    end
    return output
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_mixed_convergence()
    MPI.Finalize()
end
