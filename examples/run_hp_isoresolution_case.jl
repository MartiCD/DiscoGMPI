#!/usr/bin/env julia
# Run one prepared case using the established convergence kernels.
using MPI, TOML, SHA, DiscoGMPI

module HPCavity
include("convergence_distributed_poisson_bracket_maxwell.jl")
end
module HPPeriodic
include("convergence_distributed_periodic_poisson_bracket_maxwell.jl")
end

function initial_result_path(c)
    root = dirname(dirname(c["output"]))
    family = get(c, "mesh_family", "unstructured")
    filename = "$(c["mode"])_$(c["nodal_set"])_$(family)_p$(c["order"])_m$(c["level"]).csv"
    return joinpath(root, "initial_results", filename)
end

function write_initial_result(path, c, initial_errors)
    names = ("electric", "magnetic", "total", "ex", "ey", "ez", "hx", "hy", "hz")
    columns = ["mode", "nodal_set", "mesh_family", "order", "mesh_level", "mesh_sha256"]
    append!(columns, ["initial_l2_space_$(name)_error" for name in names])
    values = Any[c["mode"], c["nodal_set"], get(c, "mesh_family", "unstructured"),
                 c["order"], c["level"], c["mesh_sha256"]]
    append!(values, collect(initial_errors))
    mkpath(dirname(path))
    temporary = path * ".tmp"
    open(temporary, "w") do io
        println(io, join(columns, ','))
        println(io, join(values, ','))
    end
    mv(temporary, path; force=true)
end

function run_hp_case(path; initial_only=false)
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    started = time()
    try
        c = TOML.parsefile(path)
        mode = c["mode"]
        driver = mode == "plane_wave" ? HPPeriodic : HPCavity
        mode in ("plane_wave", "pec_cavity") || error("Unknown mode $mode")
        args = ["--orders=$(c["order"])", "--final-time=$(c["final_time"])",
                "--cfl=$(c["cfl"])", "--flux=$(c["flux"])",
                "--nodal-set=$(c["nodal_set"])", "--error-mode=max-time",
                "--error-every=1", "--rate-policy=report", "--output=$(c["output"])"]
        if mode == "plane_wave"
            append!(args, ["--nx-targets=$(c["resolution"])", "--wave-number=$(Float64(pi))"])
            config = driver.parse_periodic_convergence_arguments(args)
        else
            config = driver.parse_convergence_arguments(args)
        end
        root_mesh = nothing
        load_error = nothing
        if rank == 0
            try
                open(c["mesh"]) do io
                    bytes2hex(sha256(io)) == c["mesh_sha256"] || error("Mesh SHA256 mismatch")
                end
                root_mesh = mode == "plane_wave" ? driver.load_periodic_mesh(c["mesh"]) : driver.load_pec_mesh(c["mesh"])
                println("Running ", c["name"], ": nsteps=", c["nsteps"], ", ranks=", MPI.Comm_size(comm))
            catch e
                load_error = sprint(showerror, e)
            end
        end
        load_error = MPI.bcast(load_error, comm; root=0)
        load_error === nothing || error(load_error)
        diagnostics = Dict{String, Any}()
        initial_errors = Ref{Any}(nothing)
        function validity(stage, U, dg)
            materials = homogeneous_maxwell_materials(size(U.Hx, 2); epsilon=1., permeability=1.)
            for (name, values) in (("electric", electric_initialization_diagnostics(U, dg, materials)),
                                   ("magnetic", magnetic_initialization_diagnostics(U, dg, materials)))
                for (key, value) in pairs(values)
                    isfinite(value) || error("Non-finite $(stage) $(name) validity diagnostic $(key)")
                    diagnostics["$(stage)_$(name)_$(key)"] = value
                end
            end
            if stage == "initial"
                cubature_order = max(2, 2 * c["order"] + 4)
                components = if mode == "plane_wave"
                    box = driver.distributed_periodic_box(dg)
                    wave = PlaneWaveParameters(
                        Float64(pi), Float64(pi), driver.MAGNETIC_AMPLITUDE,
                        box.lower[1])
                    driver.distributed_periodic_component_l2_errors(
                        U, dg, 0.0, cubature_order;
                        epsilon=1.0, mu=1.0, wave=wave)
                else
                    driver.distributed_cavity_component_l2_errors(
                        U, dg, 0.0, cubature_order;
                        epsilon=1.0, mu=1.0, boundary_condition=:pec)
                end
                electric = sqrt(sum(components[index]^2 for index in 1:3))
                magnetic = sqrt(sum(components[index]^2 for index in 4:6))
                initial_errors[] = (electric, magnetic, hypot(electric, magnetic), components...)
            end
        end
        if initial_only
            root_partition = rank == 0 ? balanced_spatial_partition(root_mesh, MPI.Comm_size(comm)) : nothing
            distributed_dg = build_distributed_dg_from_root(
                root_mesh, root_partition, c["order"];
                comm=comm, nodal_set=config.nodal_set)
            electric, magnetic = if mode == "plane_wave"
                box = driver.distributed_periodic_box(distributed_dg)
                wave = PlaneWaveParameters(
                    Float64(pi), Float64(pi), driver.MAGNETIC_AMPLITUDE, box.lower[1])
                exact_periodic_wave_functions(0.0; epsilon=1.0, mu=1.0, wave=wave)
            else
                exact_cavity_mode_functions(
                    0.0; epsilon=1.0, mu=1.0, boundary_condition=:pec)
            end
            U = interpolate_maxwell_field(distributed_dg, electric, magnetic)
            validity("initial", U, distributed_dg)
            initial_errors[] === nothing && error("Initial interpolation errors were not sampled")
            if rank == 0
                output = initial_result_path(c)
                write_initial_result(output, c, initial_errors[])
                println("Wrote initial interpolation errors to ", output)
            end
            return
        end
        result = if mode == "plane_wave"
            driver.run_distributed_periodic_convergence_case(
                c["resolution"], c["order"], c["level"]-1, config, comm, root_mesh;
                nsteps_override=c["nsteps"], validity_callback=validity)
        else
            driver.run_distributed_convergence_case(
                c["resolution"], c["order"], c["level"]-1, config, comm,
                root_mesh, nothing, true; nsteps_override=c["nsteps"], validity_callback=validity)
        end
        wall = MPI.Allreduce(time()-started, max, comm)
        if rank == 0
            row = mode == "plane_wave" ? driver.periodic_convergence_row(result, config) : driver.cavity_convergence_row(result, config)
            initial_errors[] === nothing && error("Initial interpolation errors were not sampled")
            initial_names = (
                :electric, :magnetic, :total, :ex, :ey, :ez, :hx, :hy, :hz)
            initial_fields = (; (
                Symbol("initial_l2_space_$(name)_error") => initial_errors[][index]
                for (index, name) in enumerate(initial_names))...)
            row = merge(row, initial_fields,
                        (mesh_family=Symbol(get(c, "mesh_family", "unstructured")),
                         mesh_source=c["mesh"], seed=c["seed"]))
            driver.write_maxwell_convergence_csv(c["output"], [row])
            scheme = explicit_partitioned_symplectic_rk_scheme(c["order"]+1; first_partition=:H)
            metadata = merge(c, diagnostics, Dict(
                "actual_ranks"=>MPI.Comm_size(comm), "dt"=>result.dt,
                "rk_stages_per_step"=>length(scheme.first_weights),
                "rhs_partition_evaluations"=>result.nsteps*(count(!iszero, scheme.first_weights)+count(!iszero, scheme.second_weights)),
                "case_wall_seconds"=>wall,
                "case_process_hours"=>wall*MPI.Comm_size(comm)/3600,
                "integration_and_error_seconds"=>result.elapsed_seconds,
                "timing_scope"=>"case wall includes mesh loading/setup; excludes Julia startup/imports and final output",
                "relative_energy_drift"=>row.relative_energy_drift))
            open(replace(c["output"], r"\.csv$"=>"_metadata.toml"), "w") do io
                TOML.print(io, metadata)
            end
            println("Wrote ", c["output"])
        end
    finally
        MPI.Finalize()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (1, 2) || error(
        "Usage: run_hp_isoresolution_case.jl CASE.toml [--initial-only]")
    initial_only = length(ARGS) == 2
    initial_only && ARGS[2] != "--initial-only" && error("Unknown option $(ARGS[2])")
    run_hp_case(ARGS[1]; initial_only=initial_only)
end
