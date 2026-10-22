#!/usr/bin/env julia

"""Prepare or execute the six-run PEC-sphere campaign used by the FFD comparison.

Only Julia standard libraries are loaded here. Preparation is the default; Gmsh,
MPI, the DG driver, and the comparison are invoked only with `--execute`.
"""
module FFDSphereBenchmark

using TOML
using SHA

const REPOSITORY = normpath(joinpath(@__DIR__, "..", ".."))
const LIGHT_SPEED = 299_792_458.0
const WAVELENGTHS = (1.922, 0.961, 0.4805)

function usage(io=stdout)
    println(io, """
    Usage: julia tools/ffd/launch_sphere_benchmark.jl [options]
      --output-dir PATH             Required campaign directory
      --emagpie-dir PATH            Required supplied EMAGPIE example directory
      --execute                     Generate meshes, run six DG jobs, compare
      --dry-run                     Prepare and print the plan only (default)
      --ranks N                     MPI ranks per DG run (default 2)
      --order N                     Equal-order DG polynomial order (default 2)
      --elements-per-wavelength N   Near-sphere mesh resolution (default 8)
      --periods N                   Total incident periods (default 12)
      --accumulation-periods N      Final periods used for Fourier data (default 4)
      --air-wavelengths N           Air gap in wavelengths (default 1)
      --pml-wavelengths N           PML width in wavelengths (default 1)
      --sigma-period N              PML sigma_max times incident period (default 12)
      --cfl N                       Maxwell CFL factor (default 0.15)
      --esprk-order N               Time integrator order, 1:6 (default 3)
      --gmsh PATH                   Gmsh executable (default gmsh)
      --mpiexec PATH                MPI launcher (default mpiexec)
      --julia PATH                  Julia executable (default current executable)
      --python PATH                 Plotting Python interpreter (default python3)
      --no-plots                    Compare without generating plots

    Preparation writes plan.toml and campaign.toml and prints executable commands.
    Repeat the same command with --execute to execute an identical prepared plan.
    Existing meshes and simulation directories are never overwritten: use a fresh
    output directory after a partial or completed execution. The initial numerical
    settings and regression tolerances require a DG convergence study.
    """)
end

function parse_arguments(args)
    options = Dict{String,Any}(
        "--ranks" => 2, "--order" => 2, "--elements-per-wavelength" => 8.0,
        "--periods" => 12.0, "--accumulation-periods" => 4.0,
        "--air-wavelengths" => 1.0, "--pml-wavelengths" => 1.0,
        "--sigma-period" => 12.0, "--cfl" => 0.15, "--esprk-order" => 3,
        "--gmsh" => "gmsh", "--mpiexec" => "mpiexec",
        "--julia" => joinpath(Sys.BINDIR, Base.julia_exename()),
        "--python" => "python3", "--execute" => false, "--plots" => true)
    integer_options = ("--ranks", "--order", "--esprk-order")
    number_options = ("--elements-per-wavelength", "--periods", "--accumulation-periods",
        "--air-wavelengths", "--pml-wavelengths", "--sigma-period", "--cfl")
    string_options = ("--output-dir", "--emagpie-dir", "--gmsh", "--mpiexec", "--julia", "--python")
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--execute"
            options["--execute"] = true
        elseif arg == "--dry-run"
            options["--execute"] = false
        elseif arg == "--no-plots"
            options["--plots"] = false
        else
            tokens = split(arg, '='; limit=2)
            key = first(tokens)
            key in (integer_options..., number_options..., string_options...) || error("Unknown option $key")
            if length(tokens) == 2
                raw = tokens[2]
            else
                i += 1
                i <= length(args) || error("Missing value for $key")
                raw = args[i]
            end
            isempty(raw) && error("$key requires a nonempty value")
            options[key] = key in integer_options ? parse(Int, raw) : key in number_options ? parse(Float64, raw) : String(raw)
        end
        i += 1
    end
    for key in ("--output-dir", "--emagpie-dir")
        haskey(options, key) || error("$key is required")
        options[key] = abspath(options[key])
    end
    for key in (integer_options..., number_options...)
        isfinite(options[key]) && options[key] > 0 || error("$key must be finite and positive")
    end
    options["--esprk-order"] <= 6 || error("--esprk-order must lie in 1:6")
    options["--accumulation-periods"] < options["--periods"] || error("--accumulation-periods must be smaller than --periods")
    isdir(options["--emagpie-dir"]) || error("EMAGPIE directory does not exist")
    return options
end

"""Return commands and a comparison manifest without touching the filesystem."""
function campaign_plan(options)
    root = options["--output-dir"]
    input = options["--emagpie-dir"]
    manifest = TOML.parsefile(joinpath(@__DIR__, "emagpie_reference.toml"))
    manifest["comparison"]["required_sources"] = ["Mie", "EMAGPIE", "DiscoGMPI"]
    manifest["comparison"]["output_dir"] = joinpath(root, "comparison")
    for source in manifest["sources"]
        prefix = source["name"] == "Mie" ? "mie" : "numeric"
        source["files"] = [joinpath(input, "freq$n", "$(prefix)_$cut.ffd") for n in 1:3 for cut in ("phi", "theta")]
        for path in source["files"]
            isfile(path) || error("Missing reference FFD file: $path")
        end
    end
    dg_files = [joinpath(root, "freq$n", cut, "diagnostics", "far_field.ffd") for n in 1:3 for cut in ("phi", "theta")]
    push!(manifest["sources"], Dict{String,Any}("name" => "DiscoGMPI", "files" => dg_files,
        "frequency_scale" => LIGHT_SPEED, "expected_transform" => "identity"))
    meshes = NamedTuple[]
    runs = NamedTuple[]
    for (n, wavelength) in enumerate(WAVELENGTHS)
        mesh = joinpath(root, "meshes", "freq$n.vtk")
        air = options["--air-wavelengths"] * wavelength
        pml = options["--pml-wavelengths"] * wavelength
        gmsh_args = String[options["--gmsh"], joinpath(REPOSITORY, "examples", "meshes", "metallic_sphere_scattering.geo"),
            "-3", "-format", "vtk", "-nt", "1", "-o", mesh]
        for (key, value) in (("Radius", 1.0), ("Lambda0", wavelength),
                ("ElementsPerWavelength", options["--elements-per-wavelength"]),
                ("AirBuffer", air), ("PMLWidth", pml), ("MeshOrder", 1),
                ("SphereMeshSizeFactor", 0.75), ("PMLMeshSizeFactor", 4.0),
                ("FarMeshSizeFactor", 4.0), ("CurvatureSamples", 24), ("OptimizeMesh", 0),
                ("VolumeAlgorithm3D", 1))
            append!(gmsh_args, ["-setnumber", key, string(value)])
        end
        push!(meshes, (; case="freq$n", path=mesh, command=Cmd(gmsh_args)))
        for cut in ("phi", "theta")
            output = joinpath(root, "freq$n", cut)
            driver_args = String[options["--mpiexec"], "-n", string(options["--ranks"]), options["--julia"],
                "--project=$REPOSITORY", joinpath(REPOSITORY, "examples", "distributed_metallic_sphere_scattering.jl"),
                "--mesh=$mesh", "--radius=1", "--wavelength=$wavelength", "--epsilon=1", "--mu=1",
                "--amplitude=1", "--incident-direction=1,0,0", "--incident-polarization=0,0,1",
                "--order=$(options["--order"])", "--esprk-order=$(options["--esprk-order"])",
                "--final-time=$(options["--periods"]*wavelength)",
                "--rcs-start-time=$((options["--periods"]-options["--accumulation-periods"])*wavelength)",
                "--cfl=$(options["--cfl"])", "--pml-width=$pml", "--sigma-max=$(options["--sigma-period"]/wavelength)",
                "--sigma-degree=2", "--flux=centered", "--pml-a=0.5", "--regularization=1e-12",
                "--diagnostics-every=100", "--paraview-every=0", "--enable-rcs", "--rcs-every=1",
                "--enable-ffd", "--ffd-freq-unit=solver_time^-1", "--ffd-field-unit=m", "--output-dir=$output"]
            if cut == "phi"
                phi = join(string.(range(0.0, 180.0; length=500)), ',')
                append!(driver_args, ["--rcs-theta-count=1", "--rcs-theta-min-degrees=90", "--rcs-theta-max-degrees=90", "--rcs-phi-degrees=$phi"])
            else
                append!(driver_args, ["--rcs-theta-count=250", "--rcs-theta-min-degrees=$(90/499)", "--rcs-theta-max-degrees=90", "--rcs-phi-degrees=0,180"])
            end
            push!(runs, (; case="freq$n", cut, path=output, command=Cmd(driver_args)))
        end
    end
    compare_args = String[options["--julia"], "--project=$REPOSITORY", joinpath(@__DIR__, "compare_far_fields.jl"),
        "--config=$(joinpath(root, "campaign.toml"))", "--python=$(options["--python"])"]
    options["--plots"] || push!(compare_args, "--no-plots")
    comparison = Cmd(compare_args)
    stored_options = Dict(key => value for (key, value) in options if key != "--execute")
    input_paths = unique(reduce(vcat, [source["files"] for source in manifest["sources"][1:2]]))
    source_paths = [joinpath(REPOSITORY, "examples", "meshes", "metallic_sphere_scattering.geo"),
        joinpath(REPOSITORY, "examples", "distributed_metallic_sphere_scattering.jl"),
        joinpath(REPOSITORY, "src", "FFDFormat.jl"),
        joinpath(REPOSITORY, "src", "FarFieldConventions.jl"),
        joinpath(@__DIR__, "compare_far_fields.jl"), joinpath(@__DIR__, "plot_comparison.py"),
        @__FILE__]
    plan = Dict{String,Any}("format" => "DiscoGMPI FFD PEC sphere campaign", "version" => 1,
        "repository" => REPOSITORY, "light_speed_m_per_s" => LIGHT_SPEED,
        "wavelengths_m" => collect(WAVELENGTHS), "incident_direction" => [1.0, 0.0, 0.0],
        "incident_polarization" => [0.0, 0.0, 1.0], "radius_m" => 1.0,
        "input_sha256" => Dict(path => bytes2hex(open(sha256, path)) for path in input_paths),
        "source_sha256" => Dict(path => bytes2hex(open(sha256, path)) for path in source_paths),
        "settings" => stored_options, "mesh_commands" => [mesh.command.exec for mesh in meshes],
        "run_commands" => [job.command.exec for job in runs], "comparison_command" => comparison.exec)
    git = Sys.which("git")
    if git !== nothing
        revision = try
            readchomp(pipeline(`$git -C $REPOSITORY rev-parse HEAD`; stderr=devnull))
        catch
            nothing
        end
        revision === nothing || (plan["repository_commit"] = revision)
    end
    return (; plan, manifest, meshes, runs, comparison)
end

shell_quote(value) = "'" * replace(String(value), "'" => "'\"'\"'") * "'"
shell_command(command::Cmd) = join(shell_quote.(command.exec), ' ')

function write_identical_or_new(path, data)
    if ispath(path)
        isfile(path) && TOML.parsefile(path) == data || error("Existing plan differs: $path. Use a fresh output directory.")
    else
        open(path, "w") do io
            TOML.print(io, data; sorted=true)
        end
    end
end

function prepare(options, campaign)
    root = options["--output-dir"]
    for target in [getproperty.(campaign.meshes, :path); getproperty.(campaign.runs, :path); joinpath(root, "comparison")]
        (ispath(target) || islink(target)) && error("Refusing to overwrite existing campaign data: $target. Use a fresh output directory.")
    end
    for name in ("plan.toml", "campaign.toml")
        path = joinpath(root, name)
        data = name == "plan.toml" ? campaign.plan : campaign.manifest
        if ispath(path) || islink(path)
            isfile(path) && TOML.parsefile(path) == data || error("Existing plan differs: $path. Use a fresh output directory.")
        end
    end
    mkpath(root)
    write_identical_or_new(joinpath(root, "plan.toml"), campaign.plan)
    write_identical_or_new(joinpath(root, "campaign.toml"), campaign.manifest)
    return nothing
end

function run_logged(command, path)
    (ispath(path) || islink(path)) && error("Refusing to overwrite log $path")
    open(path, "w") do io
        run(pipeline(command; stdout=io, stderr=io))
    end
end

function main(args=ARGS; io=stdout)
    if "--help" in args || "-h" in args
        usage(io)
        return nothing
    end
    options = parse_arguments(args)
    campaign = campaign_plan(options)
    prepare(options, campaign)
    println(io, "Campaign: ", options["--output-dir"])
    println(io, "Prepared three wavelength-specific meshes and six runs with 500 observation directions each.")
    println(io, "Numerical settings are starting values; spatial, time-window, and PML convergence remain to be established.")
    println(io, "Comparison expects identity complex convention; alternate transforms are diagnostic and do not relax the gate.")
    for mesh in campaign.meshes
        println(io, shell_command(mesh.command))
    end
    for job in campaign.runs
        println(io, shell_command(job.command))
    end
    println(io, shell_command(campaign.comparison))
    if options["--execute"]
        root = options["--output-dir"]
        mkpath(joinpath(root, "logs"))
        mkpath(joinpath(root, "meshes"))
        for mesh in campaign.meshes
            println(io, "Meshing ", mesh.case)
            run_logged(mesh.command, joinpath(root, "logs", "$(mesh.case)_mesh.log"))
            isfile(mesh.path) || error("Gmsh did not create $(mesh.path)")
        end
        for job in campaign.runs
            println(io, "Running ", job.case, "/", job.cut)
            run_logged(job.command, joinpath(root, "logs", "$(job.case)_$(job.cut).log"))
        end
        println(io, "Comparing FFD results")
        run_logged(campaign.comparison, joinpath(root, "logs", "comparison.log"))
    else
        println(io, "Prepared only. Repeat with --execute to generate the meshes and run this plan.")
    end
    return campaign
end

end # module FFDSphereBenchmark

if abspath(PROGRAM_FILE) == @__FILE__
    try
        FFDSphereBenchmark.main()
    catch exception
        showerror(stderr, exception)
        println(stderr)
        exit(1)
    end
end
