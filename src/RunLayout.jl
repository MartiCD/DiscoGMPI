struct RunPaths
    root::String
    config_dir::String
    input_dir::String
    diagnostics_dir::String
    fields_dir::String
    checkpoints_dir::String
    logs_dir::String
    plots_dir::String
    tables_dir::String
end

function RunPaths(root::AbstractString)
    resolved_root = abspath(String(root))
    return RunPaths(
        resolved_root,
        joinpath(resolved_root, "config"),
        joinpath(resolved_root, "input"),
        joinpath(resolved_root, "diagnostics"),
        joinpath(resolved_root, "fields"),
        joinpath(resolved_root, "checkpoints"),
        joinpath(resolved_root, "logs"),
        joinpath(resolved_root, "plots"),
        joinpath(resolved_root, "tables"),
    )
end

function utc_run_stamp(now = Dates.now(Dates.UTC))
    return Dates.format(now, dateformat"yyyymmddTHHMMSS")
end

function sanitize_run_name(name::AbstractString)
    stripped = strip(String(name))
    isempty(stripped) && return "run"
    sanitized = replace(lowercase(stripped), r"[^a-z0-9._=-]+" => "-")
    sanitized = replace(sanitized, r"-+" => "-")
    sanitized = strip(sanitized, ['-', '.', '_'])
    return isempty(sanitized) ? "run" : sanitized
end

function isolated_run_directory(
    run_root::AbstractString,
    run_name::AbstractString;
    stamp::AbstractString = utc_run_stamp(),
)
    return joinpath(
        abspath(String(run_root)),
        string(String(stamp), "_", sanitize_run_name(run_name)),
    )
end

function run_path_dictionary(paths::RunPaths)
    return Dict{String, Any}(
        "root" => paths.root,
        "config_dir" => paths.config_dir,
        "input_dir" => paths.input_dir,
        "diagnostics_dir" => paths.diagnostics_dir,
        "fields_dir" => paths.fields_dir,
        "checkpoints_dir" => paths.checkpoints_dir,
        "logs_dir" => paths.logs_dir,
        "plots_dir" => paths.plots_dir,
        "tables_dir" => paths.tables_dir,
    )
end

function run_path_directories(paths::RunPaths)
    return (
        paths.root,
        paths.config_dir,
        paths.input_dir,
        paths.diagnostics_dir,
        paths.fields_dir,
        paths.checkpoints_dir,
        paths.logs_dir,
        paths.plots_dir,
        paths.tables_dir,
    )
end

function write_run_layout_metadata(paths::RunPaths)
    metadata = Dict{String, Any}(
        "format" => "DiscoGMPI run layout",
        "version" => 1,
        "created_utc" => string(Dates.now(Dates.UTC)),
        "paths" => run_path_dictionary(paths),
    )
    atomic_toml(joinpath(paths.config_dir, "run_layout.toml"), metadata)
    return joinpath(paths.config_dir, "run_layout.toml")
end

function create_relative_symlink_if_absent(
    target::AbstractString,
    link::AbstractString,
)
    link_path = String(link)
    (ispath(link_path) || islink(link_path)) && return link_path
    mkpath(dirname(link_path))
    relative_target = relpath(String(target), dirname(link_path))
    symlink(relative_target, link_path)
    return link_path
end

function create_run_compatibility_links(paths::RunPaths)
    links = (
        (run_diagnostics_path(paths, "energy.csv"),
         joinpath(paths.root, "energy.csv")),
        (run_diagnostics_path(paths, "quadrature_diagnostics.csv"),
         joinpath(paths.root, "quadrature_diagnostics.csv")),
        (run_config_path(paths, "run_metadata.toml"),
         joinpath(paths.root, "run_metadata.toml")),
        (run_config_path(paths, "partition_metadata.csv"),
         joinpath(paths.root, "partition_metadata.csv")),
    )
    created = String[]
    for (target, link) in links
        push!(created, create_relative_symlink_if_absent(target, link))
    end
    return created
end

function write_resolved_run_config(
    paths::RunPaths,
    configuration::AbstractDict,
)
    metadata = Dict{String, Any}(
        "format" => "DiscoGMPI resolved run configuration",
        "version" => 1,
        "created_utc" => string(Dates.now(Dates.UTC)),
        "configuration" => toml_value(configuration),
    )
    atomic_toml(run_config_path(paths, "resolved_config.toml"), metadata)
    return run_config_path(paths, "resolved_config.toml")
end

function write_run_input_manifest(
    paths::RunPaths,
    inputs::AbstractDict,
)
    metadata = Dict{String, Any}(
        "format" => "DiscoGMPI run input manifest",
        "version" => 1,
        "created_utc" => string(Dates.now(Dates.UTC)),
        "inputs" => toml_value(inputs),
    )
    atomic_toml(run_input_path(paths, "input_manifest.toml"), metadata)
    return run_input_path(paths, "input_manifest.toml")
end

function collectively_write_run_provenance(
    paths::RunPaths,
    comm::MPI.Comm;
    configuration::AbstractDict = Dict{String, Any}(),
    inputs::AbstractDict = Dict{String, Any}(),
)
    collective_root_action(comm, "Run provenance writing") do
        isempty(configuration) || write_resolved_run_config(paths, configuration)
        isempty(inputs) || write_run_input_manifest(paths, inputs)
    end
    MPI.Barrier(comm)
    return (
        run_config_path(paths, "resolved_config.toml"),
        run_input_path(paths, "input_manifest.toml"),
    )
end

function write_run_status(
    paths::RunPaths,
    status::AbstractString;
    values::AbstractDict = Dict{String, Any}(),
)
    metadata = Dict{String, Any}(
        "format" => "DiscoGMPI run status",
        "version" => 1,
        "updated_utc" => string(Dates.now(Dates.UTC)),
        "status" => String(status),
    )
    for (key, value) in values
        metadata[string(key)] = toml_value(value)
    end
    atomic_toml(joinpath(paths.root, "status.toml"), metadata)
    return joinpath(paths.root, "status.toml")
end

function collectively_write_run_status(
    paths::RunPaths,
    comm::MPI.Comm,
    status::AbstractString;
    values::AbstractDict = Dict{String, Any}(),
)
    status_path = joinpath(paths.root, "status.toml")
    collective_root_action(comm, "Run status writing") do
        write_run_status(paths, status; values = values)
    end
    MPI.Barrier(comm)
    return status_path
end

function prepare_run_directory(
    root::AbstractString,
)
    paths = RunPaths(root)
    for directory in run_path_directories(paths)
        mkpath(directory)
    end
    create_run_compatibility_links(paths)
    write_run_layout_metadata(paths)
    write_run_status(paths, "prepared")
    return paths
end

function prepare_run_directory(
    root::AbstractString,
    comm::MPI.Comm,
)
    paths = RunPaths(root)
    collective_root_action(comm, "Run directory creation") do
        for directory in run_path_directories(paths)
            mkpath(directory)
        end
        create_run_compatibility_links(paths)
        write_run_layout_metadata(paths)
        write_run_status(paths, "prepared")
    end
    MPI.Barrier(comm)
    return paths
end

run_config_path(paths::RunPaths, filename::AbstractString) =
    joinpath(paths.config_dir, String(filename))

run_input_path(paths::RunPaths, filename::AbstractString) =
    joinpath(paths.input_dir, String(filename))

run_diagnostics_path(paths::RunPaths, filename::AbstractString) =
    joinpath(paths.diagnostics_dir, String(filename))

run_log_path(paths::RunPaths, filename::AbstractString) =
    joinpath(paths.logs_dir, String(filename))

run_plot_path(paths::RunPaths, filename::AbstractString) =
    joinpath(paths.plots_dir, String(filename))

run_table_path(paths::RunPaths, filename::AbstractString) =
    joinpath(paths.tables_dir, String(filename))

function default_checkpoint_dir(
    output_dir::AbstractString,
    checkpoint_dir::AbstractString,
    paths::RunPaths,
)
    default_dir = abspath(joinpath(String(output_dir), "checkpoints"))
    selected_dir = abspath(String(checkpoint_dir))
    return selected_dir == default_dir ? paths.checkpoints_dir : selected_dir
end
