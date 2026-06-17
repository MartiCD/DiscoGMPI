using MPI
using Printf
using DiscoGMPI
using LinearAlgebra: dot

include(
    joinpath(
        @__DIR__,
        "..",
        "src",
        "solver",
        "kernels",
        "JaskowiecSukumar.jl",
    ),
)

const MPI_WAS_INITIALIZED = MPI.Initialized()
MPI_WAS_INITIALIZED || MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NRANKS = MPI.Comm_size(COMM)

function argument_value(name::String, default)
    prefix = "--$name="
    for argument in ARGS
        startswith(argument, prefix) || continue
        value = split(argument, "="; limit = 2)[2]
        return default isa Int ? parse(Int, value) :
               default isa Float64 ? parse(Float64, value) : value
    end
    return default
end

function node_id(i::Int, j::Int, k::Int, cells::Int)
    nodes_per_axis = cells + 1
    return 1 + i + nodes_per_axis * (j + nodes_per_axis * k)
end

function periodic_cube_mesh(cells::Int)
    nodes_per_axis = cells + 1
    coordinates = collect(range(0.0, 1.0; length = nodes_per_axis))
    points = zeros(Float64, 3, nodes_per_axis^3)

    for k in 0:cells, j in 0:cells, i in 0:cells
        points[:, node_id(i, j, k, cells)] .=
            (coordinates[i + 1], coordinates[j + 1], coordinates[k + 1])
    end

    tetrahedra = NTuple{4, Int}[]
    for k in 0:(cells - 1), j in 0:(cells - 1), i in 0:(cells - 1)
        v000 = node_id(i, j, k, cells)
        v100 = node_id(i + 1, j, k, cells)
        v010 = node_id(i, j + 1, k, cells)
        v110 = node_id(i + 1, j + 1, k, cells)
        v001 = node_id(i, j, k + 1, cells)
        v101 = node_id(i + 1, j, k + 1, cells)
        v011 = node_id(i, j + 1, k + 1, cells)
        v111 = node_id(i + 1, j + 1, k + 1, cells)
        append!(
            tetrahedra,
            (
                (v000, v100, v110, v111),
                (v000, v110, v010, v111),
                (v000, v010, v011, v111),
                (v000, v011, v001, v111),
                (v000, v001, v101, v111),
                (v000, v101, v100, v111),
            ),
        )
    end
    tets = reduce(hcat, collect.(tetrahedra))

    counts = Dict{NTuple{3, Int}, Int}()
    oriented = Dict{NTuple{3, Int}, NTuple{3, Int}}()
    for elem in axes(tets, 2), face in DiscoGMPI.TET_FACES
        nodes = (
            tets[face[1], elem],
            tets[face[2], elem],
            tets[face[3], elem],
        )
        key = Tuple(sort(collect(nodes)))
        counts[key] = get(counts, key, 0) + 1
        oriented[key] = nodes
    end
    boundary_faces = [
        oriented[key] for (key, count) in counts if count == 1
    ]
    tris = reduce(hcat, collect.(boundary_faces))
    ntets = size(tets, 2)
    ntris = size(tris, 2)
    boundary_id = zeros(Int, ntets + ntris)
    material_id = zeros(Int, ntets + ntris)

    for tri in axes(tris, 2)
        centroid = sum(points[:, tris[:, tri]]; dims = 2)[:, 1] / 3
        boundary_id[ntets + tri] =
            abs(centroid[1]) < 1e-12 ? 1 :
            abs(centroid[1] - 1.0) < 1e-12 ? 2 :
            abs(centroid[2]) < 1e-12 ? 3 :
            abs(centroid[2] - 1.0) < 1e-12 ? 4 :
            abs(centroid[3]) < 1e-12 ? 5 : 6
    end
    for elem in 1:ntets
        centroid_x = sum(@view points[1, tets[:, elem]]) / 4
        material_id[elem] = centroid_x < 0.5 ? 1 : 2
    end

    return RawVTUMesh(
        points,
        tets,
        tris,
        collect(1:ntets),
        collect((ntets + 1):(ntets + ntris)),
        Dict{String, Any}(
            "boundary_id" => boundary_id,
            "material_id" => material_id,
        ),
    )
end

function exact_fields(
    solution::Symbol,
    time::Float64;
    epsilon::Float64,
    permeability::Float64,
)
    wave_number = 2pi
    impedance = sqrt(permeability / epsilon)
    frequency = wave_number / sqrt(epsilon * permeability)

    if solution == :traveling
        E = (x, y, z) -> (
            0.0,
            cos(wave_number * x - frequency * time),
            0.0,
        )
        H = (x, y, z) -> (
            0.0,
            0.0,
            cos(wave_number * x - frequency * time) / impedance,
        )
    elseif solution == :standing
        E = (x, y, z) -> (
            0.0,
            cos(wave_number * x) * cos(frequency * time),
            0.0,
        )
        H = (x, y, z) -> (
            0.0,
            0.0,
            sin(wave_number * x) * sin(frequency * time) / impedance,
        )
    else
        error("Unknown analytical solution $solution.")
    end

    return E, H
end

function distributed_l2_errors(
    U::MaxwellField,
    distributed_dg::DistributedDGDiscretization,
    time::Float64,
    solution::Symbol;
    epsilon::Float64,
    permeability::Float64,
)
    Eexact, Hexact = exact_fields(
        solution,
        time;
        epsilon = epsilon,
        permeability = permeability,
    )
    dg = distributed_dg.dg
    local_sums = zeros(Float64, 3)
    cubature_order = min(20, max(2, 2 * dg.ref.N + 4))
    cubature_points, cubature_weights, number_cubature_points =
        get_JaskowiecSukumar_cubature(cubature_order)
    rq = collect(@view cubature_points[:, 1])
    sq = collect(@view cubature_points[:, 2])
    tq = collect(@view cubature_points[:, 3])
    modal_values =
        DiscoGMPI.orthonormal_vandermonde_tet(rq, sq, tq, dg.ref.basis)
    interpolation = modal_values * dg.ref.invV

    for elem in distributed_dg.distributed_mesh.partition.owned
        tet_nodes = dg.mesh.tets[:, elem]
        J = dg.mappings.tet_mappings[elem].absdetJ

        for q in 1:number_cubature_points
            x, y, z = DiscoGMPI.map_to_physical(
                dg.mesh.points,
                tet_nodes,
                cubature_points[q, 1],
                cubature_points[q, 2],
                cubature_points[q, 3],
            )
            exact_E = Eexact(x, y, z)
            exact_H = Hexact(x, y, z)
            row = @view interpolation[q, :]
            numerical = (
                dot(row, @view U.Ex[:, elem]),
                dot(row, @view U.Ey[:, elem]),
                dot(row, @view U.Ez[:, elem]),
                dot(row, @view U.Hx[:, elem]),
                dot(row, @view U.Hy[:, elem]),
                dot(row, @view U.Hz[:, elem]),
            )
            exact = (exact_E..., exact_H...)
            electric_error = sum(
                (numerical[component] - exact[component])^2
                for component in 1:3
            )
            magnetic_error = sum(
                (numerical[component] - exact[component])^2
                for component in 4:6
            )
            numerical_energy =
                0.5 * epsilon * sum(abs2, numerical[1:3]) +
                0.5 * permeability * sum(abs2, numerical[4:6])
            exact_energy =
                0.5 * epsilon * sum(abs2, exact[1:3]) +
                0.5 * permeability * sum(abs2, exact[4:6])
            weight = J * cubature_weights[q]
            local_sums[1] += weight * electric_error
            local_sums[2] += weight * magnetic_error
            local_sums[3] += weight * (numerical_energy - exact_energy)^2
        end
    end

    global_errors = MPI.Allreduce(local_sums, +, distributed_dg.comm)
    return sqrt.(max.(global_errors, 0.0))
end

function write_rows(path::String, rows)
    open(path, "w") do io
        println(
            io,
            "solution,step,time,l2_E,l2_H,l2_energy,total_energy,energy_drift," *
            "energy_exact,energy_error,electric_charge,magnetic_charge," *
            "px,py,pz,px_exact,px_error,lx,ly,lz,ly_exact,lz_exact",
        )
        for row in rows
            println(io, join(row, ","))
        end
    end
end

function run_solution(
    solution::Symbol,
    distributed_dg,
    periodic,
    registry,
    materials;
    epsilon::Float64,
    permeability::Float64,
    dt::Float64,
    steps::Int,
)
    E0, H0 = exact_fields(
        solution,
        0.0;
        epsilon = epsilon,
        permeability = permeability,
    )
    U = interpolate_maxwell_field(distributed_dg, E0, H0)
    formulation = PoissonBracketFormulation()
    scheme = explicit_partitioned_symplectic_rk_scheme(
        2;
        first_partition = :H,
    )
    work = MaxwellPartitionedRKWorkspace(U, scheme)
    initial =
        distributed_maxwell_invariants(U, distributed_dg, materials)
    exact_energy =
        solution == :traveling ? 0.5 * epsilon : 0.25 * epsilon
    exact_px =
        solution == :traveling ?
        0.5 * epsilon * sqrt(epsilon * permeability) : 0.0
    exact_ly = solution == :traveling ? 0.5 * exact_px : 0.0
    exact_lz = -exact_ly
    rows = Vector{Vector{Any}}()

    for step in 0:steps
        time = step * dt
        errors = distributed_l2_errors(
            U,
            distributed_dg,
            time,
            solution;
            epsilon = epsilon,
            permeability = permeability,
        )
        diagnostics =
            distributed_maxwell_invariants(U, distributed_dg, materials)
        drift =
            (diagnostics.energy.total - initial.energy.total) /
            max(initial.energy.total, eps(Float64))

        if RANK == 0
            push!(
                rows,
                Any[
                    solution,
                    step,
                    time,
                    errors...,
                    diagnostics.energy.total,
                    drift,
                    exact_energy,
                    diagnostics.energy.total - exact_energy,
                    diagnostics.electric_charge,
                    diagnostics.magnetic_charge,
                    diagnostics.linear_momentum...,
                    exact_px,
                    diagnostics.linear_momentum[1] - exact_px,
                    diagnostics.angular_momentum...,
                    exact_ly,
                    exact_lz,
                ],
            )
        end

        step == steps && break
        distributed_periodic_partitioned_symplectic_rk_step!(
            U,
            work,
            scheme,
            dt,
            distributed_dg,
            periodic,
            registry,
            formulation,
            materials,
        )
    end

    return rows
end

cells = argument_value("cells", 2)
order = argument_value("order", 2)
steps = argument_value("steps", 20)
dt = argument_value("dt", 0.002)
output = abspath(argument_value("output", "output/distributed_periodic_validation"))
epsilon = 1.0
permeability = 1.0

global_mesh = RANK == 0 ? periodic_cube_mesh(cells) : nothing
partition = if RANK == 0
    mod.(0:(size(global_mesh.tets, 2) - 1), NRANKS)
else
    nothing
end
distributed_dg = build_distributed_dg_from_root(
    global_mesh,
    partition,
    order;
    comm = COMM,
    root = 0,
    material_tag_name = "material_id",
)
materials = maxwell_element_materials(
    distributed_dg,
    Dict(
        1 => MaxwellMaterial(epsilon, permeability),
        2 => MaxwellMaterial(epsilon, permeability),
    ),
)
periodic = build_distributed_periodic_maxwell_exchange(
    distributed_dg;
    materials = materials,
)
registry =
    MaxwellBoundaryRegistry(Dict(id => MaxwellBC_None for id in 1:6))

rows = vcat(
    run_solution(
        :traveling,
        distributed_dg,
        periodic,
        registry,
        materials;
        epsilon = epsilon,
        permeability = permeability,
        dt = dt,
        steps = steps,
    ),
    run_solution(
        :standing,
        distributed_dg,
        periodic,
        registry,
        materials;
        epsilon = epsilon,
        permeability = permeability,
        dt = dt,
        steps = steps,
    ),
)

if RANK == 0
    mkpath(output)
    path = joinpath(output, "periodic_analytical_validation.csv")
    write_rows(path, rows)
    @printf(
        "Wrote %d analytical validation rows to %s\n",
        length(rows),
        path,
    )
end

MPI.Barrier(COMM)
MPI_WAS_INITIALIZED || MPI.Finalize()
