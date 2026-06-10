#!/usr/bin/env julia

using DiscoGMPI
using LinearAlgebra
using WriteVTK: MeshCell, VTKCellTypes, vtk_grid

const PEC_BOUNDARY_ID = 10
const TET_FACE_NODE_IDS = (
    (2, 3, 4),
    (1, 4, 3),
    (1, 2, 4),
    (1, 3, 2),
)
const REF_TET_VERTEX_COORDS = (
    (-1.0, -1.0, -1.0),
    ( 1.0, -1.0, -1.0),
    (-1.0,  1.0, -1.0),
    (-1.0, -1.0,  1.0),
)

function node_id(i::Int, j::Int, k::Int, cells_per_axis::Int)
    n = cells_per_axis + 1
    return 1 + i + n * (j + n * k)
end

function build_cube_points(cells_per_axis::Int)
    npoints_1d = cells_per_axis + 1
    coords = collect(range(-1.0, 1.0; length = npoints_1d))
    points = zeros(Float64, 3, npoints_1d^3)

    for k in 0:cells_per_axis
        for j in 0:cells_per_axis
            for i in 0:cells_per_axis
                id = node_id(i, j, k, cells_per_axis)
                points[1, id] = coords[i + 1]
                points[2, id] = coords[j + 1]
                points[3, id] = coords[k + 1]
            end
        end
    end

    return points
end

function build_cube_tets(cells_per_axis::Int)
    if cells_per_axis < 1
        error("cells_per_axis must be at least 1.")
    end

    tets = NTuple{4, Int}[]

    for k in 0:(cells_per_axis - 1)
        for j in 0:(cells_per_axis - 1)
            for i in 0:(cells_per_axis - 1)
                v000 = node_id(i,     j,     k,     cells_per_axis)
                v100 = node_id(i + 1, j,     k,     cells_per_axis)
                v010 = node_id(i,     j + 1, k,     cells_per_axis)
                v110 = node_id(i + 1, j + 1, k,     cells_per_axis)
                v001 = node_id(i,     j,     k + 1, cells_per_axis)
                v101 = node_id(i + 1, j,     k + 1, cells_per_axis)
                v011 = node_id(i,     j + 1, k + 1, cells_per_axis)
                v111 = node_id(i + 1, j + 1, k + 1, cells_per_axis)

                push!(tets, (v000, v100, v110, v111))
                push!(tets, (v000, v110, v010, v111))
                push!(tets, (v000, v010, v011, v111))
                push!(tets, (v000, v011, v001, v111))
                push!(tets, (v000, v001, v101, v111))
                push!(tets, (v000, v101, v100, v111))
            end
        end
    end

    return reduce(hcat, collect.(tets))
end

function sorted_face_key(nodes::NTuple{3, Int})
    s = sort(collect(nodes))
    return (s[1], s[2], s[3])
end

function build_boundary_tris(tets::Matrix{Int})
    counts = Dict{NTuple{3, Int}, Int}()
    oriented_nodes = Dict{NTuple{3, Int}, NTuple{3, Int}}()

    for e in axes(tets, 2)
        tet = tets[:, e]

        for local_face in 1:4
            ids = TET_FACE_NODE_IDS[local_face]
            nodes = (tet[ids[1]], tet[ids[2]], tet[ids[3]])
            key = sorted_face_key(nodes)

            counts[key] = get(counts, key, 0) + 1
            oriented_nodes[key] = nodes
        end
    end

    tris = NTuple{3, Int}[]

    for (key, count) in counts
        if count == 1
            push!(tris, oriented_nodes[key])
        elseif count != 2
            error("Non-manifold face $key appears $count times.")
        end
    end

    return reduce(hcat, collect.(tris))
end

function build_pec_cube_mesh(cells_per_axis::Int; pec_boundary_id::Int = PEC_BOUNDARY_ID)
    points = build_cube_points(cells_per_axis)
    tets = build_cube_tets(cells_per_axis)
    tris = build_boundary_tris(tets)

    ntets = size(tets, 2)
    ntris = size(tris, 2)

    tet_cell_ids = collect(1:ntets)
    tri_cell_ids = collect((ntets + 1):(ntets + ntris))

    boundary_id = zeros(Int, ntets + ntris)
    boundary_id[tri_cell_ids] .= pec_boundary_id

    mesh = RawVTUMesh(
        points,
        tets,
        tris,
        tet_cell_ids,
        tri_cell_ids,
        Dict{String, Any}("boundary_id" => boundary_id),
    )

    check_mesh_consistency(mesh)

    return mesh
end

function build_dg_discretization(mesh::RawVTUMesh, order::Int)
    dg = DGDiscretization(mesh, order)

    print_topology_summary(mesh, dg.topology)
    print_geometry_summary(dg.geometry)
    print_mapping_summary(mesh, dg.geometry, dg.mappings)
    print_reference_tet_summary(dg.ref)
    print_reference_face_operator_summary(dg.ref, dg.fops)

    return dg
end

function stationary_maxwell_field(mesh::RawVTUMesh, ref::ReferenceTet)
    ω = √3 * π
    electric = (x, y, z) -> (-1.0 * cos(π * x) * sin(π * y) * sin(π * z), 0.0, sin(π * x) * sin(π * y) * cos(π * z))
    magnetic = (x, y, z) -> ((-π/ω) * sin(π * x) * cos(π * y) * cos(π * z), (2.0 * π/ω) * cos(π * x) * sin(π * y) * cos(π * z), (-π/ω) * cos(π * x) * cos(π * y) * sin(π * z))

    return interpolate_maxwell_field(mesh, ref, electric, magnetic)
end

function copy_maxwell_field(U::MaxwellField)
    return MaxwellField(
        copy(U.Ex),
        copy(U.Ey),
        copy(U.Ez),
        copy(U.Hx),
        copy(U.Hy),
        copy(U.Hz),
    )
end

function similar_rhs(U::MaxwellField)
    return MaxwellRHS(
        similar(U.Ex),
        similar(U.Ey),
        similar(U.Ez),
        similar(U.Hx),
        similar(U.Hy),
        similar(U.Hz),
    )
end

function max_abs_rhs(rhs::MaxwellRHS)
    return maximum((
        maximum(abs.(rhs.rhsEx)),
        maximum(abs.(rhs.rhsEy)),
        maximum(abs.(rhs.rhsEz)),
        maximum(abs.(rhs.rhsHx)),
        maximum(abs.(rhs.rhsHy)),
        maximum(abs.(rhs.rhsHz)),
    ))
end

function max_abs_field_difference(U::MaxwellField, V::MaxwellField)
    return maximum((
        maximum(abs.(U.Ex .- V.Ex)),
        maximum(abs.(U.Ey .- V.Ey)),
        maximum(abs.(U.Ez .- V.Ez)),
        maximum(abs.(U.Hx .- V.Hx)),
        maximum(abs.(U.Hy .- V.Hy)),
        maximum(abs.(U.Hz .- V.Hz)),
    ))
end

function max_tangential_electric_field(U::MaxwellField, flux_faces)
    max_tangential = 0.0

    for face in flux_faces.boundary
        tr = face.trace
        n = face.normal

        for node in tr.nodes
            ex = U.Ex[node, tr.elem]
            ey = U.Ey[node, tr.elem]
            ez = U.Ez[node, tr.elem]

            cx = n[2] * ez - n[3] * ey
            cy = n[3] * ex - n[1] * ez
            cz = n[1] * ey - n[2] * ex

            max_tangential = max(max_tangential, sqrt(cx * cx + cy * cy + cz * cz))
        end
    end

    return max_tangential
end

function reference_vertex_node_ids(ref::ReferenceTet; tol::Float64 = 1e-12)
    ids = Vector{Int}(undef, 4)

    for v in 1:4
        rv, sv, tv = REF_TET_VERTEX_COORDS[v]
        matches = findall(
            i -> abs(ref.r[i] - rv) < tol &&
                 abs(ref.s[i] - sv) < tol &&
                 abs(ref.t[i] - tv) < tol,
            1:ref.Np,
        )

        if length(matches) != 1
            error(
                "Could not identify reference vertex $v in DG nodes. " *
                "VTU point-field output requires one node at each tetrahedron vertex.",
            )
        end

        ids[v] = only(matches)
    end

    return ids
end

function vtk_output_basename(path::AbstractString)
    return lowercase(splitext(path)[2]) == ".vtu" ? splitext(path)[1] : String(path)
end

function ensure_output_directory(path::AbstractString)
    dir = dirname(path)

    if !isempty(dir) && dir != "."
        mkpath(dir)
    end

    return nothing
end

function save_cell_centered_fields_vtu(
    output_path::AbstractString,
    mesh::RawVTUMesh,
    U::MaxwellField;
    time::Float64,
)
    basename = vtk_output_basename(output_path)
    ensure_output_directory(basename)

    ne = size(mesh.tets, 2)
    cells = [
        MeshCell(
            VTKCellTypes.VTK_TETRA,
            (mesh.tets[1, e], mesh.tets[2, e], mesh.tets[3, e], mesh.tets[4, e]),
        )
        for e in 1:ne
    ]

    electric = zeros(Float64, 3, ne)
    magnetic = zeros(Float64, 3, ne)

    for e in 1:ne
        electric[:, e] .= (U.Ex[1, e], U.Ey[1, e], U.Ez[1, e])
        magnetic[:, e] .= (U.Hx[1, e], U.Hy[1, e], U.Hz[1, e])
    end

    electric_norm = vec(sqrt.(sum(abs2, electric; dims = 1)))
    magnetic_norm = vec(sqrt.(sum(abs2, magnetic; dims = 1)))

    return vtk_grid(basename, mesh.points, cells; append = false, compress = false) do vtk
        vtk["ElectricField"] = electric
        vtk["MagneticField"] = magnetic
        vtk["ElectricFieldMagnitude"] = electric_norm
        vtk["MagneticFieldMagnitude"] = magnetic_norm
        vtk["ElementId"] = collect(1:ne)
        vtk["TimeValue"] = time
    end
end

function save_discontinuous_vertex_fields_vtu(
    output_path::AbstractString,
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    U::MaxwellField;
    time::Float64,
)
    basename = vtk_output_basename(output_path)
    ensure_output_directory(basename)

    vertex_ids = reference_vertex_node_ids(ref)
    ne = size(mesh.tets, 2)
    points = zeros(Float64, 3, 4 * ne)
    electric = zeros(Float64, 3, 4 * ne)
    magnetic = zeros(Float64, 3, 4 * ne)
    cells = Vector{MeshCell}(undef, ne)

    for e in 1:ne
        cell_nodes = ntuple(i -> 4 * (e - 1) + i, 4)
        cells[e] = MeshCell(VTKCellTypes.VTK_TETRA, cell_nodes)

        for v in 1:4
            vtk_node = cell_nodes[v]
            mesh_node = mesh.tets[v, e]
            local_node = vertex_ids[v]

            points[:, vtk_node] .= mesh.points[:, mesh_node]
            electric[:, vtk_node] .= (
                U.Ex[local_node, e],
                U.Ey[local_node, e],
                U.Ez[local_node, e],
            )
            magnetic[:, vtk_node] .= (
                U.Hx[local_node, e],
                U.Hy[local_node, e],
                U.Hz[local_node, e],
            )
        end
    end

    electric_norm = vec(sqrt.(sum(abs2, electric; dims = 1)))
    magnetic_norm = vec(sqrt.(sum(abs2, magnetic; dims = 1)))

    return vtk_grid(basename, points, cells; append = false, compress = false) do vtk
        vtk["ElectricField"] = electric
        vtk["MagneticField"] = magnetic
        vtk["ElectricFieldMagnitude"] = electric_norm
        vtk["MagneticFieldMagnitude"] = magnetic_norm
        vtk["ElementId"] = collect(1:ne)
        vtk["PolynomialOrder"] = ref.N
        vtk["TimeValue"] = time
    end
end

function save_final_fields_vtu(
    output_path::AbstractString,
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    U::MaxwellField;
    time::Float64,
)
    if ref.N == 0
        return save_cell_centered_fields_vtu(output_path, mesh, U; time = time)
    else
        return save_discontinuous_vertex_fields_vtu(output_path, mesh, ref, U; time = time)
    end
end

function parse_args(args)
    cells_per_axis = 2
    order = 1
    nsteps = 5
    rk_order = 4
    cfl = 0.05
    output = joinpath("output", "stationary_pec_cube_final")

    for arg in args
        if arg == "--help" || arg == "-h"
            print_usage()
            exit(0)
        elseif startswith(arg, "--cells=")
            cells_per_axis = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--order=")
            order = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--nsteps=")
            nsteps = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--rk=")
            rk_order = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--cfl=")
            cfl = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--output=")
            output = split(arg, "=", limit = 2)[2]
        else
            error("Unknown argument: $arg. Run with --help for usage.")
        end
    end

    return (
        cells_per_axis = cells_per_axis,
        order = order,
        nsteps = nsteps,
        rk_order = rk_order,
        cfl = cfl,
        output = output,
    )
end

function print_usage()
    println("Stationary Maxwell driver in [-1,1]^3 with PEC walls")
    println()
    println("Usage:")
    println("  julia --project=. examples/solver/stationary_pec_cube.jl [options]")
    println()
    println("Options:")
    println("  --cells=N     Structured cube cells per axis before tet split. Default: 2")
    println("  --order=N     DG polynomial order. Default: 1")
    println("  --nsteps=N    RK steps to run. Default: 5")
    println("  --rk=N        Explicit RK order 1, 2, 3, 4, or 5. Default: 4")
    println("  --cfl=X       CFL factor used for dt estimate. Default: 0.05")
    println("  --output=PATH Final VTU output path. Default: output/stationary_pec_cube_final.vtu")
end

function main(args = ARGS)
    config = parse_args(args)

    mesh = build_pec_cube_mesh(config.cells_per_axis)
    dg = build_dg_discretization(mesh, config.order)

    registry = MaxwellBoundaryRegistry(
        Dict(PEC_BOUNDARY_ID => DiscoGMPI.MaxwellBC_PEC),
    )

    formulation = HesthavenWarburtonFormulation(MaxwellFlux_Central)

    U = stationary_maxwell_field(mesh, dg.ref)
    U0 = copy_maxwell_field(U)

    rhs = similar_rhs(U)
    maxwell_rhs!(
        rhs,
        U,
        dg,
        registry,
        formulation,
    )

    rhs0 = max_abs_rhs(rhs)
    energy0 = maxwell_energy(U, dg.ref, dg.mappings)
    tangential_e0 = max_tangential_electric_field(U, dg.flux_faces)

    dt, sizes = estimate_maxwell_dt(
        mesh,
        dg.geometry,
        dg.ref;
        CFL = config.cfl,
    )

    println("Stationary PEC cube Maxwell run")
    println("--------------------------------")
    println("Domain:              [-1, 1]^3")
    println("Boundary condition:  PEC on all exterior faces, boundary_id = ", PEC_BOUNDARY_ID)
    println("Initial field:       PEC cavity mode with omega = sqrt(3) * pi")
    println("                     E = (-cos(pi*x)sin(pi*y)sin(pi*z), 0, sin(pi*x)sin(pi*y)cos(pi*z))")
    println("                     H = (-(pi/omega)sin(pi*x)cos(pi*y)cos(pi*z), (2pi/omega)cos(pi*x)sin(pi*y)cos(pi*z), -(pi/omega)cos(pi*x)cos(pi*y)sin(pi*z))")
    println("Cells per axis:      ", config.cells_per_axis)
    println("Tetrahedra:          ", size(mesh.tets, 2))
    println("Boundary triangles:  ", size(mesh.tris, 2))
    println("DG order:            ", config.order)
    println("Initial energy:      ", energy0.total)
    println("Initial max |rhs|:   ", rhs0)
    println("Initial max |n x E|: ", tangential_e0)
    println("Estimated dt:        ", dt)
    println()

    run_maxwell_time_steps!(
        U,
        dg,
        registry,
        formulation;
        rk_order = config.rk_order,
        dt = dt,
        nsteps = config.nsteps,
        energy_every = max(1, config.nsteps),
    )

    rhs_final = similar_rhs(U)
    maxwell_rhs!(
        rhs_final,
        U,
        dg,
        registry,
        formulation,
    )

    energy1 = maxwell_energy(U, dg.ref, dg.mappings)
    field_delta = max_abs_field_difference(U, U0)
    rhs1 = max_abs_rhs(rhs_final)
    tangential_e1 = max_tangential_electric_field(U, dg.flux_faces)
    final_time = dt * config.nsteps
    saved_files = save_final_fields_vtu(
        config.output,
        mesh,
        dg.ref,
        U;
        time = final_time,
    )

    println()
    println("Stationary diagnostics")
    println("----------------------")
    println("Final energy:         ", energy1.total)
    println("Energy drift:         ", energy1.total - energy0.total)
    println("Max field change:     ", field_delta)
    println("Final max |rhs|:      ", rhs1)
    println("Final max |n x E|:    ", tangential_e1)
    println("Saved final fields:   ", join(saved_files, ", "))

    if field_delta < 1e-12 && rhs1 < 1e-12 && tangential_e1 < 1e-12
        println("Result: PASS")
    else
        println("Result: CHECK TOLERANCES")
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
