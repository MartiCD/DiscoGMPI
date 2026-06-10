# Standalone script section
# -------------------------------------------------------------------------

using DiscoGMPI
using LinearAlgebra

function reference_tet_mass_matrix_quadratic()
    return (1 / 315) * [
         6   1   1   1  -4  -4  -4  -6  -6  -6;
         1   6   1   1  -4  -6  -6  -4  -4  -6;
         1   1   6   1  -6  -4  -6  -4  -6  -4;
         1   1   1   6  -6  -6  -4  -6  -4  -4;
        -4  -4  -6  -6  32  16  16  16  16   8;
        -4  -6  -4  -6  16  32  16  16   8  16;
        -4  -6  -6  -4  16  16  32   8  16  16;
        -6  -4  -4  -6  16  16   8  32  16  16;
        -6  -4  -6  -4  16   8  16  16  32  16;
        -6  -6  -4  -4   8  16  16  16  16  32
    ]
end

function main()
    if length(ARGS) < 1
        println("Usage:")
        println("  julia --project=. examples/solver/read_vtu_mesh.jl mesh.vtu")
        println()
        println("Example:")
        println("  julia --project=. examples/solver/read_vtu_mesh.jl box_with_pec_sphere.vtu")
        return
    end

    filename = ARGS[1]

    println("Reading VTU mesh:")
    println("  ", filename)
    println()

    mesh = read_vtu_mesh(filename)

    print_mesh_summary(mesh)

    println()
    println("Connectivity preview")
    println("--------------------")

    if size(mesh.tets, 2) > 0
        println("First tetrahedron:")
        println("  ", mesh.tets[:, 1])
    else
        println("No tetrahedra found.")
    end

    if size(mesh.tris, 2) > 0
        println("First surface triangle:")
        println("  ", mesh.tris[:, 1])
    else
        println("No surface triangles found.")
    end

    println()
    println("Cell-data previews")
    println("------------------")

    for key in sort(collect(keys(mesh.cell_data)))
        data = mesh.cell_data[key]

        println("Cell data: ", key)

        if length(data) == 0
            println("  empty")
        else
            npreview = min(10, length(data))
            println("  first values: ", data[1:npreview])

            try
                println("  unique values: ", unique(data))
            catch
                println("  unique values: unavailable")
            end
        end

        println()
    end

    # Common DG-FEM tag names.
    # These are optional and only printed if present in the VTU file.

    if haskey(mesh.cell_data, "region_id")
        tet_region_ids = tet_data(mesh, "region_id")

        println("Tetrahedral region ids")
        println("----------------------")
        println("unique(tet region_id) = ", sort(unique(tet_region_ids)))
        println()
    end

    if haskey(mesh.cell_data, "boundary_id")
        if size(mesh.tris, 2) > 0
            tri_boundary_ids = tri_data(mesh, "boundary_id")

            println("Triangle boundary ids")
            println("---------------------")
            println("unique(tri boundary_id) = ", sort(unique(tri_boundary_ids)))
            println()
        else
            println("boundary_id exists, but no triangular surface cells were found.")
            println()
        end
    end

    println(mesh.tets[:,1])


    # DG Topology
    topology = build_dg_topology(mesh)
    println()
    print_topology_summary(mesh, topology)

    # DG Geometry
    geometry = build_dg_geometry(mesh,topology)
    println()
    print_geometry_summary(geometry)

    # DG Metrics
    mappings = build_reference_mappings(mesh)
    println()
    print_mapping_summary(mesh, geometry, mappings)

    # DG Nodal basis
    N = 2

    println()
    print_orthonormal_basis_summary(N)

    ref = build_reference_tet(N)
    println()
    print_reference_tet_summary(ref)

    tri = build_reference_tri(N)
    println()
    print_reference_tri_summary(tri)


    fops = build_reference_face_operators(ref)
    println()
    print_reference_face_operator_summary(ref, fops)

    # println("Mass matrix in the reference tet:")
    # println(ref.M)

    M_exact = reference_tet_mass_matrix_quadratic()
    println("max absolute error ", maximum(abs.(ref.M .- M_exact)))
    println("Integral partition of unity squared: ", isapprox(sum(ref.M), 4. / 3.; rtol=1e-12, atol=1e-14))

    # for N in 1:6
    #     ref = build_reference_tet(N)
    #     println("N = $N, Np = $(ref.Np), cond(V) = $(cond(ref.V))")
    # end

    physops = build_physical_operators(ref, mappings)
    println()
    print_physical_operator_summary(physops)
    
    println()
    test_physical_derivatives_linear(mesh, ref, physops)

    trace_maps = build_dg_trace_maps(
        mesh,
        ref,
        topology,
        fops;
        tol = 1e-9,
    )

    println()
    print_trace_map_summary(trace_maps)

    println()
    test_trace_map_geometry(mesh, ref, trace_maps)

    println()
    test_trace_maps_linear_function(mesh, ref, trace_maps)

    trace_maps = build_dg_trace_maps(
        mesh,
        ref,
        topology,
        fops;
        tol = 1e-9,
    )

    println()
    print_trace_map_summary(trace_maps)

    println()
    test_trace_map_geometry(mesh, ref, trace_maps)

    println()
    test_trace_maps_linear_function(mesh, ref, trace_maps)

    flux_faces = build_dg_flux_faces(trace_maps, geometry)

    println()
    print_flux_face_summary(flux_faces)

    println()
    test_interior_flux_face_normals(geometry, flux_faces)

    println()
    test_boundary_box_normals(flux_faces)

    println()
    test_pec_sphere_normals(
        flux_faces;
        sphere_center = (0.5, 0.5, 0.5),
        pec_boundary_id = 10,
    )

    println()
    print_boundary_area_reference_check(
        flux_faces;
        sphere_radius = 0.2,
    )

    println()
    test_scalar_advection_volume_operator(mesh, ref, physops)

    println()
    test_scalar_advection_interior_surface_operator(
        mesh,
        ref,
        fops,
        flux_faces,
    )

    println()
    test_scalar_advection_boundary_surface_operator(
        mesh,
        ref,
        fops,
        flux_faces,
    )

    println()
    test_scalar_advection_full_surface_operator(
        mesh,
        ref,
        fops,
        flux_faces,
    )

    println()
    test_maxwell_pec_boundary_reflection(
        mesh,
        ref,
        flux_faces;
        pec_boundary_id = 10,
    )

    println()
    test_maxwell_volume_operator(
        mesh,
        ref,
        physops,
    )

    println()
    test_maxwell_interior_surface_operator(
        mesh,
        ref,
        fops,
        mappings,
        flux_faces,
    )

    println()
    test_maxwell_pec_local_flux_zero(
        flux_faces,
        ref;
        pec_boundary_id = 10,
    )

    println()
    test_maxwell_pec_boundary_surface_zero_field(
        mesh,
        ref,
        fops,
        mappings,
        flux_faces;
        pec_boundary_id = 10,
    )

    println()
    test_maxwell_pec_boundary_surface_arbitrary_field(
        mesh,
        ref,
        fops,
        mappings,
        flux_faces;
        pec_boundary_id = 10,
    )

    println()
    test_maxwell_rhs_matches_volume_without_boundaries(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
    )

    println()
    test_maxwell_rhs_zero_field_with_pec(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
    )

    println()
    test_maxwell_rhs_linear_field_no_boundaries(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
    )

    println()
    test_maxwell_energy_constant_field(
        mesh,
        ref,
        mappings;
        ε = 1.0,
        μ = 1.0,
    )

    println()
    test_maxwell_energy_rate_zero_field(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
    )

    println()
    test_maxwell_energy_rate_no_boundaries(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
    )


    rk_order = 4

    scheme = explicit_rk_scheme(rk_order)
    println()
    print_rk_scheme_summary(scheme)

    println()
    test_rk_zero_field_step(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces;
        rk_order = rk_order,
        dt = 1e-4,
    )

    println()
    test_rk_one_step_energy_diagnostic(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces;
        rk_order = rk_order,
        dt = 1e-5,
    )

    # Periodic tests
    periodic_specs = default_unit_box_periodic_specs()

    periodic_faces = build_periodic_flux_faces(
        mesh,
        ref,
        flux_faces,
        periodic_specs;
        centroid_tol = 1e-8,
        node_tol = 1e-8,
        area_rtol = 1e-8,
    )

    println()
    print_periodic_flux_face_summary(periodic_faces)

    println()
    test_periodic_flux_face_geometry(mesh, ref, periodic_faces)

    println()
    test_periodic_trace_maps_scalar_function(mesh, ref, periodic_faces)

    println()
    test_maxwell_periodic_rhs_zero_field(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        periodic_faces,
    )

    println()
    test_rk_periodic_zero_field_step(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        periodic_faces;
        rk_order = 4,
        dt = 1e-4,
    )

    println()
    test_rk_periodic_one_step_energy_diagnostic(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        periodic_faces;
        rk_order = 4,
        dt = 1e-5,
    )

    CFL = 0.05
    ε = 1.0
    μ = 1.0

    dt, sizes = estimate_maxwell_dt(
        mesh,
        geometry,
        ref;
        CFL = CFL,
        ε = ε,
        μ = μ,
    )

    println()
    print_element_size_diagnostics(sizes)

    println()
    print_maxwell_dt_estimate(
        dt,
        sizes,
        ref;
        CFL = CFL,
        ε = ε,
        μ = μ,
    )

    println()
    test_periodic_time_marching_with_cfl_dt(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        geometry,
        flux_faces,
        periodic_faces;
        rk_order = 4,
        CFL = 0.05,
        nsteps = 5,
        ε = 1.0,
        μ = 1.0,
    )

    println()
    test_maxwell_upwind_interior_surface_operator(
        mesh,
        ref,
        fops,
        mappings,
        flux_faces,
    )

    println()
    test_maxwell_upwind_periodic_surface_operator(
        mesh,
        ref,
        fops,
        mappings,
        periodic_faces,
    )

    println()
    test_maxwell_upwind_periodic_rhs_zero_field(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        periodic_faces,
    )

    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
