# -------------------------------------------------------------------------
# Shared DG infrastructure
# -------------------------------------------------------------------------

function DGDiscretization(
    mesh::RawVTUMesh,
    order::Int;
    boundary_tag_name::String = "boundary_id",
    trace_tol::Float64 = 1e-10,
    backend::AbstractBackend = SerialBackend(),
)
    topology = build_dg_topology(mesh; boundary_tag_name = boundary_tag_name)
    geometry = build_dg_geometry(mesh, topology)
    mappings = build_reference_mappings(mesh)
    ref = build_reference_tet(order)
    fops = build_reference_face_operators(ref)
    physops = build_physical_operators(ref, mappings)

    trace_maps = build_dg_trace_maps(
        mesh,
        ref,
        topology,
        fops;
        tol = trace_tol,
    )

    flux_faces = build_dg_flux_faces(trace_maps, geometry)

    return DGDiscretization(
        mesh,
        topology,
        geometry,
        mappings,
        ref,
        fops,
        physops,
        trace_maps,
        flux_faces,
        backend,
    )
end
