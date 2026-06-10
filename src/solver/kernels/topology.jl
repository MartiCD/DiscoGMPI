# -------------------------------------------------------------------------
# DG topology structs
# -------------------------------------------------------------------------


function sorted_face_key(nodes::NTuple{3, Int})
    return Tuple(sort(collect(nodes)))
end

function build_surface_tag_dict(mesh::RawVTUMesh, tag_name::String = "boundary_id")
    tri_tags = tri_data(mesh, tag_name)

    surface_tags = Dict{NTuple{3, Int}, Int}()

    for i in axes(mesh.tris, 2)
        tri = (
            mesh.tris[1, i],
            mesh.tris[2, i],
            mesh.tris[3, i],
        )

        key = sorted_face_key(tri)

        if haskey(surface_tags, key)
            error("Duplicate tagged surface triangle detected: $key")
        end

        surface_tags[key] = Int(tri_tags[i])
    end

    return surface_tags
end

function build_dg_topology(mesh::RawVTUMesh; boundary_tag_name::String = "boundary_id")
    surface_tags = build_surface_tag_dict(mesh, boundary_tag_name)

    face_map = Dict{NTuple{3, Int}, Vector{FaceRef}}()

    ntets = size(mesh.tets, 2)

    for e in 1:ntets
        tet = mesh.tets[:, e]

        for lf in 1:4
            local_nodes = TET_FACES[lf]

            face_nodes = (
                tet[local_nodes[1]],
                tet[local_nodes[2]],
                tet[local_nodes[3]],
            )

            key = sorted_face_key(face_nodes)

            if !haskey(face_map, key)
                face_map[key] = FaceRef[]
            end

            push!(face_map[key], FaceRef(e, lf, face_nodes))
        end
    end

    interior_faces = InteriorFace[]
    boundary_faces = BoundaryFace[]

    for (key, refs) in face_map
        if length(refs) == 2
            a, b = refs

            push!(
                interior_faces,
                InteriorFace(
                    a.elem,
                    a.local_face,
                    b.elem,
                    b.local_face,
                    a.nodes,
                ),
            )

        elseif length(refs) == 1
            a = refs[1]

            if !haskey(surface_tags, key)
                error(
                    "Boundary tet face $key was not found in tagged surface triangles. " *
                    "This usually means mesh.tris does not match the exterior tet faces."
                )
            end

            boundary_id = surface_tags[key]

            push!(
                boundary_faces,
                BoundaryFace(
                    a.elem,
                    a.local_face,
                    a.nodes,
                    boundary_id,
                ),
            )

        else
            error("Non-manifold face $key has $(length(refs)) adjacent tetrahedra.")
        end
    end

    return DGTopology(interior_faces, boundary_faces)
end

function print_topology_summary(mesh::RawVTUMesh, topology::DGTopology)
    ntets = size(mesh.tets, 2)
    ntris = size(mesh.tris, 2)

    ninterior = length(topology.interior_faces)
    nboundary = length(topology.boundary_faces)

    total_tet_face_refs = 4 * ntets
    total_unique_faces = ninterior + nboundary

    expected_tet_face_refs = 2 * ninterior + nboundary

    println("DG topology")
    println("-----------")
    println("Number of tetrahedra:              ", ntets)
    println("Tet face references, 4 × Nt:       ", total_tet_face_refs)
    println("Interior faces:                    ", ninterior)
    println("Boundary faces:                    ", nboundary)
    println("Unique mesh faces, int + bnd:      ", total_unique_faces)
    println("Expected tet face refs, 2int+bnd:  ", expected_tet_face_refs)
    println("Surface triangles from VTU:        ", ntris)

    println()
    println("Consistency checks")
    println("------------------")

    if expected_tet_face_refs == total_tet_face_refs
        println("✓ 2 × interior_faces + boundary_faces == 4 × tetrahedra")
    else
        println("✗ 2 × interior_faces + boundary_faces != 4 × tetrahedra")
        println("  left  = ", expected_tet_face_refs)
        println("  right = ", total_tet_face_refs)
    end

    if nboundary == ntris
        println("✓ boundary_faces == surface triangles from VTU")
    else
        println("✗ boundary_faces != surface triangles from VTU")
        println("  boundary_faces = ", nboundary)
        println("  surface_tris   = ", ntris)
    end

    boundary_ids = sort(unique(f.boundary_id for f in topology.boundary_faces))

    println()
    println("Boundary ids")
    println("------------")
    println(boundary_ids)

    println()
    println("Boundary-face counts")
    println("--------------------")

    for bid in boundary_ids
        n = count(f -> f.boundary_id == bid, topology.boundary_faces)
        println("  boundary_id = ", bid, " : ", n, " faces")
    end

    return nothing
end
