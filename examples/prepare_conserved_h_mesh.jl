#!/usr/bin/env julia
# Reuse the established periodic generators and VTK boundary validator.
using DiscoGMPI
include("meshes/generate_periodic_box_2x1x1_unstructured.jl")

length(ARGS) == 5 || error("Usage: prepare_conserved_h_mesh.jl FAMILY NX NY NZ OUTPUT")
family, sx, sy, sz, output = ARGS
nx, ny, nz = parse.(Int, (sx, sy, sz))
ispath(output) && error("Refusing to overwrite $output")
if family == "unstructured"
    main(["--nx", sx, "--ny", sy, "--nz", sz, "--output", output])
elseif family == "structured"
    mesh = structured_box_mesh(nx, ny, nz; upper=(2., 1., 1.))
    triangles = [Tuple(t) for t in eachcol(mesh.tris)]
    tetrahedra = [Tuple(t) for t in eachcol(mesh.tets)]
    tags = [boundary_tag(mesh.points, t) for t in triangles]
    validate_periodic_boundaries(mesh.points, triangles, tags)
    mkpath(dirname(output))
    write_vtk(output, mesh.points, triangles, tetrahedra, tags)
else
    error("Unknown mesh family $family")
end
