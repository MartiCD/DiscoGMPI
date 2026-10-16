"""Parse the optional magnetic initialization mode; interpolation remains the default."""
function parse_magnetic_initialization(value::AbstractString)
    mode = Symbol(replace(lowercase(value), '-' => '_'))
    mode in (:interpolate, :curl_potential) ||
        throw(ArgumentError("--magnetic-initialization must be interpolate or curl-potential."))
    return mode
end

"""Parse the optional electric initialization mode; interpolation remains the default."""
function parse_electric_initialization(value::AbstractString)
    mode = Symbol(replace(lowercase(value), '-' => '_'))
    mode in (:interpolate, :curl_potential) ||
        throw(ArgumentError("--electric-initialization must be interpolate or curl-potential."))
    return mode
end

"""
    MagneticCurlInitializationWorkspace(ref; nodal_set=:equispaced)

Cache degree-(p+1) potential interpolation and rectangular derivatives evaluated
at the actual degree-p solver nodes. `nodal_set` selects the potential nodes;
the target ordering is taken from `ref`. Reuse with that reference object only.
Scratch buffers are not thread safe.
"""
struct CurlInitializationWorkspace
    target::ReferenceTet
    potential::ReferenceTet
    Rr::Matrix{Float64}
    Rs::Matrix{Float64}
    Rt::Matrix{Float64}
    A::Matrix{Float64}
    Ar::Matrix{Float64}
    As::Matrix{Float64}
    At::Matrix{Float64}
end

function CurlInitializationWorkspace(ref::ReferenceTet; nodal_set::Symbol = :equispaced)
    hi = build_reference_tet(ref.N + 1; nodal_set = nodal_set)
    Vr, Vs, Vt = orthonormal_grad_vandermonde_tet(ref.r, ref.s, ref.t, hi.basis)
    return CurlInitializationWorkspace(ref, hi,
        Vr * hi.invV, Vs * hi.invV, Vt * hi.invV,
        zeros(hi.Np, 3), zeros(ref.Np, 3), zeros(ref.Np, 3), zeros(ref.Np, 3))
end

# Reference operators are independent of the field and material coefficient.
const MagneticCurlInitializationWorkspace = CurlInitializationWorkspace
"""Cached degree-(p+1) curl operators for D=epsilon*E; same workspace as for B=mu*H."""
const ElectricCurlInitializationWorkspace = CurlInitializationWorkspace

"""
    initialize_magnetic_from_potential!(U, mesh, ref, mappings, materials, A, workspace;
                                        time=0.0, elements=axes(U.Hx, 2))

Sample Cartesian `A(x,y,z,time)` at degree p+1 and evaluate its polynomial curl
directly at degree p. Only H is changed. Affine four-vertex tetrahedra and finite
positive elementwise scalar permeability are required. This ensures broken
div(mu H)=0 to scaled roundoff, NOT normal continuity, conservation of energy,
or preservation of divergence by subsequent DG evolution.
"""
function initialize_magnetic_from_potential!(
    U::MaxwellField, mesh::RawVTUMesh, ref::ReferenceTet,
    mappings::DGReferenceMapping, materials,
    potential, ws::MagneticCurlInitializationWorkspace;
    time::Real = 0.0, elements = axes(U.Hx, 2),
)
    return initialize_field_from_potential!(U, mesh, ref, mappings, materials, potential, ws;
        time, elements, field = :magnetic)
end

"""
    initialize_electric_from_potential!(U, mesh, ref, mappings, materials, A, workspace;
                                        time=0.0, elements=axes(U.Ex, 2))

Construct D=curl(A_{p+1}) and E=D/epsilon at the degree-p solver nodes.
Only E is changed. Requires affine tetrahedra and finite positive elementwise
scalar permittivity. Guarantees only elementwise div(epsilon E)=0 to scaled
roundoff at initialization, not normal continuity, energy preservation, or
divergence preservation during subsequent DG evolution.
"""
function initialize_electric_from_potential!(
    U::MaxwellField, mesh::RawVTUMesh, ref::ReferenceTet,
    mappings::DGReferenceMapping, materials,
    potential, ws::ElectricCurlInitializationWorkspace;
    time::Real = 0.0, elements = axes(U.Ex, 2),
)
    return initialize_field_from_potential!(U, mesh, ref, mappings, materials, potential, ws;
        time, elements, field = :electric)
end

function initialize_field_from_potential!(
    U::MaxwellField, mesh::RawVTUMesh, ref::ReferenceTet,
    mappings::DGReferenceMapping, materials, potential, ws::CurlInitializationWorkspace;
    time::Real, elements, field::Symbol,
)
    field in (:electric, :magnetic) || throw(ArgumentError("Unknown field $field."))
    electric = field == :electric
    materials isa MaxwellElementMaterials ||
        throw(ArgumentError("Use elementwise scalar MaxwellElementMaterials; within-element material variation is unsupported."))
    coefficient = electric ? materials.epsilon : materials.permeability
    components = electric ? (U.Ex, U.Ey, U.Ez) : (U.Hx, U.Hy, U.Hz)
    ref === ws.target || throw(ArgumentError("Workspace belongs to a different reference element."))
    size(mesh.tets, 1) == 4 || throw(ArgumentError("Only affine four-vertex tetrahedra are supported."))
    ne = size(mesh.tets, 2)
    all(size(component) == (ref.Np, ne) for component in components) ||
        throw(ArgumentError("Field dimensions do not match the mesh/reference."))
    length(mappings.tet_mappings) == ne || throw(ArgumentError("Mapping count mismatch."))
    validate_maxwell_materials(materials, ne)
    length(coefficient) == ne || throw(ArgumentError("Material coefficient count mismatch."))
    all(x -> isfinite(x) && x > 0, coefficient) ||
        throw(ArgumentError("The $field material coefficient must be finite, positive, and elementwise constant."))
    isfinite(time) || throw(ArgumentError("Initialization time must be finite."))
    ids = collect(elements)
    all(e -> 1 <= e <= ne, ids) || throw(ArgumentError("Invalid local element index."))
    for e in ids
        mapping = mappings.tet_mappings[e]
        all(isfinite, mapping.J) && all(isfinite, mapping.invJ) &&
            isfinite(mapping.detJ) && abs(mapping.detJ) > 0 ||
            throw(ArgumentError("Nonfinite or singular affine mapping in element $e."))
    end
    hi = ws.potential
    for e in ids
        nodes = @view mesh.tets[:, e]
        for i in 1:hi.Np
            x, y, z = map_to_physical(mesh.points, nodes, hi.r[i], hi.s[i], hi.t[i])
            value = potential(x, y, z, Float64(time))
            length(value) == 3 && all(isfinite, value) ||
                throw(ArgumentError("Potential must return three finite Cartesian components."))
            ws.A[i, 1], ws.A[i, 2], ws.A[i, 3] = value
        end
        mul!(ws.Ar, ws.Rr, ws.A)
        mul!(ws.As, ws.Rs, ws.A)
        mul!(ws.At, ws.Rt, ws.A)
        g = mappings.tet_mappings[e].invJ
        material = coefficient[e]
        # Columns of invJ map reference derivatives to physical derivatives.
        for i in 1:ref.Np
            derivative(axis, component) = g[1, axis] * ws.Ar[i, component] +
                g[2, axis] * ws.As[i, component] + g[3, axis] * ws.At[i, component]
            components[1][i, e] = (derivative(2, 3) - derivative(3, 2)) / material
            components[2][i, e] = (derivative(3, 1) - derivative(1, 3)) / material
            components[3][i, e] = (derivative(1, 2) - derivative(2, 1)) / material
        end
        all(c -> all(isfinite, @view(c[:, e])), components) ||
            throw(ArgumentError("$field initialization overflow."))
    end
    return U
end

function initialize_electric_from_potential!(U::MaxwellField, dg::DGDiscretization,
    materials, potential, ws::ElectricCurlInitializationWorkspace; kwargs...)
    return initialize_electric_from_potential!(U, dg.mesh, dg.ref, dg.mappings,
        materials, potential, ws; kwargs...)
end

function initialize_magnetic_from_potential!(U::MaxwellField, dg::DGDiscretization,
    materials, potential, ws::MagneticCurlInitializationWorkspace; kwargs...)
    return initialize_magnetic_from_potential!(U, dg.mesh, dg.ref, dg.mappings,
        materials, potential, ws; kwargs...)
end

"""
Distributed initialization. Only owned H coefficients are constructed; ghost H
is zeroed before the usual face-trace exchange. Ghost interiors are not valid
volume fields. Optional `periodic` refreshes the separate periodic trace buffers.
Callback/validation errors are propagated collectively before communication.
"""
function initialize_magnetic_from_potential!(U::MaxwellField, ddg::DistributedDGDiscretization,
    materials, potential; workspace = nothing, nodal_set::Symbol = :equispaced,
    time::Real = 0.0, periodic = nothing)
    return initialize_distributed_field_from_potential!(U, ddg, materials, potential;
        workspace, nodal_set, time, periodic, field = :magnetic)
end

"""Initialize owned E and refresh ordinary/optional periodic traces, with collective errors.
Ghost E interiors are not valid volume fields; H owned coefficients are unchanged.
"""
function initialize_electric_from_potential!(U::MaxwellField, ddg::DistributedDGDiscretization,
    materials, potential; workspace = nothing, nodal_set::Symbol = :equispaced,
    time::Real = 0.0, periodic = nothing)
    return initialize_distributed_field_from_potential!(U, ddg, materials, potential;
        workspace, nodal_set, time, periodic, field = :electric)
end

function initialize_distributed_field_from_potential!(U, ddg, materials, potential;
    workspace, nodal_set, time, periodic, field)
    collective_rank_action(ddg.comm, "$field curl-potential initialization") do
        materials isa MaxwellElementMaterials ||
            throw(ArgumentError("Use elementwise scalar MaxwellElementMaterials, not a spatial material callback."))
        ws = workspace === nothing ? CurlInitializationWorkspace(ddg.dg.ref; nodal_set) : workspace
        initialize_field_from_potential!(U, ddg.dg.mesh, ddg.dg.ref, ddg.dg.mappings,
            materials, potential, ws; time, elements = ddg.distributed_mesh.partition.owned, field)
        components = field == :electric ? (U.Ex, U.Ey, U.Ez) : (U.Hx, U.Hy, U.Hz)
        for e in ddg.distributed_mesh.partition.ghosts, component in components
            component[:, e] .= 0.0
        end
    end
    exchange_maxwell_ghost_traces!(U, ddg)
    periodic === nothing || exchange_distributed_periodic_traces!(U, ddg, periodic)
    return U
end

"""Owned-element broken divergence diagnostics for elementwise constant mu.

The L2 norm and signed integral are distinct; the latter can hide cancellation.
Uses the solver's physical derivative matrices and reference mass matrix.
"""
function magnetic_initialization_diagnostics(U::MaxwellField, dg::DGDiscretization,
    materials::MaxwellElementMaterials; elements = axes(U.Hx, 2))
    validate_maxwell_materials(materials, size(U.Hx, 2))
    return field_initialization_diagnostics((U.Hx, U.Hy, U.Hz), dg,
        materials.permeability, elements)
end

"""Broken div(epsilon E) from solver derivatives/mass; report L2 squared and signed integral separately."""
function electric_initialization_diagnostics(U::MaxwellField, dg::DGDiscretization,
    materials::MaxwellElementMaterials; elements = axes(U.Ex, 2))
    validate_maxwell_materials(materials, size(U.Ex, 2))
    return field_initialization_diagnostics((U.Ex, U.Ey, U.Ez), dg,
        materials.epsilon, elements)
end

function field_initialization_diagnostics(components, dg, coefficient, elements)
    l2_squared = 0.0
    signed = 0.0
    max_nodal = 0.0
    mass_ones = dg.ref.M * ones(dg.ref.Np)
    for e in elements
        op = dg.physops.elements[e]
        d = coefficient[e] .* (op.Dx * components[1][:, e] +
            op.Dy * components[2][:, e] + op.Dz * components[3][:, e])
        jac = dg.mappings.tet_mappings[e].absdetJ
        l2_squared += jac * dot(d, dg.ref.M * d)
        signed += jac * dot(mass_ones, d)
        max_nodal = max(max_nodal, maximum(abs, d))
    end
    return (; l2_squared, signed, max_nodal)
end

function electric_initialization_diagnostics(U::MaxwellField, ddg::DistributedDGDiscretization,
    materials::MaxwellElementMaterials)
    local_values = electric_initialization_diagnostics(U, ddg.dg, materials;
        elements = ddg.distributed_mesh.partition.owned)
    sums = MPI.Allreduce([local_values.l2_squared, local_values.signed], +, ddg.comm)
    max_nodal = MPI.Allreduce(local_values.max_nodal, max, ddg.comm)
    return (broken_divergence_l2 = sqrt(max(0.0, sums[1])),
        signed_electric_charge = sums[2], max_nodal_divergence = max_nodal)
end

function magnetic_initialization_diagnostics(U::MaxwellField, ddg::DistributedDGDiscretization,
    materials::MaxwellElementMaterials)
    local_values = magnetic_initialization_diagnostics(U, ddg.dg, materials;
        elements = ddg.distributed_mesh.partition.owned)
    sums = MPI.Allreduce([local_values.l2_squared, local_values.signed], +, ddg.comm)
    max_nodal = MPI.Allreduce(local_values.max_nodal, max, ddg.comm)
    return (broken_divergence_l2 = sqrt(max(0.0, sums[1])),
        signed_magnetic_charge = sums[2], max_nodal_divergence = max_nodal)
end

"""Legacy checkpoints used interpolation; never silently relabel their initial state."""
function validate_magnetic_initialization_restart(metadata, mode::Symbol)
    stored = parse_magnetic_initialization(string(get(metadata, "magnetic_initialization", "interpolate")))
    stored == mode || throw(ArgumentError("Checkpoint magnetic initialization $stored does not match $mode."))
    return nothing
end

"""Legacy checkpoints used electric interpolation; reject inconsistent restart labels."""
function validate_electric_initialization_restart(metadata, mode::Symbol)
    stored = parse_electric_initialization(string(get(metadata, "electric_initialization", "interpolate")))
    stored == mode || throw(ArgumentError("Checkpoint electric initialization $stored does not match $mode."))
    return nothing
end
