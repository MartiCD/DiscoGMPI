# Data containers and symbolic constants used by the DG kernels.
# Keep this file free of numerical assembly logic so storage layout stays easy to audit.

"""
    RawVTUMesh

Raw mesh container read from a `.vtu` file.

Fields
------
- `points`: 3 × Np matrix of point coordinates.
- `tets`: 4 × Nt matrix of tetrahedral connectivity.
- `tris`: 3 × Ns matrix of triangular surface connectivity.
- `tet_cell_ids`: original VTK cell indices corresponding to tetrahedra.
- `tri_cell_ids`: original VTK cell indices corresponding to triangles.
- `cell_data`: dictionary containing raw VTK cell-data arrays.
"""
struct RawVTUMesh
    points::Matrix{Float64}
    tets::Matrix{Int}
    tris::Matrix{Int}
    tet_cell_ids::Vector{Int}
    tri_cell_ids::Vector{Int}
    cell_data::Dict{String, Any}
end

const TET_FACES = (
    (2, 3, 4),  # face opposite local node 1
    (1, 4, 3),  # face opposite local node 2
    (1, 2, 4),  # face opposite local node 3
    (1, 3, 2),  # face opposite local node 4
)

struct FaceRef
    elem::Int
    local_face::Int
    nodes::NTuple{3, Int}
end

struct InteriorFace
    left_elem::Int
    left_local_face::Int
    right_elem::Int
    right_local_face::Int
    nodes::NTuple{3, Int}
end

struct BoundaryFace
    elem::Int
    local_face::Int
    nodes::NTuple{3, Int}
    boundary_id::Int
end

struct DGTopology
    interior_faces::Vector{InteriorFace}
    boundary_faces::Vector{BoundaryFace}
end

struct FaceGeometry
    centroid::NTuple{3, Float64}
    normal::NTuple{3, Float64}   # unit normal
    area::Float64
end

struct CellGeometry
    centroid::NTuple{3, Float64}
    volume::Float64
end

struct DGGeometry
    cells::Vector{CellGeometry}
    interior_faces::Vector{FaceGeometry}
    boundary_faces::Vector{FaceGeometry}
end

struct TetMapping
    J::Matrix{Float64}       # 3 × 3
    invJ::Matrix{Float64}    # 3 × 3
    detJ::Float64
    absdetJ::Float64
end

struct DGReferenceMapping
    tet_mappings::Vector{TetMapping}
end

const REF_TET_NODES = (
    (-1.0, -1.0, -1.0),
    ( 1.0, -1.0, -1.0),
    (-1.0,  1.0, -1.0),
    (-1.0, -1.0,  1.0),
)

const REF_TET_VOLUME = 4.0 / 3.0

struct OrthonormalTetBasis
    N::Int
    Np::Int
    exponents::Vector{NTuple{3, Int}}

    # modal_coeffs[m, q] is the coefficient of monomial m
    # in orthonormal modal basis function q.
    modal_coeffs::Matrix{Float64}

    # Gram matrix of the raw monomial basis.
    gram::Matrix{Float64}
end

struct ReferenceTet
    N::Int
    Np::Int

    r::Vector{Float64}
    s::Vector{Float64}
    t::Vector{Float64}

    # exponents::Vector{NTuple{3, Int}}
    basis::OrthonormalTetBasis

    V::Matrix{Float64}
    invV::Matrix{Float64}

    Dr::Matrix{Float64}
    Ds::Matrix{Float64}
    Dt::Matrix{Float64}

    M::Matrix{Float64}

    # Reference weak derivative matrices:
    # Sα[i,j] = ∫_Kref ℓ_i ∂αℓ_j dV.
    Sr::Matrix{Float64}
    Ss::Matrix{Float64}
    St::Matrix{Float64}
end

struct TriangleQuadrature
    rq::Vector{Float64}
    sq::Vector{Float64}
    wq::Vector{Float64}
    degree::Int
end

struct ReferenceTetFaceOperators
    face_nodes::NTuple{4, Vector{Int}}
    face_mass::NTuple{4, Matrix{Float64}}
    Emat::Matrix{Float64}
    LIFT::Matrix{Float64}
end

const REF_TET_VERTEX_COORDS = (
    (-1.0, -1.0, -1.0),
    ( 1.0, -1.0, -1.0),
    (-1.0,  1.0, -1.0),
    (-1.0, -1.0,  1.0),
)

const REF_TET_FACE_VERTEX_IDS = (
    (2, 3, 4),  # opposite vertex 1
    (1, 4, 3),  # opposite vertex 2
    (1, 2, 4),  # opposite vertex 3
    (1, 3, 2),  # opposite vertex 4
)

struct ReferenceTri
    N::Int
    Np::Int

    a::Vector{Float64}
    b::Vector{Float64}

    exponents::Vector{NTuple{2, Int}}

    V::Matrix{Float64}
    invV::Matrix{Float64}

    Da::Matrix{Float64}
    Db::Matrix{Float64}

    M::Matrix{Float64}
end

struct OrthonormalTriBasis
    N::Int
    Np::Int
    exponents::Vector{NTuple{2, Int}}

    # modal_coeffs[m, q] is coefficient of monomial m
    # in orthonormal modal basis function q.
    modal_coeffs::Matrix{Float64}

    gram::Matrix{Float64}
end

struct PhysicalWeakDerivativeOperators
    # Weak derivative matrices on the reference-volume scaling:
    # Sα = M * Dα, where Dα is the physical derivative operator.
    # The physical mass factor is stored separately in mass_scale and
    # cancels in affine-element volume RHS solves.
    Sx::Matrix{Float64}
    Sy::Matrix{Float64}
    Sz::Matrix{Float64}

    # Lazy transpose views used by the Poisson-bracket adjoint curl.
    SxT::Transpose{Float64, Matrix{Float64}}
    SyT::Transpose{Float64, Matrix{Float64}}
    SzT::Transpose{Float64, Matrix{Float64}}
end

function PhysicalWeakDerivativeOperators(
    Sx::Matrix{Float64},
    Sy::Matrix{Float64},
    Sz::Matrix{Float64},
)
    return PhysicalWeakDerivativeOperators(
        Sx,
        Sy,
        Sz,
        transpose(Sx),
        transpose(Sy),
        transpose(Sz),
    )
end

struct PhysicalElementOperators
    # Strong nodal derivative operators.
    Dx::Matrix{Float64}
    Dy::Matrix{Float64}
    Dz::Matrix{Float64}

    weak::PhysicalWeakDerivativeOperators

    mass_scale::Float64
end

struct DGPhysicalOperators
    elements::Vector{PhysicalElementOperators}
end

struct InteriorTraceMap
    minus_elem::Int
    minus_face::Int

    plus_elem::Int
    plus_face::Int

    # Local volume-node ids on the minus/plus elements.
    # These are indices in 1:ref.Np.
    minus_nodes::Vector{Int}
    plus_nodes::Vector{Int}

    # plus_nodes[plus_to_minus_perm] is ordered like minus_nodes.
    plus_to_minus_perm::Vector{Int}
end

struct BoundaryTraceMap
    elem::Int
    face::Int
    boundary_id::Int

    # Local volume-node ids on this boundary face.
    nodes::Vector{Int}
end

struct DGTraceMaps
    interior::Vector{InteriorTraceMap}
    boundary::Vector{BoundaryTraceMap}
end

struct InteriorFluxFace
    trace::InteriorTraceMap

    # Outward unit normal from the minus element.
    normal::NTuple{3, Float64}

    # Physical face area.
    area::Float64

    # Face centroid in physical space.
    centroid::NTuple{3, Float64}
end

struct BoundaryFluxFace
    trace::BoundaryTraceMap

    # Outward unit normal from the element/domain.
    normal::NTuple{3, Float64}

    # Physical face area.
    area::Float64

    # Face centroid in physical space.
    centroid::NTuple{3, Float64}

    boundary_id::Int
end

struct DGFluxFaces
    interior::Vector{InteriorFluxFace}
    boundary::Vector{BoundaryFluxFace}
    interior_colors::Vector{Vector{Int}}
    boundary_colors::Vector{Vector{Int}}
end

abstract type AbstractBackend end

struct SerialBackend <: AbstractBackend end

struct ThreadedBackend <: AbstractBackend
    maxwell_volume_workspace::Base.RefValue{Any}
end

ThreadedBackend() = ThreadedBackend(Ref{Any}(nothing))

struct DGDiscretization{B<:AbstractBackend}
    mesh::RawVTUMesh
    topology::DGTopology
    geometry::DGGeometry
    mappings::DGReferenceMapping
    ref::ReferenceTet
    fops::ReferenceTetFaceOperators
    physops::DGPhysicalOperators
    trace_maps::DGTraceMaps
    flux_faces::DGFluxFaces
    backend::B
end

struct MaxwellField
    Ex::Matrix{Float64}   # Np × Ne
    Ey::Matrix{Float64}
    Ez::Matrix{Float64}

    Hx::Matrix{Float64}
    Hy::Matrix{Float64}
    Hz::Matrix{Float64}
end

struct MaxwellRHS
    rhsEx::Matrix{Float64}
    rhsEy::Matrix{Float64}
    rhsEz::Matrix{Float64}

    rhsHx::Matrix{Float64}
    rhsHy::Matrix{Float64}
    rhsHz::Matrix{Float64}
end

@enum MaxwellBoundaryKind begin
    MaxwellBC_None = 0
    MaxwellBC_PEC = 1
end

struct MaxwellBoundaryRegistry
    kinds::Dict{Int, MaxwellBoundaryKind}
end

@enum MaxwellFluxKind begin
    MaxwellFlux_Central = 0
    MaxwellFlux_Upwind = 1
end

abstract type AbstractDGFormulation end

abstract type AbstractMaxwellDGFormulation <: AbstractDGFormulation end

struct HesthavenWarburtonFormulation <: AbstractMaxwellDGFormulation
    flux_kind::MaxwellFluxKind
end

HesthavenWarburtonFormulation() = HesthavenWarburtonFormulation(MaxwellFlux_Central)

struct PoissonBracketFormulation <: AbstractMaxwellDGFormulation 
    flux_kind::MaxwellFluxKind
end

PoissonBracketFormulation() = PoissonBracketFormulation(MaxwellFlux_Central)

struct MaxwellEnergy
    electric::Float64
    magnetic::Float64
    total::Float64

    Ex::Float64
    Ey::Float64
    Ez::Float64

    Hx::Float64
    Hy::Float64
    Hz::Float64
end

struct ExplicitRKScheme
    order::Int
    name::String
    A::Matrix{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
end

struct ExplicitPartitionedSymplecticRKScheme
    order::Int
    name::String
    first_partition::Symbol
    first_weights::Vector{Float64}
    second_weights::Vector{Float64}
end

struct MaxwellRKWorkspace
    U0::MaxwellField
    Ustage::MaxwellField
    K::Vector{MaxwellRHS}
end

struct MaxwellPartitionedRKWorkspace
    rhs::MaxwellRHS
end

struct PeriodicBoundarySpec
    minus_boundary_id::Int
    plus_boundary_id::Int

    # Maps plus-side physical coordinates into minus-side coordinates.
    # Example: xmax -> xmin uses (-Lx, 0, 0).
    plus_to_minus_shift::NTuple{3, Float64}

    name::Symbol
end

struct PeriodicTraceMap
    minus_elem::Int
    minus_face::Int
    minus_boundary_id::Int
    minus_nodes::Vector{Int}

    plus_elem::Int
    plus_face::Int
    plus_boundary_id::Int
    plus_nodes::Vector{Int}

    # plus_nodes[plus_to_minus_perm] is ordered like minus_nodes
    # after applying plus_to_minus_shift to the plus coordinates.
    plus_to_minus_perm::Vector{Int}

    plus_to_minus_shift::NTuple{3, Float64}
end

struct PeriodicFluxFace
    trace::PeriodicTraceMap

    # Normal is outward from the minus element/domain side.
    normal::NTuple{3, Float64}

    area::Float64
    centroid::NTuple{3, Float64}

    name::Symbol
end

struct DGPeriodicFluxFaces
    faces::Vector{PeriodicFluxFace}
    colors::Vector{Vector{Int}}
end

struct ElementSizeDiagnostics
    hmin::Float64
    hmax::Float64
    hmean::Float64

    vmin::Float64
    vmax::Float64

    amin::Float64
    amax::Float64

    worst_elem::Int
end
