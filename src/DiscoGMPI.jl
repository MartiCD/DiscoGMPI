
module DiscoGMPI

using LinearAlgebra
using MPI
using Printf
using ReadVTK
using VTKBase
using WriteVTK

# Serial/threaded solver API inherited from DiscoG3D.
export RawVTUMesh,
       read_vtu_mesh,
       print_mesh_summary,
       tet_data,
       tri_data,
       list_cell_data,
       check_mesh_consistency,
       FaceRef,
       InteriorFace,
       BoundaryFace,
       DGTopology,
       build_dg_topology,
       print_topology_summary,
       FaceGeometry,
       CellGeometry,
       DGGeometry,
       build_dg_geometry,
       print_geometry_summary,
       TetMapping,
       DGReferenceMapping,
       build_reference_mappings,
       print_mapping_summary,
       reference_gradient_to_physical,
       physical_shape_gradients,
       OrthonormalTetBasis,
       ReferenceTet,
       TriangleQuadrature,
       build_reference_tet,
       print_reference_tet_summary,
       print_orthonormal_basis_summary,
       num_tet_nodes,
       equispaced_tet_nodes,
       ReferenceTri,
       OrthonormalTriBasis,
       build_reference_tri,
       print_reference_tri_summary,
       ReferenceTetFaceOperators,
       build_reference_face_operators,
       print_reference_face_operator_summary,
       reference_face_nodes,
       build_physical_operators,
       physical_weak_derivative_matrices,
       physical_weak_derivative_transpose_matrices,
       print_physical_operator_summary,
       test_physical_stiffness_consistency,
       test_physical_derivatives_linear,
       build_dg_trace_maps,
       print_trace_map_summary,
       test_trace_map_geometry,
       test_trace_maps_linear_function,
       build_dg_flux_faces,
       print_flux_face_summary,
       test_interior_flux_face_normals,
       test_boundary_box_normals,
       test_pec_sphere_normals,
       print_boundary_area_reference_check,
       AbstractBackend,
       SerialBackend,
       ThreadedBackend,
       DGDiscretization,
       test_scalar_advection_volume_operator,
       test_scalar_advection_interior_surface_operator,
       test_scalar_advection_boundary_surface_operator,
       test_scalar_advection_full_surface_operator,
       MaxwellField,
       MaxwellRHS,
       MaxwellBoundaryKind,
       MaxwellBC_None,
       MaxwellBC_PEC,
       MaxwellBC_PMC,
       MaxwellBC_Absorbing,
       MaxwellFluxKind,
       MaxwellFlux_Central,
       MaxwellFlux_Upwind,
       MaxwellBoundaryRegistry,
       MaxwellMaterial,
       MaxwellElementMaterials,
       MaxwellNonlinearPML,
       MaxwellInvariantDiagnostics,
       polynomial_pml_sigma,
       build_maxwell_nonlinear_pml,
       nonlinear_pml_source,
       add_maxwell_nonlinear_pml_source!,
       maxwell_nonlinear_pml_rhs!,
       maxwell_nonlinear_pml_rhs_periodic!,
       make_maxwell_nonlinear_pml_rhs_function,
       make_distributed_maxwell_nonlinear_pml_rhs_function,
       make_distributed_periodic_maxwell_nonlinear_pml_rhs_function,
       maxwell_nonlinear_pml_rk_step!,
       distributed_maxwell_nonlinear_pml_rk_step!,
       distributed_periodic_maxwell_nonlinear_pml_rk_step!,
       run_maxwell_nonlinear_pml_time_steps!,
       run_distributed_maxwell_nonlinear_pml_time_steps!,
       homogeneous_maxwell_materials,
       maxwell_element_materials,
       AbstractDGFormulation,
       AbstractMaxwellDGFormulation,
       HesthavenWarburtonFormulation,
       PoissonBracketFormulation,
       default_maxwell_boundary_registry,
       empty_maxwell_boundary_registry,
       interpolate_maxwell_field,
       maxwell_rhs!,
       maxwell_rhs_periodic!,
       maxwell_volume_rhs!,
       maxwell_energy,
       maxwell_energy_rate,
       maxwell_invariants,
       MaxwellRKWorkspace,
       run_maxwell_time_steps!,
       run_maxwell_partitioned_symplectic_time_steps!,
       test_maxwell_pec_boundary_reflection,
       test_maxwell_volume_operator,
       test_maxwell_interior_surface_operator,
       test_maxwell_pec_local_flux_zero,
       test_maxwell_pec_boundary_surface_zero_field,
       test_maxwell_pec_boundary_surface_arbitrary_field,
       test_maxwell_rhs_matches_volume_without_boundaries,
       test_maxwell_rhs_zero_field_with_pec,
       test_maxwell_rhs_linear_field_no_boundaries,
       test_maxwell_energy_constant_field,
       test_maxwell_energy_rate_zero_field,
       test_maxwell_energy_rate_no_boundaries,
       explicit_rk_scheme,
       ExplicitPartitionedSymplecticRKScheme,
       MaxwellPartitionedRKWorkspace,
       explicit_partitioned_symplectic_rk_scheme,
       print_rk_scheme_summary,
       print_partitioned_symplectic_rk_scheme_summary,
       partitioned_symplectic_rk_step!,
       test_rk_zero_field_step,
       test_rk_one_step_energy_diagnostic,
       test_partitioned_symplectic_rk_zero_field_step,
       test_partitioned_symplectic_rk_one_step_energy_diagnostic,
       default_unit_box_periodic_specs,
       build_periodic_flux_faces,
       print_periodic_flux_face_summary,
       test_periodic_flux_face_geometry,
       test_periodic_trace_maps_scalar_function,
       test_maxwell_periodic_rhs_zero_field,
       test_rk_periodic_zero_field_step,
       test_rk_periodic_one_step_energy_diagnostic,
       estimate_maxwell_dt,
       print_element_size_diagnostics,
       print_maxwell_dt_estimate,
       test_periodic_time_marching_with_cfl_dt,
       test_maxwell_upwind_interior_surface_operator,
       test_maxwell_upwind_periodic_surface_operator,
       test_maxwell_upwind_periodic_rhs_zero_field,
       DistributedDGInterfaceFace,
       DistributedMaxwellExchange,
       DistributedDGDiscretization,
       DistributedPeriodicFace,
       DistributedPeriodicMaxwellExchange,
       build_distributed_dg,
       build_distributed_dg_from_root,
       build_distributed_dg_from_partition,
       prepare_distributed_mesh_partition,
       prepare_distributed_mesh_partition_collective,
       write_distributed_mesh_partition,
       load_distributed_mesh_partition,
       DistributedCheckpointState,
       write_distributed_checkpoint,
       load_distributed_checkpoint,
       write_distributed_run_metadata,
       localize_maxwell_field,
       exchange_maxwell_ghost_traces!,
       zero_ghost_maxwell_rhs!,
       profile_distributed_maxwell_rhs!,
       make_distributed_maxwell_rhs_function,
       distributed_maxwell_energy,
       distributed_maxwell_invariants,
       build_distributed_periodic_maxwell_exchange,
       exchange_distributed_periodic_traces!,
       make_distributed_periodic_maxwell_rhs_function,
       distributed_periodic_rk_step!,
       distributed_periodic_partitioned_symplectic_rk_step!,
       distributed_rk_step!,
       distributed_partitioned_symplectic_rk_step!,
       run_distributed_maxwell_time_steps!,
       run_distributed_maxwell_partitioned_symplectic_time_steps!,
       get_JaskowiecSukumar_cubature,
       PlaneWaveParameters,
       exact_cavity_mode_functions,
       exact_cavity_mode_curl_functions,
       exact_periodic_wave_functions,
       exact_periodic_wave_curl_functions,
       optical_chirality_density,
       MaxwellQuadratureDiagnostics,
       DistributedMeshQualityMetrics,
       distributed_mesh_quality_metrics,
       reference_interpolation_matrix,
       distributed_maxwell_quadrature_diagnostics,
       distributed_cavity_quadrature_diagnostics,
       distributed_periodic_quadrature_diagnostics,
       distributed_maxwell_linf_errors,
       distributed_cavity_linf_errors,
       distributed_periodic_linf_errors,
       DistributedMaxwellComponentL2Workspace,
       distributed_maxwell_component_l2_errors!,
       distributed_maxwell_component_l2_errors,
       distributed_cavity_component_l2_errors,
       distributed_periodic_component_l2_errors!,
       distributed_periodic_component_l2_errors,
       write_energy_header,
       write_energy_row,
       write_quadrature_diagnostics_header,
       write_quadrature_diagnostics_row,
       reference_vertex_node_ids,
       vtk_lagrange_tetra_barycentric_indices,
       vtk_lagrange_tetra_node_ids,
       write_parallel_maxwell_fields,
       xml_escape,
       atomic_output_file,
       checkpoint_step_dir,
       write_latest_checkpoint,
       collectively_write_latest_checkpoint,
       read_paraview_series,
       write_paraview_series,
       collective_root_action,
       collective_rank_action,
       write_maxwell_paraview_snapshot!,
       write_maxwell_integration_points

include("MetisIO.jl")
using .MetisIO
export read_mesh_file_tet_vtk,
        write_metis_mesh_from_vtk

include("DistributedMesh3D.jl")
using .DistributedMesh3D

export ElementOwnership,
       FaceNeighborInfo,
       LocalNodeGeometry,
       LocalElementTopology,
       ElementPartition,
       InterfaceFace,
       NeighborComm,
       MPICommunicationTable,
       DistributedMesh,
       build_distributed_mesh,
       build_distributed_mesh_from_metis,
       tet_face_local_vertices,
       TRI_PERMUTATIONS,
       read_metis_epart

include("TraceMaps.jl")
using .TraceMaps

export build_mpi_trace_maps_buffers_and_permutations,
       exchange_face_traces_blocking!,
       get_face_traces

include("PartitionVTK.jl")
using .PartitionVTK

export write_partition_piece_vtks, write_partition_vtk

include("solver/data_containers.jl")
include("solver/kernels/mesh_io.jl")
include("solver/kernels/topology.jl")
include("solver/kernels/geometry.jl")
include("solver/kernels/metrics.jl")
include("solver/kernels/reference_tet.jl")
include("solver/kernels/JaskowiecSukumar.jl")
include("solver/kernels/face_operators.jl")
include("solver/kernels/physical_operators.jl")
include("solver/kernels/trace_maps.jl")
include("solver/kernels/flux_faces.jl")
include("solver/kernels/dg_discretization.jl")
include("solver/kernels/scalar_advection.jl")
include("solver/kernels/maxwell.jl")
include("solver/kernels/time_integration.jl")
include("solver/kernels/periodic.jl")
include("solver/kernels/maxwell_fluxes.jl")
include("DistributedDG.jl")
include("PhysicalCoverage.jl")
include("NonlinearPML.jl")
include("DistributedIO.jl")
include("MaxwellExactSolutions.jl")
include("MaxwellDiagnostics.jl")
include("MaxwellOutput.jl")

# # ------------------------------------------------------------------------------------------------------------------------------
# # I'll organize the code in a way that allows for easy extension in the future. For now, I'll just set up the basic structure.
# # 1.- local mesh storage
# # 2.- ownership and ghost bookkeeping
# # 3.- local/global maps 
# # 4.- MPI interface communicationt tables
# # ------------------------------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------------------------------
# # CORE STRCUTS 
# # ------------------------------------------------------------------------------------------------------------------------------
# # Element ownership kind: cleaner than encoding ownership implicitly
# @enum ElementOwnership begin
#     OWNED = 1
#     GHOST = 2
# end

# # Face identification: a face is the real MPI communication unit in DG, not the full element 
# # using sorted global node ids makes face matching across ranks easy and robust 
# struct FaceKey
#     verts::NTuple{3, Int} # sorted global node ids
# end

# # Element-to-face local orientation info 
# # needed for numerical fluxes
# # mirrors SeisSol's approach of SideNeighbor, LocalNeighborSide, LocalNeighborVrtx, Reference(iSide, iElem)
# struct FaceNeighborInfo
#     neighbor_local_elem::Int      # local index of neighbor element, 0 if boundary
#     neighbor_local_face::Int      # which face on the neighbor
#     orientation::Int              # local orientation/permutation code
#     bc_tag::Int                   # 0 for interior, >0 boundary/periodic/etc.
# end

# # Local element table 
# # main topological object solver will use 
# # for tets: Nvpe = 4, Nfaces = 4
# struct LocalElementTopology
#     vertices::Matrix{Int}         # (Nvpe, nlocal_elems), local node ids
#     global_ids::Vector{Int}       # (nlocal_elems)
#     owner_rank::Vector{Int}       # owning rank of each local element
#     ownership::Vector{ElementOwnership}  # OWNED or GHOST
#     material_id::Vector{Int}      # volume zone / material tag
#     faceinfo::Matrix{FaceNeighborInfo}   # (Nfaces, nlocal_elems)
# end

# # Local node table 
# # corresponds to NodeLocal2Global, NodeGlobal2Local in SeisSol
# # Dict is safer and usually more memory efficient in distributed runs than using Vector{Int}
# struct LocalNodeGeometry{T}
#     coords::Matrix{T}             # (dim, nlocal_nodes)
#     global_ids::Vector{Int}       # local -> global
#     global_to_local::Dict{Int,Int}
# end

# # Owned/Ghost split 
# # worth making explicit
# # owned and ghost vectors are used constantly in DG loops
# # typical useage: volume kernels run over owned, ghost states are updated by MPI, flux kernels run over owned boundary/interface faces
# struct ElementPartition
#     owned::Vector{Int}            # local element ids owned by this rank
#     ghosts::Vector{Int}           # local element ids stored but owned elsewhere
#     global_owner::Dict{Int,Int}   # global element id => owner rank
# end

# # ------------------------------------------------------------------------------------------------------------------------------


# # ------------------------------------------------------------------------------------------------------------------------------
# # MPI face communication model
# # ------------------------------------------------------------------------------------------------------------------------------
# # which local faces touch another rank 
# # what data must be sent/received 
# # how to map incoming data onto local ghost faces 

# # One interface 
# # minimal high-value object for MPI DG flux exchange
# struct InterfaceFace
#     local_elem::Int
#     local_face::Int
#     neighbor_rank::Int
#     neighbor_global_elem::Int
#     neighbor_face::Int
#     orientation::Int
#     face_nodes_local::Vector{Int}     # local node ids on this face
#     face_nodes_global::NTuple{3,Int}   # global node ids for matching/debugging
# end

# # Rank-to-rank communication block 
# # groups all communication with one neighbor rank
# struct NeighborComm
#     rank::Int
#     send_faces::Vector{InterfaceFace}
#     recv_faces::Vector{InterfaceFace}
#     send_elem_ids::Vector{Int}      # optional cached unique local owned elems to pack from
#     recv_elem_ids::Vector{Int}      # optional cached unique local ghost elems to unpack into
# end

# # Global MPI communication table 
# # this is what timestepper will consult before each RHS evaluation
# struct MPICommunicationTable
#     neighbors::Vector{Int}
#     comms::Dict{Int,NeighborComm}
# end

# # ------------------------------------------------------------------------------------------------------------------------------



# # ------------------------------------------------------------------------------------------------------------------------------
# # FULL DISTRIBUTED MESH OBJECT 
# # ------------------------------------------------------------------------------------------------------------------------------
# struct DistributedMesh{T}
#     dim::Int
#     elemtype::Symbol                 # :tet
#     nfaces::Int                      # 4
#     nvpe::Int                        # 4

#     nodes::LocalNodeGeometry{T}
#     elements::LocalElementTopology
#     partition::ElementPartition

#     volumes::Vector{T}
#     barycenters::Matrix{T}           # (dim, nlocal_elems)

#     mpi::MPICommunicationTable

#     periodic_face_pairs::Vector{Tuple{Int,Int,Int,Int}}  # optional
#     boundary_faces::Vector{Tuple{Int,Int,Int}}           # (elem, face, bc_tag)
# end

# # ------------------------------------------------------------------------------------------------------------------------------



# # ------------------------------------------------------------------------------------------------------------------------------
# # TOPOLOGY VS DG EXECUTION METADATA
# # ------------------------------------------------------------------------------------------------------------------------------
# # mesh object should store only geometry/topology/ownership 
# # DG-specific execution info should live elsewhere
# struct DGFaceMaps
#     vmapM::Matrix{Int}
#     vmapP::Matrix{Int}
#     mapM::Vector{Int}
#     mapP::Vector{Int}
#     mapB::Vector{Int}
#     mapI::Vector{Int}
#     mapMPI::Vector{Int}
# end

# struct MPIFaceExchange
#     send_indices::Dict{Int,Vector{Int}}   # rank => indices into packed trace arrays
#     recv_indices::Dict{Int,Vector{Int}}   # rank => indices into local ghost trace arrays
#     send_requests::Vector{Any}
#     recv_requests::Vector{Any}
# end

# ------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------------------------------
# RECOMMENDED OWNED/GHOST SEMANTICS
# ------------------------------------------------------------------------------------------------------------------------------
# Owned element 
# - a local element whose solution DFs are advanced by this rank 
#
# Ghost element
# - a local copy of an element owned by another rank, stored only because it shares an interface needed for:
#   - numerical fluxes 
#   - periodic boundaries 
#   - maybe postprocessing or geometry quieries 
#
# in DiscoGMPI, the rule should be: 
# - time integration updates only owned elements 
# - ghost elements are read-only during local RHS assembly 
# - ghost states are refreshed every stage/substep as needed 
# ------------------------------------------------------------------------------------------------------------------------------




# ------------------------------------------------------------------------------------------------------------------------------
# POPULATE THIS FROM THE FORTRAN LOGIC IN SeisSol's MeshReader
# ------------------------------------------------------------------------------------------------------------------------------
#
# Phase A: global partition input 
# For every global element, we need:
# - global connecitivity
# - global owner rank 
# Example: 
# elem_to_rank::Vector{Int}
# global_elem_vertices::Matrix{Int}   # (4, ntotelem), global node ids
# global_node_coords::Matrix{Float64} # (3, ntotnode)
#
# Phase B: select local stored elements 
# For rank r:
# - owned elements: owned_global = findall(==(r), elem_to_rank)
# - collect their vertices 
# - add fhost elements according to halo policy:
#   - face halo for minimal DG communication 
#   - vertex halo if needed to match the Fortran conservatively
# 
# Phase C: build local node subsed 
# Collect all global nodes touched by owned + ghost elements 
# Build:
# - local_node_global_ids
# - global_to_local 
#
# Phase D: localize connecitivity 
# Convert global vertex ids in each stored element to local node ids
#
# Phase E: build face neighbors 
# For every local element face: 
# - form a face key from global node ids 
# - use a dictionary FaceKey => [(elem, face)]
# - if two entries share a key, they are neghbors 
# - if only one entry exists, it is a physical or periodic boundary 

# ------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------------------------------
# BEST JULIA APPROACH FOR NEIGHBOR DISCOVERY
# ------------------------------------------------------------------------------------------------------------------------------
# Instead of recunstructing neighbors from node-to-element adjency as Fortran does, in Julia:
# face_dict = Dict{NTuple{3,Int}, Vector{Tuple{Int,Int}}}()
# For each lcoal element e and local face f:
# 1.- get that face's three gloval node ids 
# 2 .- sort them to get a unique face key 
# 3 .- push (e,f) into face_dict[key]
# Then:
# - length 2 -> interior/shared face,
# - length 1 -> boundary/periodic face 
# - more than 2 -> non-manifold mesh error 
# ------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------------------------------
# INTERFACE-FACE COMMUNICATION TABLES 
# ------------------------------------------------------------------------------------------------------------------------------
# After local face matching is built, identify faces whose neighboring element is ghost-owned:
# if elements.ownership[e] == OWNED &&
#    faceinfo[f,e].neighbor_local_elem != 0 &&
#    elements.owner_rank[faceinfo[f,e].neighbor_local_elem] != myrank
#     # MPI interface face
# end
#
# For such face, create an InterfaceFace object with all the info needed for MPI communication
# Then group them by neighbor_rank.
# Usable structure:
# - send_faces: faces on owned elements whose trace we must send 
# - recv_faces: local ghost interfaces whose neighbor traces are expected to be received 
#
# In practice, many codes only explicitly store one side and derive the other packing/unpacking maps, but keeping both is easier at first

# ------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------------------------------
# WHAT DATA SHOULD BE EXCHANGED
# ------------------------------------------------------------------------------------------------------------------------------
# For nodal DG Maxwell code, across MPI faces typically exchange face trace values such as:
# - E and H 
# evaluated at face nodes/quadrature points 
#
# Communicationt tables should immediately support
# - packing owned face DOFs into send buffers
# - unpacking received buffers into ghost-face buffers 
#
# Add precomputed packing maps later:
# struct NeighborComm
#     rank::Int
#     send_faces::Vector{InterfaceFace}
#     recv_faces::Vector{InterfaceFace}
#     send_dof_indices::Vector{Int}
#     recv_dof_indices::Vector{Int}
# end
#
# Indices would point into flattened solution arrays or trace arrays 
# That is the real runtime-efficient form 

# ------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------------------------------
# RECOMMENDED PHASE 1 MPI DESING FOR DiscoGMPI 
# ------------------------------------------------------------------------------------------------------------------------------
# To keep implementation manageable:

# Stage 1:
# Build distributed mesh with: 
# - owned elements 
# - ghost elements
# - local/global maps 
# - local face neighbors 
# - MPI interface faces 

# Stage 2:
# Exchange only face trace values using blocking communication 

# Stage 3:
# Switch to nonblocking MPI and prepacked send/recv buffers 

# Stage 4:
# Optimize halo policy if needed:
# - face halo insted of vertex halo 
# - compressed communication maps 
# - better periodic handling 
# ------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------------------------------
# SUGGESTED HELPER FUNCTIONS
# ------------------------------------------------------------------------------------------------------------------------------
# build_owned_element_set(...)
# build_ghost_element_set(...)
# build_local_node_maps(...)
# build_local_element_maps(...)
# build_face_neighbors!(mesh)
# build_boundary_tags!(mesh, ...)
# build_periodic_faces!(mesh, ...)
# build_mpi_interface_table(mesh, myrank)
# compute_element_geometry!(mesh)
# ------------------------------------------------------------------------------------------------------------------------------




# ------------------------------------------------------------------------------------------------------------------------------
# IMPORTANT DiscoGMPI-specific recommendation 
# # ------------------------------------------------------------------------------------------------------------------------------
# Because DiscoG is DG-FEM, follow ownership convention:
# - solution arrays store values only for nlocal_elems = owned + ghosts 
# - RHS/time updates are applied only on partition.owned 
# - ghost entries are overwritten by MPI exchange before flux evaluation 
# That lets most of the kernels work with local element indexing, while the outher driver simply restricts updates to owned elements 

# ------------------------------------------------------------------------------------------------------------------------------


end # module DiscoGMPI
