using Test

include("test_run_layout.jl")
include("test_empty_domain_incident_pml_gate.jl")
include("test_pec_sphere_boundary_residual_gate.jl")
include("test_scattered_ex_hy_fields_gate.jl")
include("test_coarse_sphere_rcs_gate.jl")
include("test_solver.jl")
include("test_metis_distributed_mesh.jl")
include("test_validation_matrix.jl")
include("test_periodic_isoresolution.jl")

# MPI tests should be launched explicitly with mpiexec, for example:
#   mpiexec -n 2 julia --project=test test/test_mpi_face_permutation.jl
#
# Or from the package root:
#   mpiexec -n 2 julia --project=. test/test_mpi_face_permutation.jl
