using Test

include("test_solver.jl")
include("test_metis_distributed_mesh.jl")
include("test_validation_matrix.jl")

# MPI tests should be launched explicitly with mpiexec, for example:
#   mpiexec -n 2 julia --project=test test/test_mpi_face_permutation.jl
#
# Or from the package root:
#   mpiexec -n 2 julia --project=. test/test_mpi_face_permutation.jl
