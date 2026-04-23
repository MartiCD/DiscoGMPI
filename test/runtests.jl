using Test

@testset "DiscoG serial tests" begin
    # include("test_serial_something.jl")
end

# MPI tests should be launched explicitly with mpiexec, for example:
#   mpiexec -n 2 julia --project=test test/test_mpi_face_permutation.jl
#
# Or from the package root:
#   mpiexec -n 2 julia --project=. test/test_mpi_face_permutation.jl