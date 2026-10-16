using Test
using MPI
using DiscoGMPI

module CavityConvergenceTest
include(joinpath(@__DIR__, "..", "examples", "convergence_distributed_poisson_bracket_maxwell.jl"))
end
module PeriodicConvergenceTest
include(joinpath(@__DIR__, "..", "examples", "convergence_distributed_periodic_poisson_bracket_maxwell.jl"))
end

MPI.Init()
try
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    @testset "Convergence driver nodal sets, fluxes, and MPI reductions" begin
        for nodes in ("equispaced", "warp-blend"), flux in ("centered", "alternating")
            P = PeriodicConvergenceTest
            config = P.parse_periodic_convergence_arguments([
                "--mesh-family=structured", "--nx-targets=2,4", "--orders=3",
                "--final-time=0.01", "--nodal-set=$nodes", "--flux=$flux", "--rate-policy=report"])
            root_mesh = rank == 0 ? P.build_structured_periodic_convergence_mesh(2) : nothing
            result = P.run_distributed_periodic_convergence_case(2, 3, 0, config, comm, root_mesh)
            row = P.periodic_convergence_row(result, config)
            serial = P.run_distributed_periodic_convergence_case(
                2, 3, 0, config, MPI.COMM_SELF, P.build_structured_periodic_convergence_mesh(2))
            reference = P.periodic_convergence_row(serial, config)
            @test row.nodal_set == parse_convergence_nodal_set(nodes)
            @test row.mpi_ranks == nranks
            @test row.error_samples == row.nsteps + 1
            @test join(string.(keys(row)), ',') == maxwell_convergence_csv_header()
            for name in MAXWELL_CONVERGENCE_QUANTITIES
                key = Symbol("linf_time_l2_space_$(name)_error")
                @test row[key] >= row[Symbol("final_l2_space_$(name)_error")] - 1e-13
                @test row[key] ≈ reference[key] rtol=1e-10 atol=1e-12
            end
        end
        C = CavityConvergenceTest
        for nodes in ("equispaced", "warp-blend"), bc in ("pec", "pmc")
            config = C.parse_convergence_arguments([
                "--cells=1,2", "--orders=3", "--jitter=0", "--final-time=0.01",
                "--nodal-set=$nodes", "--boundary-condition=$bc", "--rate-policy=report"])
            result = C.run_distributed_convergence_case(1, 3, 0, config, comm)
            row = C.cavity_convergence_row(result, config)
            serial = C.run_distributed_convergence_case(1, 3, 0, config, MPI.COMM_SELF)
            reference = C.cavity_convergence_row(serial, config)
            @test join(string.(keys(row)), ',') == maxwell_convergence_csv_header()
            for name in MAXWELL_CONVERGENCE_QUANTITIES
                key = Symbol("linf_time_l2_space_$(name)_error")
                @test row[key] ≈ reference[key] rtol=1e-10 atol=1e-12
            end
        end
    end

    @testset "Collective convergence failure writes CSV first" begin
        directory = MPI.bcast(rank == 0 ? mktempdir() : "", comm; root=0)
        path = joinpath(directory, "failed.csv")
        rows = [maxwell_convergence_row((order=2, mesh_level=l, h_max=0.5^l,
                    dt=0.01, nodal_set=:equispaced, boundary_condition=:periodic),
                    ntuple(_ -> 1.0, 9), ntuple(_ -> 0.1, 9)) for l in 0:1]
        @test_throws ErrorException finish_maxwell_convergence(rows, path, comm)
        @test MPI.bcast(rank == 0 && isfile(path), comm; root=0)
        MPI.Barrier(comm)
        rank == 0 && rm(directory; recursive=true)
    end
finally
    MPI.Finalize()
end
