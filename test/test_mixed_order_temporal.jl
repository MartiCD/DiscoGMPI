# Explicit opt-in: convergence checks are deferred to the future study.
using Test, LinearAlgebra, DiscoGMPI

@testset "ESPRK temporal order with unequal partitions" begin
    B = [1. 0.3 0.1; 0.2 0.8 -0.1]
    A = [zeros(2,2) B; -B' zeros(3,3)]
    initial = [0.7,-0.2,0.3,0.4,-0.1]
    target = exp(A)*initial
    function rectangular_rhs!(R,U)
        DiscoGMPI.fill_maxwell_rhs!(R,0.)
        mul!(R.rhsEx,B,U.Hx)
        mul!(R.rhsHx,B',U.Ex,-1.,0.)
    end
    for p in 1:4
        errors = Float64[]
        for steps in (4,8,16)
            U = MaxwellField(reshape(copy(initial[1:2]),2,1),zeros(2,1),zeros(2,1),
                reshape(copy(initial[3:5]),3,1),zeros(3,1),zeros(3,1))
            scheme = mixed_esprk_scheme(p)
            work = MaxwellPartitionedRKWorkspace(U,scheme)
            for _ in 1:steps
                DiscoGMPI.partitioned_symplectic_rk_step!(U,work,scheme,1.0/steps,rectangular_rhs!)
            end
            push!(errors,norm(vcat(vec(U.Ex),vec(U.Hx))-target))
        end
        @test log2(errors[2]/errors[3]) > p+2-0.25
    end
end
