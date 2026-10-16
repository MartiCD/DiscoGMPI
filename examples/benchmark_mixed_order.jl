using DiscoGMPI, LinearAlgebra, Printf
BLAS.set_num_threads(1)
mesh = structured_box_mesh(2,2,2)
function benchmark_rhs(rhs!,R,U)
    rhs!(R,U)
    bytes = @allocated rhs!(R,U)
    samples = [@elapsed rhs!(R,U) for _ in 1:30]
    return sort(samples)[15]*1000,bytes
end
for p in 1:4
    for mixed in (false,true)
        dg = mixed ? MixedDGDiscretization(mesh,p) : DGDiscretization(mesh,p)
        E(x,y,z) = (sin(x),cos(y),sin(z))
        H(x,y,z) = (cos(x),sin(y),cos(z))
        U = mixed ? interpolate_maxwell_field(dg,E,H) : interpolate_maxwell_field(mesh,dg.ref,E,H)
        rhs! = mixed ? DiscoGMPI.make_maxwell_rhs_function(dg) :
            DiscoGMPI.make_maxwell_rhs_function(dg,MaxwellBoundaryRegistry(Dict(1=>MaxwellBC_PEC)),PoissonBracketFormulation(MaxwellFlux_Central))
        R = DiscoGMPI.similar_maxwell_rhs(U)
        milliseconds,bytes = benchmark_rhs(rhs!,R,U)
        @printf("p=%d mixed=%s median_ms=%.6f bytes=%d\n",p,mixed,milliseconds,bytes)
    end
end
