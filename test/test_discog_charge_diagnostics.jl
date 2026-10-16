using Test, LinearAlgebra, DiscoGMPI

@testset "DiscoG contraction fidelity" begin
    I = [1.0 0.0; 0.25 0.75; 0.0 1.0]
    w = [0.2, 0.7, 0.3]
    U = ([2.0, 3.0], [-1.0, 4.0], [5.0, -2.0])
    G = ntuple(d -> d .* [-1.0 1.0; -1.0 1.0; -1.0 1.0], 3)
    C = ntuple(d -> d .* [0.2 -0.2; -0.7 0.7], 3)
    expected = ntuple(d -> sum(w[q] * I[q,a] * C[d][a,b] * U[d][b]
                              for q in 1:3, a in 1:2, b in 1:2), 3)
    @test compute_electric_charge(U, I, w, G; epsilon=2) == 0
    @test compute_electric_charge_directional(U, I, w, G) == (0,0,0)
    @test collect(compute_magnetic_charge_directional(U, I, w, C; mu=7)) ≈ collect(expected)
    @test compute_magnetic_charge(U, I, w, C; mu=7) ≈ sum(expected)
    @test compute_electric_charge_strong(U, I, w, C; epsilon=2) ≈ 2sum(expected)
    @test compute_magnetic_charge_strong(U, I, w, C; mu=3) ≈ 3sum(expected)
    @test collect(compute_electric_charge_strong_directional(U,I,w,C;epsilon=2)) ≈ 2collect(expected)
    @test collect(compute_magnetic_charge_strong_directional(U,I,w,C;mu=3)) ≈ 3collect(expected)
end
