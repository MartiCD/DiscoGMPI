using Test
include(joinpath(@__DIR__, "..", "examples", "run_mixed_hp_isoresolution_case.jl"))
MPI.Initialized() || MPI.Init()
BLAS.set_num_threads(1)

# Direct physical integration is an independent oracle for the cached kernel.
function direct_errors(ws, U, ddg, mode, t)
    E, H = mixed_hp_exact(mode,t)
    sums = zeros(6)
    fields = DiscoGMPI.mixed_components(U)
    for e in ddg.owned
        values = hcat([(c<=3 ? ws.VE : ws.VH)*fields[c][:,e] for c in 1:6]...)
        for q in eachindex(ws.weights)
            x = DiscoGMPI.map_to_physical(ddg.dg.mesh.points, view(ddg.dg.mesh.tets,:,e),
                ws.points[q,1], ws.points[q,2], ws.points[q,3])
            exact = (E(x...)..., H(x...)...)
            for c in 1:6
                sums[c] += ws.weights[q]*ddg.dg.mappings.tet_mappings[e].absdetJ*(values[q,c]-exact[c])^2
            end
        end
    end
    sums = MPI.Allreduce(sums,+,ddg.comm)
    return (sqrt(sum(sums[1:3])),sqrt(sum(sums[4:6])),sqrt(sum(sums)),sqrt.(sums)...)
end

@testset "Cached every-step errors" begin
    mesh = structured_box_mesh(1,1,1;lower=(0.,0.,0.),upper=(2.,1.,1.),boundary_id=periodic_box_boundary_id)
    dg = MixedDGDiscretization(mesh,4)
    # Noncontiguous owned elements catch confusion between cache and mesh indices.
    ddg = (dg=dg, owned=[1,3], comm=MPI.COMM_WORLD)
    @test mixed_cfl_dt(dg) ≈ 7.2*mixed_cfl_dt(dg;CFL=.1,degree_rule=:quadratic)
    @test_throws ArgumentError mixed_cfl_dt(dg;degree_rule=:invalid)
    for mode in ("plane_wave","pec_cavity")
        E,H = mixed_hp_exact(mode,0.)
        U = interpolate_maxwell_field(dg,E,H)
        ws = mixed_hp_error_workspace(ddg,22,mode)
        @test !hasproperty(ws,:coords)
        diagnostics_ws = DiscoGMPI.mixed_diagnostic_workspace(dg;
            points_weights=(ws.points,ws.weights),basis=(ws.VE,ws.VH))
        @test diagnostics_ws.points === ws.points
        @test diagnostics_ws.V[1] === ws.VE
        @test mixed_diagnostics(U,dg;elements=ddg.owned,workspace=diagnostics_ws) ≈
            mixed_diagnostics(U,dg;elements=ddg.owned,degree=22)
        for t in (0.,.137,.5,1.,2.)
            @test collect(mixed_hp_errors!(ws,U,ddg,mode,t)) ≈ collect(direct_errors(ws,U,ddg,mode,t)) rtol=2e-12 atol=1e-14
        end
        reduction_buffer = ws.global_sums
        mixed_hp_errors!(ws,U,ddg,mode,.137)
        @test ws.global_sums === reduction_buffer
        println(mode," cached_error_bytes=",@allocated(mixed_hp_errors!(ws,U,ddg,mode,.137)))
        fast = @elapsed for _ in 1:10; mixed_hp_errors!(ws,U,ddg,mode,.137); end
        direct = @elapsed for _ in 1:10; direct_errors(ws,U,ddg,mode,.137); end
        println(mode, " cached_seconds=",fast," direct_seconds=",direct)
    end
end
MPI.Finalize()
