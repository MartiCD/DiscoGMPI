using Test, LinearAlgebra, MPI, DiscoGMPI
MPI.Initialized() || MPI.Init()
comm = MPI.COMM_WORLD
rank,nranks = MPI.Comm_rank(comm),MPI.Comm_size(comm)
@testset "Mixed-order MPI traces and residuals" begin
    for periodic in (false,true), flux in (MaxwellFlux_Central,MaxwellFlux_Alternating)
        mesh = rank==0 ? structured_box_mesh(2,1,1;lower=(0.,0.,0.),upper=(1.,1.,1.),boundary_id=periodic_box_boundary_id) : nothing
        parts = rank==0 ? [mod(i-1,nranks) for i in axes(mesh.tets,2)] : nothing
        specs = periodic ? DiscoGMPI.box_periodic_specs(1.,1.,1.) : DiscoGMPI.PeriodicBoundarySpec[]
        ddg = build_distributed_mixed_dg_from_root(mesh,parts,2;comm,periodic=specs)
        E(x,y,z) = (sin(2pi*y),cos(2pi*x),sin(2pi*z))
        H(x,y,z) = (cos(2pi*z),sin(2pi*y),cos(2pi*x))
        U = interpolate_maxwell_field(ddg,E,H)
        R = DiscoGMPI.similar_maxwell_rhs(U)
        rhs! = DiscoGMPI.make_maxwell_rhs_function(ddg;flux_kind=flux)
        rhs!(R,U)
        refR = if rank==0
            dg = MixedDGDiscretization(mesh,2;periodic=specs)
            V = interpolate_maxwell_field(dg,E,H)
            S = DiscoGMPI.similar_maxwell_rhs(V)
            DiscoGMPI.make_maxwell_rhs_function(dg;flux_kind=flux)(S,V)
            S
        else
            nothing
        end
        refR = MPI.bcast(refR,comm;root=0)
        for c in 1:6,e in ddg.owned
            @test DiscoGMPI.mixed_components(R)[c][:,e] ≈
                DiscoGMPI.mixed_components(refR)[c][:,ddg.global_ids[e]] atol=1e-10 rtol=1e-10
        end
        for r in ddg.exchange.neighbors, (i,key) in enumerate(ddg.exchange.receives[r])
            @test ddg.exchange.traces[key] == ddg.exchange.recv_buffers[r][:,6(i-1)+1:6i]
        end
        d = mixed_diagnostic_values(mixed_diagnostics(U,ddg))
        @test d.volume ≈ 1.
        @test d.energy > 0
        @test isfinite(mixed_cfl_dt(ddg))
        if periodic && flux==MaxwellFlux_Central
            # Broadcast the root temporary path: all rank pieces share a folder.
            dir = MPI.bcast(rank==0 ? mktempdir() : nothing,comm;root=0)
            write_distributed_checkpoint(dir,U,ddg;step=3,time=0.03,dt=0.01)
            V,state = load_distributed_checkpoint(dir,ddg)
            @test state.step==3
            for c in 1:6
                @test DiscoGMPI.mixed_components(V)[c][:,ddg.owned] == DiscoGMPI.mixed_components(U)[c][:,ddg.owned]
            end
            write_parallel_maxwell_fields(joinpath(dir,"mixed"),ddg,U;time=0.)
            @test isfile(joinpath(dir,"mixed.pvtu")) || rank!=0
            @test isfile(write_maxwell_integration_points(dir,ddg,U,6))
            scheme = mixed_esprk_scheme(2)
            workU,workV = MaxwellPartitionedRKWorkspace(U,scheme),MaxwellPartitionedRKWorkspace(V,scheme)
            for _ in 1:2
                DiscoGMPI.partitioned_symplectic_rk_step!(U,workU,scheme,0.001,rhs!)
                DiscoGMPI.partitioned_symplectic_rk_step!(V,workV,scheme,0.001,rhs!)
            end
            for c in 1:6
                @test DiscoGMPI.mixed_components(V)[c][:,ddg.owned] == DiscoGMPI.mixed_components(U)[c][:,ddg.owned]
            end
            wrong = build_distributed_mixed_dg_from_root(mesh,parts,1;comm,periodic=specs)
            @test_throws Exception load_distributed_checkpoint(dir,wrong)
            if rank==0
                open(joinpath(dir,"mixed_rank0000.bin"),"a") do io
                    write(io,UInt8(0))
                end
            end
            MPI.Barrier(comm)
            @test_throws Exception load_distributed_checkpoint(dir,ddg)
        end
    end
end
MPI.Finalize()
