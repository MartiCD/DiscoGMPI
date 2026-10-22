# Fixed tiny runs test history/field recovery, not a convergence experiment.
# julia --project=. test/test_mixed_invariant_restart.jl
# mpiexec -n 2 julia --project=. test/test_mixed_invariant_restart.jl
using Test, Serialization
include(joinpath(@__DIR__,"..","examples","run_mixed_conserved_quantities.jl"))
include(joinpath(@__DIR__,"..","examples","meshes","generate_periodic_box_2x1x1_unstructured.jl"))
MPI.Init()
comm=MPI.COMM_WORLD
rank=MPI.Comm_rank(comm)
root=MPI.bcast(rank==0 ? mktempdir() : nothing,comm;root=0)

@testset "Invariant fields and full history survive continuation" begin
    for (p,flux) in ((1,"centered"),(2,"alternating"))
        directory=joinpath(root,"p$p")
        config=joinpath(directory,"case.toml")
        c=nothing
        collective_root_action(comm,"Prepare invariant restart test") do
            mkpath(directory)
            mesh=structured_box_mesh(2,1,1;upper=(2.,1.,1.))
            vtk=joinpath(directory,"mesh.vtk")
            write_vtk(vtk,mesh.points,Tuple.(eachcol(mesh.tris)),Tuple.(eachcol(mesh.tets)),ones(Int,size(mesh.tris,2)))
            c=Dict("name"=>"test_p$p","discretization"=>"mixed","p_E"=>p,"p_H"=>p+1,
                "esprk_order"=>p+2,"flux"=>flux,"sample_every"=>2,"checkpoint_every"=>2,
                "print_every"=>1000,"final_time"=>.01,"cfl"=>.002,"cfl_degree_rule"=>"linear",
                "cubature_degree"=>4*(p+1)+2,"electric_initialization"=>"interpolate",
                "magnetic_initialization"=>"curl-potential","mesh"=>vtk,
                "mesh_sha256"=>bytes2hex(sha256(read(vtk))),"nelements"=>size(mesh.tets,2),
                "output"=>joinpath(directory,"reference"))
            DiscoGMPI.atomic_toml(config,c)
        end
        c=MPI.bcast(c,comm;root=0)
        ref=c["output"]
        run_mixed_invariants(config;warmup_steps=1,finalize_mpi=false)
        expected=read(joinpath(ref,"diagnostics","invariant_diagnostics.csv"),String)
        nsteps=TOML.parsefile(joinpath(ref,"completed.toml"))["steps"]
        @test nsteps>6
        resumed=joinpath(directory,"resumed")
        run_mixed_invariants(config;output_override=resumed,segment_steps=4,warmup_steps=1,finalize_mpi=false)
        @test !isfile(joinpath(resumed,"completed.toml"))
        candidates=invariant_checkpoint_candidates(resumed,comm)
        @test length(candidates)==2
        collective_root_action(comm,"Corrupt newest invariant field piece") do
            open(joinpath(first(candidates),"mixed_rank0000.bin"),"a") do io
                write(io,UInt8(0))
            end
        end
        run_mixed_invariants(config;output_override=resumed,warmup_steps=1,finalize_mpi=false)
        @test read(joinpath(resumed,"diagnostics","invariant_diagnostics.csv"),String)==expected
        @test TOML.parsefile(joinpath(resumed,"config","run_metadata.toml"))["first_step"]==2
        filename="mixed_rank$(DiscoGMPI.rank_file_label(rank)).bin"
        refslot=TOML.parsefile(joinpath(ref,"checkpoints","latest.toml"))["slot"]
        slot=TOML.parsefile(joinpath(resumed,"checkpoints","latest.toml"))["slot"]
        payload=deserialize(joinpath(resumed,"checkpoints",slot,filename))
        @test payload.components==deserialize(joinpath(ref,"checkpoints",refslot,filename)).components
        @test payload.metadata["initial_diagnostics"]==deserialize(joinpath(ref,"checkpoints",refslot,filename)).metadata["initial_diagnostics"]
        collective_root_action(comm,"Ignore an uncommitted truncated history suffix") do
            segment=last(payload.metadata["history_segments"])
            open(segment,"a") do io
                write(io,"crash left an incomplete row")
            end
            @test invariant_merge_history(payload.metadata["history_segments"],
                payload.metadata["history_ends"],first(split(expected,'\n')),payload.dt)==nsteps
        end
        # A completed dependent job is a cheap no-op.
        run_mixed_invariants(config;output_override=resumed,finalize_mpi=false)
        @test read(joinpath(resumed,"diagnostics","invariant_diagnostics.csv"),String)==expected

        # Diagnostic cadence does not alter the integrated fields.
        sparse=joinpath(directory,"sparse")
        run_mixed_invariants(config;output_override=sparse,sample_every=100,checkpoint_every=0,
            warmup_steps=1,finalize_mpi=false)
        sparseslot=TOML.parsefile(joinpath(sparse,"checkpoints","latest.toml"))["slot"]
        @test payload.components==deserialize(joinpath(sparse,"checkpoints",sparseslot,filename)).components
        @test length(readlines(joinpath(sparse,"diagnostics","invariant_diagnostics.csv")))==3

        # The legacy flat checkpoint + single CSV is upgraded in place.
        legacy=joinpath(directory,"legacy")
        step4path=joinpath(legacy,"checkpoints","step4")
        bench=joinpath(directory,"pilot")
        run_mixed_invariants(config;output_override=bench,benchmark_steps=4,warmup_steps=2,finalize_mpi=false)
        @test !isfile(joinpath(bench,"completed.toml"))
        @test TOML.parsefile(joinpath(bench,"benchmark.toml"))["measured_steps"]==4
        benchslot=TOML.parsefile(joinpath(bench,"checkpoints","latest.toml"))["slot"]
        b=deserialize(joinpath(bench,"checkpoints",benchslot,filename))
        collective_root_action(comm,"Prepare legacy history") do
            mkpath(step4path)
            mkpath(joinpath(legacy,"diagnostics"))
            # Includes the pilot prefix up to step 6.
            segment=last(b.metadata["history_segments"])
            cp(segment,joinpath(legacy,"diagnostics","invariant_diagnostics.csv"))
        end
        old=merge(b,(metadata=Dict{String,Any}(c),))
        piece=joinpath(step4path,filename)
        DiscoGMPI.atomic_serialize(piece,old)
        checksums=MPI.gather(bytes2hex(sha256(read(piece))),comm;root=0)
        collective_root_action(comm,"Commit legacy checkpoint") do
            manifest=TOML.parsefile(joinpath(bench,"checkpoints",benchslot,"mixed_manifest.toml"))
            manifest["checksums"]=checksums
            DiscoGMPI.atomic_toml(joinpath(step4path,"mixed_manifest.toml"),manifest)
        end
        run_mixed_invariants(config;output_override=legacy,warmup_steps=1,finalize_mpi=false)
        @test read(joinpath(legacy,"diagnostics","invariant_diagnostics.csv"),String)==expected

        # Cooperative wall-time exit commits step zero, then continues safely.
        wall=joinpath(directory,"wall")
        run_mixed_invariants(config;output_override=wall,max_wall_seconds=1e-9,finalize_mpi=false)
        @test TOML.parsefile(joinpath(wall,"segment_status.toml"))["reason"]=="walltime"
        @test !isfile(joinpath(wall,"completed.toml"))
        run_mixed_invariants(config;output_override=wall,warmup_steps=1,finalize_mpi=false)
        @test read(joinpath(wall,"diagnostics","invariant_diagnostics.csv"),String)==expected
        # A final checkpoint can rebuild the final CSV after interrupted merge.
        collective_root_action(comm,"Simulate interruption before completion marker") do
            rm(joinpath(wall,"completed.toml"))
        end
        run_mixed_invariants(config;output_override=wall,warmup_steps=1,finalize_mpi=false)
        @test read(joinpath(wall,"diagnostics","invariant_diagnostics.csv"),String)==expected
        collective_root_action(comm,"Change physics") do
            c["flux"]=flux=="centered" ? "alternating" : "centered"
            DiscoGMPI.atomic_toml(config,c)
            rm(joinpath(wall,"completed.toml"))
        end
        @test_throws Exception run_mixed_invariants(config;output_override=wall,finalize_mpi=false)
    end
end
rank==0 && println("Invariant restart test outputs: $root")
MPI.Finalize()
