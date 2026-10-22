# Run explicitly: mpiexec -n 2 julia --project=. test/test_mixed_hp_restart.jl
# Tiny fixed-step runs exercise restart semantics, not convergence rates.
using Test, Serialization
include(joinpath(@__DIR__,"..","examples","run_mixed_hp_isoresolution_case.jl"))
include(joinpath(@__DIR__,"..","examples","meshes","generate_periodic_box_2x1x1_unstructured.jl"))
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
root = MPI.bcast(rank==0 ? mktempdir() : nothing,comm;root=0)

function checkpoint_test_case(root,mode,flux,p)
    mesh = joinpath(root,"$mode.vtk")
    m = structured_box_mesh(2,1,1;upper=mode=="plane_wave" ? (2.,1.,1.) : (1.,1.,1.))
    write_vtk(mesh,m.points,Tuple.(eachcol(m.tris)),Tuple.(eachcol(m.tets)),ones(Int,size(m.tris,2)))
    c = Dict{String,Any}("name"=>"test_$(mode)_$flux","discretization"=>"mixed",
        "p_E"=>p,"p_H"=>p+1,"order"=>p,"esprk_order"=>p+2,
        "Np_E"=>binomial(p+3,3),"Np_H"=>binomial(p+4,3),
        "mode"=>mode,"flux"=>flux,"nsteps"=>8,"final_time"=>.004,
        "cubature_degree"=>4*(p+1)+2,"mesh"=>mesh,"mesh_sha256"=>bytes2hex(sha256(read(mesh))),
        "nelements"=>size(m.tets,2),"nodal_set"=>"equispaced","cfl"=>.1,
        "cfl_degree_rule"=>"linear","level"=>1,"mesh_family"=>"structured",
        "h_max"=>1.,"mean_ratio_min"=>.5,"mean_ratio_avg"=>.5,
        "electric_dofs"=>3*size(m.tets,2)*binomial(p+3,3),
        "magnetic_dofs"=>3*size(m.tets,2)*binomial(p+4,3))
    c["field_dofs"]=c["electric_dofs"]+c["magnetic_dofs"]
    return c
end

@testset "Mixed hp field and max-time restart" begin
    for (mode,flux,p) in (("plane_wave","centered",1),("pec_cavity","alternating",2))
        directory=joinpath(root,mode)
        casefile=joinpath(directory,"case.toml")
        c=nothing
        DiscoGMPI.collective_root_action(comm,"Prepare restart test") do
            mkpath(directory)
            c=checkpoint_test_case(directory,mode,flux,p)
            c["output"]=joinpath(directory,"results","reference.csv")
            DiscoGMPI.atomic_toml(casefile,c)
        end
        c=MPI.bcast(c,comm;root=0)
        reference=joinpath(directory,"reference-checkpoints")
        resumed=joinpath(directory,"resumed-checkpoints")
        run_mixed_hp_case(casefile;checkpoint_dir=reference,checkpoint_every=2,finalize_mpi=false)
        DiscoGMPI.collective_root_action(comm,"Save reference result") do
            cp(c["output"],joinpath(directory,"reference.csv"))
            rm(c["output"])
        end
        run_mixed_hp_case(casefile;checkpoint_dir=resumed,checkpoint_every=2,
            stop_after_step=3,finalize_mpi=false)
        @test !isfile(c["output"])
        candidates=mixed_hp_checkpoint_candidates(resumed,comm)
        @test length(candidates)==2
        @test TOML.parsefile(joinpath(first(candidates),"mixed_manifest.toml"))["step"]==3
        # Corrupt the newest rank piece. Every rank must fall back together to
        # step 2, without losing prior samples or accepting a partial generation.
        DiscoGMPI.collective_root_action(comm,"Corrupt latest checkpoint") do
            open(joinpath(first(candidates),"mixed_rank0000.bin"),"a") do io
                write(io,UInt8(0))
            end
        end
        run_mixed_hp_case(casefile;checkpoint_dir=resumed,checkpoint_every=2,finalize_mpi=false)
        @test read(c["output"],String)==read(joinpath(directory,"reference.csv"),String)
        @test TOML.parsefile(replace(c["output"],".csv"=>"_metadata.toml"))["restart_step"]==2
        # Compare the actual distributed field pieces, not just error norms.
        file="mixed_rank$(DiscoGMPI.rank_file_label(rank)).bin"
        latestA=TOML.parsefile(joinpath(reference,"latest.toml"))["slot"]
        latestB=TOML.parsefile(joinpath(resumed,"latest.toml"))["slot"]
        a=deserialize(joinpath(reference,latestA,file))
        b=deserialize(joinpath(resumed,latestB,file))
        @test a.components==b.components
        @test a.metadata["max_errors"]==b.metadata["max_errors"]
        @test b.metadata["error_samples"]==9
        # A final-step checkpoint can regenerate output without another step.
        run_mixed_hp_case(casefile;checkpoint_dir=resumed,checkpoint_every=2,finalize_mpi=false)
        @test read(c["output"],String)==read(joinpath(directory,"reference.csv"),String)
        @test TOML.parsefile(replace(c["output"],".csv"=>"_metadata.toml"))["restart_step"]==8
        # Ensure an earlier maximum larger than every later error survives.
        b.metadata["max_errors"][9]=1234.
        piece=joinpath(resumed,latestB,file)
        DiscoGMPI.atomic_serialize(piece,b)
        checksums=MPI.gather(bytes2hex(sha256(read(piece))),comm;root=0)
        DiscoGMPI.collective_root_action(comm,"Commit synthetic prefix maximum") do
            manifest_path=joinpath(resumed,latestB,"mixed_manifest.toml")
            manifest=TOML.parsefile(manifest_path)
            manifest["checksums"]=checksums
            DiscoGMPI.atomic_toml(manifest_path,manifest)
        end
        run_mixed_hp_case(casefile;checkpoint_dir=resumed,finalize_mpi=false)
        lines=readlines(c["output"])
        column=findfirst(==("linf_time_l2_space_hz_error"),split(lines[1],','))
        @test parse(Float64,split(lines[2],',')[column])==1234.
        # Changed physics must be rejected instead of silently starting over.
        DiscoGMPI.collective_root_action(comm,"Change restart physics") do
            c["flux"]=flux=="centered" ? "alternating" : "centered"
            DiscoGMPI.atomic_toml(casefile,c)
        end
        @test_throws Exception run_mixed_hp_case(casefile;checkpoint_dir=resumed,finalize_mpi=false)
    end
end
rank==0 && println("Restart test outputs: ",root)
MPI.Finalize()
