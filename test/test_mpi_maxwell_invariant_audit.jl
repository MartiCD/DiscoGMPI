using Test, DiscoGMPI, MPI, LinearAlgebra
const Audit=DiscoGMPI.MaxwellInvariantAudit
MPI.Initialized() || MPI.Init()
try
    comm=MPI.COMM_WORLD;rank=MPI.Comm_rank(comm)
    mesh=structured_box_mesh(4,2,2;lower=(0.,0.,0.),upper=(2.,1.,1.),boundary_id=periodic_box_boundary_id)
    @testset "Periodic invariant audit" begin
        for flux in (MaxwellFlux_Central,MaxwellFlux_Alternating)
            case=Audit.build_case(rank==0 ? mesh : nothing,2,comm;flux)
            (;ddg,periodic,materials,ws,rhs!)=case
            U=Audit.initial_field(case;electric=:curl_potential)
            @test size(mesh.tets,2)==96
            @test length(ws.keys)==ws.total_tests==16
            initial=Audit.constraint_diagnostics(U,ddg,periodic,materials,ws)
            @test maximum(initial.div_l2)<1e-10
            @test sum(initial.facecount)==192
            for (c,field) in enumerate(Audit.components(U)),e in axes(field,2),i in axes(field,1)
                g=ddg.distributed_mesh.elements.global_ids[e]
                field[i,e]=(mod(13i+7g+3c,31)-15)/16
            end
            d=Audit.constraint_diagnostics(U,ddg,periodic,materials,ws)
            scale=max(1.0,maximum(abs,d.weak_volume),maximum(abs,d.weak_face))
            tol=4096eps(Float64)*ddg.dg.ref.Np*scale
            @test maximum(abs,d.weak_reconstruction_error)<=tol
            @test maximum(abs,d.charge-vec(sum(d.face_charge;dims=2)))<=tol
            @test minimum(d.jump_l2)>0
            R=DiscoGMPI.similar_maxwell_rhs(U);rhs!(R,U)
            rd=Audit.constraint_diagnostics(Audit.asfield(R),ddg,periodic,materials,ws)
            rhs_scale=MPI.Allreduce(maximum(maximum(abs,c) for c in Audit.components(R)),max,comm)
            @test maximum(abs,rd.weak)<=tol*max(rhs_scale,1.0)
            m=Audit.momentum_diagnostics(U,R,ddg,materials,ws)
            delta=2.0^-10; plus=deepcopy(U); minus=deepcopy(U)
            for (a,b,r) in zip(Audit.components(plus),Audit.components(minus),Audit.components(R))
                a .+= delta*r; b .-= delta*r
            end
            dp=(Audit.momentum_diagnostics(plus,R,ddg,materials,ws).momentum-
                Audit.momentum_diagnostics(minus,R,ddg,materials,ws).momentum)/(2delta)
            @test maximum(abs,dp-m.momentum_rate)<=tol/delta
            # Compare distributed values with COMM_SELF on exactly the same global mesh.
            serial=Audit.build_case(mesh,2,MPI.COMM_SELF;flux)
            S=Audit.initial_field(serial)
            for (c,field) in enumerate(Audit.components(S)),e in axes(field,2),i in axes(field,1)
                g=serial.ddg.distributed_mesh.elements.global_ids[e]
                field[i,e]=(mod(13i+7g+3c,31)-15)/16
            end
            sd=Audit.constraint_diagnostics(S,serial.ddg,serial.periodic,serial.materials,serial.ws)
            @test d.facecount==sd.facecount
            @test isapprox(d.jump_l2,sd.jump_l2;atol=tol,rtol=tol)
            @test isapprox(d.face_charge,sd.face_charge;atol=tol,rtol=tol)
            @test isapprox(d.weak,sd.weak;atol=tol,rtol=tol)
            @test isapprox(d.charge,sd.charge;atol=tol,rtol=tol)
            @test isapprox(d.div_l2,sd.div_l2;atol=tol,rtol=tol)
            # A constant field has zero jump, broken divergence and weak functional.
            constant=deepcopy(U)
            foreach(c->fill!(c,1.0),Audit.components(constant))
            cd=Audit.constraint_diagnostics(constant,ddg,periodic,materials,ws)
            @test maximum(cd.jump_l2)==0
            @test maximum(cd.div_l2)<=tol
            @test maximum(abs,cd.weak)<=tol
            # Instrumentation uses identical RHS/update operations, bitwise on owned DOFs.
            scheme=explicit_partitioned_symplectic_rk_scheme(3;first_partition=:H)
            plain=deepcopy(U); traced=deepcopy(U); n=Ref(0)
            observer=(i,p,a,u,r,phase)->(phase==:after && (n[]+=1);nothing)
            Audit.audited_step!(plain,MaxwellPartitionedRKWorkspace(plain,scheme),scheme,0.001,rhs!)
            Audit.audited_step!(traced,MaxwellPartitionedRKWorkspace(traced,scheme),scheme,0.001,rhs!;observer)
            @test n[]==Audit.update_counts(scheme).rhs
            @test all(a[:,ddg.distributed_mesh.partition.owned]==b[:,ddg.distributed_mesh.partition.owned]
                for (a,b) in zip(Audit.components(plain),Audit.components(traced)))
            high=setprecision(128) do;Audit.linear_values(U,ddg,materials,ws;T=BigFloat);end
            low=Audit.linear_values(U,ddg,materials,ws)
            @test maximum(abs,Float64.(high)-low)<=tol
            rank==0 && println("flux=$flux ranks=$(MPI.Comm_size(comm)) reconstruction=",
                maximum(abs,d.weak_reconstruction_error)," faces=",d.facecount,
                " momentum derivative error=",maximum(abs,dp-m.momentum_rate))
            rank==0 && println("  RHS charge=",rd.charge,"; weak max=",maximum(abs,rd.weak),
                "; broken RHS divergence L2=",rd.div_l2)
        end
    end
finally
    MPI.Finalize()
end
