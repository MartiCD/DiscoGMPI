using Test, DiscoGMPI, MPI, LinearAlgebra
const Audit=DiscoGMPI.MaxwellInvariantAudit
MPI.Initialized() || MPI.Init()
try
    @test MPI.Comm_size(MPI.COMM_WORLD)==1
    @testset "Full periodic precision oracle" begin
        for p in (1,2),flux in (MaxwellFlux_Central,MaxwellFlux_Alternating)
            mesh=structured_box_mesh(2,1,1;lower=(0.,0.,0.),upper=(2.,1.,1.),boundary_id=periodic_box_boundary_id)
            case=Audit.build_case(mesh,p,MPI.COMM_SELF;flux)
            U=Audit.initial_field(case)
            for (c,field) in enumerate(Audit.components(U)),e in axes(field,2),i in axes(field,1)
                field[i,e]=(mod(13i+7e+3c,31)-15)/16
            end
            R=DiscoGMPI.similar_maxwell_rhs(U);case.rhs!(R,U)
            x=Audit.coefficient_array(U);r=Audit.coefficient_array(R)
            op=Audit.precision_operators(case.ddg,case.periodic,case.ws,Float64)
            predicted=Audit.precision_rhs(op,x,flux)
            tolerance=1024eps(Float64)*case.ddg.dg.ref.Np*cond(op.M)*max(maximum(abs,r),1)
            @test maximum(abs,predicted-r)<=tolerance
            b128=nothing
            for bits in (128,256)
                setprecision(bits) do
                    B=Audit.precision_operators(case.ddg,case.periodic,case.ws,BigFloat)
                    C=Audit.precision_operators(case.ddg,case.periodic,case.ws,BigFloat;rebuild=true)
                    rb=Audit.precision_rhs(B,BigFloat.(x),flux)
                    rc=Audit.precision_rhs(C,BigFloat.(x),flux)
                    db=Audit.precision_residuals(B,rb);dc=Audit.precision_residuals(C,rc)
                    budget=BigFloat(4096)*eps(BigFloat)*case.ddg.dg.ref.Np*BigFloat(cond(op.M))
                    @test maximum(dc.charge_normalized)<=budget
                    @test maximum(dc.weak_normalized)<=budget
                    @test all(isfinite,rb) && all(isfinite,rc)
                    if bits==128
                        b128=db.charge
                    else
                        # B converges to a fixed rounded-coefficient operator, not C.
                        @test maximum(abs,db.charge-b128)<=
                            BigFloat(4096)*BigFloat(2)^(-127)*maximum(db.charge_scale)*cond(op.M)
                    end
                    println("p=$p flux=$flux bits=$bits B charge=",db.charge,
                        "; C normalized charge=",dc.charge_normalized)
                end
            end
        end
    end
finally
    MPI.Finalize()
end
