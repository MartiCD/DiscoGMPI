#!/usr/bin/env julia
using DiscoGMPI, MPI, LinearAlgebra, TOML, Dates
const Audit=DiscoGMPI.MaxwellInvariantAudit

"""Small serial A/B/C experiment. Output paths must not already exist."""
function precision_main(args=ARGS)
    cfg=Dict("order"=>"2","bits"=>"128,256","flux"=>"centered",
        "output-dir"=>"output/invariant_precision/"*Dates.format(now(),"yyyymmdd_HHMMSSsss"))
    for a in args
        a=="--help" && (println("--order=2 --bits=128,256 --flux=centered|alternating --output-dir=NEW_PATH");return)
        k,v=split(a[3:end],'=';limit=2);haskey(cfg,k) || error("Unknown option $k");cfg[k]=v
    end
    initialized=MPI.Initialized(); initialized || MPI.Init()
    try
        MPI.Comm_size(MPI.COMM_WORLD)==1 || error("BigFloat oracle must run with one rank.")
        root=abspath(cfg["output-dir"]);ispath(root) && error("Refusing overwrite: $root")
        p=parse(Int,cfg["order"]); bits=parse.(Int,split(cfg["bits"],','))
        all(b->b>=64,bits) || error("Precision must be >=64 bits.")
        mesh=structured_box_mesh(2,1,1;lower=(0.,0.,0.),upper=(2.,1.,1.),boundary_id=periodic_box_boundary_id)
        flux=parse_maxwell_flux_kind(cfg["flux"])
        case=Audit.build_case(mesh,p,MPI.COMM_SELF;flux)
        U=Audit.initial_field(case)
        # Identical exactly representable coefficient vector for A, B and C.
        for (c,field) in enumerate(Audit.components(U)),e in axes(field,2),i in axes(field,1)
            field[i,e]=(mod(13i+7e+3c,31)-15)/16
        end
        R=DiscoGMPI.similar_maxwell_rhs(U);case.rhs!(R,U)
        v=Audit.coefficient_array(U); r=Audit.coefficient_array(R)
        op=Audit.precision_operators(case.ddg,case.periodic,case.ws,Float64)
        oracle=Audit.precision_rhs(op,v,flux)
        oracle_error=maximum(abs,oracle-r);scale=max(maximum(abs,r),1.0)
        tolerance=1024eps(Float64)*case.ddg.dg.ref.Np*cond(case.ddg.dg.ref.M)*scale
        oracle_error<=tolerance || error("Float64 oracle disagrees with production: $oracle_error > $tolerance")
        mkpath(root)
        metadata=Dict("options"=>cfg,"julia_version"=>string(VERSION),"mpi_library"=>MPI.Get_library_version(),
            "production_oracle_max_error"=>oracle_error,"production_oracle_tolerance"=>tolerance,
            "tetrahedra"=>size(mesh.tets,2),"weak_tests"=>length(case.ws.keys),
            "coefficient_data"=>"identical Float64 dyadic vector promoted exactly",
            "construction_C"=>"equispaced rational barycentric nodes, exact simplex moments; BigFloat geometry/basis/mass/solves",
            "construction_C_limit"=>"production integer topology/permutations and exact binary input vertices reused; no curved geometry, warp blend, or variable material",
            "command"=>join(vcat(Base.julia_cmd().exec,PROGRAM_FILE,ARGS)," "),
            "git_revision"=>try readchomp(`git rev-parse HEAD`) catch; "unavailable" end,
            "git_status"=>try readchomp(`git status --short`) catch; "unavailable" end)
        open(joinpath(root,"metadata.toml"),"w") do io;TOML.print(io,metadata);end
        open(joinpath(root,"precision.csv"),"w") do io
            println(io,"variant,bits,data,field,charge_residual,charge_scale,charge_normalized,weak_max_abs,weak_max_normalized,rhs_max_abs")
            function record(label,bits,data,ops,rhs)
                d=Audit.precision_residuals(ops,rhs)
                for f in 1:2
                    println(io,join((label,bits,data,f,d.charge[f],d.charge_scale[f],d.charge_normalized[f],
                        maximum(abs,d.weak[f,:]),maximum(d.weak_normalized[f,:]),maximum(abs,rhs)),','))
                end
            end
            record("A_production",53,"fixed_coefficients",op,r)
            record("A_oracle",53,"fixed_coefficients",op,oracle)
            for b in bits
                setprecision(b) do
                    promoted=Audit.precision_operators(case.ddg,case.periodic,case.ws,BigFloat)
                    rebuilt=Audit.precision_operators(case.ddg,case.periodic,case.ws,BigFloat;rebuild=true)
                    x=BigFloat.(v)
                    rb=Audit.precision_rhs(promoted,x,flux)
                    rc=Audit.precision_rhs(rebuilt,x,flux)
                    record("A_promoted_diagnostic",b,"fixed_coefficients",promoted,BigFloat.(r))
                    record("B_promoted",b,"fixed_coefficients",promoted,rb)
                    record("C_rebuilt",b,"fixed_coefficients",rebuilt,rc)
                    record("B_rhs_C_diagnostic",b,"fixed_coefficients",rebuilt,rb)
                    record("C_rhs_B_diagnostic",b,"fixed_coefficients",promoted,rc)
                    # Separate experiment: regenerate a polynomial/trigonometric coefficient
                    # pattern using high precision; never conflate with fixed-vector A/B/C.
                    xnew=[sin(BigFloat(i+2e+3c)/7) for i in axes(v,1),e in axes(v,2),c in 1:6]
                    record("C_rebuilt",b,"regenerated_data",rebuilt,Audit.precision_rhs(rebuilt,xnew,flux))
                end
            end
        end
        println("Float64 production/oracle max difference: ",oracle_error,"; results: ",root)
    finally
        initialized || MPI.Finalize()
    end
end
abspath(PROGRAM_FILE)==abspath(@__FILE__) && precision_main()
