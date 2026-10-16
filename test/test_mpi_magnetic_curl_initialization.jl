using Test, LinearAlgebra, MPI, DiscoGMPI

MPI.Init()
try
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    mesh = structured_box_mesh(2,2,1; lower=(0.0,0.0,0.0), upper=(1.0,1.0,1.0),
        boundary_id=periodic_box_boundary_id)
    potential = (x,y,z,t)->(z^3+x*y, x^3+y*z, y^3+z*x)
    @testset "Owned initialization and ordinary/periodic MPI traces" begin
        for nodes in (:equispaced,:warp_blend), p in (2,4), electric_mode in (false,true)
            initialize! = electric_mode ? initialize_electric_from_potential! : initialize_magnetic_from_potential!
            diagnose = electric_mode ? electric_initialization_diagnostics : magnetic_initialization_diagnostics
            names = electric_mode ? (:Ex,:Ey,:Ez) : (:Hx,:Hy,:Hz)
            other_names = electric_mode ? (:Hx,:Hy,:Hz) : (:Ex,:Ey,:Ez)
            ddg = build_distributed_dg_from_root(rank == 0 ? mesh : nothing,
                rank == 0 ? balanced_spatial_partition(mesh,nranks) : nothing,p;
                comm,nodal_set=nodes)
            serial = build_distributed_dg_from_root(mesh,zeros(Int,size(mesh.tets,2)),p;
                comm=MPI.COMM_SELF,nodal_set=nodes)
            electric = (x,y,z)->(x,y,z)
            magnetic = (x,y,z)->(z,x,y)
            U = interpolate_maxwell_field(ddg,electric,magnetic)
            S = interpolate_maxwell_field(serial,electric,magnetic)
            ids = ddg.distributed_mesh.elements.global_ids
            coefficients = [0.7+0.3*(g%3) for g in ids]
            scoefficients = [0.7+0.3*(g%3) for g in 1:size(mesh.tets,2)]
            materials = electric_mode ? MaxwellElementMaterials(coefficients,ones(length(ids))) :
                MaxwellElementMaterials(ones(length(ids)),coefficients)
            smaterials = electric_mode ? MaxwellElementMaterials(scoefficients,ones(size(mesh.tets,2))) :
                MaxwellElementMaterials(ones(size(mesh.tets,2)),scoefficients)
            periodic = build_distributed_periodic_maxwell_exchange(ddg)
            ws = MagneticCurlInitializationWorkspace(ddg.dg.ref;nodal_set=nodes)
            other = map(name -> copy(getproperty(U,name)),other_names)
            initialize!(U,ddg,materials,potential;workspace=ws,periodic)
            initialize!(S,serial,smaterials,potential;nodal_set=nodes)
            @test map(name -> getproperty(U,name),other_names) == other
            # A <= 2 on the unit cube; derivative and conditioning scales are measured.
            D = maximum(opnorm(op.Dx,Inf)+opnorm(op.Dy,Inf)+opnorm(op.Dz,Inf)
                for op in ddg.dg.physops.elements)
            G = maximum(opnorm(m.invJ,1) for m in ddg.dg.mappings.tet_mappings) *
                (opnorm(ws.Rr,Inf)+opnorm(ws.Rs,Inf)+opnorm(ws.Rt,Inf))
            field_tol = 64eps(Float64)*ws.potential.Np*max(cond(ws.potential.V),cond(ws.target.V))*(2G+6)/0.7
            div_tol = field_tol*D*maximum(coefficients)
            owned = ddg.distributed_mesh.partition.owned
            for name in names
                a,b = getproperty(U,name),getproperty(S,name)
                @test maximum(abs,a[:,owned]-b[:,ids[owned]]) <= field_tol
                for faces in values(ddg.exchange.faces), face in faces
                    @test maximum(abs,a[face.ghost_nodes,face.ghost_elem]-
                        b[face.ghost_nodes,ids[face.ghost_elem]]) <= field_tol
                end
            end
            for face in periodic.faces
                plus = DiscoGMPI.distributed_periodic_plus_trace(U,periodic,face,rank)
                partner_nodes = serial.dg.fops.face_nodes[face.partner_face][face.partner_to_local_perm]
                for name in names
                    @test maximum(abs,getproperty(plus,name)-
                        getproperty(S,name)[partner_nodes,face.partner_global_elem]) <= field_tol
                end
            end
            d = diagnose(U,ddg,materials)
            s = diagnose(S,serial,smaterials)
            @test d.broken_divergence_l2 <= div_tol
            charge = electric_mode ? d.signed_electric_charge : d.signed_magnetic_charge
            @test abs(charge) <= div_tol
            @test abs(d.broken_divergence_l2-s.broken_divergence_l2) <= div_tol
            if !electric_mode
                @test abs(distributed_strong_magnetic_charge(U,ddg,materials)-charge) <= div_tol
            end
            rank == 0 && println("ranks=$nranks nodes=$nodes p=$p electric=$electric_mode diagnostics=$d tolerance=$div_tol")
            # Errors on just one rank must reach every peer before halo exchange.
            bad = rank == 0 ? ((x,y,z,t)->(NaN,0.0,0.0)) : potential
            @test_throws ErrorException initialize!(U,ddg,materials,bad;workspace=ws)
            invalid_material = rank == 0 ? ((x,y,z)->1.0) : materials
            @test_throws ErrorException initialize!(U,ddg,invalid_material,potential;workspace=ws)
        end
    end
finally
    MPI.Finalize()
end
