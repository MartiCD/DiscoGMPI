using Test, LinearAlgebra, Random, DiscoGMPI

function mixed_rhs_allocations(f::F,R,U) where F
    f(R,U)
    return @allocated f(R,U)
end

function mixed_energy_pair(U,R,dg)
    sum(begin
        M = c<=3 ? dg.reference.M_E : dg.reference.M_H
        material = c<=3 ? dg.materials.epsilon[e] : dg.materials.permeability[e]
        dg.mappings.tet_mappings[e].absdetJ * material *
            dot(view(DiscoGMPI.mixed_components(U)[c],:,e),M*view(DiscoGMPI.mixed_components(R)[c],:,e))
    end for c in 1:6,e in axes(U.Ex,2))
end

@testset "Mixed-order Maxwell serial kernels" begin
    Random.seed!(7801)
    mesh = structured_box_mesh(1,1,1;lower=(0.,0.,0.),upper=(1.,1.,1.),boundary_id=periodic_box_boundary_id)
    for p in 1:4, flux in (MaxwellFlux_Central,MaxwellFlux_Alternating)
        dg = MixedDGDiscretization(mesh,p)
        U = MaxwellField(dg)
        R = DiscoGMPI.similar_maxwell_rhs(U)
        rhs! = DiscoGMPI.make_maxwell_rhs_function(dg;flux_kind=flux)
        rhs!(R,U)
        @test all(iszero,Iterators.flatten(DiscoGMPI.mixed_components(R)))
        for a in DiscoGMPI.mixed_components(U)
            randn!(a)
        end
        rhs!(R,U)
        @test abs(mixed_energy_pair(U,R,dg)) < 2e-10
        # Skew-adjointness tested with independent vectors as well as energy.
        V = MaxwellField(dg)
        foreach(randn!,DiscoGMPI.mixed_components(V))
        S = DiscoGMPI.similar_maxwell_rhs(V)
        rhs!(S,V)
        @test abs(mixed_energy_pair(U,S,dg)+mixed_energy_pair(V,R,dg)) < 2e-10
        scheme = mixed_esprk_scheme(p)
        @test scheme.order == p+2
        work = MaxwellPartitionedRKWorkspace(U,scheme)
        DiscoGMPI.partitioned_symplectic_rk_step!(U,work,scheme,1e-5,rhs!)
        @test all(isfinite,Iterators.flatten(DiscoGMPI.mixed_components(U)))
        @test size(U.Hx,1) > size(U.Ex,1)
        rhs!(R,U)
        bytes = @allocated rhs!(R,U)
        @test bytes == 0
    end
    @test_throws ArgumentError mixed_esprk_scheme(5)
    ne = size(mesh.tets,2)
    heterogeneous = MixedDGDiscretization(mesh,2;
        materials=MaxwellElementMaterials(collect(1.:ne),collect(2.:ne+1)))
    U = MaxwellField(heterogeneous)
    foreach(randn!,DiscoGMPI.mixed_components(U))
    R = DiscoGMPI.similar_maxwell_rhs(U)
    DiscoGMPI.make_maxwell_rhs_function(heterogeneous)(R,U)
    @test abs(mixed_energy_pair(U,R,heterogeneous)) < 2e-10
end

@testset "Mixed polynomial curls, periodic constants, and threading" begin
    mesh = structured_box_mesh(1,1,1;lower=(0.,0.,0.),upper=(1.,1.,1.),boundary_id=periodic_box_boundary_id)
    for permutation in ((1,2,3,4),(1,3,2,4),(2,3,1,4),(3,1,2,4),(2,1,3,4),(3,2,1,4))
        permuted = RawVTUMesh(mesh.points,mesh.tets[collect(permutation),:],mesh.tris,
            mesh.tet_cell_ids,mesh.tri_cell_ids,mesh.cell_data)
        for p in 1:3,flux in (MaxwellFlux_Central,MaxwellFlux_Alternating)
            dg = MixedDGDiscretization(permuted,p)
            E(x,y,z) = (0.,0.,0.)
            H(x,y,z) = (z^(p+1),x^(p+1),y^(p+1))
            U = interpolate_maxwell_field(dg,E,H)
            R = DiscoGMPI.similar_maxwell_rhs(U)
            DiscoGMPI.make_maxwell_rhs_function(dg;flux_kind=flux)(R,U)
            exact = interpolate_maxwell_field(dg,(x,y,z)->((p+1)*y^p,(p+1)*z^p,(p+1)*x^p),E)
            for c in 1:6
                @test DiscoGMPI.mixed_components(R)[c] ≈ DiscoGMPI.mixed_components(exact)[c] atol=2e-10 rtol=2e-10
            end
        end
    end
    for p in 1:4,flux in (MaxwellFlux_Central,MaxwellFlux_Alternating)
        specs = DiscoGMPI.box_periodic_specs(1.,1.,1.)
        dg = MixedDGDiscretization(mesh,p;periodic=specs)
        dt = MixedDGDiscretization(mesh,p;periodic=specs,backend=ThreadedBackend())
        U = interpolate_maxwell_field(dg,(x,y,z)->(1.,2.,3.),(x,y,z)->(4.,5.,6.))
        R,S = DiscoGMPI.similar_maxwell_rhs(U),DiscoGMPI.similar_maxwell_rhs(U)
        DiscoGMPI.make_maxwell_rhs_function(dg;flux_kind=flux)(R,U)
        @test maximum(maximum(abs,c) for c in DiscoGMPI.mixed_components(R)) < 2e-9
        foreach(randn!,DiscoGMPI.mixed_components(U))
        DiscoGMPI.make_maxwell_rhs_function(dg;flux_kind=flux)(R,U)
        DiscoGMPI.make_maxwell_rhs_function(dt;flux_kind=flux)(S,U)
        @test abs(mixed_energy_pair(U,R,dg)) < 2e-10
        @test all(a≈b for (a,b) in zip(DiscoGMPI.mixed_components(R),DiscoGMPI.mixed_components(S)))
    end
end

@testset "Mixed curl initialization" begin
    mesh = structured_box_mesh(1,1,1)
    dg = MixedDGDiscretization(mesh,2)
    U = MaxwellField(dg)
    A(x,y,z,t) = (0.,0.,x*y)
    initialize_electric_from_potential!(U,dg,A)
    initialize_magnetic_from_potential!(U,dg,A)
    exact = interpolate_maxwell_field(dg,(x,y,z)->(x,-y,0.),(x,y,z)->(x,-y,0.))
    @test all(a≈b for (a,b) in zip(DiscoGMPI.mixed_components(U),DiscoGMPI.mixed_components(exact)))
end

@testset "Mixed driven PEC and projected sources" begin
    mesh = structured_box_mesh(1,1,1)
    dg = MixedDGDiscretization(mesh,2)
    E(x,y,z) = (x,y,z)
    H(x,y,z) = (0.,0.,0.)
    U = interpolate_maxwell_field(dg,E,H)
    R = DiscoGMPI.similar_maxwell_rhs(U)
    rhs! = DiscoGMPI.make_maxwell_rhs_function(dg;boundary_electric=(tag,x,y,z,t)->E(x,y,z))
    rhs!(R,U)
    @test maximum(maximum(abs,c) for c in DiscoGMPI.mixed_components(R)) < 1e-11
    source = (x,y,z,t,E,H,eps,mu)->(electric=(x^2,y^2,z^2),magnetic=(x^3,y^3,z^3))
    driven! = with_mixed_source(rhs!,dg,source)
    driven!(R,U)
    @test mixed_rhs_allocations(driven!,R,U) == 0
    exact = interpolate_maxwell_field(dg,(x,y,z)->(x^2,y^2,z^2),(x,y,z)->(x^3,y^3,z^3))
    @test all(a≈b for (a,b) in zip(DiscoGMPI.mixed_components(R),DiscoGMPI.mixed_components(exact)))
    # Isolate nonautonomous forcing to verify ordinary RK stage times.
    zero_rhs! = (R,U,t)->DiscoGMPI.fill_maxwell_rhs!(R,0.)
    timed! = with_mixed_source(zero_rhs!,dg,(x,y,z,t,E,H,eps,mu)->(electric=(t^3,0.,0.),magnetic=(0.,t^3,0.)))
    V = MaxwellField(dg)
    scheme = explicit_rk_scheme(4)
    workspace = MaxwellRKWorkspace(V,scheme)
    mixed_rk_step!(V,workspace,scheme,0.,0.1,timed!)
    @test maximum(abs.(V.Ex .- 0.1^4/4)) < 1e-15
    @test maximum(abs.(V.Hy .- 0.1^4/4)) < 1e-15
    damped! = with_mixed_nonlinear_pml(zero_rhs!,dg,(x,y,z)->(1.,1.,1.))
    W = interpolate_maxwell_field(dg,(x,y,z)->(0.,1.,0.),(x,y,z)->(0.,0.,1.))
    damped!(R,W)
    @test mixed_energy_pair(W,R,dg) < 0
end
