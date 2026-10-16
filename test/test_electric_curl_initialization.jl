using Test, LinearAlgebra, DiscoGMPI

# Reuse the mapped-element fixtures and scaled roundoff budget; this file also
# remains directly runnable without the suite's include order.
if !isdefined(@__MODULE__, :MagneticInitializationFixtures)
    include("test_magnetic_curl_initialization.jl")
end

@testset "Electric polynomial curl and independent field initialization" begin
    for nodes in (:equispaced, :warp_blend), p in 0:4, reversed in (false, true)
        mesh = MIF.mesh(; reversed)
        ref = build_reference_tet(p; nodal_set=nodes)
        mappings = DiscoGMPI.build_reference_mappings(mesh)
        ws = ElectricCurlInitializationWorkspace(ref; nodal_set=nodes)
        U = interpolate_maxwell_field(mesh, ref, (x,y,z)->(0.0,0.0,0.0), (x,y,z)->(x,y,z))
        H = (copy(U.Hx),copy(U.Hy),copy(U.Hz))
        materials = homogeneous_maxwell_materials(1; epsilon=2.3, permeability=1.7)
        initialize_electric_from_potential!(U,mesh,ref,mappings,materials,MIF.potential(p),ws)
        @test (U.Hx,U.Hy,U.Hz) == H
        displacement = 2.3 .* hcat(U.Ex[:,1],U.Ey[:,1],U.Ez[:,1])
        expected = similar(displacement)
        for i in 1:ref.Np
            x,y,z = DiscoGMPI.map_to_physical(mesh.points,mesh.tets[:,1],ref.r[i],ref.s[i],ref.t[i])
            expected[i,:] .= MIF.curl(p,x,y,z)
        end
        tol = MIF.budgets(ws,mappings.tet_mappings[1],maximum(abs,ws.A),maximum(abs,displacement))
        @test maximum(abs,displacement-expected) <= tol.field
        @test norm(displacement) > 0
        op = DiscoGMPI.build_physical_element_operator(ref,mappings.tet_mappings[1])
        div = op.Dx*displacement[:,1] + op.Dy*displacement[:,2] + op.Dz*displacement[:,3]
        @test maximum(abs,div) <= tol.divergence
        E = (copy(U.Ex),copy(U.Ey),copy(U.Ez))
        initialize_magnetic_from_potential!(U,mesh,ref,mappings,materials,MIF.potential(p),ws)
        @test (U.Ex,U.Ey,U.Ez) == E
        initialize_electric_from_potential!(U,mesh,ref,mappings,materials,MIF.potential(p),ws)
        @test (U.Ex,U.Ey,U.Ez) == E
        if p > 0
            dg = DGDiscretization(mesh,p; nodal_set=nodes)
            d = electric_initialization_diagnostics(U,dg,materials)
            volume = mappings.tet_mappings[1].absdetJ*4/3
            @test sqrt(max(0,d.l2_squared)) <= sqrt(volume)*tol.divergence
            @test abs(d.signed) <= volume*tol.divergence
            @test d.max_nodal <= tol.divergence
            U.Ex[:,1] .+= 0.01 .* [DiscoGMPI.map_to_physical(mesh.points,mesh.tets[:,1],
                ref.r[i],ref.s[i],ref.t[i])[1] for i in 1:ref.Np]
            @test electric_initialization_diagnostics(U,dg,materials).max_nodal > 0.02
        end
    end
end

@testset "Electric nonpolynomial potential, scaling and validation" begin
    for nodes in (:equispaced,:warp_blend), amplitude in (1e-6,1.0,1e6)
        dg = DGDiscretization(MIF.mesh(),3; nodal_set=nodes)
        ws = ElectricCurlInitializationWorkspace(dg.ref; nodal_set=nodes)
        U = MIF.field(dg.ref,dg.mesh)
        materials = homogeneous_maxwell_materials(1;epsilon=1.7)
        potential = (x,y,z,t)->amplitude .* (sin(y*z+x+t),cos(z*x+y),sin(x*y+z))
        initialize_electric_from_potential!(U,dg,materials,potential,ws;time=0.13)
        Dmax = 1.7*max(maximum(abs,U.Ex),maximum(abs,U.Ey),maximum(abs,U.Ez))
        tol = MIF.budgets(ws,dg.mappings.tet_mappings[1],maximum(abs,ws.A),Dmax)
        @test Dmax > amplitude/100
        @test electric_initialization_diagnostics(U,dg,materials).max_nodal <= tol.divergence
        @test_throws ArgumentError initialize_electric_from_potential!(U,dg,materials,potential,ws;time=NaN)
        @test_throws ArgumentError initialize_electric_from_potential!(U,dg,(x,y,z)->1.0,potential,ws)
        @test_throws ArgumentError initialize_electric_from_potential!(U,dg,materials,(x,y,z,t)->(NaN,0,0),ws)
        @test_throws ArgumentError initialize_electric_from_potential!(U,dg,materials,potential,
            ElectricCurlInitializationWorkspace(build_reference_tet(3)))
        for bad in (NaN,Inf,0.0,-1.0)
            materials.epsilon[1] = bad
            @test_throws ArgumentError initialize_electric_from_potential!(U,dg,materials,potential,ws)
        end
    end
end

@testset "Electric CLI, potential and restart contracts" begin
    scope = MagneticInitializationCLI
    args = ["--mesh-family=structured", "--nx-targets=2,4", "--orders=2"]
    @test scope.parse_periodic_convergence_arguments(args).electric_initialization == :interpolate
    unstructured = scope.parse_periodic_convergence_arguments([
        "--mesh-family=unstructured", "--nx-targets=5,6,7,8", "--orders=2"])
    @test unstructured.nx_targets == [5, 6, 7, 8]
    unstructured_alias = scope.parse_periodic_convergence_arguments([
        "--mesh-family=unstructured", "--nx-target=5,6,7,8", "--orders=2"])
    @test unstructured_alias.nx_targets == [5, 6, 7, 8]
    first_order = scope.parse_periodic_convergence_arguments([
        "--mesh-family=unstructured", "--nx-targets=4,5,6,7,8", "--orders=1"])
    @test first_order.orders == [1]
    @test_throws ArgumentError scope.parse_periodic_convergence_arguments([
        "--mesh-family=unstructured", "--nx-targets=5,7,6,8", "--orders=2"])
    @test_throws ArgumentError scope.parse_periodic_convergence_arguments([
        "--mesh-family=structured", "--nx-targets=4,6,8", "--orders=2"])
    config = scope.parse_periodic_convergence_arguments(vcat(args,["--electric-initialization=curl-potential"]))
    @test config.electric_initialization == :curl_potential
    @test config.magnetic_initialization == :interpolate
    @test scope.parse_arguments(String[],1,dirname(@__DIR__),"test").electric_initialization == :interpolate
    for arg in (["--electric-initialization=curl-potential"], ["--electric-initialization","curl-potential"])
        cfg = scope.parse_arguments(arg,1,dirname(@__DIR__),"test")
        @test cfg.electric_initialization == :curl_potential
        @test cfg.magnetic_initialization == :interpolate
    end
    @test parse_electric_initialization("curl_potential") == :curl_potential
    @test_throws ArgumentError parse_electric_initialization("projection")
    @test validate_electric_initialization_restart(Dict(),:interpolate) === nothing
    @test_throws ArgumentError validate_electric_initialization_restart(Dict(),:curl_potential)
    md = Dict("electric_initialization"=>"curl_potential")
    @test validate_electric_initialization_restart(md,:curl_potential) === nothing
    @test_throws ArgumentError validate_electric_initialization_restart(md,:interpolate)
    epsilon=2.3; mu=1.7; k=2pi
    wave = PlaneWaveParameters(k,k/sqrt(epsilon*mu),1.3,0.2)
    x=0.37; t=0.13
    dx = imag(periodic_electric_vector_potential(x+1e-30im,0,0,t;epsilon,wave)[2])/1e-30
    E,_ = exact_periodic_wave_functions(t;epsilon,mu,wave)
    @test dx ≈ epsilon*E(x,0,0)[3]
    @test abs(dx) > 0.1
    @test_throws ArgumentError periodic_electric_vector_potential(x,0,0,t;epsilon=0.0,wave)
end
