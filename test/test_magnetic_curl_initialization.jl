using Test
using LinearAlgebra
using DiscoGMPI

module MagneticInitializationFixtures
using LinearAlgebra, DiscoGMPI
function mesh(transform = [1.2 0.3 -0.1; 0.2 0.8 0.25; -0.15 0.1 0.6]; reversed = false)
    points = transform * [-1.0 1 -1 -1; -1 -1 1 -1; -1 -1 -1 1] .+ [0.2, -0.1, 0.15]
    tets = reshape(reversed ? [2, 1, 3, 4] : [1, 2, 3, 4], 4, 1)
    tris = [2 1 1 1; 3 4 2 3; 4 3 4 2]
    RawVTUMesh(points, tets, tris, [1], collect(2:5),
        Dict{String, Any}("boundary_id" => [0, 10, 10, 10, 10]))
end
function field(ref, mesh)
    interpolate_maxwell_field(mesh, ref, (x,y,z)->(x,y,z), (x,y,z)->(0.0,0.0,0.0))
end
function potential(p)
    p == 0 && return (x,y,z,t)->(0.5*(2z-3y), 0.5*(3x-z), 0.5*(y-2x))
    return (x,y,z,t)->(z^(p+1)+x*y, x^(p+1)+y*z, y^(p+1)+z*x)
end
curl(p,x,y,z) = p == 0 ? (1.0,2.0,3.0) :
    ((p+1)*y^p-y, (p+1)*z^p-z, (p+1)*x^p-x)
function budgets(ws, mapping, Amax, Bmax)
    # Two floating-point differentiations; use actual reference and metric norms.
    op = DiscoGMPI.build_physical_element_operator(ws.target, mapping)
    D = opnorm(op.Dx, Inf) + opnorm(op.Dy, Inf) + opnorm(op.Dz, Inf)
    G = opnorm(mapping.invJ, 1) *
        (opnorm(ws.Rr, Inf) + opnorm(ws.Rs, Inf) + opnorm(ws.Rt, Inf))
    amplification = max(cond(ws.potential.V), cond(ws.target.V), 1.0)
    gamma = 64 * eps(Float64) * ws.potential.Np * amplification
    field = gamma * (G*Amax + Bmax)
    divergence = gamma * D * (G*Amax + Bmax)
    return (; field, divergence)
end
end

using .MagneticInitializationFixtures: budgets
const MIF = MagneticInitializationFixtures

@testset "Polynomial curl at solver nodes" begin
    transforms = ([1.2 0.3 -0.1; 0.2 0.8 0.25; -0.15 0.1 0.6],
                  [0.0 -0.3 0.0; 2.0 0.0 0.0; 0.0 0.0 0.7])
    for nodes in (:equispaced, :warp_blend), p in 0:5, T in transforms, reversed in (false,true)
        mesh = MIF.mesh(T; reversed)
        ref = build_reference_tet(p; nodal_set=nodes)
        mappings = DiscoGMPI.build_reference_mappings(mesh)
        ws = MagneticCurlInitializationWorkspace(ref; nodal_set=nodes)
        U = MIF.field(ref, mesh)
        E = (copy(U.Ex), copy(U.Ey), copy(U.Ez))
        materials = homogeneous_maxwell_materials(1; permeability=2.3)
        initialize_magnetic_from_potential!(U, mesh, ref, mappings, materials, MIF.potential(p), ws)
        @test (U.Ex,U.Ey,U.Ez) == E
        B = 2.3 .* hcat(U.Hx[:,1],U.Hy[:,1],U.Hz[:,1])
        expected = zeros(ref.Np,3)
        for i in 1:ref.Np
            x,y,z = DiscoGMPI.map_to_physical(mesh.points, mesh.tets[:,1],ref.r[i],ref.s[i],ref.t[i])
            expected[i,:] .= MIF.curl(p,x,y,z)
        end
        op = DiscoGMPI.build_physical_element_operator(ref, mappings.tet_mappings[1])
        d = op.Dx*B[:,1] + op.Dy*B[:,2] + op.Dz*B[:,3]
        tol = budgets(ws, mappings.tet_mappings[1], maximum(abs, ws.A), maximum(abs,B))
        @test maximum(abs, B-expected) <= tol.field
        @test norm(B) > 0
        @test maximum(abs,d) <= tol.divergence
        # Repeat with cached operators and scratch storage.
        first = copy(U.Hy)
        initialize_magnetic_from_potential!(U, mesh, ref, mappings, materials, MIF.potential(p), ws)
        @test U.Hy == first
        if p > 0 && T === transforms[1] && !reversed
            dg = DGDiscretization(mesh,p; nodal_set=nodes)
            measured = magnetic_initialization_diagnostics(U,dg,materials)
            volume = mappings.tet_mappings[1].absdetJ * 4/3
            @test sqrt(measured.l2_squared) <= sqrt(volume)*tol.divergence
            charge = DiscoGMPI.local_strong_magnetic_charge_from_components(
                U.Hx,U.Hy,U.Hz,dg,materials,1:1)
            @test abs(charge-measured.signed) <= volume*tol.divergence
            println("nodes=$nodes p=$p: L2=",sqrt(measured.l2_squared),
                " signed=", measured.signed, " max/tolerance=", measured.max_nodal/tol.divergence)
            for i in 1:ref.Np
                x,_,_ = DiscoGMPI.map_to_physical(mesh.points,mesh.tets[:,1],ref.r[i],ref.s[i],ref.t[i])
                U.Hx[i,1] += 0.01*x
            end
            @test sqrt(magnetic_initialization_diagnostics(U,dg,materials).l2_squared) > 0.001*sqrt(volume)
        end
    end
end

@testset "Nonpolynomial potential and amplitude scaling" begin
    mesh = MIF.mesh()
    for nodes in (:equispaced,:warp_blend), amplitude in (1e-6,1.0,1e6)
        dg = DGDiscretization(mesh,3; nodal_set=nodes)
        ws = MagneticCurlInitializationWorkspace(dg.ref; nodal_set=nodes)
        U = MIF.field(dg.ref,mesh)
        materials = homogeneous_maxwell_materials(1; permeability=1.7)
        potential = (x,y,z,t)->amplitude .* (sin(y*z+x),cos(z*x+y),sin(x*y+z))
        initialize_magnetic_from_potential!(U,dg,materials,potential,ws)
        Bmax = 1.7*max(maximum(abs,U.Hx),maximum(abs,U.Hy),maximum(abs,U.Hz))
        tol = budgets(ws,dg.mappings.tet_mappings[1],maximum(abs,ws.A),Bmax)
        d = magnetic_initialization_diagnostics(U,dg,materials)
        @test Bmax > amplitude/100
        @test d.max_nodal <= tol.divergence
        @test abs(d.signed) <= dg.mappings.tet_mappings[1].absdetJ*4/3*tol.divergence
    end
end

@testset "Validation and checkpoint contracts" begin
    mesh = MIF.mesh()
    ref = build_reference_tet(2)
    mappings = DiscoGMPI.build_reference_mappings(mesh)
    ws = MagneticCurlInitializationWorkspace(ref)
    U = MIF.field(ref,mesh)
    materials = homogeneous_maxwell_materials(1)
    @test_throws ArgumentError initialize_magnetic_from_potential!(U,mesh,build_reference_tet(2),mappings,materials,MIF.potential(2),ws)
    @test_throws ArgumentError initialize_magnetic_from_potential!(U,mesh,ref,mappings,materials,(x,y,z,t)->(NaN,0,0),ws)
    @test_throws ArgumentError initialize_magnetic_from_potential!(U,mesh,ref,mappings,(x,y,z)->1.0,MIF.potential(2),ws)
    for value in (NaN,Inf,0.0,-1.0)
        materials.permeability[1] = value
        @test_throws ArgumentError initialize_magnetic_from_potential!(U,mesh,ref,mappings,materials,MIF.potential(2),ws)
    end
    @test parse_magnetic_initialization("curl-potential") == :curl_potential
    @test_throws ArgumentError parse_magnetic_initialization("projection")
    @test validate_magnetic_initialization_restart(Dict(),:interpolate) === nothing
    @test_throws ArgumentError validate_magnetic_initialization_restart(Dict(),:curl_potential)
    @test validate_magnetic_initialization_restart(Dict("magnetic_initialization"=>"curl_potential"),:curl_potential) === nothing
end

module MagneticInitializationCLI
include(joinpath(@__DIR__, "..", "examples", "convergence_distributed_periodic_poisson_bracket_maxwell.jl"))
end

@testset "Periodic CLI defaults and potential" begin
    scope = MagneticInitializationCLI
    args = ["--mesh-family=structured", "--nx-targets=2,4", "--orders=2"]
    @test scope.parse_periodic_convergence_arguments(args).magnetic_initialization == :interpolate
    @test scope.parse_periodic_convergence_arguments(vcat(args,["--magnetic-initialization=curl-potential"])).magnetic_initialization == :curl_potential
    @test scope.parse_arguments(String[],1,dirname(@__DIR__),"initialization-test").magnetic_initialization == :interpolate
    @test scope.parse_arguments(["--magnetic-initialization=curl-potential"],1,dirname(@__DIR__),"initialization-test").magnetic_initialization == :curl_potential
    wave = PlaneWaveParameters(2pi,2pi,1.0,0.2)
    x=0.37; t=0.13; mu=2.3
    # Complex-step verification of the analytical helper's curl sign and phase.
    dx = imag(periodic_magnetic_vector_potential(x+1e-30im,0,0,t;mu,wave)[3])/1e-30
    _, H = exact_periodic_wave_functions(t;epsilon=1.0,mu,wave)
    @test -dx ≈ mu*H(x,0,0)[2]
    @test abs(dx) > 0.1
end
