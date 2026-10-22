using Test, LinearAlgebra, Random, DiscoGMPI
BLAS.set_num_threads(1)
# Frozen pre-buffer implementation: independent oracle for all 24 diagnostics.
@eval DiscoGMPI begin
"""Quadrature diagnostics on the actual two spaces; errors are measured independently.

Use higher quadrature for nonpolynomial exact solutions. Degree 4(p+1)+2
also resolves polynomial energy-density products without aliasing.
"""
function reference_mixed_diagnostics(U,dg::MixedDGDiscretization;
        elements=axes(U.Ex,2),time=0.,exact_electric=nothing,exact_magnetic=nothing,
        degree=4dg.reference.ref_H.N+2)
    validate_mixed_field(U,dg)
    points,w = mixed_tetrahedron_quadrature(degree)
    V = (mixed_basis(dg.reference.ref_E,points),mixed_basis(dg.reference.ref_H,points))
    G = (mixed_basis_gradient(dg.reference.ref_E,points),mixed_basis_gradient(dg.reference.ref_H,points))
    sums = zeros(24)
    for e in elements
        jac = dg.mappings.tet_mappings[e]
        fields = mixed_components(U)
        values = [V[c<=3 ? 1 : 2]*view(fields[c],:,e) for c in 1:6]
        gradients = [ntuple(a->sum(jac.invJ[b,a]*(G[c<=3 ? 1 : 2][b]*view(fields[c],:,e)) for b in 1:3),3) for c in 1:6]
        eps,mu = dg.materials.epsilon[e],dg.materials.permeability[e]
        for q in eachindex(w)
            weight = w[q]*jac.absdetJ
            x = map_to_physical(dg.mesh.points,view(dg.mesh.tets,:,e),points[q,1],points[q,2],points[q,3])
            E,H = ntuple(c->values[c][q],3),ntuple(c->values[c+3][q],3)
            sums[1] += weight*eps*sum(abs2,E)/2
            sums[2] += weight*mu*sum(abs2,H)/2
            if exact_electric!==nothing
                ex = exact_electric(x...,time)
                sums[3] += weight*sum(abs2(E[c]-ex[c]) for c in 1:3)
                sums[5] += weight*sum(abs2,ex)
            end
            if exact_magnetic!==nothing
                ex = exact_magnetic(x...,time)
                sums[4] += weight*sum(abs2(H[c]-ex[c]) for c in 1:3)
                sums[6] += weight*sum(abs2,ex)
            end
            divE = eps*sum(gradients[c][c][q] for c in 1:3)
            divH = mu*sum(gradients[c+3][c][q] for c in 1:3)
            sums[7] += weight*divE
            sums[8] += weight*divH
            sums[9] += weight*divE^2
            sums[10] += weight*divH^2
            momentum = eps*mu .* cross3(E,H)
            angular = cross3(x,momentum)
            for c in 1:3
                sums[10+c] += weight*eps*E[c]
                sums[13+c] += weight*mu*H[c]
                sums[16+c] += weight*momentum[c]
                sums[19+c] += weight*angular[c]
            end
            curlE = ntuple(c->gradients[mod1(c+2,3)][mod1(c+1,3)][q]-gradients[mod1(c+1,3)][mod1(c+2,3)][q],3)
            curlH = ntuple(c->gradients[mod1(c+2,3)+3][mod1(c+1,3)][q]-gradients[mod1(c+1,3)+3][mod1(c+2,3)][q],3)
            sums[23] += weight*(eps*dot3(E,curlE)+mu*dot3(H,curlH))/2
            sums[24] += weight
        end
    end
    return sums
end

end

@testset "Buffered mixed diagnostics" begin
    Random.seed!(640)
    mesh = structured_box_mesh(1,1,1;lower=(0.,0.,0.),upper=(2.,1.,.7),boundary_id=periodic_box_boundary_id)
    for p in (1,4)
        dg = MixedDGDiscretization(mesh,p)
        U = MaxwellField(dg)
        foreach(randn!,DiscoGMPI.mixed_components(U))
        ws = DiscoGMPI.mixed_diagnostic_workspace(dg)
        E(x,y,z,t) = (x+t,y,z)
        H(x,y,z,t) = (z,x,y-t)
        options = (;elements=[1,3],time=.27,exact_electric=E,exact_magnetic=H)
        reference = DiscoGMPI.reference_mixed_diagnostics(U,dg;options...)
        actual = mixed_diagnostics(U,dg;workspace=ws,options...)
        @test actual ≈ reference rtol=2e-12 atol=1e-11
        @test actual === ws.sums
        @test mixed_diagnostics(U,dg;workspace=ws,options...) ≈ reference
        mixed_diagnostics(U,dg;workspace=ws)
        DiscoGMPI.reference_mixed_diagnostics(U,dg)
        old_bytes = @allocated DiscoGMPI.reference_mixed_diagnostics(U,dg)
        new_bytes = @allocated mixed_diagnostics(U,dg;workspace=ws)
        println("p=",p," diagnostic allocations: old=",old_bytes," new=",new_bytes)
        @test new_bytes < old_bytes ÷ 5
    end
end

