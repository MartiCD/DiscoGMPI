using DiscoGMPI, LinearAlgebra, Printf
BLAS.set_num_threads(1)
mesh = structured_box_mesh(1,1,1;lower=(0.,0.,0.),upper=(1.,1.,1.),boundary_id=periodic_box_boundary_id)
for p in 1:2,flux in (MaxwellFlux_Central,MaxwellFlux_Alternating)
    dg = MixedDGDiscretization(mesh,p;periodic=DiscoGMPI.box_periodic_specs(1.,1.,1.))
    U = MaxwellField(dg)
    R = DiscoGMPI.similar_maxwell_rhs(U)
    rhs! = DiscoGMPI.make_maxwell_rhs_function(dg;flux_kind=flux)
    fields,residuals = DiscoGMPI.mixed_components(U),DiscoGMPI.mixed_components(R)
    lengths = length.(fields)
    offsets = cumsum((0,lengths...))
    A = zeros(sum(lengths),sum(lengths))
    for c in 1:6,i in eachindex(fields[c])
        fields[c][i] = 1.
        rhs!(R,U)
        for d in 1:6
            A[offsets[d]+1:offsets[d+1],offsets[c]+i] = vec(residuals[d])
        end
        fields[c][i] = 0.
    end
    eigenvalues = eigvals(A)
    ne = sum(lengths[1:3])
    C = A[1:ne,ne+1:end]
    nullity = size(C,2)-rank(C;rtol=1e-11)
    dt = mixed_cfl_dt(dg;CFL=0.1)
    scheme = mixed_esprk_scheme(p)
    G = Matrix{Float64}(I,size(A)...)
    BE = copy(A); BE[ne+1:end,:] .= 0
    BH = copy(A); BH[1:ne,:] .= 0
    for (a,b) in zip(scheme.first_weights,scheme.second_weights)
        G = (I+dt*b*BE)*(I+dt*a*BH)*G
    end
    @printf("p=%d flux=%s max_real=%.3e spectral_radius=%.6f dt=%.6e amplification_radius=%.12f H_nullity=%d/%d\n",
        p,DiscoGMPI.maxwell_flux_kind_label(flux),maximum(abs,real.(eigenvalues)),
        maximum(abs,eigenvalues),dt,maximum(abs,eigvals(G)),nullity,size(C,2))
end
