using Test, LinearAlgebra, MPI, DiscoGMPI

const Audit = DiscoGMPI.MaxwellInvariantAudit
const integral_names = (:integrated_d_x, :integrated_d_y, :integrated_d_z,
                        :integrated_b_x, :integrated_b_y, :integrated_b_z)
const directional_names = (:electric_charge_x, :electric_charge_y, :electric_charge_z,
                           :magnetic_charge_x, :magnetic_charge_y, :magnetic_charge_z)
integrals(d) = [getproperty(d, name) for name in integral_names]

MPI.Init()
try
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    mesh = structured_box_mesh(4,2,2; lower=(0.,0.,0.), upper=(2.,1.,1.),
                               boundary_id=periodic_box_boundary_id)
    @testset "Integrated D and B components" begin
        for flux in (MaxwellFlux_Central, MaxwellFlux_Alternating)
            case = Audit.build_case(rank == 0 ? mesh : nothing, 2, comm; flux)
            ddg = case.ddg
            lightweight_workspace =
                DistributedMaxwellLightweightWorkspace(ddg, 8)
            zero_field = (x,y,z) -> (0.,0.,0.)
            function diagnose(U; epsilon=1.0, mu=1.0, E=zero_field, H=zero_field)
                DiscoGMPI.distributed_maxwell_quadrature_diagnostics(U,ddg,8;
                    epsilon,mu,exact_electric=E,exact_magnetic=H,
                    exact_curl_electric=zero_field,exact_curl_magnetic=zero_field)
            end
            tol = 4096eps(Float64)*ddg.dg.ref.Np*cond(ddg.dg.ref.M)
            for (E,H,expected,expected_charge,expected_directional) in (
                ((x,y,z)->(1.,-2.,3.), (x,y,z)->(4.,5.,-6.),
                 [4.,-8.,12.,24.,30.,-36.], 0., zeros(6)),
                ((x,y,z)->(x,y,z), (x,y,z)->(z,x,y),
                 [4.,2.,2.,3.,6.,3.], 12., [4.,4.,4.,0.,0.,0.]),
                ((x,y,z)->(x,2y,-3z), (x,y,z)->(-2x,3y,4z),
                 [4.,4.,-6.,-12.,9.,12.], 0., [4.,8.,-12.,-12.,18.,24.]))
                U = interpolate_maxwell_field(ddg,E,H)
                # Ghost elements must never contribute to volume diagnostics.
                owned = Set(ddg.distributed_mesh.partition.owned)
                for c in Audit.components(U), e in axes(c,2)
                    e in owned || fill!(view(c,:,e),NaN)
                end
                d = diagnose(U; epsilon=2.0, mu=3.0, E,H)
                lightweight = distributed_maxwell_lightweight_diagnostics!(
                    lightweight_workspace,
                    U,
                    ddg;
                    epsilon=2.0,
                    mu=3.0,
                )
                combined =
                    distributed_maxwell_lightweight_diagnostics_and_energy!(
                        lightweight_workspace,
                        U,
                        ddg;
                        epsilon=2.0,
                        mu=3.0,
                    )
                reference_energy = distributed_maxwell_energy(
                    U,
                    ddg;
                    ε=2.0,
                    μ=3.0,
                )
                @test combined.diagnostics == lightweight
                @test isapprox(combined.energy.total, reference_energy.total;
                    atol=tol, rtol=tol)
                @test isapprox(combined.energy.electric, reference_energy.electric;
                    atol=tol, rtol=tol)
                @test isapprox(combined.energy.magnetic, reference_energy.magnetic;
                    atol=tol, rtol=tol)
                @test isapprox(
                    [combined.energy.Ex, combined.energy.Ey, combined.energy.Ez,
                     combined.energy.Hx, combined.energy.Hy, combined.energy.Hz],
                    [reference_energy.Ex, reference_energy.Ey, reference_energy.Ez,
                     reference_energy.Hx, reference_energy.Hy, reference_energy.Hz];
                    atol=tol,
                    rtol=tol,
                )
                @test isapprox(lightweight.optical_chirality,
                    d.optical_chirality; atol=tol, rtol=tol)
                @test isapprox(collect(lightweight.linear_momentum),
                    [d.linear_momentum_x,d.linear_momentum_y,d.linear_momentum_z];
                    atol=tol, rtol=tol)
                @test isapprox(collect(lightweight.angular_momentum),
                    [d.angular_momentum_x,d.angular_momentum_y,d.angular_momentum_z];
                    atol=tol, rtol=tol)
                @test isapprox(collect(lightweight.integrated_d),integrals(d)[1:3];
                    atol=tol, rtol=tol)
                @test isapprox(collect(lightweight.integrated_b),integrals(d)[4:6];
                    atol=tol, rtol=tol)
                @test all(isfinite, d.discog_charges)
                @test abs(d.discog_charges[1]) <= tol
                reference = zeros(16)
                mass = ddg.dg.ref.M
                for e in owned
                    jac = ddg.dg.mappings.tet_mappings[e].absdetJ
                    op = ddg.dg.physops.elements[e]
                    derivatives = (op.Dx, op.Dy, op.Dz)
                    for (offset, components, coefficient, weak) in (
                        (4, (U.Ex,U.Ey,U.Ez), 2.0, false),
                        (8, (U.Hx,U.Hy,U.Hz), 1.0, true),
                        (12, (U.Hx,U.Hy,U.Hz), 3.0, false))
                        for axis in 1:3
                            stiffness = jac .* (mass * derivatives[axis])
                            C = weak ? transpose(stiffness) : stiffness
                            value = coefficient * jac * dot(ones(size(mass,1)),
                                mass * (C * components[axis][:,e]))
                            reference[offset+axis+1] += value
                            reference[offset+1] += value
                        end
                    end
                end
                reference = MPI.Allreduce(reference,+,comm)
                @test collect(d.discog_charges[5:16]) ≈ reference[5:16] atol=tol rtol=tol
                for offset in (0,4,8,12)
                    @test isapprox(d.discog_charges[offset+1],
                        sum(d.discog_charges[offset+2:offset+4]); atol=tol)
                end
                @test isapprox(integrals(d),expected; rtol=tol,atol=tol)
                @test isapprox(d.electric_charge,expected_charge; atol=tol)
                directional = [getproperty(d,name) for name in directional_names]
                @test isapprox(directional,expected_directional; rtol=tol,atol=tol)
                @test isapprox(sum(directional[1:3]),d.electric_charge; atol=tol)
                @test isapprox(sum(directional[4:6]),d.magnetic_charge; atol=tol)
                io = IOBuffer()
                write_quadrature_diagnostics_header(io)
                write_quadrature_diagnostics_row(io,0,0.0,d)
                lines = split(chomp(String(take!(io))),'\n')
                header = split(lines[1],','); values = split(lines[2],',')
                @test length(header) == length(values)
                for (i,name) in enumerate(integral_names)
                    column = findfirst(==(string(name)),header)
                    @test column !== nothing
                    @test parse(Float64,values[column]) == integrals(d)[i]
                end
                for name in directional_names
                    column = findfirst(==(string(name)),header)
                    @test column !== nothing
                    @test parse(Float64,values[column]) == getproperty(d,name)
                end
            end
            # Test constant-vector functionals on the actual periodic RHS.
            U = Audit.initial_field(case)
            for (c,field) in enumerate(Audit.components(U)), e in axes(field,2), i in axes(field,1)
                g = ddg.distributed_mesh.elements.global_ids[e]
                field[i,e] = (mod(13i+7g+3c,31)-15)/16
            end
            R = DiscoGMPI.similar_maxwell_rhs(U)
            case.rhs!(R,U)
            residual = integrals(diagnose(Audit.asfield(R)))
            scale = 2MPI.Allreduce(maximum(maximum(abs,c[:,ddg.distributed_mesh.partition.owned])
                for c in Audit.components(R)),max,comm)
            @test maximum(abs,residual) <= tol*max(scale,1.0)
            rank == 0 && println("flux=$flux ranks=$(MPI.Comm_size(comm)) integrated RHS=$residual scale=$scale")
        end
    end
finally
    MPI.Finalize()
end
