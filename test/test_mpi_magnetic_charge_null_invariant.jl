using Test
using MPI
using DiscoGMPI

# Run from the repository root, for example:
#   mpiexec -n 1 julia --project=. test/test_mpi_magnetic_charge_null_invariant.jl
#   mpiexec -n 2 julia --project=. test/test_mpi_magnetic_charge_null_invariant.jl

const MPI_WAS_INITIALIZED = MPI.Initialized()
MPI_WAS_INITIALIZED || MPI.Init()

const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NPROCS = MPI.Comm_size(COMM)

function magnetic_null_test_mesh()
    return structured_box_mesh(
        2,
        2,
        1;
        lower = (0.0, 0.0, 0.0),
        upper = (2.0, 1.0, 1.0),
        boundary_id = periodic_box_boundary_id,
    )
end

function deterministic_electric_only_field(distributed_dg)
    ref = distributed_dg.dg.ref
    nelements = size(distributed_dg.dg.mesh.tets, 2)
    global_ids = distributed_dg.distributed_mesh.elements.global_ids

    Ex = zeros(Float64, ref.Np, nelements)
    Ey = similar(Ex)
    Ez = similar(Ex)
    Hx = similar(Ex)
    Hy = similar(Ex)
    Hz = similar(Ex)

    for elem in 1:nelements
        global_elem = global_ids[elem]
        for node in 1:ref.Np
            Ex[node, elem] =
                sin(0.37 * global_elem + 0.19 * node) +
                0.07 * cos(0.11 * global_elem * node)
            Ey[node, elem] =
                cos(0.23 * global_elem - 0.31 * node) +
                0.05 * sin(0.17 * (global_elem + node))
            Ez[node, elem] =
                sin(0.29 * global_elem + 0.41 * node) -
                0.03 * cos(0.13 * (2 * global_elem - node))
        end
    end

    return MaxwellField(Ex, Ey, Ez, Hx, Hy, Hz)
end

function max_abs_owned_rhs_difference(
    left::MaxwellRHS,
    right::MaxwellRHS,
    distributed_dg,
)
    owned = distributed_dg.distributed_mesh.partition.owned
    value = 0.0
    for (lhs, rhs) in zip(
        (left.rhsEx, left.rhsEy, left.rhsEz, left.rhsHx, left.rhsHy, left.rhsHz),
        (right.rhsEx, right.rhsEy, right.rhsEz, right.rhsHx, right.rhsHy, right.rhsHz),
    )
        value = max(
            value,
            maximum(abs.(@view(lhs[:, owned]) .- @view(rhs[:, owned]))),
        )
    end
    return value
end

function magnetic_charge_null_invariant_case(flux_kind::MaxwellFluxKind)
    root_mesh = RANK == 0 ? magnetic_null_test_mesh() : nothing
    nelements = MPI.bcast(
        RANK == 0 ? size(root_mesh.tets, 2) : 0,
        COMM;
        root = 0,
    )
    if nelements < NPROCS
        RANK == 0 &&
            @warn "Skipping magnetic-charge null-invariant test: mesh has fewer elements than MPI ranks." nelements NPROCS
        return nothing
    end

    root_partition =
        RANK == 0 ? balanced_spatial_partition(root_mesh, NPROCS) : nothing
    distributed_dg = build_distributed_dg_from_root(
        root_mesh,
        root_partition,
        2;
        comm = COMM,
    )
    materials = homogeneous_maxwell_materials(
        size(distributed_dg.dg.mesh.tets, 2);
        epsilon = 1.0,
        permeability = 1.0,
    )
    box = distributed_axis_aligned_box(distributed_dg)
    registry = MaxwellBoundaryRegistry(
        Dict(boundary_id => MaxwellBC_None for boundary_id in 1:6),
    )
    periodic = build_distributed_periodic_maxwell_exchange(
        distributed_dg,
        periodic_box_specs(box),
        materials = materials,
    )
    formulation = PoissonBracketFormulation(flux_kind)
    U = deterministic_electric_only_field(distributed_dg)

    full_rhs = DiscoGMPI.similar_maxwell_rhs(U)
    split_rhs = DiscoGMPI.similar_maxwell_rhs(U)
    DiscoGMPI.fill_maxwell_rhs!(split_rhs, 0.0)

    breakdown = distributed_magnetic_charge_rhs_breakdown(
        U,
        distributed_dg,
        periodic,
        registry,
        formulation,
        materials,
    )

    exchange_maxwell_ghost_traces!(U, distributed_dg)
    exchange_distributed_periodic_traces!(U, distributed_dg, periodic)
    maxwell_rhs_periodic!(
        full_rhs,
        U,
        distributed_dg,
        periodic,
        registry,
        formulation,
        materials,
    )
    maxwell_volume_rhs!(
        split_rhs,
        U,
        distributed_dg.dg.ref,
        distributed_dg.dg.physops,
        formulation,
        materials;
        reset = false,
    )
    DiscoGMPI.add_material_interior_surface_rhs!(
        split_rhs,
        U,
        distributed_dg.dg.ref,
        distributed_dg.dg.fops,
        distributed_dg.dg.mappings,
        distributed_dg.dg.flux_faces,
        formulation,
        materials,
    )
    DiscoGMPI.add_distributed_periodic_surface_rhs!(
        split_rhs,
        U,
        distributed_dg,
        periodic,
        formulation;
        materials = materials,
    )
    DiscoGMPI.add_material_boundary_surface_rhs!(
        split_rhs,
        U,
        distributed_dg.dg.ref,
        distributed_dg.dg.fops,
        distributed_dg.dg.mappings,
        distributed_dg.dg.flux_faces,
        registry,
        formulation,
        materials,
    )

    rhs_split_error = MPI.Allreduce(
        max_abs_owned_rhs_difference(full_rhs, split_rhs, distributed_dg),
        max,
        COMM,
    )
    directed_periodic_faces =
        MPI.Allreduce(length(periodic.faces), +, COMM)
    mpi_interface_faces = MPI.Allreduce(
        sum(
            length(faces)
            for faces in values(distributed_dg.exchange.faces);
            init = 0,
        ),
        +,
        COMM,
    )

    contribution_scale = maximum(
        abs,
        (
            breakdown.volume,
            breakdown.interior_faces,
            breakdown.periodic_faces,
            breakdown.boundary_faces,
            1.0,
        ),
    )
    null_tolerance =
        5.0e3 * eps(Float64) * contribution_scale * sqrt(Float64(nelements))
    split_tolerance = 1.0e-10 * max(1.0, contribution_scale)

    if RANK == 0
        println("Poisson-bracket magnetic-charge null-invariant residual")
        println("  flux:                       ", maxwell_flux_kind_label(flux_kind))
        println("  MPI ranks:                  ", NPROCS)
        println("  tetrahedra:                 ", nelements)
        println("  directed periodic faces:    ", directed_periodic_faces)
        println("  MPI interface faces:        ", mpi_interface_faces)
        println("  full dQ_H/dt:               ", breakdown.full)
        println("  volume contribution:        ", breakdown.volume)
        println("  interior-face contribution: ", breakdown.interior_faces)
        println("  periodic-face contribution: ", breakdown.periodic_faces)
        println("  boundary-face contribution: ", breakdown.boundary_faces)
        println("  reconstructed dQ_H/dt:      ", breakdown.reconstructed)
        println("  full - reconstructed:       ", breakdown.full_minus_reconstructed)
        println("  null tolerance:             ", null_tolerance)
    end

    @test directed_periodic_faces > 0
    @test NPROCS == 1 || mpi_interface_faces > 0
    @test rhs_split_error <= split_tolerance
    @test abs(breakdown.full_minus_reconstructed) <= split_tolerance
    @test abs(breakdown.full) <= null_tolerance

    return breakdown
end

try
    @testset "Poisson-bracket magnetic-charge null invariant" begin
        for flux_kind in (MaxwellFlux_Central, MaxwellFlux_Alternating)
            magnetic_charge_null_invariant_case(flux_kind)
        end
    end
finally
    MPI.Barrier(COMM)
    if !MPI_WAS_INITIALIZED && !MPI.Finalized()
        MPI.Finalize()
    end
end
