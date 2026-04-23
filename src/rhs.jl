# Workspace for RHS  
struct RHSWorkspaceMPI{T}
    rhsEx::Matrix{T}
    rhsEy::Matrix{T}
    rhsEz::Matrix{T}
    rhsHx::Matrix{T}
    rhsHy::Matrix{T}
    rhsHz::Matrix{T}

    fluxEx::Matrix{T}
    fluxEy::Matrix{T}
    fluxEz::Matrix{T}
    fluxHx::Matrix{T}
    fluxHy::Matrix{T}
    fluxHz::Matrix{T}

    traceE::Array{T,4}      # (Nfp, Nfaces, nlocal_elems, 3)
    traceH::Array{T,4}      # (Nfp, Nfaces, nlocal_elems, 3)

    ghostE::Dict{Int,Array{T,3}}  # rank => (Nfp, nrecvfaces, 3)
    ghostH::Dict{Int,Array{T,3}}

    avgx::Vector{T}
    avgy::Vector{T}
    avgz::Vector{T}
    jumpx::Vector{T}
    jumpy::Vector{T}
    jumpz::Vector{T}
    tempx::Vector{T}
    tempy::Vector{T}
    tempz::Vector{T}
end

function allocate_rhs_workspace_mpi(
    Np::Int,
    Nfp::Int,
    Nfaces::Int,
    nlocal_elems::Int,
    ghostE::Dict{Int,Array{Float64,3}},
    ghostH::Dict{Int,Array{Float64,3}},
)
    rhsEx = zeros(Float64, Np, nlocal_elems)
    rhsEy = similar(rhsEx)
    rhsEz = similar(rhsEx)
    rhsHx = similar(rhsEx)
    rhsHy = similar(rhsEx)
    rhsHz = similar(rhsEx)

    fluxEx = similar(rhsEx)
    fluxEy = similar(rhsEx)
    fluxEz = similar(rhsEx)
    fluxHx = similar(rhsEx)
    fluxHy = similar(rhsEx)
    fluxHz = similar(rhsEx)

    traceE = zeros(Float64, Nfp, Nfaces, nlocal_elems, 3)
    traceH = similar(traceE)

    avgx  = zeros(Float64, Nfp)
    avgy  = similar(avgx)
    avgz  = similar(avgx)
    jumpx = similar(avgx)
    jumpy = similar(avgx)
    jumpz = similar(avgx)
    tempx = similar(avgx)
    tempy = similar(avgx)
    tempz = similar(avgx)

    return RHSWorkspaceMPI(
        rhsEx, rhsEy, rhsEz, rhsHx, rhsHy, rhsHz,
        fluxEx, fluxEy, fluxEz, fluxHx, fluxHy, fluxHz,
        traceE, traceH, ghostE, ghostH,
        avgx, avgy, avgz, jumpx, jumpy, jumpz, tempx, tempy, tempz,
    )
end

struct TraceCommData
    trace_maps::MPITraceExchangeTable
    face_lookup::MPIRemoteFaceLookup
    perm_maps::MPIFacePermutationMap
end

# This function allocates a fresh vector every time 
# That is okay for getting the MPI version working, but later we may want to precompute/store these local face 
# row ids once, because right now we are rebuilding them for every face at each stage
# Correctness-wise: okay. Performance-wise: not ideal, but should be fine for now.
@inline function local_face_node_ids(problem::Maxwell3D, f::Int, e::Int)
    Np, _, _ = get_number_of_nodes_simplex(problem.feb3D)
    return @views problem.vmapM[:, f, e] .- (e - 1) * Np
end

# Trace construction  
# build_face_traces_E_owned!(traceE, Ex, Ey, Ez, problem, mesh)
function build_face_traces_E_owned!(
    traceE::Array{Float64,4},
    Ex::Matrix{Float64},
    Ey::Matrix{Float64},
    Ez::Matrix{Float64},
    problem::Maxwell3D,
    mesh,
)
    _, Nfp, Nfaces = get_number_of_nodes_simplex(problem.feb3D)

    for e in mesh.partition.owned
        for f in 1:Nfaces
            ids = local_face_node_ids(problem, f, e)   # length Nfp, local row ids in element e
            @views begin
                traceE[:, f, e, 1] .= Ex[ids, e]
                traceE[:, f, e, 2] .= Ey[ids, e]
                traceE[:, f, e, 3] .= Ez[ids, e]
            end
        end
    end
    return nothing
end

# build_face_traces_H_owned!(traceH, Hx, Hy, Hz, problem, mesh)
function build_face_traces_H_owned!(
    traceH::Array{Float64,4},
    Hx::Matrix{Float64},
    Hy::Matrix{Float64},
    Hz::Matrix{Float64},
    problem::Maxwell3D,
    mesh,
)
    _, Nfp, Nfaces = get_number_of_nodes_simplex(problem.feb3D)

    for e in mesh.partition.owned
        for f in 1:Nfaces
            ids = local_face_node_ids(problem, f, e)   # length Nfp, local row ids in element e
            @views begin
                traceH[:, f, e, 1] .= Hx[ids, e]
                traceH[:, f, e, 2] .= Hy[ids, e]
                traceH[:, f, e, 3] .= Hz[ids, e]
            end
        end
    end
    return nothing
end 

# H-face kernel
function face_integral_H_from_traces!(
    fluxHx_k::Vector{Float64},
    fluxHy_k::Vector{Float64},
    fluxHz_k::Vector{Float64},
    face_rows::AbstractVector{Int},
    mass_face::AbstractMatrix{Float64},
    nx::Float64, ny::Float64, nz::Float64,
    ExM::AbstractVector{Float64}, EyM::AbstractVector{Float64}, EzM::AbstractVector{Float64},
    ExP::AbstractVector{Float64}, EyP::AbstractVector{Float64}, EzP::AbstractVector{Float64},
    avgEx::Vector{Float64}, avgEy::Vector{Float64}, avgEz::Vector{Float64},
    tempx::Vector{Float64}, tempy::Vector{Float64}, tempz::Vector{Float64},
)
    @. avgEx = 0.5 * (ExM + ExP)
    @. avgEy = 0.5 * (EyM + EyP)
    @. avgEz = 0.5 * (EzM + EzP)

    mul!(tempx, mass_face, avgEx)
    mul!(tempy, mass_face, avgEy)
    mul!(tempz, mass_face, avgEz)

    @views begin
        @. fluxHx_k[face_rows] += nz * tempy - ny * tempz
        @. fluxHy_k[face_rows] += nx * tempz - nz * tempx
        @. fluxHz_k[face_rows] += ny * tempx - nx * tempy
    end
    return nothing
end

# E-face kernel 
function face_integral_E_from_traces!(
    fluxEx_k::Vector{Float64},
    fluxEy_k::Vector{Float64},
    fluxEz_k::Vector{Float64},
    face_rows::AbstractVector{Int},
    mass_face::AbstractMatrix{Float64},
    nx::Float64, ny::Float64, nz::Float64,
    HxM::AbstractVector{Float64}, HyM::AbstractVector{Float64}, HzM::AbstractVector{Float64},
    HxP::AbstractVector{Float64}, HyP::AbstractVector{Float64}, HzP::AbstractVector{Float64},
    jumpHx::Vector{Float64}, jumpHy::Vector{Float64}, jumpHz::Vector{Float64},
    tempx::Vector{Float64}, tempy::Vector{Float64}, tempz::Vector{Float64},
)
    @. jumpHx = HxM - HxP
    @. jumpHy = HyM - HyP
    @. jumpHz = HzM - HzP

    mul!(tempx, mass_face, jumpHx)
    mul!(tempy, mass_face, jumpHy)
    mul!(tempz, mass_face, jumpHz)

    @views begin
        @. fluxEx_k[face_rows] += 0.5 * (ny * tempz - nz * tempy)
        @. fluxEy_k[face_rows] += 0.5 * (nz * tempx - nx * tempz)
        @. fluxEz_k[face_rows] += 0.5 * (nx * tempy - ny * tempx)
    end
    return nothing
end

# Communication 
# exchange_E_traces_blocking!(workspace, commdata, comm_mpi)
# exchange_H_traces_blocking!(workspace, commdata, comm_mpi)

# Face fluxes 
# accumulate_face_flux_H_owned!(workspace, problem, mesh, commdata)
function accumulate_face_flux_H_owned!(
    ws::RHSWorkspaceMPI{Float64},
    problem::Maxwell3D,
    mesh,
    commdata::TraceCommData,
)

    _, Nfp, Nfaces = get_number_of_nodes_simplex(problem.feb3D)

    ws.fluxHx .= 0.0
    ws.fluxHy .= 0.0
    ws.fluxHz .= 0.0

    for e in mesh.partition.owned
        fluxHx_k = @view ws.fluxHx[:, e]
        fluxHy_k = @view ws.fluxHy[:, e]
        fluxHz_k = @view ws.fluxHz[:, e]

        for f in 1:Nfaces
            nx = problem.n[1, f, e]
            ny = problem.n[2, f, e]
            nz = problem.n[3, f, e]

            mass_face = @view problem.Mfp[:, :, f, e]
            rows = local_face_node_ids(problem, f, e)

            uM, uP, facetype = get_face_traces(
                mesh,
                ws.traceE,
                ws.ghostE,
                commdata.face_lookup,
                commdata.perm_maps,
                e,
                f,
            )

            if facetype == :boundary
                ExP, EyP, EzP = boundary_plus_trace_E(problem, e, f, uM)
            else
                ExP = @view uP[:, 1]
                EyP = @view uP[:, 2]
                EzP = @view uP[:, 3]
            end

            ExM = @view uM[:, 1]
            EyM = @view uM[:, 2]
            EzM = @view uM[:, 3]

            face_integral_H_from_traces!(
                fluxHx_k, fluxHy_k, fluxHz_k,
                rows, mass_face, nx, ny, nz,
                ExM, EyM, EzM,
                ExP, EyP, EzP,
                ws.avgx, ws.avgy, ws.avgz,
                ws.tempx, ws.tempy, ws.tempz,
            )
        end
    end

    return nothing
end

# Still hardcoded to PEC-like behavior and ignore bc_tag.
# allocates new arrays because of unitary minus on the views
# okay for now, but later we may want a no-allocation boundary path
function boundary_plus_trace_E(problem, e, f, uM)
    ExP = -@view uM[:, 1]
    EyP = -@view uM[:, 2]
    EzP = -@view uM[:, 3]
    return ExP, EyP, EzP
end

# accumulate_face_flux_E_owned!(workspace, problem, mesh, commdata)
# RHS routines index problem.n, problem.Mfp, etc, with local element id e, so those arrays must already be built for the 
# local distributed mesh layout, including any stored ghost elements if their element slot exists in mesh
# second major assumption after vmapM
function accumulate_face_flux_E_owned!(
    ws::RHSWorkspaceMPI{Float64},
    problem::Maxwell3D,
    mesh,
    commdata::TraceCommData,
)

    _, Nfp, Nfaces = get_number_of_nodes_simplex(problem.feb3D)

    ws.fluxEx .= 0.0
    ws.fluxEy .= 0.0
    ws.fluxEz .= 0.0

    for e in mesh.partition.owned
        fluxEx_k = @view ws.fluxEx[:, e]
        fluxEy_k = @view ws.fluxEy[:, e]
        fluxEz_k = @view ws.fluxEz[:, e]

        for f in 1:Nfaces
            nx = problem.n[1, f, e]
            ny = problem.n[2, f, e]
            nz = problem.n[3, f, e]

            mass_face = @view problem.Mfp[:, :, f, e]
            rows = local_face_node_ids(problem, f, e)

            uM, uP, facetype = get_face_traces(
                mesh,
                ws.traceH,
                ws.ghostH,
                commdata.face_lookup,
                commdata.perm_maps,
                e,
                f,
            )

            if facetype == :boundary
                HxP, HyP, HzP = boundary_plus_trace_H(problem, e, f, uM)
            else
                HxP = @view uP[:, 1]
                HyP = @view uP[:, 2]
                HzP = @view uP[:, 3]
            end

            HxM = @view uM[:, 1]
            HyM = @view uM[:, 2]
            HzM = @view uM[:, 3]

            face_integral_E_from_traces!(
                fluxEx_k, fluxEy_k, fluxEz_k,
                rows, mass_face, nx, ny, nz,
                HxM, HyM, HzM,
                HxP, HyP, HzP,
                ws.jumpx, ws.jumpy, ws.jumpz,
                ws.tempx, ws.tempy, ws.tempz,
            )
        end
    end

    return nothing
end


# Still hardcoded to PEC-like behavior and ignore bc_tag.
function boundary_plus_trace_H(problem, e, f, uM)
    HxP = @view uM[:, 1]
    HyP = @view uM[:, 2]
    HzP = @view uM[:, 3]
    return HxP, HyP, HzP
end

# Volume terms 
# accumulate_volume_rhs_H_owned!(workspace, Ex, Ey, Ez, problem, mesh)
function accumulate_volume_rhs_H_owned!(
    ws::RHSWorkspaceMPI{Float64},
    Ex::Matrix{Float64},
    Ey::Matrix{Float64},
    Ez::Matrix{Float64},
    problem::Maxwell3D,
    mesh,
)
    for e in mesh.partition.owned
        invmass_k = @view problem.invMp[:, :, e]
        Cx_k = @view problem.Cwp[:, :, 1, e]
        Cy_k = @view problem.Cwp[:, :, 2, e]
        Cz_k = @view problem.Cwp[:, :, 3, e]

        Ex_k = @view Ex[:, e]
        Ey_k = @view Ey[:, e]
        Ez_k = @view Ez[:, e]

        fluxHx_k = @view ws.fluxHx[:, e]
        fluxHy_k = @view ws.fluxHy[:, e]
        fluxHz_k = @view ws.fluxHz[:, e]

        rhsHx_k = @view ws.rhsHx[:, e]
        rhsHy_k = @view ws.rhsHy[:, e]
        rhsHz_k = @view ws.rhsHz[:, e]

        rhsHx_k .= invmass_k * (Cy_k * Ez_k .- Cz_k * Ey_k .+ fluxHx_k)
        rhsHy_k .= invmass_k * (Cz_k * Ex_k .- Cx_k * Ez_k .+ fluxHy_k)
        rhsHz_k .= invmass_k * (Cx_k * Ey_k .- Cy_k * Ex_k .+ fluxHz_k)
    end
    return nothing
end

# accumulate_volume_rhs_E_owned!(workspace, Hx, Hy, Hz, problem, mesh)
function accumulate_volume_rhs_E_owned!(
    ws::RHSWorkspaceMPI{Float64},
    Hx::Matrix{Float64},
    Hy::Matrix{Float64},
    Hz::Matrix{Float64},
    problem::Maxwell3D,
    mesh,
)
    for e in mesh.partition.owned
        invmass_k = @view problem.invMp[:, :, e]
        Cx_k = @view problem.Csp[:, :, 1, e]
        Cy_k = @view problem.Csp[:, :, 2, e]
        Cz_k = @view problem.Csp[:, :, 3, e]

        Hx_k = @view Hx[:, e]
        Hy_k = @view Hy[:, e]
        Hz_k = @view Hz[:, e]

        fluxEx_k = @view ws.fluxEx[:, e]
        fluxEy_k = @view ws.fluxEy[:, e]
        fluxEz_k = @view ws.fluxEz[:, e]

        rhsEx_k = @view ws.rhsEx[:, e]
        rhsEy_k = @view ws.rhsEy[:, e]
        rhsEz_k = @view ws.rhsEz[:, e]

        rhsEx_k .= invmass_k * (Cy_k * Hz_k .- Cz_k * Hy_k .- fluxEx_k)
        rhsEy_k .= invmass_k * (Cz_k * Hx_k .- Cx_k * Hz_k .- fluxEy_k)
        rhsEz_k .= invmass_k * (Cx_k * Hy_k .- Cy_k * Hx_k .- fluxEz_k)
    end
    return nothing
end

# Full MPI RHS entry points 
# rhs_hamiltonian_H_mpi!(Ex, Ey, Ez, workspace, problem, mesh, commdata, comm_mpi)
function rhs_hamiltonian_H_mpi!(
    Ex::Matrix{Float64},
    Ey::Matrix{Float64},
    Ez::Matrix{Float64},
    ws::RHSWorkspaceMPI{Float64},
    problem::Maxwell3D,
    mesh,
    commdata::TraceCommData,
    comm_mpi,
)
    build_face_traces_E_owned!(ws.traceE, Ex, Ey, Ez, problem, mesh)
    exchange_face_traces_blocking!(ws.traceE, ws.ghostE, commdata.trace_maps, comm_mpi)
    accumulate_face_flux_H_owned!(ws, problem, mesh, commdata)
    accumulate_volume_rhs_H_owned!(ws, Ex, Ey, Ez, problem, mesh)
    return nothing
end

# rhs_hamiltonian_E_mpi!(Hx, Hy, Hz, workspace, problem, mesh, commdata, comm_mpi)
function rhs_hamiltonian_E_mpi!(
    Hx::Matrix{Float64},
    Hy::Matrix{Float64},
    Hz::Matrix{Float64},
    ws::RHSWorkspaceMPI{Float64},
    problem::Maxwell3D,
    mesh,
    commdata::TraceCommData,
    comm_mpi,
)
    build_face_traces_H_owned!(ws.traceH, Hx, Hy, Hz, problem, mesh)
    exchange_face_traces_blocking!(ws.traceH, ws.ghostH, commdata.trace_maps, comm_mpi)
    accumulate_face_flux_E_owned!(ws, problem, mesh, commdata)
    accumulate_volume_rhs_E_owned!(ws, Hx, Hy, Hz, problem, mesh)
    return nothing
end

# Time marching 
# ti_ESPRK_mpi!(Hx, Hy, Hz, Ex, Ey, Ez, workspace, problem, mesh, commdata, comm_mpi, time)
function ti_ESPRK_mpi!(
    Hx::Matrix{Float64},
    Hy::Matrix{Float64},
    Hz::Matrix{Float64},
    Ex::Matrix{Float64},
    Ey::Matrix{Float64},
    Ez::Matrix{Float64},
    ws::RHSWorkspaceMPI{Float64},
    problem::Maxwell3D,
    mesh,
    commdata::TraceCommData,
    comm_mpi,
    time::ODEParams,
)
    tiorder = time.order
    dt = time.Δt
    ntimesteps = compute_time_steps(time)

    COEFFS = Dict(
        2 => (a=(0.0, 1.0), b=(0.5, 0.5)),
        3 => (a=(0.2916666666666667, 0.75, -0.041666666666666664),
              b=(0.6666666666666666, -0.6666666666666666, 1.0)),
        4 => (a=(7.0/48.0, 3.9/8.0, -1.0/48.0, -1.0/48.0, 3.0/8.0, 7.0/48.0),
              b=(1.0/3.0, -1.0/3.0, 1.0, -1.0/3.0, 1.0/3.0, 0.0))
    )

    haskey(COEFFS, tiorder) || error("Selected ESPRK scheme not available.")
    a_coeffs, b_coeffs = COEFFS[tiorder]
    nstages = length(a_coeffs)

    for timestep in 1:ntimesteps
        for s in 1:nstages
            rhs_hamiltonian_H_mpi!(Ex, Ey, Ez, ws, problem, mesh, commdata, comm_mpi)

            for e in mesh.partition.owned
                @views begin
                    Hx[:, e] .+= dt * a_coeffs[s] .* ws.rhsHx[:, e]
                    Hy[:, e] .+= dt * a_coeffs[s] .* ws.rhsHy[:, e]
                    Hz[:, e] .+= dt * a_coeffs[s] .* ws.rhsHz[:, e]
                end
            end

            rhs_hamiltonian_E_mpi!(Hx, Hy, Hz, ws, problem, mesh, commdata, comm_mpi)

            for e in mesh.partition.owned
                @views begin
                    Ex[:, e] .+= dt * b_coeffs[s] .* ws.rhsEx[:, e]
                    Ey[:, e] .+= dt * b_coeffs[s] .* ws.rhsEy[:, e]
                    Ez[:, e] .+= dt * b_coeffs[s] .* ws.rhsEz[:, e]
                end
            end
        end

        # Later: global reductions for norms/energy
    end

    return nothing
end