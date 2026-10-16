#!/usr/bin/env julia

include(joinpath(@__DIR__, "distributed_metallic_sphere_scattering.jl"))

const PEC_ALMOND_C0 = 299_792_458.0
const DEFAULT_PEC_ALMOND_FREQUENCY_HZ = 3.5e9
const DEFAULT_PEC_ALMOND_WAVELENGTH =
    PEC_ALMOND_C0 / DEFAULT_PEC_ALMOND_FREQUENCY_HZ
const DEFAULT_PEC_ALMOND_PEC_BOUNDARY_ID = 10
const DEFAULT_PEC_ALMOND_MESH = joinpath(
    @__DIR__,
    "meshes",
    "pec_almond_scattering_step_fx1_epw4.vtk",
)
const DEFAULT_PEC_ALMOND_OUTPUT_DIR = joinpath(
    normpath(joinpath(@__DIR__, "..")),
    "output",
    "pec_almond_scattering",
)

function pec_almond_usage(io::IO = stdout)
    println(io, """
Distributed PEC almond scattering driver

Usage:
  mpiexec -n <ranks> julia --project=. \\
    examples/distributed_pec_almond_scattering.jl [options]

This is a thin production frontend over the generic PEC-scatterer Maxwell/PML
driver. It solves for the scattered field around the NASA/Austin PEC almond
with a +z incident plane wave, x polarization, and the same diagnostics as the
metallic-sphere workflow: scattered/incident/total ParaView fields, PEC
boundary residuals, invariant diagnostics, and optional RCS extraction.

Default mesh:
  examples/meshes/pec_almond_scattering_step_fx1_epw4.vtk

Generate it with:
  bash examples/meshes/generate_pec_almond_scattering_step_mesh.sh

Almond-specific options:
  --frequency-hz F             Physical benchmark frequency used only to set
                               wavelength = c0/F in the normalized solver.
                               Default: 3.5e9

Inherited core options:
  --mesh PATH                  Almond VTK mesh. Default: generated EPW4 mesh.
  --wavelength L               Normalized incident wavelength. Overrides the
                               default implied by --frequency-hz.
  --frequency F                Normalized solver frequency. If F > 0, it
                               overrides wavelength through L = c/F.
  --amplitude A                Incident electric-field amplitude. Default: 1
  --order N                    DG polynomial order. Default: 2
  --esprk-order N              ESPRK order in 1:6. Default: 3
  --final-time T               Final simulation time. Default: two periods
  --cfl C                      Maxwell CFL factor. Default: 0.15
  --pml-width W                Cartesian PML width on each box side. Default:
                               one wavelength at --frequency-hz.
  --sigma-max S                Maximum PML damping. Default: 12
  --sigma-degree N             Polynomial PML degree. Default: 2
  --flux centered|alternating|upwind
                               Poisson-bracket numerical flux. Default: centered
  --paraview-every N           ParaView interval; 0 disables snapshots.
                               Default: 50
  --enable-rcs / --disable-rcs Enable or disable RCS extraction. Default:
                               enabled for this almond frontend.
  --rcs-start-time T           Start Fourier extraction at T. Default: last
                               incident period, or 0 for shorter runs
  --rcs-every N                RCS sampling interval in time steps. Default: 10
  --rcs-theta-count N          Number of polar theta samples. Default: 181
  --rcs-phi-degrees LIST       Comma-separated azimuth angles. Default: 0
  --output-dir PATH            Isolated run output directory.
  --help                       Show this message.

The solver is nondimensional internally. With epsilon = mu = 1, the wave speed
is 1, so --frequency-hz is converted to wavelength using the physical speed of
light and then the normalized solver frequency becomes 1/wavelength.
""")
end

function pec_almond_default_args()
    wavelength = DEFAULT_PEC_ALMOND_WAVELENGTH
    return [
        "--mesh=$(DEFAULT_PEC_ALMOND_MESH)",
        "--radius=0.0",
        "--scatterer-name=PEC almond",
        "--pec-boundary-id=$(DEFAULT_PEC_ALMOND_PEC_BOUNDARY_ID)",
        "--wavelength=$wavelength",
        "--frequency=0.0",
        "--pml-width=$wavelength",
        "--enable-rcs",
        "--output-dir=$(DEFAULT_PEC_ALMOND_OUTPUT_DIR)",
    ]
end

function translate_pec_almond_args(args::Vector{String})
    translated = String[]
    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--frequency-hz" || startswith(argument, "--frequency-hz=")
            raw_value, index = option_value(args, index, "--frequency-hz")
            frequency_hz = parse(Float64, raw_value)
            frequency_hz > 0.0 || error("--frequency-hz must be positive.")
            wavelength = PEC_ALMOND_C0 / frequency_hz
            push!(translated, "--wavelength=$wavelength")
            push!(translated, "--frequency=0.0")
        else
            push!(translated, argument)
        end
        index += 1
    end
    return translated
end

function pec_almond_main(args::Vector{String})
    if any(argument -> argument == "--help" || argument == "-h", args)
        pec_almond_usage()
        return nothing
    end

    MPI.Init()
    comm = MPI.COMM_WORLD
    try
        config = parse_config(
            vcat(pec_almond_default_args(), translate_pec_almond_args(args)),
        )
        run_experiment(config, comm)
    catch error
        message = sprint(showerror, error)
        println(stderr, "ERROR: ", message)
        rethrow()
    finally
        MPI.Finalize()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    pec_almond_main(ARGS)
end
