# Metallic-Sphere RCS Workflow

This note documents the DiscoGMPI workflow for radar-cross-section (RCS)
diagnostics of a perfectly electrically conducting (PEC) sphere illuminated by
a time-dependent plane wave. It is a code-oriented companion to the user guide:
it explains where the implementation lives, what is computed, how to run it,
and how to interpret the generated Mie-comparison figures.

## Code Map

The RCS path is intentionally split into three layers.

`examples/distributed_metallic_sphere_scattering.jl` is the production
scattering driver. It loads and partitions a tetrahedral box-minus-sphere mesh,
solves for the scattered Maxwell field, applies the incident-aware PEC
condition on the sphere, writes ParaView scattered/incident/total fields, and
optionally accumulates a time-domain near-to-far RCS diagnostic.

The RCS-specific structures and functions in that driver are:

- `RCSObservationDirection`: stores one far-field direction
  `(theta, phi)`, the unit vector `rhat`, and the spherical basis vectors
  `e_theta`, `e_phi`.
- `RCSFaceSample`: stores owned quadrature nodes, normals, physical weights,
  and coordinates on PEC sphere faces.
- `RCSWorkspace`: stores observation directions, PEC surface samples, complex
  current accumulators, Fourier-window metadata, incident frequency, wave
  number, impedance, and incident amplitude.
- `rcs_observation_directions`: builds the angular sampling grid from
  `--rcs-theta-*` and `--rcs-phi-degrees`.
- `build_rcs_face_samples`: extracts owned PEC sphere boundary faces tagged
  with physical surface `10`.
- `rcs_time_window`: evaluates the Hann window over
  `[--rcs-start-time, --final-time]`.
- `accumulate_rcs_sample!`: integrates the PEC surface current over the sphere
  and accumulates the complex Fourier coefficient.
- `global_rcs_accumulator`: reduces the rank-local complex accumulators.
- `far_field_from_surface_current`: converts the accumulated surface-current
  integral into the transverse far-field amplitude.
- `write_rcs_results`: writes `diagnostics/rcs.csv`.

`examples/validate_coarse_sphere_rcs.jl` is a release-gate wrapper. It runs the
scattering driver with RCS enabled, checks that `rcs.csv` contains finite,
positive, windowed samples, computes a lightweight Julia PEC Mie comparison,
and writes `diagnostics/coarse_mie_rcs_comparison.csv` plus
`coarse_sphere_rcs_summary.csv/json`. Its default thresholds are intentionally
loose; strict error thresholds are optional.

`examples/postprocess_metallic_sphere_mie_rcs.py` is the publication-oriented
postprocessor. It reads `diagnostics/rcs.csv`, evaluates the exact PEC sphere
Mie solution at the same angles, writes
`diagnostics/mie_rcs_comparison.csv`, writes
`tables/mie_rcs_comparison.tex`, and generates
`plots/mie_rcs_comparison.pdf` or any path supplied by `--plot-output`.

The main supporting library code is:

- `src/MaxwellExactSolutions.jl`: `IncidentPlaneWaveParameters`,
  `incident_electric_plane_wave`, and `incident_magnetic_plane_wave`.
- `src/solver/kernels/maxwell.jl`: `MaxwellBoundaryData`,
  `incident_pec_boundary_data`, and `set_boundary_data_time!`.
- `src/PhysicalCoverage.jl` and `src/NonlinearPML.jl`: Maxwell RHS closures,
  material coefficients, boundary data, and nonlinear PML source coupling.
- `src/MaxwellDriverSupport.jl`: six-sided Cartesian nonlinear PML setup.
- `src/RunLayout.jl` and `src/DistributedIO.jl`: isolated run directories,
  diagnostics paths, metadata, manifests, and run status.
- `src/MaxwellOutput.jl`: collective output helpers and ParaView PVD writing.
- `src/solver/kernels/*`: DG geometry, face maps, normals, face mass matrices,
  halo exchange, fluxes, and ESPRK integration.

## Physical Model

The numerical unknown in the scattering driver is the scattered field

```math
E_h = E_{\rm scat,h}, \qquad H_h = H_{\rm scat,h}.
```

The incident field is a plane wave traveling in the positive `z` direction,
with electric polarization in the `x` direction:

```math
E_{\rm inc}(x,y,z,t)
  = A\,\hat x\cos(kz-\omega t),
```

```math
H_{\rm inc}(x,y,z,t)
  = \frac{A}{Z}\,\hat y\cos(kz-\omega t),
```

where

```math
k=\frac{2\pi}{\lambda},\qquad
\omega = c k,\qquad
c=\frac{1}{\sqrt{\epsilon\mu}},\qquad
Z=\sqrt{\frac{\mu}{\epsilon}}.
```

The total field is reconstructed as

```math
E_{\rm tot}=E_{\rm scat}+E_{\rm inc},\qquad
H_{\rm tot}=H_{\rm scat}+H_{\rm inc}.
```

On the metallic sphere, the incident-aware PEC condition is

```math
n\times E_{\rm tot}=0,
\qquad
n\times E_{\rm scat}=-n\times E_{\rm inc}.
```

The production driver also computes the residual

```math
\|n\times E_{\rm tot}\|_{L^2(\Gamma_{\rm PEC})}
```

as an independent diagnostic of the incident-aware boundary condition.

## RCS Definition and Normalization

The RCS diagnostic follows the normalization written by the driver:

```math
\sigma(\theta,\phi)
  =
  4\pi
  \frac{|E^\infty_{\rm scat}(\theta,\phi)|^2}
       {|E_{\rm inc}|^2}.
```

The postprocessor plots the dimensionless quantity

```math
10\log_{10}\left(\frac{\sigma}{\pi a^2}\right),
```

where `a` is the sphere radius. Its error panel reports

```math
\frac{|\sigma_h-\sigma_{\rm Mie}|}{|\sigma_{\rm Mie}|}.
```

The comparison CSV also stores absolute error, dB error, and
`|sigma_h-sigma_Mie|/(pi*a^2)`.

## Near-To-Far Surface-Current Extraction

For a PEC scatterer, the equivalent electric surface current is computed from
the total magnetic field on the metallic surface:

```math
J_s = n\times H_{\rm tot}.
```

At each sampled time, the driver integrates this current over owned PEC sphere
faces. For an observation direction `rhat`, it accumulates

```math
I(\hat r,t)
  =
  \int_{\Gamma_{\rm PEC}}
  J_s(x,t)\,
  \exp(-i k\,\hat r\cdot x)\,dS.
```

MPI ranks contribute owned face quadrature only. The complex integrals are
summed across ranks before the far field is reconstructed.

The transverse far-field amplitude used by the driver is

```math
E^\infty(\hat r)
  =
  \frac{i k Z}{4\pi}
  \left(\hat r(\hat r\cdot I)-I\right).
```

The reported components are the projections

```math
E_\theta = e_\theta\cdot E^\infty,\qquad
E_\phi = e_\phi\cdot E^\infty.
```

## Time-Windowed Fourier Extraction

The simulation is time-domain. A monochromatic RCS requires extracting the
phasor at the incident angular frequency. DiscoGMPI accumulates the windowed
Fourier coefficient

```math
\widehat I(\hat r)
  \approx
  \frac{2}{\sum_m w_m}
  \sum_m w_m I(\hat r,t_m)\exp(i\omega t_m),
```

where the sum is over times selected by `--rcs-every`. The window `w_m` is a
Hann window over `[--rcs-start-time, --final-time]`:

```math
w(t)=
\frac12\left[
1-\cos\left(2\pi\frac{t-t_0}{T-t_0}\right)
\right],
\qquad t_0\le t\le T.
```

Before `--rcs-start-time`, samples are ignored. If `--final-time <=
--rcs-start-time`, the code returns a unit window; this is useful for one-step
smoke tests but not for quantitative RCS validation.

## CLI Reference

The production driver is:

```bash
mpiexec -n <ranks> julia --project=. \
  examples/distributed_metallic_sphere_scattering.jl [options]
```

The release-gate wrapper is:

```bash
mpiexec -n <ranks> julia --project=. \
  examples/validate_coarse_sphere_rcs.jl [gate options] [scattering options]
```

The RCS options are:

- `--enable-rcs`: enable RCS extraction. The production driver defaults to
  disabled; the coarse validation gate enables it by default.
- `--disable-rcs`: explicitly disable RCS extraction.
- `--rcs-start-time T`: start Fourier accumulation at time `T`. Use this to
  skip startup transients.
- `--rcs-every N`: accumulate an RCS sample every `N` time steps, and always at
  the final step. `N=1` maximizes sampling but can be unnecessarily expensive.
- `--rcs-theta-count N`: number of polar scattering angles between
  `--rcs-theta-min-degrees` and `--rcs-theta-max-degrees`.
- `--rcs-theta-min-degrees A`, `--rcs-theta-max-degrees B`: polar angle range.
  The default is `0` to `180` degrees.
- `--rcs-phi-degrees LIST`: comma-separated azimuth angles. The default is
  `0`, matching the common E-plane comparison for an `x`-polarized incident
  wave propagating in `+z`.
- `--min-max-rcs V`: coarse gate threshold for nonzero numerical RCS.
- `--max-absolute-normalized-error V`: optional strict gate for
  `max |sigma_h-sigma_Mie|/(pi*a^2)`.
- `--max-rms-relative-error V`: optional strict gate for RMS relative error.
- `--max-db-error V`: optional strict gate for maximum dB error.
- `--mie-terms N`: Mie-series truncation order for the Julia validation gate.
  The default is automatic.

The Mie postprocessor is:

```bash
python3 examples/postprocess_metallic_sphere_mie_rcs.py \
  --run-dir <run-directory>
```

Useful options are:

- `--rcs-csv PATH`: read an RCS CSV outside the run directory.
- `--radius A`, `--wavelength L`, `--ka X`: override geometry/frequency
  metadata.
- `--terms N`: set the Mie truncation order.
- `--comparison-csv PATH`, `--latex-output PATH`, `--plot-output PATH`:
  override output paths.

## Meshes

The primary mesh generator is
`examples/meshes/metallic_sphere_scattering.geo`. It creates a Cartesian box
minus an inner sphere. Physical surface tags are:

- `1`: x-min outer PML boundary.
- `2`: x-max outer PML boundary.
- `3`: y-min outer PML boundary.
- `4`: y-max outer PML boundary.
- `5`: z-min outer PML boundary.
- `6`: z-max outer PML boundary.
- `10`: inner PEC metallic sphere.

The PML is not a separate volume. The solver applies Cartesian nonlinear PML
profiles based on coordinates and `--pml-width`.

Current sphere/RCS meshes include:

- `examples/meshes/metallic_sphere_scattering.vtk`: larger production-like
  scattering mesh.
- `examples/meshes/metallic_sphere_scattering_validation.vtk`: default coarse
  validation mesh.
- `examples/meshes/metallic_sphere_scattering_epw4.vtk`: lighter high-order
  smoke mesh with four first-order elements per wavelength near the sphere.
- `examples/meshes/metallic_sphere_scattering_epw2.vtk`: very coarse smoke
  mesh when present; use for pipeline checks, not final RCS validation.

Example EPW4 generation command:

```bash
gmsh examples/meshes/metallic_sphere_scattering.geo -3 -format vtk \
  -o examples/meshes/metallic_sphere_scattering_epw4.vtk \
  -setnumber ElementsPerWavelength 4 \
  -setnumber SphereMeshSizeFactor 1.0 \
  -setnumber PMLMeshSizeFactor 3.0 \
  -setnumber FarMeshSizeFactor 4.0 \
  -setnumber CurvatureSamples 8 \
  -setnumber PMLWidth 1.0 \
  -setnumber AirBuffer 1.0 \
  -setnumber OptimizeMesh 0
```

For `p=4`, the approximate interpolation points per wavelength are
`ElementsPerWavelength * (p+1)`. This does not remove the need to resolve the
geometry. On very coarse meshes the faceted sphere, not the field polynomial
order, may dominate the RCS error.

## Recommended Runs

EPW2 or EPW4 smoke tests check that the setup, RCS extraction, CSV writing,
Mie postprocessing, and plot generation work. They are not quantitative Mie
validation:

```bash
mpiexec -n 2 -genv OPENBLAS_NUM_THREADS 1 julia --project=. \
  examples/validate_coarse_sphere_rcs.jl \
  --mesh=examples/meshes/metallic_sphere_scattering_epw4.vtk \
  --order=4 --esprk-order=5 \
  --final-time=1e-6 --rcs-start-time=1e-6 \
  --rcs-every=1 --diagnostics-every=1 \
  --min-max-rcs=0 \
  --output-dir=output/validation_sequence/coarse_sphere_rcs_epw4_p4_smoke
```

A first transient-free quantitative experiment should run for several incident
periods and start Fourier accumulation after the startup transient has left the
near field:

```bash
mpiexec -n 2 -genv OPENBLAS_NUM_THREADS 1 julia --project=. \
  examples/validate_coarse_sphere_rcs.jl \
  --mesh=examples/meshes/metallic_sphere_scattering_epw2.vtk \
  --order=4 --esprk-order=5 \
  --final-time=12.0 --rcs-start-time=5.0 \
  --rcs-every=5 --diagnostics-every=250 \
  --min-max-rcs=0 \
  --output-dir=output/validation_sequence/coarse_sphere_rcs_epw2_p4_transient
```

For a stronger EPW4 quantitative run, use the same idea but expect a much
larger runtime:

```bash
mpiexec -n 4 -genv OPENBLAS_NUM_THREADS 1 julia --project=. \
  examples/validate_coarse_sphere_rcs.jl \
  --mesh=examples/meshes/metallic_sphere_scattering_epw4.vtk \
  --order=4 --esprk-order=5 \
  --final-time=12.0 --rcs-start-time=5.0 \
  --rcs-every=10 --diagnostics-every=500 \
  --min-max-rcs=0 \
  --output-dir=output/validation_sequence/coarse_sphere_rcs_epw4_p4_transient
```

After the run:

```bash
python3 examples/postprocess_metallic_sphere_mie_rcs.py \
  --run-dir output/validation_sequence/coarse_sphere_rcs_epw4_p4_transient
```

## Choosing `--rcs-start-time`

`--rcs-start-time` should be after the incident wave has interacted with the
sphere and after nonphysical startup content has mostly left the near field.
It does not reduce the time integration cost; it only controls which samples
enter the Fourier coefficient.

For the current default `lambda=1`, `epsilon=mu=1`, the wave speed is one and
the period is one. In the box with radius `a=1`, air gap `1`, and PML width
`1`, a conservative first estimate is

```text
rcs_start_time ~= 3 to 5 periods.
```

Use diagnostics to refine this:

1. Plot total scattered energy and PEC residual versus time.
2. Inspect ParaView snapshots of `E_scat` and `H_scat`.
3. Choose a start time after the initial growth and after early reflections
   have settled.
4. Repeat with later start times. A reliable RCS estimate should not change
   significantly when `--rcs-start-time` is shifted modestly.

## Outputs

The scattering driver writes:

- `diagnostics/rcs.csv`: numerical far-field table with one row per
  observation direction.
- `diagnostics/scattering_diagnostics.csv`: electromagnetic diagnostics of the
  scattered solution.
- `diagnostics/pec_boundary_residual.csv`: `n x E_total` residual on the
  metallic sphere.
- `fields.pvd` and `fields/*.pvtu`: optional ParaView output with
  `E_scat`, `H_scat`, `E_inc`, `H_inc`, `E_total`, `H_total`,
  `E_scat_magnitude`, and `H_scat_magnitude`.
- `config/run_metadata.toml`, `config/resolved_config.toml`,
  `config/partition_metadata.csv`, `input/input_manifest.toml`, and
  `status.toml`.

The coarse validation gate adds:

- `diagnostics/coarse_mie_rcs_comparison.csv`.
- `coarse_sphere_rcs_summary.csv`.
- `coarse_sphere_rcs_summary.json`.

The Python postprocessor adds:

- `diagnostics/mie_rcs_comparison.csv`.
- `tables/mie_rcs_comparison.tex`.
- `plots/mie_rcs_comparison.pdf`, or the file selected with `--plot-output`.

## Interpreting Plots

The top panel compares
`10 log10(sigma/(pi*a^2))` for the numerical DG RCS and the exact Mie series.
The bottom panel shows the relative RCS error.

A smoke-test plot validates the pipeline if:

- the numerical markers are finite;
- the angular grid is correct;
- the Mie curve is present;
- CSV, LaTeX, and plot files are produced.

It does not validate the RCS quantitatively. A one-step run or a run with
Fourier extraction during startup can produce large, structured errors even
when all files are correct.

A quantitative Mie validation requires:

- enough runtime for the scattered field to reach a nearly time-periodic state;
- Fourier accumulation over several periods;
- a start time after startup transients;
- adequate sphere geometry resolution;
- adequate near-field and PML resolution;
- RCS stability under changes in `--rcs-start-time`, `--rcs-every`, mesh, and
  polynomial order.

## Common Failure Modes

`RCS CSV contains no rows` usually means `--enable-rcs` was not active, the
simulation failed before final output, or no owned faces with physical tag `10`
were found.

`RCS was enabled but no nonzero windowed samples were accumulated` means the
sampling times did not enter the Fourier window. Check `--final-time`,
`--rcs-start-time`, and `--rcs-every`.

Large relative errors in a one-step smoke plot are expected. Run longer and
start the Fourier window after the transient before interpreting the Mie
comparison.

Large relative errors in a long run can indicate insufficient mesh resolution,
faceted sphere geometry error, PML reflection, an RCS sign/phase convention
issue, an incorrect physical surface tag, or a poor Fourier window.

If the PEC residual remains large on the sphere, inspect
`diagnostics/pec_boundary_residual.csv` before trusting RCS data.

If the Mie postprocessor cannot infer `radius` or `wavelength`, pass
`--radius`, `--wavelength`, or `--ka` explicitly.
