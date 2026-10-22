# Comparing complex FFD far fields

`compare_far_fields.jl` compares FFD-producing applications without loading the
DG solver, MPI, or an EMAGPIE implementation. It includes the production
`src/FFDFormat.jl` codec and uses `read_ffd` and `each_ffd_point` for every input.
Julia 1.9+ and its standard libraries suffice for numerical comparison.
Plotting uses the adjacent Python helper and Python 3.9+ with matplotlib.
No Python code parses FFD data or computes comparison metrics.

## Supplied EMAGPIE/Mie reference campaign

From the repository root:

```bash
julia --startup-file=no tools/ffd/compare_far_fields.jl \
  --config tools/ffd/emagpie_reference.toml
```

The example manifest points to `/home/mcircuns/Desktop/N2/EMAGPIE`. Edit its
paths when moving the campaign. Relative paths resolve against the manifest
directory, independently of the invoking working directory.
`--output-dir PATH` overrides its output directory. `--no-plots` removes the
Python dependency; `--python /path/to/python3` selects a plotting interpreter.
The CLI accepts either `--option value` or `--option=value`.

This manifest explicitly requires **two** sources. Success establishes
EMAGPIE/Mie interoperability and the stated tolerances; it does not establish
agreement with DiscoGMPI. The default required source list, when omitted from
a new manifest, is `Mie`, `EMAGPIE`, and `DiscoGMPI`.

The directory has twelve files: Mie and numerical results for two cuts and
three frequencies, with 500 samples per file. The physical example is a
radius-1 m PEC sphere at the origin, propagation along +x, electric polarization
along +z, and wavelengths 1.922, 0.961, 0.4805 m. Frequencies use 299792458 m/s.
The files round frequency and angles to seven significant digits; matching
allows `frequency_rtol=1e-6` and Cartesian chord distance `direction_atol=2e-6`.

Three explicit assumptions adapt this legacy example:

1. `legacy_polarization_column=true` replaces only the `[DATA]` table token
   `polarization` with standard `pol` in a temporary file. The strict production
   codec parses that file. Source files are never edited. The package reader's
   grammar is unchanged; other malformed syntax still fails.
2. `field_unit_override="m"` interprets the files' `V/m` label as normalized
   scattering length. `sphere_comparison.jl` supports this interpretation: the
   Mie amplitude contains `i/k` and RCS is computed as `4*pi*|F|^2`. Arbitrary
   `V/m` files are rejected without a justified override.
3. `field_scale=[-1.0,0.0]` divides the supplied physical +z response by its
   signed amplitude in the canonical incident V basis. At emitter -x,
   `e_theta_inc=-z`. DiscoGMPI already performs this normalization on export
   and needs no such scale.

The referenced EMAGPIE/Mie implementation sources are absent from the directory.
Their time/radial convention is therefore **unverified**. The shared minus sign
does not change EMAGPIE/Mie errors or RCS. Complex agreement with DiscoGMPI must
still establish the temporal convention; the tool does not silently fit it.

## Cartesian alignment and signed cuts

Each sample is converted using its original spherical basis:
`F = Etheta*e_theta + Ephi*e_phi`. This occurs before angle canonicalization.
Radians/degrees, frequency units Hz/kHz/MHz/GHz, and normalized length units
m/cm/mm are converted to SI. Other frequency units require `frequency_scale`
to Hz. `field_scale=[real,imaginary]` applies an explicit complex scale. Original
unit labels, all scales, adapter counts, and input SHA-256 hashes appear in
`report.toml`.

| Cut | Cartesian condition | Signed plotting coordinate |
| --- | --- | --- |
| phi | z=0, y>=0 | atan(y,x), from 0 to pi |
| theta | y=0, z>=0 | atan(x,z), from -pi/2 to pi/2 |

For negative signed theta, canonical coordinates are `theta=-alpha, phi=pi`
and **both** spherical field components change sign. Cartesian conversion
handles this, periodic azimuths, and transverse gauges at observation poles.
`samples.csv` stores canonical colatitude/azimuth and signed `angle_rad`.

Each cut must cover both endpoints, contain at least three distinct points,
and have no gap above `max_gap_rad`. Shared endpoints are deduplicated only
when their Cartesian fields agree within `duplicate_field_rtol` (default 1e-4).
Incident emitter and V/H basis vectors must agree across sources.

`alignment="exact"` matches each reference direction to a source direction
within the Cartesian chord tolerance. Extra source points are allowed; missing
matches fail. `alignment="linear"` explicitly permits interpolation of complex
Cartesian fields in signed cut angle. It forbids extrapolation and gaps above
`max_gap_rad`, and projects out the radial interpolation residual. Reports count
interpolated samples. Use exact alignment for the sphere launcher.

## RCS and complex conventions

The normalized amplitudes give RCS in square metres:

```text
sigma_total = 4*pi*(abs2(Etheta) + abs2(Ephi))
sigma_theta = 4*pi*abs2(Etheta)
sigma_phi   = 4*pi*abs2(Ephi)
```

Co-polar direction is the projection of the incident V/H electric basis onto
the observation tangent plane. Its sign follows the cut's continuous meridional
gauge, avoiding an artificial complex phase jump at signed theta=0. At a
vanishing projection, that cut basis fixes the gauge. Cross-polar direction is
`cross(rhat,co)`. For this vertical-polarization sphere example, co/cross powers
agree with theta/phi powers. This is not a general radar convention such as
Ludwig-3. Tiny cross-polar components also reflect rounded angle metadata.

The tool evaluates eight transformations in the common canonical spherical
basis: `identity`, `conjugate`, `negative_identity`, `negative_conjugate`,
`flip_theta`, `flip_phi`, `conjugate_flip_theta`, `conjugate_flip_phi`.
`transformations.csv` lists every result. Only the source's declared
`expected_transform` (default `identity`) determines plotted data and pass/fail.
There is no fitted phase or scale. Equivalent diagnostic minima are recorded:
a zero reference cross-polar component cannot identify that component's sign.

## Quantitative gates

The primary measures use discrete sample norms, without solid-angle weighting:

```text
complex_l2 = sqrt(sum(norm(F-G)^2) / sum(norm(G)^2))
rcs_l2     = norm(sigma-sigma_reference) / norm(sigma_reference)
```

Both gates are always active (default limits 0.03 and 0.04). Optional limits are
`max_normalized_rcs_error`, `max_db_error`, and `max_cross_power_fraction`.
Normalized maximum RCS error is `max(abs(delta_sigma))/(pi*radius_m^2)`.
Cross-power fraction is summed cross-polar power divided by summed total power.
`[sources.tolerances]` overrides global limits for one source. Limits must be
finite and nonnegative; unknown global `max_*` tolerance names and unknown
source tolerance names fail. Duplicate-field tolerances must also be finite
and nonnegative.

The maximum dB gate excludes reference samples at or below
`null_relative_floor * maximum(reference_sigma)`. `db_floor` in dBsm also clamps
powers in logarithmic differences, avoiding infinities. Pointwise CSVs retain
every direction and unclipped linear RCS; plotting floors do not affect linear
regression metrics.

The supplied reference manifest passes with maximum complex L2 about 0.01772,
maximum RCS L2 about 0.02815, and integrated cross-power fraction below 3.6e-7.
Deep nulls give dB differences up to about 3.4 dB, so the example does not impose
an unqualified maximum dB limit.

Metric failures write CSV/TOML diagnostics and plots before returning nonzero.
Missing files, malformed data, incompatible incidence, incomplete cuts, and
invalid configuration fail immediately. A better diagnostic transform never
overrides a failed declared transform. Cut data are loaded into memory; the
underlying codec separately supports lazy scans for larger data sets.

| Output | Contents |
| --- | --- |
| samples.csv | Aligned complex fields, Cartesian directions, signed angles, total and polarized RCS |
| pairs.csv | Signed RCS error divided by pi*a^2, dB error, vector error divided by peak reference amplitude |
| summary.csv | Selected-transform metrics and pass/fail per source, frequency, and cut |
| transformations.csv | Metrics for all eight transforms |
| report.toml | Configuration, hashes, units, equivalent minima, interpolation counts, limits and failures |
| plots/ | Total/polarized RCS, complex fields and error panels in PDF/SVG/300-dpi PNG |

Repeated comparisons replace generated files. Preserve a campaign directory
when its exact report or plot versions are needed for a publication.

## Reproducible DiscoGMPI campaign

After the reference comparison succeeds, prepare a three-source campaign:

```bash
julia tools/ffd/launch_sphere_benchmark.jl \
  --emagpie-dir /home/mcircuns/Desktop/N2/EMAGPIE \
  --output-dir output/ffd/sphere_benchmark --dry-run
```

This writes `plan.toml` and `campaign.toml` and prints three Gmsh commands,
six MPI driver commands, and the final comparison command. Repeating the same
invocation with `--execute` runs that plan with per-command logs. Existing meshes
or run/comparison directories are refused; use a fresh directory after a partial
or completed execution. `--help` lists ranks, order, mesh, PML, time-window and
plotting controls. Julia `Cmd` vectors preserve paths with spaces.
The plan fingerprints all reference inputs and the mesh template, driver, codec,
convention adapter, comparison, renderer and launcher sources, and records the
Git commit when available. Changed inputs or sources invalidate a prepared plan.

Three wavelength-specific meshes each serve two solves. The phi run uses
theta=90 degrees and 500 phi values. The theta run uses phi=0 and 180 degrees
and 250 theta values from 90/499 to 90 degrees. Those 500 directions match the
supplied even-length signed grid without interpolation or a large rectangular
observation grid.

The solver uses epsilon=mu=1, so c_solver=1 m per solver time unit and
f_solver=1/lambda. The generated source declares `frequency_scale=299792458`
to obtain Hz. FFD labels frequency `solver_time^-1`; relabeling alone never
rescales values. Normalized lengths already use metres.

Default mesh/order, PML, CFL, and time-window choices are explicit starting
parameters, not a convergence certificate. Vary spatial/geometry resolution,
timestep, absorber and time window before accepting physical DG agreement.
The launcher currently uses the equal-order production scattering driver.
Its full comparison requires all three sources and starts with the declared
identity transform; diagnose a phase discrepancy before changing that setting.

## Tests

```bash
julia --startup-file=no test/test_ffd_comparison.jl
julia --startup-file=no test/test_ffd_benchmark_launcher.jl
```

Synthetic fixtures exercise the real codec with signed/canonical angles,
legacy adapters, units, incidence, interpolation, three source labels,
intentional phase corruption and failure reports. They require neither Desktop
data nor a DG simulation.
