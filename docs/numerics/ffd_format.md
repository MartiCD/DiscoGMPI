# FFD v1.0 far-field output

DiscoGMPI supports the FFD v1.0 text format for complex electromagnetic
far-field data. FFD retains phase and both spherical electric-field components,
so exported results can support coherent comparisons and later postprocessing
that cannot be performed from RCS magnitudes alone.

`src/FFDFormat.jl` implements the format independently of the Maxwell solver,
DG order, time stepping, MPI, and surface quadrature. The production
metallic-sphere driver supplies its completed near-to-far results to this codec.
The existing [metallic-sphere RCS workflow](rcs_metallic_sphere.md) describes
the surface-current and Fourier extraction in detail.

The [standalone comparison tools](../../tools/ffd/README.md) validate EMAGPIE,
Mie, and DiscoGMPI FFD data in Cartesian coordinates, including signed cuts,
explicit convention diagnostics, plots, and quantitative regression gates.

## Enabling output

Request both RCS extraction and FFD output:

```bash
mpiexec -n 2 julia --project=. \
  examples/distributed_metallic_sphere_scattering.jl \
  --enable-rcs --enable-ffd \
  --rcs-start-time=5.0 --final-time=12.0 \
  --rcs-theta-count=37 --rcs-phi-degrees=0
```

The final time and window start above are illustrative. Choose a mesh and a
window that resolve the scattering problem and exclude startup transients;
writing a valid FFD file does not establish physical convergence.

FFD output is disabled by default. `--disable-ffd` explicitly disables it.
`--enable-ffd` requires RCS extraction to be enabled. Output is written once,
by MPI rank zero, to:

```text
<run-directory>/diagnostics/far_field.ffd
```

The run summary reports this path. Run configuration and metadata record the
FFD setting and unit labels. RCS accumulation and MPI reduction are shared
with the CSV output; enabling FFD does not initiate a second reduction.

The unit-label controls are:

| Option | Default | Meaning |
| --- | --- | --- |
| `--ffd-freq-unit` | `Hz` | Unit of the stored numerical frequency |
| `--ffd-field-unit` | `m` | Unit of the unit-incident scattering amplitude |

The driver always writes angles in degrees with `angle_unit deg`. The controls
label the numerical units supplied to the solver; they do not rescale frequency,
geometry, material parameters, or fields. For example, setting
`--ffd-freq-unit=GHz` does not divide the computed frequency by `1e9`. Only use
that label if the simulation's numerical frequency is already measured in GHz.
The default `Hz` and `m` labels require a corresponding SI interpretation of
the solver inputs. Nondimensional runs need appropriate explicit labels and a
documented physical scaling before comparison with dimensional measurements.

The almond frontend provides a concrete example. In
`examples/distributed_pec_almond_scattering.jl`, `--frequency-hz=3.5e9` sets the
benchmark wavelength to `lambda = c0 / 3.5e9 = 0.085654988`, with
`c0 = 299792458` in physical SI units. The default normalized solver has
`epsilon = mu = 1` and wave speed one, so the exported frequency is
`f_solver = 1 / lambda = 11.674743331935321` cycles per solver time unit.
It is not the physical `3.5e9` Hz frequency. For this run, a label such as
`--ffd-freq-unit=inverse_solver_time` makes that distinction explicit. Setting
the label to `Hz` or `GHz` alone does not perform the dimensional conversion.
This integration exports the computed solver frequency and provides no
physical-frequency conversion option.

## Phasor and incident-field conventions

FFD uses the temporal convention

```math
E(x,t)=\operatorname{Re}\{\widehat E(x)\exp(+i\omega t)\},
```

and the outgoing radial factor `exp(-i*k*r)/r`. The current DiscoGMPI driver
instead samples a real incident wave

```math
A\,\hat p\cos(k\hat k\cdot x-\omega t+\psi)
```

and multiplies time samples by `exp(+i*omega*t)` during Fourier extraction.
Its internal phasor therefore uses `exp(-i*omega*t)`. The conversion to FFD
conjugates the completed internal far field.

The adapter also normalizes by the full complex incident amplitude. Let `s` be
the signed component of the incident polarization vector along the selected
FFD incident basis vector. For a representable pure V or H wave, `s` is `+1`
or `-1`, to the polarization tolerance. Then

```math
a_{\rm inc}=s A\exp(i\psi),\qquad
F_{\theta,\phi}
 =\operatorname{conj}\!\left(
   \frac{E^\infty_{\theta,\phi,\rm internal}}{a_{\rm inc}}
  \right).
```

The file stores `F_theta` and `F_phi` under the standard FFD column names
`Etheta_re`, `Etheta_im`, `Ephi_re`, and `Ephi_im`. Dividing by `abs(A)` alone
would preserve RCS magnitudes but lose the source sign and reference phase.
The named helper `canonical_ffd_field` performs the convention and complex
amplitude conversion; `ffd_incident_coordinates` determines the incident
coordinates and signed basis component. Both helpers are implemented in
`src/FarFieldConventions.jl`.

The specification's term *normalized far field* removes the radial decay.
In SI, removing `1/r` from an electric field measured in V/m yields a
far-field coefficient measured in V. DiscoGMPI additionally divides by the
incident electric-field amplitude measured in V/m, so its stored scattering
response has units of length. This is the reason for the SI label
`field_unit m`; an illustrative `V/m` label is not the interpretation used by
this integration. The stored convention and incident normalization are also
recorded as file-header metadata.

For the exported unit-incident response, total and component RCS are

```math
\sigma=4\pi\bigl(|F_\theta|^2+|F_\phi|^2\bigr),\qquad
\sigma_\theta=4\pi|F_\theta|^2,\qquad
\sigma_\phi=4\pi|F_\phi|^2.
```

With `field_unit m`, these quantities have units of square metres. Conjugation
and division by the complex incident amplitude preserve the RCS obtained by
dividing the internal field intensity by the incident intensity. A zero
incident amplitude has no normalized scattering response and is rejected.

## Coordinates and polarization

Both observation and incidence coordinates use the spherical basis

```math
\hat s(\theta,\phi)
 =(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta),
```

```math
\hat e_\theta
 =(\cos\theta\cos\phi,\cos\theta\sin\phi,-\sin\theta),\qquad
\hat e_\phi=(-\sin\phi,\cos\phi,0).
```

Observation angles describe `rhat = s(theta_obs, phi_obs)`. Incidence angles
describe the emitter direction, so the propagation direction is
`khat = -s(theta_inc, phi_inc)`. Equal incidence and observation angles
therefore describe monostatic backscattering.

Away from the incidence poles, the adapter accepts `V` for incident electric
polarization parallel or antiparallel to `e_theta_inc`, and `H` for polarization
parallel or antiparallel to `e_phi_inc`. It includes an antiparallel sign in
`a_inc` before normalization. FFD v1's V/H tokens cannot represent an arbitrary
slanted or elliptical incident polarization. FFD export rejects an
unrepresentable incident polarization with a specific error; CSV-only RCS
output remains available.

At `theta_inc = 0` or `180` degrees, the emitter direction does not determine
an azimuth. The adapter selects `V` and chooses `phi_inc` to align `e_theta_inc`
positively with the supplied polarization. Thus the production driver's
default wave, propagating along `+z` with electric polarization `+x`, is encoded
as:

```text
theta_inc 180 degrees
phi_inc   180 degrees
pol       V
```

The existing observation spherical basis is used directly, including its
azimuth-dependent transverse basis at the observation poles. The exporter
preserves the solver's spatial origin; it does not recenter the scatterer or
remove an origin-dependent phase.

## Stored blocks and syntax

An FFD file contains one global `[FILE_HEADER]` and one or more `[BLOCK]`
sections. The mandatory header fields are `version`, `freq_unit`,
`angle_unit`, and `field_unit`. Optional metadata includes `solver`, `problem`,
and an ISO-8601 `date`. The production driver writes the date in UTC with a
`Z` suffix. The codec validates the extended datetime profile
`YYYY-MM-DDTHH:MM:SS[.fraction][Z|+HH:MM|-HH:MM]`, including calendar dates;
fractions and timezone suffixes are optional. It accepts local datetimes but
does not accept leap-second `:60` or the `24:00` extension. Additional
file-header fields are preserved.
File-header values may contain spaces, for example `solver DiscoGMPI 0.1.0`.
Blocks cannot override the global metadata.

Explicit blocks store all six coordinates and both complex field components
on every row. Gridded blocks declare selected coordinates as axes and omit
those coordinates from data rows. Length-one axes encode fixed coordinates.
The codec supports fully explicit blocks, partly gridded blocks, fully gridded
blocks, and files containing multiple blocks.

The production driver writes one fully gridded block:

```text
storage gridded
sweep_order freq pol theta_inc phi_inc phi_obs theta_obs
```

Frequency, polarization, and both incidence angles have length-one axes. The
configured observation azimuths form the outer observation axis and polar
angles form the inner axis. Hence `theta_obs` changes fastest, and
`n_points = length(phi_obs) * length(theta_obs)`. The data header is:

```text
Etheta_re Etheta_im Ephi_re Ephi_im
```

For general gridded blocks, the first declared axis is the outermost loop.
There may be `M > 1` explicit rows per axis cell, where
`M = n_points / prod(axis_lengths)`. Their within-cell order is preserved.
Every coordinate must occur exactly once, either as an axis or a data column.
Axis declarations must match `sweep_order`, including declaration order.

The reader validates section structure, coordinate coverage, field-column
order, axis lengths, and row counts. The writer emits physical floating-point
values in deterministic scientific notation with 16 digits after the decimal
point. `version` is a version token, `n_points` is a decimal integer, and `V`
and `H` are case-sensitive polarization tokens. Blank lines and lines whose
first non-whitespace character is `#` are ignored. Trailing inline comments
are not accepted. Delimiters must occupy their own lines. Unknown block-header
fields are rejected for v1.0.

## Julia API and large files

The public types are `FFDHeader`, `FFDPolarization` (`FFD_V`, `FFD_H`),
`FFDSample`, `FFDAxis`, `AbstractFFDBlock`, `FFDExplicitBlock`,
`FFDGriddedBlock`, and `FFDFile`.
For example, a small explicit file can be written as:

```julia
using DiscoGMPI

header = FFDHeader(; freq_unit="Hz", angle_unit="deg", field_unit="m",
                   solver="DiscoGMPI", problem="example")
samples = [FFDSample(1.0e9, FFD_V, 180.0, 180.0, 90.0, 0.0,
                     1.0e-3 + 2.0e-4im, -3.0e-4 + 1.0e-4im)]
write_ffd("example.ffd", header, [FFDExplicitBlock(samples)])
```

`write_ffd(path, header, blocks)` accepts iterable rows; for iterators without
a known length, supply `n_points` when constructing a block. The path writer
uses a temporary sibling file, validates actual row counts, and replaces the
destination only after successful completion. On failure it removes the
temporary file and preserves an existing destination.

`read_ffd(path; lazy=true)` validates the complete file with a bounded-memory
scan and returns file-backed blocks. The initial validation still reads the
entire input, but it does not retain all field samples. The source file must
remain available and unmodified while these blocks are used.
`read_ffd(path; lazy=false)` is the convenience mode that loads stored rows into
memory. File-backed iteration reads bounded chunks and closes each file handle
before yielding its samples, so stopping iteration early does not leave an open
file handle. `validate_ffd` provides explicit validation. Use `each_ffd_row` for
stored records and `each_ffd_point` or `expand_block` for canonical samples:

```julia
file = read_ffd("example.ffd")
for block in file.blocks
    for sample in each_ffd_point(block)
        sigma = 4pi * (abs2(sample.Etheta) + abs2(sample.Ephi))
        # Consume or aggregate the result without collecting all samples.
    end
end
```

Expanded point iteration reconstructs gridded coordinates lazily. Calling
`collect` explicitly materializes the iterator and is appropriate only when
the dataset fits in memory.

## CSV diagnostics and solver coverage

`diagnostics/rcs.csv` remains the auxiliary diagnostic output. It includes
field magnitudes, RCS and dB values, sample count, window sum, and Fourier
interval information that FFD does not encode. Its complex field columns use
the historical internal solver convention and incident amplitude; the FFD
fields use the normalized convention described above. Compare their RCS
values, or perform the documented conversion before comparing complex fields.
The CSV RCS now divides each complex field component by the nonzero incident
amplitude before squaring; it does not clip the incident intensity to machine
epsilon. This preserves agreement with the unit-incident FFD result even for
very small representable source amplitudes. It also avoids underflow from
forming `abs2(A)` before division.

The codec has no dependency on DG order and accepts identical canonical data
from any solver. The integrated production scattering driver currently uses
equal-order DG. Actual mixed-order near-to-far extraction is a follow-up:
magnetic coefficients in `P_(p+1)` must be evaluated with `MixedFace.H` at the
common face quadrature points and integrated with physical face weights.
Codec independence alone does not validate those surface traces.

## Phase-sensitive postprocessing

The v1.0 specification outlines aggregation, filtering, addition, subtraction,
scaling, rotation, and translation. This implementation provides their data
interchange foundation; it does not implement these postprocessing operators.

The translation phase needs a precise interpretation. Translating a fixed
radiating current distribution by `d`, while preserving its complex current
amplitudes, produces in the FFD convention

```math
F_d(\hat r)=F_0(\hat r)\exp(+ik\hat r\cdot d).
```

Moving a passive scatterer under the same globally referenced plane wave
also changes the incident phase at the scatterer. In a translation-invariant
homogeneous medium, its unit-incident scattering response instead obeys

```math
F_d(\hat r,\hat k)
 =F_0(\hat r,\hat k)\exp\!\bigl(+ik(\hat r-\hat k)\cdot d\bigr).
```

These are different physical operations. Coherent comparisons or future
addition and translation operators must use compatible spatial origins,
incident phase references, spherical bases, units, and coordinate samples.
