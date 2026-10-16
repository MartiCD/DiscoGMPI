# Integrated electric and magnetic field diagnostics

The quadrature diagnostics now append six signed volume integrals:

\[
 I_{D,i}(t)=\int_\Omega \varepsilon E_{h,i}(x,t)\,dx,
 \qquad I_{B,i}(t)=\int_\Omega \mu H_{h,i}(x,t)\,dx,
 \qquad i=x,y,z.
\]

These are the fields D and B tested against constant Cartesian basis vectors.
They are **not electric or magnetic charges**. The existing `electric_charge`
and `magnetic_charge` columns still measure signed integrals of broken
divergence. Changing the diagnostic does not change the solution or remove
drift from those existing quantities.

## Implementation and interpretation

`src/MaxwellDiagnostics.jl` interpolates the field to the existing quadrature
points, multiplies by scalar epsilon/mu and the physical quadrature weight,
accumulates owned elements only, then applies the existing MPI sum reduction.
The shared diagnostic currently accepts spatially constant scalar materials.
No numerical derivative is used for these six integrals. No volume
normalization is applied: divide by the mesh volume to obtain mean D/B.

In a source-free periodic Maxwell problem, constant-vector tests suggest
conservation through cancellation of volume and face contributions. The
focused test evaluates these functionals on the actual complete periodic
RHS, for centered and alternating fluxes. This is separate from testing
broken divergence or any claim about a full discrete Poisson structure.
Floating-point evaluation and updates can still produce drift.

The periodic sine benchmark has zero analytical values for all six integrals.
Measure conservation using signed `I(t)-I(0)` and its absolute value, not
relative error divided by the nearly zero initial integral. A nonzero initial
discrete value can result from interpolation and quadrature, independently
of subsequent conservation.

## CSV output

New runs write these columns at the end of
`diagnostics/quadrature_diagnostics.csv`, using the `--energy-every` cadence:

| Columns | Quantity |
|---|---|
| `integrated_d_x`, `integrated_d_y`, `integrated_d_z` | Integral of epsilon times each E component |
| `integrated_b_x`, `integrated_b_y`, `integrated_b_z` | Integral of mu times each H component |

The quantities are also available on `MaxwellQuadratureDiagnostics`. Legacy
aggregate vectors passed to `maxwell_quadrature_diagnostics_from_sums` without
these measurements yield NaN for the six new properties, not fictitious zeros.
Old CSV files cannot recover these integrals from charge or energy alone.
Use a fresh output directory rather than appending new-schema rows to an old
CSV history.

## Measured short-run results (2026-09-24)

The table below concerns the original integrated-field extension. The later
directional-charge extension retains all of these diagnostics.

The production command below was executed with two ranks, p=2, ESPRK3,
centered flux, magnetic curl-potential initialization, CFL=0.05 and T=10.
The actual constant time step was 0.0022476961114857273 (4,449 steps).
Unlike the separate audit driver, this production driver adjusts dt to
reach the final time with an integer number of equal steps.
There are 46 diagnostic samples, including both endpoints, in
`output/integrated_fields_smoke_20260924/diagnostics/quadrature_diagnostics.csv`.

| Quantity | Signed final change from t=0 | Maximum absolute sampled change |
|---|---:|---:|
| Integrated D_x | -4.10e-16 | 1.61e-14 |
| Integrated D_y | 2.83e-16 | 5.72e-16 |
| Integrated D_z | 6.94e-16 | 2.00e-15 |
| Integrated B_x | 1.20e-15 | 1.63e-14 |
| Integrated B_y | -8.33e-17 | 6.94e-16 |
| Integrated B_z | 1.46e-15 | 1.54e-15 |
| Original electric charge | -3.98e-14 | 6.91e-14 |
| Original magnetic charge | -4.43e-13 | 4.43e-13 |

The integrated components were better preserved in this test, but this does
not remove or disprove drift in the distinct divergence-based charge.
These are sampled maxima, not bounds over every step. No T=1000 test was
executed for this change, and no exact-arithmetic conservation claim follows
from one short run.

The focused regression passed 62 assertions per process at each of 1, 2 and
4 ranks, covering both fluxes. On its dyadic test state, the largest absolute
integrated RHS component was 1.33e-15 across those runs, with RHS amplitude
times volume scales of approximately 125 (centered) and 143 (alternating).
`git diff --check` also passed.

## Periodic plane-wave command

Run from the repository root, with an MPI launcher matching MPI.jl:

```bash
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1
mpiexec -n 2 julia --project=. \
  examples/distributed_periodic_poisson_bracket_maxwell.jl \
  --mesh=examples/meshes/periodic_box_structured_nx4_ny2_nz2.vtk \
  --partitions=2 --order=2 --esprk-order=3 \
  --final-time=10.0 --cfl=0.05 --flux=centered \
  --magnetic-initialization=curl-potential \
  --energy-every=100 --paraview-every=1000000000 \
  --output-dir=output/integrated_fields_T10
```

No additional switch enables these diagnostics. To reproduce the long-time
setup, change to `--final-time=1000.0` and a new output directory. Initial and
final ParaView states are still written with the large snapshot interval.
Use `--flux=alternating` for a separate flux comparison.

## Publication plots

### Directional charge contributions

In addition to the integrated fields, new simulations record

\[
 Q_{E,i}=\sum_K\int_K\varepsilon\,\partial_i E_{h,i}\,dx,
 \qquad Q_{H,i}=\sum_K\int_K\mu\,\partial_i H_{h,i}\,dx,
 \qquad i=x,y,z.
\]

Columns are `electric_charge_x/y/z` and `magnetic_charge_x/y/z`.
The implementation reuses the actual physical derivative matrices and the
same quadrature points, material coefficients, owned-element selection and
MPI reduction as the original charges. These are signed, broken-element
volume integrals, without face terms or volume normalization. The current
shared quadrature path accepts constant scalar epsilon and mu.

Their sums reproduce the original charges up to floating-point summation
differences. The original total is still computed independently by the
existing divergence calculation; it is not overwritten by the new sum.
Individual directional contributions are not automatically invariants even
when their sum is conserved. Splitting the diagnostic does not enforce
conservation or change the evolution operator.

The separate integrated D/B columns remain unchanged. Old aggregate vectors
without directional measurements yield NaN for the new diagnostic properties.

### Commands and outputs

```bash
python3 examples/plot_integrated_fields_and_charge.py \
  output/integrated_fields_T10
```

A CSV path is also accepted. Outputs default to `<run>/plots/integrated_fields/`:

- `integrated_fields_charge_signed.pdf/png`: electric/magnetic columns;
  rows show the three integrated component drifts, raw charge and charge drift.
- `integrated_fields_charge_absolute.pdf/png`: logarithmic absolute component
  drifts and charge drifts, with power-of-ten ticks.
- `drift_summary.csv`: initial/final values, signed final drift, maximum
  sampled absolute drift and sample RMS drift for all eight quantities.
- `plot_metadata.json`: input hash, baseline time, versions and conventions.

When all six directional columns are present, the script also produces:

- `directional_charge_comparison.pdf/png`: a 3-row, 2-column panel (electric
  left, magnetic right), with directional values, signed drifts, and absolute
  drifts on power-of-ten logarithmic axes. The original total charge is
  overlaid in black on every row; these quantities have the same dimensions.
- `directional_charge_reconstruction.csv`: each directional sum, the
  original charge and their signed difference at each recorded time.
- Six additional rows in `drift_summary.csv` for directional contributions.

For old histories without any directional columns, the original panels are
still generated and a message explains why the extra panel is unavailable.
Partial directional schemas are rejected. The missing contributions cannot
be recovered from a saved total charge alone: rerun into a fresh directory.

The directional extension was validated with 182 assertions per process at
1, 2 and 4 MPI ranks (both fluxes), and five Python plotting tests. A new
two-rank T=10 run is in `output/directional_charge_smoke_20260924`. Its maximum
sampled absolute directional-sum-minus-total residuals were 6.10e-16 for E
and 1.13e-15 for H. All three plot panels were generated; the new directional
panel was visually inspected. The magnetic contributions partially cancel:
their maximum sampled drifts were 5.09e-13, 5.82e-13 and 8.44e-13 for x/y/z,
respectively. Splitting the charge did not eliminate the total-charge drift.

```bash
python3 examples/plot_integrated_fields_and_charge.py \
  output/directional_charge_smoke_20260924
```

Those plots already exist after validation; add `--overwrite` to regenerate
them, or choose a new `--output-dir`.

Drift is relative to the first recorded sample, even when it is not at t=0.
All samples are used without smoothing or detrending. Exact zeros are masked
only on logarithmic panels, never replaced by a numerical floor; all-zero
panels are explicitly labelled. Different dimensional quantities remain on
separate axes without ratios or near-zero relative normalization.

Use `--output-dir`, `--title`, `--dpi` and `--overwrite` as needed. The script
exports vector PDF with embedded TrueType fonts and 400-dpi PNG by default.
Existing generated files are protected unless overwrite is requested.

The script was run on `output/integrated_fields_smoke_20260924`; the generated
logarithmic panel was visually checked. Plotting regression tests cover
baseline subtraction, malformed data and all-zero panels:

```bash
python3 test/test_plot_integrated_fields_and_charge.py
```

## Focused regression command

The test checks constant and affine fields on a volume-two domain with
epsilon=2 and mu=3, NaN-poisoned ghosts to detect double counting, CSV column
alignment, and complete periodic RHS cancellation. It uses global-element
dependent dyadic coefficients and scale/precision-based tolerances.

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=. --compiled-modules=no -e '
using MPI
for n in (1,2,4)
    run(`$(MPI.mpiexec()) -n $n $(Base.julia_cmd()) --project=. --compiled-modules=no test/test_mpi_integrated_fields.jl`)
end'
```
