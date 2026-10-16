# Long-time magnetic-charge and divergence diagnostics

## Scope and conclusion

This report analyzes the periodic Maxwell run in
`output/magnetic_charge_longtime/curl_p2`, completed at final time T = 1000.
It distinguishes three properties that must not be conflated:

1. Elementwise divergence-free magnetic initialization.
2. Preservation of the signed, volume-integrated magnetic charge.
3. Preservation of elementwise magnetic divergence during DG evolution.

The curl-potential initialization succeeds: the initial broken-divergence
L2 norm is approximately 4.39e-15. Subsequent evolution generates substantial
local divergence, although its signed integral remains small. The final
global charge drift is approximately -3.90e-11 after 444,839 time steps.
Checkpoint RHS evaluations and a short replay support accumulated, biased
floating-point residuals as the explanation of this small global drift.
They do not establish its precise arithmetic origin.

The order-one local divergence and momentum changes are not roundoff effects.
No numerical method or flux was changed in the diagnostic investigation.

## Reproducible configuration

| Parameter | Value |
|---|---|
| Domain | [0,2] x [0,1] x [0,1] |
| Boundary conditions | Periodic in all three directions |
| Mesh | `examples/meshes/periodic_box_structured_nx4_ny2_nz2.vtk` |
| Global tetrahedra | 96 |
| MPI ranks | 2, with 48 owned elements per rank |
| Polynomial degree | p = 2, 10 nodes per element |
| Formulation and flux | Poisson bracket, centered |
| Permittivity and permeability | epsilon = mu = 1 |
| Time integrator | ESPRK order 3, H-first |
| Magnetic initialization | `curl-potential` |
| Wave parameters | k = omega = 2 pi, H0 = 1 |
| Final time and CFL | 1000 and 0.05 |
| Actual time step | 0.0022480043341523565 |
| Number of steps | 444,839 |
| Diagnostic cubature order | 8 |

The analytical traveling wave has wavelength and period equal to one.
The Cartesian mesh spacing in x is 0.5: two Cartesian intervals per
wavelength. This is not a count of tetrahedra intersected by an arbitrary ray.

Run from the repository root:

```bash
export OPENBLAS_NUM_THREADS=1
export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1

mpiexec -n 2 julia --project=. \
  examples/distributed_periodic_poisson_bracket_maxwell.jl \
  --mesh=examples/meshes/periodic_box_structured_nx4_ny2_nz2.vtk \
  --partitions=2 --order=2 --esprk-order=3 \
  --final-time=1000.0 --cfl=0.05 --flux=centered \
  --magnetic-initialization=curl-potential \
  --energy-every=100 --paraview-every=1000000000 \
  --checkpoint-every=10000 \
  --output-dir=output/magnetic_charge_longtime/curl_p2
```

Use a different output directory for a new run to retain the analyzed data.
The configuration above comes from the command and recorded run metadata;
the actual time step and step count are recorded execution values.

## Definitions and algebraic distinctions

For elementwise constant scalar permeability, the monitored charge is

$$
Q_{H,h}=\sum_{K\in\mathcal T_h}\int_K
\nabla\cdot(\mu_K\mathbf H_h)\,dx.
$$

This is a signed integral of the broken, elementwise divergence. It is not
an independently defined numerical normal-flux divergence. Its smallness
does not imply that the integrand is small.

The local constraint diagnostic instead measures

$$
\delta_H=\left(\sum_K\int_K
|\nabla\cdot(\mu_K\mathbf H_h)|^2\,dx\right)^{1/2}.
$$

The reported infinity diagnostic is a maximum over diagnostic quadrature
samples, not a certified continuous supremum. Charge drift is
Delta Q(t) = Q(t) - Q(0); it differs from error against the exact zero charge
by the small initial numerical charge.

Let the implemented magnetic equation be

$$
M_H\dot{\mathbf h}=C_{HE}\mathbf e,
\qquad Q_{H,h}=\mathbf q_H^T M_H\mathbf h.
$$

Global semidiscrete conservation requires

$$
\mathbf q_H^T C_{HE}=0.
$$

If D_b maps magnetic coefficients to the monitored broken divergence,
preservation of that full local constraint requires the stronger identity

$$
D_b M_H^{-1}C_{HE}=0.
$$

These conditions are different. Signed integration can annihilate a
nonzero divergence field. Here C_HE includes volume, numerical-trace,
lifted-face, periodic and MPI-interface contributions. Its action cannot
be replaced by the elementwise analytical curl when testing the identity.
The appropriate terminology is a discrete null invariant, not an asserted
discrete Casimir without a demonstrated Poisson structure.

## Initialization and subsequent local divergence

The initializer interpolates a magnetic vector potential in degree p+1,
evaluates its physical polynomial curl at degree-p solver nodes, and sets
H = curl(A)/mu. On affine tetrahedra the resulting polynomial satisfies
div(curl(A)) = 0 in exact arithmetic. It neither enforces normal continuity
across faces nor changes the DG evolution operator.

| Time | Magnetic broken-divergence L2 norm | Evidence |
|---:|---:|---|
| 0 | 4.3898e-15 | Production CSV |
| 0.002248 | 3.0836e-4 | One-step replay |
| 0.022480 | 3.0544e-2 | Ten-step replay |
| 0.224800 | 2.2756 | CSV and 100-step replay |
| 1000 | 2.0730 | Production CSV |

The maximum sampled magnetic divergence L2 norm over the run is 8.4508.
The initial electric divergence L2 norm is 1.3550e-12; it also grows to
order-one values. Thus the initial magnetic construction works, but the
evolved fields do not retain small broken divergence.

Evaluating the full magnetic RHS at saved states gives:

| Time | L2 norm of div(mu Hdot) | Signed charge derivative |
|---:|---:|---:|
| 0 | 1.5071e-10 | 5.9897e-15 |
| 22.4800 | 31.0955 | -9.8643e-14 |
| 449.6009 | 25.5465 | 5.0723e-14 |
| 1000 | 30.0348 | 9.5181e-14 |

The large local RHS divergence on evolved states directly demonstrates
that the broken-divergence identity does not hold there. This is not just
a plotting effect or accumulated evaluation noise. The very small initial
RHS divergence does not guarantee preservation along the evolving trajectory.

This finding alone does not establish an incorrect implementation of the
bracket weak formulation. It establishes that the monitored local constraint
is not preserved by this evolution. Any theoretical claim of local constraint
preservation must specify an operator and verify the corresponding identity,
including its skeleton terms.

## Global magnetic-charge drift

The production CSV contains 4,450 saved samples. Its initial and final
magnetic charges are approximately -1.5364e-16 and -3.9023e-11.
A least-squares fit of signed drift versus time gives slope -3.9020e-14
per unit physical time. The RMS departure from the fitted line is about
2.09e-14. The drift is therefore strongly systematic, not an observed
unbiased random walk.

A two-rank replay using the same time step and initial data gives:

| Replay measurement | Value |
|---|---:|
| Steps | 2,000 |
| Final time | 4.49601 |
| Charge drift | -1.6593e-13 |
| Drift divided by elapsed physical time | -3.6906e-14 |
| Mean sampled RHS charge derivative | -3.8336e-14 |

The sampled RHS mean is an arithmetic sample mean, not an exact integral
over every composition substep. Nevertheless, its agreement in sign and
scale with the observed slopes supports accumulation of small signed
residuals. The production homogeneous RHS and the material-aware breakdown
agree in this unit-material case at the reported precision.

Separating volume, interior-face and periodic-face signed contributions
at the audited states gives residuals of approximately 1e-16 to 1e-13;
physical boundary contributions are zero. Their reconstructed sum differs
from the assembled evaluation by approximately 1e-14. There is no order-one
global face imbalance in these measurements. They are not an exhaustive
pair-by-pair topology proof or a fresh rank-count comparison.

The evidence is consistent with floating-point residual accumulation,
including roundoff in stored operators and their repeated evaluation.
It does not uniquely distinguish coefficient construction, RHS arithmetic,
time-step updates and summation. A higher-precision comparison is required
for that attribution. In particular, a small absolute residual should not
be called machine precision without accounting for operator and field scales.

## Energy, momentum and chirality

The initial energy is 1.04120576634. Its maximum relative deviation from
that initial value is approximately 4.17e-8. The analytical total energy
is one, so the approximately 0.0412 error offset is already present in
the initial approximation. Conservation error and analytical-solution
error must be reported separately.

The numerical x-momentum starts at 1.03981, reaches approximately -0.99802,
and ends at -0.91113. These order-one changes are not roundoff. The optical
chirality ranges approximately from -1.27957 to 1.23999 despite an initial
value near zero. Energy preservation alone does not establish preservation
of either diagnostic. Likewise, an unchanged vector norm would not establish
unchanged momentum components.

Dispersion and interference of discrete modes are possible explanations
of the oscillatory momentum pattern, but no spectral attribution has been
established by this audit. Changing magnetic initialization also changes the
discrete initial state, so comparisons against interpolation must account for
that difference. Angular momentum on a periodic cell additionally requires
care with position weighting and boundary fluxes; it is not automatically
a global invariant solely because the fields are periodic.

## Recommended verification sequence

1. Compare charge drift at several time steps with a fixed spatial mesh.
   Track both physical time and number of updates; smaller CFL is not an
   automatic cure for roundoff accumulation.
2. Repeat the small algebraic charge test with higher-precision operator
   construction and arithmetic where supported. Normalize residuals by
   appropriate operator and field scales.
3. Test the derivative of the implemented momentum functional using the
   actual semidiscrete RHS. Distinguish a spatial identity failure from
   time-integration error before changing the numerical method.
4. Compare interpolation and curl-potential initialization on the same mesh,
   then vary h and p. Report initial approximation errors separately.
5. Retain signed charge, charge drift and local divergence norms together.
   Do not remove drift by subtracting it from the data or impose a correction
   and describe the result as unmodified conservation.

## Evidence and implementation map

- Production data: `output/magnetic_charge_longtime/curl_p2/diagnostics/quadrature_diagnostics.csv`.
- Run configuration: `output/magnetic_charge_longtime/curl_p2/config/run_metadata.toml`.
- Saved states: `output/magnetic_charge_longtime/curl_p2/checkpoints/`.
- Driver: `examples/distributed_periodic_poisson_bracket_maxwell.jl`.
- Initialization: `src/MaxwellInitialization.jl`, especially
  `initialize_magnetic_from_potential!` and `magnetic_initialization_diagnostics`.
- Signed functional and RHS decomposition: `src/MaxwellDiagnostics.jl`,
  `distributed_strong_magnetic_charge`,
  `distributed_strong_magnetic_charge_derivative`, and
  `distributed_magnetic_charge_rhs_breakdown`.
- Periodic RHS and time stepping: `src/PhysicalCoverage.jl`,
  `maxwell_rhs_periodic!` and
  `distributed_periodic_partitioned_symplectic_rk_step!`.
- Volume kernels: `src/solver/kernels/maxwell.jl`.
- Existing focused test: `test/test_mpi_magnetic_charge_null_invariant.jl`.
- Initialization documentation: [magnetic_curl_initialization.md](magnetic_curl_initialization.md).

CSV statistics and checkpoint/replay measurements above were obtained during
the preceding diagnostic investigation. The replay used a temporary audit
script rather than a committed regression test; the measurements should not
be described as a newly established automated release gate. This report
records those findings without modifying the numerical method.
