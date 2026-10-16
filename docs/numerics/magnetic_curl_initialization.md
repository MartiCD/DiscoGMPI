# Elementwise Divergence-Free Electric and Magnetic Initialization

Both periodic Poisson-bracket drivers accept
`--magnetic-initialization=interpolate|curl-potential`. Interpolation is the
unchanged default. This is an initialization option, not divergence cleaning.

They also accept `--electric-initialization=interpolate|curl-potential`,
independently of the magnetic option and with the same unchanged default.
Enable either or both. The electric construction is `D=curl(A_E,p+1)` followed
by `E=D/epsilon_K`, using the same rectangular reference derivatives described
below. It changes only electric owned coefficients, not magnetic ones.

For the periodic analytical wave, the electric potential is
`A_E=(0,H0*cos(k*(x-x_origin)-omega*t)/omega,0)`. Its curl is
`(0,0,-k*H0*sin(k*(x-x_origin)-omega*t)/omega)=epsilon*E`.
The potential is interpolated at degree `p+1` before differentiating;
the analytical curl is **not** interpolated independently at degree `p`.

## Construction

On each affine tetrahedron, sample the Cartesian magnetic vector potential at
degree `p+1` nodes. The reference-element workspace caches the resulting nodal
derivative evaluation operators at the solver's actual degree `p` nodes:

\[
R_\alpha = V_{p+1,\alpha}(r_p,s_p,t_p)V_{p+1}^{-1},
\qquad G_j=\sum_{\alpha=1}^3(J^{-1})_{\alpha j}R_\alpha.
\]

Construct `B = (Gy*Az-Gz*Ay, Gz*Ax-Gx*Az, Gx*Ay-Gy*Ax)`, then `H = B/mu_K`.
There is no preliminary degree reduction of the potential, no independently
interpolated analytical curl, and no change to the electric field. Affine
mapping and differentiation imply `B` has degree at most `p`, and its broken
divergence vanishes in exact arithmetic. Reference ordering and signed inverse
Jacobians come from the existing solver infrastructure.

For the periodic benchmark, use
`A = (0,0,mu*H0*cos(k*(x-x_origin)-omega*t)/k)`.
Its curl has y component `mu*H0*sin(k*(x-x_origin)-omega*t)`, as required.
This generally changes magnetic approximation errors and initial magnetic
energy relative to direct interpolation; the analytical reference is unchanged.

## API and MPI

`src/MaxwellInitialization.jl` provides `MagneticCurlInitializationWorkspace`,
`initialize_magnetic_from_potential!`, and `magnetic_initialization_diagnostics`.
The corresponding electric APIs are `ElectricCurlInitializationWorkspace`,
`initialize_electric_from_potential!`, and `electric_initialization_diagnostics`.
Both workspace names alias a common implementation and can reuse the same
reference cache for successive electric and magnetic initializations.
The callback signature is `potential(x,y,z,time)`, returning three finite
Cartesian components. Use `MaxwellElementMaterials` for positive finite scalar
permeability per local element; `homogeneous_maxwell_materials` is a convenience
for homogeneous media. The serial API accepts either a DG discretization or
the affine mesh/reference/mapping objects directly (including degree zero).
The production CLI retains its existing supported polynomial degrees.

The workspace belongs to one target `ReferenceTet` object; its `nodal_set`
keyword selects the degree `p+1` potential nodes. The convergence driver passes
its selected nodal family. Reuse workspaces for the same reference; do not share
their scratch buffers across threads or mutate their reference operators.

The distributed overload initializes owned elements, propagates validation
errors collectively, zeros ghost magnetic storage, and synchronizes ordinary
face traces. Passing `periodic=...` also synchronizes periodic trace buffers.
Ghost interiors are **not** complete volume fields. Diagnostics integrate only
owned elements and reduce globally. The serial diagnostic returns `l2_squared`,
`signed`, and `max_nodal`; the distributed version returns the global
`broken_divergence_l2`, `signed_magnetic_charge`, and `max_nodal_divergence`.
Electric diagnostics have the same local return fields and use
`signed_electric_charge` instead of `signed_magnetic_charge` in the distributed
result. They differentiate `epsilon_K*E` using the solver's physical derivatives.
The electric initializer uses finite, positive, elementwise scalar permittivity;
curved geometry, tensors and within-element variation remain unsupported.

## Guarantees and Limitations

- Only affine four-vertex tetrahedral geometry is supported; curved geometry
  needs a separate compatible construction.
- Spatially varying within-element permeability and tensor permeability are
  unsupported. Material callbacks are not silently converted to constants.
- There is no guarantee of normal continuity across faces, distributional
  divergence freedom, divergence preservation during evolution, or energy
  conservation by changing the initializer.
- The broken divergence L2 norm and signed integral must be inspected
  separately: cancellation can hide large local errors.
- Tolerances depend on floating-point precision, potential/field magnitude,
  polynomial basis conditioning, and physical derivative norms. Tests use
  explicit conservative two-differentiation error budgets, not an absolute
  `1e-16` criterion. Existing high-degree modal-basis warnings are not suppressed.

## Running and Restarting

For a short periodic run initializing both fields:

```bash
mpiexec -n 2 julia --project=. \
  examples/distributed_periodic_poisson_bracket_maxwell.jl \
  --mesh=examples/meshes/periodic_box_structured_nx4_ny2_nz2.vtk \
  --partitions=2 --order=2 --esprk-order=3 --flux=centered \
  --magnetic-initialization=curl-potential \
  --electric-initialization=curl-potential \
  --final-time=0.01 --cfl=0.05 \
  --output-dir=output/electric_magnetic_curl_smoke
```

The same electric flag is supported by the convergence driver below. Either
curl initializer can change initial approximation error and energy relative to
interpolation. Neither is a divergence-cleaning or time-evolution modification.

```bash
mpiexec -n 2 julia --project=. \
  examples/convergence_distributed_periodic_poisson_bracket_maxwell.jl \
  --mesh-family=structured --nx-targets=4,8 --orders=2 \
  --nodal-set=warp-blend --magnetic-initialization=curl-potential \
  --final-time=0.001 --rate-policy=report \
  --output=output/magnetic_curl_initialization_smoke/convergence.csv
```

Use a launcher matching Julia's MPI library. The short command is a workflow
smoke test, not an expected-rate validation. Fresh curl-potential runs print
initial broken divergence and charge separately. The production driver also
writes its usual time-zero quadrature diagnostics and records the mode in run
metadata and checkpoints. Restart never reinitializes the checkpoint fields.
Specify the checkpoint's mode when restarting; a mismatch is rejected. Legacy
checkpoints without the field are interpreted as `interpolate`.
This applies independently to `electric_initialization`: both modes are stored
in metadata/checkpoints, checked on restart, and never reapplied to restored
fields. Electric curl initialization cannot be requested when resuming a legacy
checkpoint lacking that mode.

The additive CSV field `magnetic_initialization` separates convergence series
in the Julia rate calculation and Python postprocessor. Old CSVs default to
`interpolate`; modes are never mixed into one convergence sequence.
The additive `electric_initialization` column likewise separates series in both
Julia and Python; a missing column in older CSVs defaults to interpolation.

Focused tests are `test/test_magnetic_curl_initialization.jl` and
`test/test_mpi_magnetic_curl_initialization.jl` (run with 1, 2, and 4 ranks).
They cover nonzero polynomial curls, mapped/permuted tetrahedra, both nodal
families, material scaling, owned/global-ID agreement, halo and periodic traces,
and collective input failures. The existing magnetic-charge null-invariant
regression tests exercise the unchanged evolution operator separately.
`test/test_electric_curl_initialization.jl` adds electric polynomial accuracy,
nonpolynomial/scaled potentials, field independence, material validation and
CLI/restart tests. The MPI curl-initialization test now exercises both fields.
