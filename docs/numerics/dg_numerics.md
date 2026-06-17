# Numerical Implementation Notes for DiscoGMPI

This note documents the numerical building blocks used by the DiscoGMPI Maxwell solvers and convergence drivers. The main implementation files are `src/solver/kernels/reference_tet.jl`, `src/solver/kernels/dg_discretization.jl`, `src/solver/kernels/geometry.jl`, `src/solver/kernels/trace_maps.jl`, `src/solver/kernels/flux_faces.jl`, `src/solver/kernels/maxwell_fluxes.jl`, `src/solver/kernels/time_integration.jl`, `src/DistributedDG.jl`, and `src/MaxwellDiagnostics.jl`.

## Polynomial approximation space

Each scalar component of `E` and `H` is approximated independently on every tetrahedron by a polynomial in

```math
\mathbb P^N(K)=\{p(x,y,z): \deg p\le N\}.
```

The number of volume nodes is

```math
N_p = \frac{(N+1)(N+2)(N+3)}{6},
```

and the number of nodes on a triangular face is

```math
N_{fp}=\frac{(N+1)(N+2)}{2}.
```

The Maxwell state therefore stores six arrays of size `Np x K`, where `K` is the number of rank-local tetrahedra including ghosts in distributed runs.

## Reference tetrahedron and nodal sets

The reference tetrahedron has vertices

```math
(-1,-1,-1),\quad (1,-1,-1),\quad (-1,1,-1),\quad (-1,-1,1).
```

`build_reference_tet(N)` constructs equispaced nodal points on this reference tetrahedron. The current nodal set is simple and deterministic; it is not an optimized warp-blend or Fekete set. This matters at high order because interpolation conditioning is affected by the nodal distribution.

The code builds an orthonormal modal tetrahedral basis, evaluates it at the nodal points, and forms a Vandermonde matrix

```math
V_{ij}=\phi_j(r_i,s_i,t_i).
```

The inverse Vandermonde maps nodal values to modal coefficients. Nodal basis functions are represented as modal combinations through `invV`.

## Quadrature and cubature rules

The reference mass matrix is constructed from the orthonormal modal basis and the inverse Vandermonde. Independent diagnostic integration uses Jaskowiec-Sukumar tetrahedral cubature rules through `get_JaskowiecSukumar_cubature(order)`.

The production convergence and diagnostic paths commonly use a cubature order such as

```math
q = \max(2,2N+4),
```

which is intentionally higher than the polynomial degree needed for basic mass-matrix operations. This reduces the chance that field, energy-density, chirality, and momentum diagnostics are dominated by quadrature error.

## Mass matrices

On the reference tetrahedron, the nodal mass matrix is

```math
M = V^{-T} V^{-1},
```

because the modal basis is orthonormal. For an affine physical element `K`, integrals are scaled by the absolute Jacobian determinant:

```math
M_K = |J_K| M.
```

The energy diagnostic uses this mass matrix directly:

```math
\mathcal H_h
=\frac12\sum_K |J_K|\left[
\epsilon \sum_a E_{a,K}^TME_{a,K}
+\mu \sum_a H_{a,K}^TMH_{a,K}\right].
```

## Derivative and stiffness matrices

Reference derivative matrices are formed by differentiating the modal basis and multiplying by the inverse Vandermonde:

```math
D_r = V_r V^{-1},
\qquad
D_s = V_s V^{-1},
\qquad
D_t = V_t V^{-1}.
```

The reference weak stiffness matrices are

```math
S_r = M D_r,
\qquad
S_s = M D_s,
\qquad
S_t = M D_t.
```

For every physical tetrahedron, metric terms transform these into physical derivative matrices `Dx`, `Dy`, `Dz` and weak operators `Sx`, `Sy`, `Sz`. The Hesthaven-Warburton path uses strong derivative operators for the volume curl. The Poisson-bracket path uses weak operators for one partition and transposed weak operators for the complementary partition.

## Face operators

The reference face operators identify the `Nfp` nodes on each of the four triangular faces and build face mass/lifting operators. A numerical face residual is first evaluated on face nodes, multiplied by the physical face area, projected through the face mass matrix, and lifted into the element volume.

Conceptually, a face residual `f` contributes

```math
M_K^{-1} M_{f,K} f
```

to the nodal right-hand side. The implementation folds the reference and physical scaling into the lifting helper routines.

## Mesh topology and connectivity

A tetrahedral mesh is represented by point coordinates, tetrahedron connectivity, boundary triangles, and optional cell data such as `boundary_id` and `material_id`. DG topology is constructed by sorting the three vertex IDs of each face:

- a face that appears once is a physical boundary face;
- a face that appears twice is an interior face;
- a face that appears more than twice is rejected as non-manifold.

For every interior face the trace map stores the minus element, plus element, local face IDs, face-node lists, the plus-to-minus nodal permutation, and the minus-side outward normal. Correct face-node permutation is essential for distributed runs because two ranks may enumerate the same physical face with different local node orderings.

## Distributed ownership and ghost elements

In MPI runs, every rank owns a subset of global tetrahedra and stores a one-face halo of ghost tetrahedra. Local volume operations are computed for owned elements. Ghost fields are updated by exchanging face traces with neighboring ranks before surface flux evaluation.

Global diagnostics use owned elements only and then apply MPI reductions. This prevents double counting of ghost elements.

## Periodic face maps

Periodic boundaries are represented as directed face pairs. The pairing logic matches boundary face centroids after applying the expected translation vector, then constructs node permutations that align the plus face with the minus face. Periodic tags should be handled by the periodic exchange object and should remain `MaxwellBC_None` in the ordinary physical boundary registry.

For the default rectangular periodic box, the boundary tags are paired as

```text
1 <-> 2   x-min / x-max
3 <-> 4   y-min / y-max
5 <-> 6   z-min / z-max
```

The periodic convergence driver can use structured or unstructured periodic mesh families.

## Metric terms

For each affine tetrahedron, DiscoGMPI computes the Jacobian matrix of the map from reference coordinates `(r,s,t)` to physical coordinates `(x,y,z)`. Physical derivatives follow

```math
\nabla_x u = J_K^{-T}\nabla_{r,s,t} u.
```

The determinant `|J_K|` scales volume integrals. Face geometry supplies unit normals and face areas for surface terms.

The production convergence CSV now records mesh quality metrics computed from owned elements and reduced globally:

- minimum, maximum, total, and ratio of element volumes;
- minimum, maximum, and ratio of element size `h` based on maximum edge length;
- minimum, maximum, and ratio of all edge lengths;
- minimum and average tetrahedron mean-ratio quality.

The mean-ratio quality used by the diagnostics is

```math
q_K = \frac{12(3V_K)^{2/3}}{\sum_{e\in K} |e|^2},
```

where the denominator sums the squared lengths of the six tetrahedron edges. Larger values indicate more regular tetrahedra; degraded elements lower the minimum quality.

## Flux evaluation

Flux evaluation is trace based:

1. gather the minus trace from the local element face;
2. gather the plus trace from the neighboring element, periodic partner, ghost element, or reflected boundary state;
3. align plus-side nodes with the minus-side face ordering;
4. evaluate centered, upwind, or Poisson-bracket face corrections;
5. lift the resulting residual into the element volume.

The centered Hesthaven-Warburton correction is

```math
F_E^c = \frac12 n\times(H^+ - H^-),
\qquad
F_H^c = -\frac12 n\times(E^+ - E^-).
```

The upwind correction adds

```math
-\frac{Y}{2}n\times(n\times(E^+-E^-))
```

to the electric correction and

```math
-\frac{Z}{2}n\times(n\times(H^+-H^-))
```

to the magnetic correction.

The Poisson-bracket face operator uses the implementation-specific skew form

```math
F_E^{PB}= -\frac{1}{2\epsilon}n\times(H^- - H^+),
\qquad
F_H^{PB}= -\frac{1}{2\mu}n\times(E^- + E^+).
```

Only the central Poisson-bracket flux is supported.

## Boundary trace construction

PEC, PMC, and absorbing boundaries are handled by constructing exterior traces. The production Poisson-bracket drivers currently focus on PEC, PMC, and periodic validation. PEC and PMC reflections are

```math
E^+_{PEC}=-E^-+2(n\cdot E^-)n,
\qquad
H^+_{PEC}=H^-,
```

```math
E^+_{PMC}=E^-,
\qquad
H^+_{PMC}=-H^-+2(n\cdot H^-)n.
```

Periodic boundaries do not use a reflected state; they use a true trace from the paired periodic face.

## Time integration

DiscoGMPI contains conventional explicit Runge-Kutta methods and explicit partitioned symplectic Runge-Kutta methods.

For a semidiscrete ODE

```math
\dot U = R(U),
```

an explicit RK method uses

```math
U^{(i)} = U^n + \Delta t \sum_{j<i} a_{ij}K_j,
\qquad
K_i = R(U^{(i)}),
\qquad
U^{n+1}=U^n+\Delta t\sum_i b_iK_i.
```

For Poisson-bracket Maxwell runs, the recommended method is ESPRK with magnetic-first partitioning. The electric and magnetic partitions are updated in a staged sequence that is compatible with the partitioned Hamiltonian structure.

The convergence drivers use `--esprk-order`, usually set to `N+1` for DG order `N`, so temporal error does not dominate the spatial convergence study.

## CFL condition

The basic timestep estimate is

```math
\Delta t
= \mathrm{CFL}\,\frac{\min_K h_K}{(2N+1)c},
```

where `c` is the material wave speed. The user-selected `--cfl` multiplies this estimate. The final timestep is adjusted so that the integration lands exactly on the requested final time.

This is an engineering stability estimate, not a proof. Highly skewed meshes, high order, upwind damping, periodic pairings, and material variation can require additional timestep sensitivity checks.

## Mesh-size definitions

Several mesh-size definitions appear in the code and CSV files:

- `target_h`: requested Gmsh size, usually `Lx/NxTarget` for generated unstructured families;
- `characteristic_h`: volume-based mesh scale, usually `(volume_total / number_of_elements)^(1/3)`;
- `h_min`, `h_max`: quality-diagnostic element sizes based on maximum tetrahedron edge length;
- CFL `h_K`: the element insphere-like estimate `3V_K/A_K`, where `A_K` is total surface area.

Convergence rates should use the same `h` definition throughout a study. The production convergence drivers compute rates from the measured characteristic mesh size because generated unstructured meshes are not exactly nested.

## Structured and unstructured convergence families

The cavity convergence driver supports:

- structured unit-cube meshes split into six tetrahedra per cube cell;
- unstructured unit-cube Gmsh families generated from `examples/meshes/unit_cube_unstructured.geo`.

The periodic convergence driver supports:

- structured `[0,2] x [0,1] x [0,1]` periodic box meshes;
- unstructured periodic Gmsh families generated from `examples/meshes/periodic_box_unstructured.geo`.

For unstructured studies, rank zero generates or loads the mesh family, partitions it, and distributes rank-local data. The CSV stores `mesh_family` so post-processing can separate structured and unstructured results.

## Expected convergence rates

The production convergence checks are explicit rate checks, not monotonicity checks. For polynomial degree `N`, the current regression targets are:

- aggregate electric field error: expected rate `N+1`;
- aggregate magnetic field error: expected rate `N`;
- total field error: reported as an aggregate diagnostic;
- component errors: recorded for `Ex`, `Ey`, `Ez`, `Hx`, `Hy`, `Hz`, with component rates suppressed when the coarse error is at roundoff level.

The pass/fail logic uses a tolerance around the expected aggregate rates. This is intentionally stricter than a monotone-decrease check. On general unstructured periodic meshes, exact-zero transverse components can measure polarization leakage and may converge differently from active wave components; the CSV exposes those component errors so this behavior is visible.

## Output and diagnostics

The convergence CSV files include:

- DG order and ESPRK order;
- mesh family and mesh level;
- element count and owned-element distribution;
- target and measured mesh sizes where applicable;
- mesh quality metrics;
- aggregate `L2` errors and rates;
- per-component `L2` errors and rates;
- energy, charge, momentum, angular momentum, and chirality diagnostics where available;
- elapsed runtime.

The production transient drivers additionally write time-series ParaView output, quadrature diagnostics, run metadata, partition metadata, integration-point samples, and checkpoints.
