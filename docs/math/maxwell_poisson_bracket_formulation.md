# Mathematical Formulation in DiscoGMPI

This note documents the mathematical model implemented by the DiscoGMPI Maxwell drivers and convergence studies. It is meant to be read together with the code in `src/MaxwellExactSolutions.jl`, `src/MaxwellDiagnostics.jl`, `src/MaxwellOutput.jl`, `src/DistributedDG.jl`, and `src/solver/kernels/maxwell_fluxes.jl`.

## Source-free Maxwell system

DiscoGMPI advances the source-free three-dimensional Maxwell equations on tetrahedral meshes. On each element `K`, with scalar positive permittivity `epsilon_K` and permeability `mu_K`, the physical fields satisfy

```math
\epsilon_K \partial_t E = \nabla \times H,
\qquad
\mu_K \partial_t H = -\nabla \times E.
```

The electric field is

```math
E=(E_x,E_y,E_z),
```

and the magnetic field is

```math
H=(H_x,H_y,H_z).
```

The material wave speed, impedance, and admittance are

```math
c_K = (\epsilon_K \mu_K)^{-1/2},
\qquad
Z_K = \sqrt{\mu_K/\epsilon_K},
\qquad
Y_K = Z_K^{-1}.
```

The divergence constraints are

```math
\nabla\cdot(\epsilon E)=0,
\qquad
\nabla\cdot(\mu H)=0.
```

They are monitored diagnostically through electric and magnetic charge. They are not advanced as separate equations.

## Role of the electric and magnetic fields

The Maxwell state is stored as six nodal DG fields. The electric variables `Ex`, `Ey`, `Ez` determine electric displacement through `D = epsilon E`. The magnetic variables `Hx`, `Hy`, `Hz` determine magnetic flux density through `B = mu H`. The source-free continuous equations exchange energy between these two partitions through curl coupling.

For the Poisson-bracket path, this partitioning is essential: the electric and magnetic variables are updated as complementary Hamiltonian variables. In production Poisson-bracket runs the explicit partitioned symplectic Runge-Kutta step advances the magnetic partition first, then the electric partition, at each stage.

## Hamiltonian structure

For homogeneous source-free materials, the continuous electromagnetic Hamiltonian is the total energy

```math
\mathcal H(E,H)
= \frac12 \int_\Omega
\left(\epsilon |E|^2 + \mu |H|^2\right)\,dx.
```

The variational derivatives are

```math
\frac{\delta \mathcal H}{\delta E} = \epsilon E,
\qquad
\frac{\delta \mathcal H}{\delta H} = \mu H.
```

Maxwell's equations can be written as a skew Hamiltonian system

```math
\partial_t
\begin{bmatrix} E \\ H \end{bmatrix}
=
\begin{bmatrix}
0 & \epsilon^{-1} \nabla\times \\
-\mu^{-1} \nabla\times & 0
\end{bmatrix}
\begin{bmatrix} E \\ H \end{bmatrix},
```

with the corresponding Poisson bracket

```math
\{F,G\}
=
\int_\Omega
\frac{\delta F}{\delta E}\cdot
\epsilon^{-1}\nabla\times
\frac{\delta G}{\delta H}
-
\frac{\delta F}{\delta H}\cdot
\mu^{-1}\nabla\times
\frac{\delta G}{\delta E}\,dx.
```

The implemented Poisson-bracket DG operator mirrors this skew pairing at the discrete level. Electric volume derivatives use weak derivative matrices, while magnetic volume derivatives use the corresponding transposed operators. The central surface terms close the operator across element faces. Upwind penalties are deliberately excluded from `PoissonBracketFormulation()` because they add dissipation and do not define the implemented skew partitioned operator.

## Weak DG formulation

Let the computational domain be partitioned into affine tetrahedra `K`. On each element, every scalar field component is represented in `P^N(K)`. For a test function `phi` in the same polynomial space, the elementwise weak form is obtained by multiplying the Maxwell equations by `phi`, integrating over `K`, and integrating curl terms by parts.

A representative electric equation is

```math
\int_K \epsilon \partial_t E \cdot \Phi\,dx
=
\int_K (\nabla\times H)\cdot \Phi\,dx
+ \int_{\partial K} \mathcal F_E(E^-,H^-,E^+,H^+;n)\cdot \Phi\,ds,
```

and the magnetic equation is

```math
\int_K \mu \partial_t H \cdot \Psi\,dx
=-\int_K (\nabla\times E)\cdot \Psi\,dx
+ \int_{\partial K} \mathcal F_H(E^-,H^-,E^+,H^+;n)\cdot \Psi\,ds.
```

Here `-` denotes the trace from the current element, `+` denotes the neighboring or boundary exterior trace, and `n` is the outward unit normal of the current element. DiscoGMPI stores face maps, node permutations, face normals, and face areas so that the plus-side trace is evaluated in the minus-side face-node ordering before the numerical flux is applied.

The semidiscrete form is a system of ODEs for the six nodal coefficient arrays:

```math
M_K \dot U_K = R_K(U),
```

where `M_K` is the physical mass matrix and `R_K` contains volume curl terms plus lifted face residuals.

## Numerical fluxes

### Centered flux

For the Hesthaven-Warburton strong-form path, DiscoGMPI uses the jumps

```math
[E] = E^+ - E^- ,
\qquad
[H] = H^+ - H^- .
```

The centered Maxwell correction is

```math
\mathcal F_E^c = \frac12 n\times [H],
\qquad
\mathcal F_H^c = -\frac12 n\times [E].
```

Centered fluxes are nondissipative in compatible settings. They are the natural choice for energy-conserving or energy-compatible studies, but they provide no jump damping.

### Alternating flux

An alternating flux selects one trace for one partition and the opposite trace for the complementary partition. In Maxwell language, one may take the magnetic trace from the plus side in the electric equation and the electric trace from the minus side in the magnetic equation, or the reverse. Such a choice is useful in Hamiltonian DG methods because the two one-sided choices can preserve a discrete skew pairing when applied consistently.

In DiscoGMPI, the production Poisson-bracket surface operator is the energy-compatible central/skew operator implemented in `maxwell_poisson_bracket_surface_flux_values`. It is not exposed as a separate `alternating` CLI option. The term "alternating" should therefore be understood as the broader DG concept of complementary one-sided traces, not as an additional current driver setting.

The implemented Poisson-bracket face corrections are

```math
\mathcal F_E^{PB}
= -\frac{1}{2\epsilon} n\times(H^- - H^+),
\qquad
\mathcal F_H^{PB}
= -\frac{1}{2\mu} n\times(E^- + E^+).
```

These signs follow the code convention for outward minus-side normals and the partitioned weak/transposed derivative pairing.

### Upwind flux

The upwind Hesthaven-Warburton flux adds impedance-weighted tangential penalties to the centered flux:

```math
\mathcal F_E^{up}
= \mathcal F_E^c
-\frac{Y}{2} n\times\left(n\times [E]\right),
```

```math
\mathcal F_H^{up}
= \mathcal F_H^c
-\frac{Z}{2} n\times\left(n\times [H]\right).
```

The penalty terms damp unresolved tangential jumps. This improves robustness but introduces numerical dissipation. For that reason, upwind fluxes belong to the Hesthaven-Warburton path and are rejected by the Poisson-bracket formulation.

## Boundary conditions

Boundary conditions are imposed by constructing an exterior trace `(E^+,H^+)` and then applying the same numerical flux machinery used for interior faces.

### PEC boundary

The perfect electric conductor condition is

```math
n\times E = 0.
```

DiscoGMPI enforces this by reflecting the tangential electric trace:

```math
E^+ = -E^- + 2(n\cdot E^-)n,
\qquad
H^+ = H^-.
```

This reverses the tangential electric field and keeps the normal electric component.

### PMC boundary

The perfect magnetic conductor condition is

```math
n\times H = 0.
```

The corresponding trace reflection reverses the tangential magnetic field and keeps the electric trace:

```math
E^+ = E^-,
\qquad
H^+ = -H^- + 2(n\cdot H^-)n.
```

The PMC convergence driver uses the electromagnetic dual of the PEC cavity mode:

```math
E_{PMC} = Z H_{PEC},
\qquad
H_{PMC} = -Z^{-1} E_{PEC}.
```

### Periodic boundary

Periodic boundaries pair two physical boundary faces by translation. The exterior trace on one side is the interior trace from the paired face on the opposite side, after applying the face-node permutation that aligns both face nodal sets.

For the production periodic plane-wave driver, the analytical solution is

```math
E=(0,0,-kH_0/(\omega\epsilon_0)\sin(k(x-x_0)-\omega t)),
\qquad
H=(0,H_0\sin(k(x-x_0)-\omega t),0),
```

with `k = omega = 2*pi`, `H0 = 1`, `epsilon0 = mu0 = 1`. The coordinate bounds are inferred from the mesh so the phase origin can be made consistent with the selected periodic box.

## Discrete diagnostics and invariants

### Discrete energy

The mass-matrix DG energy is

```math
\mathcal H_h
=\frac12\sum_K |J_K|\left[
\epsilon \sum_{a\in\{x,y,z\}} E_{a,K}^T M E_{a,K}
+
\mu \sum_{a\in\{x,y,z\}} H_{a,K}^T M H_{a,K}
\right].
```

In MPI runs, only owned elements contribute locally. The global value is obtained by an `MPI.Allreduce` sum, so ghost elements are not double counted.

The quadrature diagnostics also compute an independent cubature energy density

```math
w(E,H)=\frac12(\epsilon |E|^2+\mu |H|^2),
```

and compare it with the analytical energy density when an exact solution is available.

### Electric and magnetic charge

The monitored charges are volume integrals of the divergence constraints:

```math
Q_E = \int_\Omega \nabla\cdot(\epsilon E)\,dx,
\qquad
Q_H = \int_\Omega \nabla\cdot(\mu H)\,dx.
```

For exact source-free solutions these quantities are zero. In the code they are evaluated at cubature points using the DG physical derivative matrices and then reduced across MPI ranks.

### Linear momentum

The electromagnetic momentum density monitored by the diagnostics is

```math
p = \epsilon\mu\, E\times H.
```

The global linear momentum is

```math
P = \int_\Omega \epsilon\mu\, E\times H\,dx.
```

The output columns store `P_x`, `P_y`, and `P_z`, together with exact values when supplied by the diagnostic exact solution.

### Angular momentum

The angular momentum density is computed about the coordinate origin:

```math
\ell = x \times p
= x\times(\epsilon\mu\,E\times H).
```

The monitored angular momentum is

```math
L = \int_\Omega x\times(\epsilon\mu\,E\times H)\,dx.
```

Changing the origin changes this diagnostic. Current production outputs use the physical coordinates exactly as stored in the mesh.

### Optical chirality

The optical chirality density is

```math
\chi
=\frac12\left(\epsilon E\cdot(\nabla\times E)
+\mu H\cdot(\nabla\times H)\right).
```

The global chirality diagnostic is

```math
\mathcal{C}_\chi = \int_\Omega \chi\,dx.
```

The periodic plane wave used in the current validation has zero analytical chirality. Nonzero numerical chirality in that case is therefore a discretization diagnostic.

## Error norms and rates

For an exact field `U_exact` and numerical DG field `U_h`, the convergence drivers evaluate componentwise and aggregate `L2` errors by cubature:

```math
\|E_a-E_{a,h}\|_{L^2}
=\left(\sum_K\int_K (E_a-E_{a,h})^2\,dx\right)^{1/2},
```

and similarly for `H_a`. Aggregate electric and magnetic errors are

```math
\|E-E_h\|_{L^2}
=\left(\sum_a \|E_a-E_{a,h}\|_{L^2}^2\right)^{1/2},
```

```math
\|H-H_h\|_{L^2}
=\left(\sum_a \|H_a-H_{a,h}\|_{L^2}^2\right)^{1/2}.
```

Observed rates are computed from two consecutive mesh levels as

```math
p = \frac{\log(e_{coarse}/e_{fine})}{\log(h_{coarse}/h_{fine})}.
```

The production convergence checks compare the final observed rates against explicit expected rates rather than merely checking monotone error decrease.
