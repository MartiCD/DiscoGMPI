# Maxwell Poisson-Bracket Implementation Audit

## Deliverables and Scope

- `poisson_bracket_implementation.tex` and its PDF: full mathematical reconstruction, worked degree-one example, source map, and reproducibility appendix.
- `verification_summary.tex`: measured results included by the main document.
- `verify_operator.jl`: independent coefficient-matrix reconstruction and matrix-free/MPI comparisons. It does not modify or replace the solver.
- `checks_rank{1,2,4}.csv`, `run_rank{1,2,4}.log`: executed numerical evidence.
- `existing_reference_tests.log`: separate pre-existing reference-node tests.
- `provenance.txt`: branch, commit, dirty-tree inventory, and SHA-256 source hashes. Local changes are part of the inspected implementation.
- `capture_provenance.py`: regenerates the identity/hash inventory for a new snapshot without modifying the repository.

The reconstruction covers the affine, Cartesian-component, E-H tetrahedral spatial operator. It distinguishes `PoissonBracketFormulation` from `HesthavenWarburtonFormulation`, scalar and element-material overloads, physical ghosts and MPI halos, diagnostic cubature and evolution matrices. Time integration is discussed only to separate spatial invariants from fully discrete behavior and describe the timed boundary interface.

## Snapshot and Inspection

Branch: `main`. Baseline commit: `e5bd7abe249efab7806acfb197af1141313523f9`. The worktree was already dirty; no production source, driver, or existing test was modified for this audit. No applicable `AGENTS.md` was found in this repository or its ancestors.

Inspected source paths include `src/DiscoGMPI.jl`, `src/solver/data_containers.jl`, and the mesh, topology, geometry, metrics, reference, face, physical-operator, trace-map, flux-face, Maxwell, periodic, time-integration, and cubature kernels. Supporting paths include `src/PhysicalCoverage.jl`, `src/DistributedDG.jl`, `src/DistributedMesh3D.jl`, `src/DistributedIO.jl`, `src/TraceMaps.jl`, `src/MetisIO.jl`, `src/MaxwellDiagnostics.jl`, `src/MaxwellInitialization.jl`, `src/MaxwellDriverSupport.jl`, `src/MaxwellExactSolutions.jl`, and `src/NonlinearPML.jl`. Representative executable drivers are the periodic Poisson-bracket Maxwell and metallic-sphere scattering examples. Existing reference-node and magnetic-charge tests and the related math/numerics documentation were also inspected. The report gives symbols and line ranges for each major definition, not merely file-level citations.

## Main Verified Properties

1. The reference basis comes from a monomial Gram matrix and Cholesky orthonormalization, not a direct Dubiner recurrence. Evolution mass and face matrices use analytic moments through this construction; Jaskowiec-Sukumar cubature is used by diagnostics.
2. Electric volume coupling is a strong curl written as a mass-inverted stiffness action. Magnetic coupling is the negative transpose of the mass-weighted electric volume block. The magnetic surface term contains the full electric numerical trace, not just a jump correction.
3. True scalar alternating traces select by the sign of `nx + ny + nz`; exactly zero gives centered traces. The opposite element uses the complementary selector. There is no tunable `C12` or beta parameter on this path.
4. Under matching face masses, opposite normals, fixed positive elementwise materials, and supported closed boundary conditions, the conceptual mass-weighted operator is skew. The coefficient Poisson tensor is `M^-1 A M^-1`, not `M^-1 A`. A constant skew tensor has Jacobi by direct derivation; this does not extend the claim to nonlinear PML, absorbing boundaries, or floating-point exactness.
5. The monitored magnetic charge is the signed integral of **broken strong divergence**. Its null-row condition is much weaker than a local divergence-of-curl identity. Neither normal continuity nor an independently defined numerical-normal-trace divergence is imposed by that diagnostic.
6. Ordinary MPI exchanges pack six components at face nodes, use tag 24017, and wait before residual evaluation. Distributed periodic faces use tag 24018. Only owned coefficients contribute to the global state and reduced diagnostics.

## Discrepancies and Boundaries of the Evidence

### Material-aware alternating requests execute centered traces

`src/PhysicalCoverage.jl:254-274` and `421-437`, `maxwell_material_face_flux!` and `maxwell_material_face_flux`, omit `flux_kind=formulation.flux_kind` when calling the Poisson-bracket flux routine. The default is centered. Interior, boundary, and material-periodic paths inherit this behavior, including homogeneous arrays. The scalar periodic driver path passes the selector and is unaffected by this particular discrepancy.

Numerical evidence: material-aware alternating and scalar centered RHS arrays match exactly in all tested cases. They differ substantially from true scalar alternating RHS arrays, e.g. approximately 134 in maximum entry at p=3 on the 12-tet single-rank mesh. The CSV material comparisons on multi-rank runs record rank zero's local-array difference, not a global maximum. No fix is included.

This also limits existing alternating-labeled tests using the material overload, including `test/test_mpi_magnetic_charge_null_invariant.jl`: their label alone does not establish coverage of true alternating fluxes.

### Unused face-quadrature helper fails

`src/solver/kernels/face_operators.jl:234`, `eval_nodal_basis_at_points`, broadcasts a row matrix into a vector slice. The independent script observes `DimensionMismatch` when exercising this legacy route. The active triangular Gram face constructor does not use it. The face-moment cross-check evaluates the modal basis at quadrature points directly, without repairing the helper.

### Matrix construction is not exact in Float64

Reference integration-by-parts residuals increase from about 5e-16 at p=1 to 6.12e-11 at p=4 (equispaced), despite exact real-arithmetic moment formulas. Conditioning and cancellation in the Gram construction are relevant; the audit does not isolate every rounding source. A tolerance pass must not be rewritten as a claim of a few-epsilon identity at every degree.

### Unsupported or unproved extensions

- No curved-element metric, Piola-conforming field, within-element anisotropic material, or nonmatching periodic mortar is established on this path.
- Periodic setup checks translated points, permutations, and areas but does not separately reject nonopposite normals. Small structured-mesh success does not certify arbitrary mesh input.
- Centered PMC is energy-conservative, but the monitored signed magnetic-charge row is nonzero on the tested PMC problem. Periodic/PEC charge results cannot be generalized to all boundary closures.
- Global periodic charge rows were tested column-by-column only on the stated small meshes. No general-mesh charge-null theorem or time-evolution preservation of broken divergence is claimed.
- `docs/math/maxwell_poisson_bracket_formulation.tex:134` says alternating is not exposed; that statement is outdated for the current scalar path. Unrelated documentation was left unchanged.
- The report proves Jacobi only conditionally for the reconstructed fixed closed real-arithmetic skew matrix. It does not claim a general nonlinear Poisson structure or exact ESPRK energy conservation.

## Executed Checks

Each independent 1-, 2-, and 4-rank run completed with exit status zero and wrote 173 measurements, including 110 tolerance checks, all passing. Positive mass definiteness is additionally asserted. In the CSV, `tolerance=NaN` means **measurement only**; its `pass=true` is vacuous. In particular, the nonzero broken-divergence rows are not passing divergence-preservation tests.

The existing `test/test_reference_tet_nodes.jl` passed all 761 assertions (673 + 85 + 3). Its higher-degree reference construction emitted a modal-orthonormality warning of 1.38e-9; this warning remains in `existing_reference_tests.log` and is not hidden by the passing test status.

| Check | Scope | Measured result |
|---|---|---|
| Constant / coordinate differentiation | p=1:4, both nodal sets | maxima 4.28e-15 / 1.08e-14 |
| Affine coordinate differentiation | skew maps with both determinant signs | maximum 2.29e-14 |
| Mass symmetry / positive definiteness | p=1:4, both nodal sets | symmetry residual zero; all minimum eigenvalues positive |
| Reference IBP | p=1:4, both nodal sets | maximum 6.12e-11 at p=4 |
| Independent face quadrature | p=1:4, both nodal sets | maximum 1.44e-11 |
| Cubature monomials | all total degrees through 2, 4, 8 | maximum 2.23e-16 |
| Complete matrix reconstruction | p=1,2; 6/12 tets; periodic, PEC, PMC, absorbing | scaled maximum 2.68e-16 |
| Closed weighted skew identity | same closed cases | scaled maximum 8.44e-16 |
| Heterogeneous PEC mass scaling | p=1, 12 tets | scaled residual 1.39e-16 |
| Closed energy production | deterministic full coefficient vector | absolute maximum 4.72e-16 |
| Periodic signed-charge null row | p=1,2; 6/12 tets; both scalar fluxes | scaled maximum 2.70e-17 |
| Broken divergence-of-curl | periodic matrices above | induced infinity norms 50 to 1384, not zero |
| MPI vs serial owned RHS | p=1:3; 12 tets; both nodal sets and scalar fluxes | 0 (1 rank), 2.84e-14 (2 and 4 ranks) |

Matrix reconstruction and skew checks use a 3e-12 relative scaling threshold; MPI uses 2e-10 absolute. Reference thresholds are 2000 epsilon times `max(1,cond(V)^2)`, multiplied by five for affine maps. Charge residuals use `max(1, norm(cH,1)*opnorm(LHE,Inf))` as denominator. The matrix-free and reconstructed operators share source geometry/reference matrices but **not** numerical-flux or lifting routines; therefore this cross-check is independent at the residual-assembly level, not a second implementation of the entire basis/mesh stack.

MPI tests compare every owned entry for one deterministic discontinuous coefficient field and deliberately poison ghosts beforehand. Complete serial matrices apply the actual RHS to every coefficient unit vector. Those are different scopes of evidence.

## Reproduction

From the DiscoGMPI repository root (using an already instantiated environment):

```bash
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1
julia --project=. --compiled-modules=no \
  docs/math/poisson_bracket_implementation/verify_operator.jl /tmp/pb_rank1.csv

julia --project=. --compiled-modules=no -e '
using MPI
run(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --project=. --compiled-modules=no docs/math/poisson_bracket_implementation/verify_operator.jl /tmp/pb_rank2.csv`)
'
# Repeat with -n 4 and /tmp/pb_rank4.csv.

julia --project=. --compiled-modules=no test/test_reference_tet_nodes.jl
```

MPI runs were executed with process/socket permission outside the filesystem sandbox and the MPI.jl-selected launcher. The independent script was staged under `/tmp/pb_reconstruction` for execution; its installed content is identical. No full package suite, production-duration run, PML validation, or general mesh convergence study was run for this report.

Runtime: Julia 1.10.11, MPI.jl 0.20.23, MPICH 4.3.1 (ch3/nemesis), ILP64 OpenBLAS, one Julia thread and one BLAS thread per process.

Compile from this documentation directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error poisson_bracket_implementation.tex
pdflatex -interaction=nonstopmode -halt-on-error poisson_bracket_implementation.tex
```

The report's conclusions are specific to the saved working-tree snapshot. Subsequent solver changes require rerunning the checks and updating the source line references and hashes.

## Document Checks

The standalone report compiled with `pdflatex` to 21 pages. After repeat passes, the log contained no undefined references, LaTeX warnings, or overfull/underfull boxes. All pages were rendered to PNG and inspected, including the low-order matrices, operator equations, evidence tables and code map. Source-reference paths and line-range bounds were checked against this snapshot; no TODO placeholders were introduced. These document checks do not strengthen the stated numerical scope.
