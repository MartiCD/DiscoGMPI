# DiscoG charge compatibility diagnostics

The quadrature CSV retains all existing broken-divergence charges and integrated
D/B components. Sixteen appended columns start with discog_. Each field has
charge and charge_strong totals and their _x, _y, _z contributions.

These reproduce the algebra in DiscoG's src/Analysis/src/Analysis.jl using
DiscoGMPI's tetrahedral basis and geometry. No evolution operator is changed.
They are NOT interchangeable definitions of physical charge.

With interpolation I, physical quadrature weights w, component coefficients u_i,
physical basis derivatives G_i, and physical stiffness
S_i = integral l_a derivative_i(l_b):

- Electric legacy: -epsilon sum_i w^T[(I u_i) .* (G_i 1)].
  DiscoG's compute_grad_phi sums ALL basis gradients. Partition of unity
  makes this zero in exact arithmetic for ANY field; a small result is
  not evidence of a divergence constraint.
- Magnetic legacy: w^T I sum_i S_i^T u_i. The call in DiscoG's
  src/ODEs/src/ODEs.jl passes Cwp. Its mu keyword is unused, preserved here.
- Electric/magnetic strong: coefficient times w^T I sum_i S_i u_i.
  Original Csp is stiffness, not differentiation: there is no mass inverse.
  Physical Jacobians enter BOTH S_i and w, as in the original PullbackOps.jl.
  These contractions have additional mesh/basis scaling compared with the
  integral of div(coefficient U).

The public compute_* functions accept three component vectors, I, w, and the
appropriate three matrices; *_directional returns a tuple. The distributed
wrapper visits owned elements only and globally sums results. This path
supports scalar constant materials. Equivalent contractions need not reproduce
original rounding: G_i is evaluated as I D_i with the solver's derivatives.
No face or jump term is added.

After a NEW simulation run:

    python3 examples/plot_integrated_fields_and_charge.py output/YOUR_RUN

Besides existing figures, this writes PDF/PNG discog_charge_comparison
(values, signed changes, absolute changes) and discog_charge_directional_drift.
Drift uses the first recorded sample. Exact zeros are omitted on log axes.
Old CSVs remain supported but cannot reconstruct these additional quantities.
No claim is made that these legacy contractions remove drift.

Checks:

    julia --project=. --compiled-modules=no test/test_discog_charge_diagnostics.jl
    mpiexec -n 2 julia --project=. --compiled-modules=no test/test_mpi_integrated_fields.jl
    python3 -m unittest discover -s test -p test_plot_integrated_fields_and_charge.py
