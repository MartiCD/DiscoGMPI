#!/usr/bin/env python3

"""Create invariant panels for periodic cuboid alternating-flux runs."""

from __future__ import annotations

import sys

from plot_pec_box_alternating_flux_panel import main


PERIODIC_DEFAULTS = (
    "--prefix=periodic_cuboid_af_p",
    "--summary-output=output/periodic_cuboid_af_invariants_panel.pdf",
    "--components-output=output/periodic_cuboid_af_momentum_components_panel.pdf",
    "--error-output=output/periodic_cuboid_af_invariants_error_panel.pdf",
)


if __name__ == "__main__":
    sys.argv[1:1] = PERIODIC_DEFAULTS
    main()
