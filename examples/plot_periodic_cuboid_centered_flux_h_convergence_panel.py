#!/usr/bin/env python3

"""Create invariant panels for periodic cuboid centered-flux h-convergence runs."""

from __future__ import annotations

import argparse
import sys

from plot_pec_box_alternating_flux_panel import main as plot_main


H_CONVERGENCE_BASE_DEFAULTS = (
    "--prefix=periodic_cuboid_cf_structured_p2_h",
    "--legend-title=h refinement",
    "--summary-output=output/periodic_cuboid_cf_h_convergence_invariants_panel.pdf",
    "--components-output=output/periodic_cuboid_cf_h_convergence_momentum_components_panel.pdf",
    "--error-output=output/periodic_cuboid_cf_h_convergence_invariants_error_panel.pdf",
)


def has_option(args: list[str], option: str) -> bool:
    return any(argument == option or argument.startswith(f"{option}=") for argument in args)


def parse_mesh_levels(args: list[str]) -> tuple[int, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mesh-levels",
        type=int,
        default=4,
        help="Number of h-refinement levels to plot. Default: 4.",
    )
    namespace, remaining = parser.parse_known_args(args)
    if namespace.mesh_levels < 1:
        parser.error("--mesh-levels must be at least 1.")
    return namespace.mesh_levels, remaining


def mesh_level_label(level: int) -> str:
    if level == 0:
        return r"$h_0$"
    return rf"$h_0/{2 ** level}$"


def h_convergence_defaults(mesh_levels: int, remaining_args: list[str]) -> list[str]:
    defaults = list(H_CONVERGENCE_BASE_DEFAULTS)
    levels = tuple(range(mesh_levels))

    if not has_option(remaining_args, "--orders"):
        defaults.append("--orders=" + ",".join(str(level) for level in levels))

    if not has_option(remaining_args, "--series-labels"):
        defaults.append(
            "--series-labels="
            + ",".join(mesh_level_label(level) for level in levels)
        )

    return defaults


if __name__ == "__main__":
    mesh_levels, remaining_args = parse_mesh_levels(sys.argv[1:])
    sys.argv[1:] = h_convergence_defaults(mesh_levels, remaining_args) + remaining_args
    plot_main()
