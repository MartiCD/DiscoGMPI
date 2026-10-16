#!/usr/bin/env python3

"""Create CMAME-style panels for centered-flux periodic h-convergence runs."""

from __future__ import annotations

import argparse
import sys

from plot_pec_box_alternating_flux_panel import main as plot_main


def has_option(args: list[str], option: str) -> bool:
    return any(argument == option or argument.startswith(f"{option}=") for argument in args)


def parse_wrapper_args(args: list[str]) -> tuple[int, int, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-ready invariant panels for periodic-cuboid "
            "centered-flux h-convergence runs. Remaining options are forwarded "
            "to plot_pec_box_alternating_flux_panel.py."
        )
    )
    parser.add_argument(
        "--order",
        type=int,
        default=2,
        help=(
            "Fixed DG polynomial order used in the h-convergence run. "
            "Default: 2."
        ),
    )
    parser.add_argument(
        "--mesh-levels",
        type=int,
        default=4,
        help="Number of h-refinement levels to plot. Default: 4.",
    )
    namespace, remaining = parser.parse_known_args(args)
    if namespace.order < 0:
        parser.error("--order must be non-negative.")
    if namespace.mesh_levels < 1:
        parser.error("--mesh-levels must be at least 1.")
    return namespace.order, namespace.mesh_levels, remaining


def mesh_level_label(level: int) -> str:
    if level == 0:
        return r"$h_0$"
    return rf"$h_0/{2 ** level}$"


def output_stem(order: int) -> str:
    if order == 2:
        return "periodic_cuboid_cf_h_convergence"
    return f"periodic_cuboid_cf_p{order}_h_convergence"


def h_convergence_defaults(
    order: int,
    mesh_levels: int,
    remaining_args: list[str],
) -> list[str]:
    stem = output_stem(order)
    defaults: list[str] = []
    levels = tuple(range(mesh_levels))

    if not has_option(remaining_args, "--prefix"):
        defaults.append(f"--prefix=periodic_cuboid_cf_structured_p{order}_h")

    if not has_option(remaining_args, "--legend-title"):
        defaults.append("--legend-title=h refinement")

    if not has_option(remaining_args, "--series-id-label"):
        defaults.append("--series-id-label=h")

    if not has_option(remaining_args, "--summary-output"):
        defaults.append(f"--summary-output=output/{stem}_invariants_panel.pdf")

    if not has_option(remaining_args, "--components-output"):
        defaults.append(f"--components-output=output/{stem}_momentum_components_panel.pdf")

    if not has_option(remaining_args, "--error-output"):
        defaults.append(f"--error-output=output/{stem}_invariants_error_panel.pdf")

    if not has_option(remaining_args, "--divergence-output"):
        defaults.append(f"--divergence-output=output/{stem}_divergence_constraints_panel.pdf")

    if not has_option(remaining_args, "--orders"):
        defaults.append("--orders=" + ",".join(str(level) for level in levels))

    if not has_option(remaining_args, "--series-labels"):
        defaults.append(
            "--series-labels="
            + ",".join(mesh_level_label(level) for level in levels)
        )

    return defaults


if __name__ == "__main__":
    order, mesh_levels, remaining_args = parse_wrapper_args(sys.argv[1:])
    sys.argv[1:] = h_convergence_defaults(order, mesh_levels, remaining_args) + remaining_args
    plot_main()
