#!/usr/bin/env python3

"""Plot DiscoGMPI integration-point Maxwell diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    "time",
    "cubature_order",
    "electric_energy",
    "magnetic_energy",
    "total_energy",
    "exact_electric_energy",
    "exact_magnetic_energy",
    "exact_total_energy",
    "energy_error",
    "relative_energy_error",
    "electric_l2",
    "exact_electric_l2",
    "electric_error_l2",
    "electric_relative_error",
    "magnetic_l2",
    "exact_magnetic_l2",
    "magnetic_error_l2",
    "magnetic_relative_error",
    "field_error_l2",
    "field_relative_error",
    "energy_density_l2",
    "exact_energy_density_l2",
    "energy_density_error_l2",
    "energy_density_relative_error",
    "electric_charge",
    "magnetic_charge",
    "exact_electric_charge",
    "exact_magnetic_charge",
    "linear_momentum_x",
    "linear_momentum_y",
    "linear_momentum_z",
    "exact_linear_momentum_x",
    "exact_linear_momentum_y",
    "exact_linear_momentum_z",
    "angular_momentum_x",
    "angular_momentum_y",
    "angular_momentum_z",
    "exact_angular_momentum_x",
    "exact_angular_momentum_y",
    "exact_angular_momentum_z",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot energy, analytical L2, charge, and momentum diagnostics from "
            "quadrature_diagnostics.csv."
        )
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path(
            "output/distributed_poisson_bracket/quadrature_diagnostics.csv"
        ),
        help=(
            "Path to quadrature_diagnostics.csv "
            "(default: output/distributed_poisson_bracket/"
            "quadrature_diagnostics.csv)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output image path "
            "(default: quadrature_diagnostics.png beside the CSV)"
        ),
    )
    parser.add_argument(
        "--charge-momentum-output",
        type=Path,
        help=(
            "Charge and momentum image path "
            "(default: charge_momentum_diagnostics.png beside the CSV)"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output resolution in dots per inch (default: 180)",
    )
    return parser.parse_args()


def read_diagnostics(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        available = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - available)
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(missing)}"
            )

        diagnostics = {name: [] for name in REQUIRED_COLUMNS}
        for row_number, row in enumerate(reader, start=2):
            try:
                for name in diagnostics:
                    diagnostics[name].append(float(row[name]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value in {path} at row {row_number}"
                ) from error

    if not diagnostics["time"]:
        raise ValueError(f"No diagnostic samples found in {path}")

    return diagnostics


def configure_axis(axis: plt.Axes, title: str, ylabel: str) -> None:
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)


def positive_magnitudes(values: list[float]) -> list[float]:
    return [abs(value) if value != 0.0 else float("nan") for value in values]


def component_error(
    diagnostics: dict[str, list[float]],
    numerical_name: str,
    exact_name: str,
) -> list[float]:
    return [
        numerical - exact
        for numerical, exact in zip(
            diagnostics[numerical_name],
            diagnostics[exact_name],
        )
    ]


def vector_magnitude(
    x_values: list[float],
    y_values: list[float],
    z_values: list[float],
) -> list[float]:
    return [
        math.sqrt(x_value**2 + y_value**2 + z_value**2)
        for x_value, y_value, z_value in zip(
            x_values,
            y_values,
            z_values,
        )
    ]


def plot_diagnostics(
    diagnostics: dict[str, list[float]],
    output: Path,
    dpi: int,
) -> None:
    time = diagnostics["time"]
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(14, 13),
        sharex=True,
        constrained_layout=True,
    )
    (
        energy_axis,
        energy_error_axis,
        field_norm_axis,
        field_error_axis,
        relative_error_axis,
        density_axis,
    ) = axes.flat

    energy_axis.plot(
        time,
        diagnostics["electric_energy"],
        label="Electric",
    )
    energy_axis.plot(
        time,
        diagnostics["magnetic_energy"],
        label="Magnetic",
    )
    energy_axis.plot(
        time,
        diagnostics["total_energy"],
        label="Total",
        linewidth=2.0,
    )
    energy_axis.plot(
        time,
        diagnostics["exact_electric_energy"],
        "--",
        label="Exact electric",
    )
    energy_axis.plot(
        time,
        diagnostics["exact_magnetic_energy"],
        "--",
        label="Exact magnetic",
    )
    energy_axis.plot(
        time,
        diagnostics["exact_total_energy"],
        "--",
        label="Exact total",
        linewidth=2.0,
    )
    configure_axis(energy_axis, "Electromagnetic Energy", "Energy")
    energy_axis.legend(ncol=2, fontsize="small")

    energy_error_axis.plot(
        time,
        diagnostics["energy_error"],
        label="Absolute energy error",
    )
    energy_error_axis.axhline(0.0, color="black", linewidth=0.8)
    configure_axis(
        energy_error_axis,
        "Total-Energy Error",
        r"$\mathcal{E}_h-\mathcal{E}_{exact}$",
    )
    energy_error_axis.legend(fontsize="small")

    field_norm_axis.plot(
        time,
        diagnostics["electric_l2"],
        label=r"$\|E_h\|_{L^2}$",
    )
    field_norm_axis.plot(
        time,
        diagnostics["exact_electric_l2"],
        "--",
        label=r"$\|E\|_{L^2}$",
    )
    field_norm_axis.plot(
        time,
        diagnostics["magnetic_l2"],
        label=r"$\|H_h\|_{L^2}$",
    )
    field_norm_axis.plot(
        time,
        diagnostics["exact_magnetic_l2"],
        "--",
        label=r"$\|H\|_{L^2}$",
    )
    configure_axis(field_norm_axis, "Electric and Magnetic L2 Norms", "L2 norm")
    field_norm_axis.legend(ncol=2, fontsize="small")

    field_error_axis.plot(
        time,
        positive_magnitudes(diagnostics["electric_error_l2"]),
        label=r"$\|E_h-E\|_{L^2}$",
    )
    field_error_axis.plot(
        time,
        positive_magnitudes(diagnostics["magnetic_error_l2"]),
        label=r"$\|H_h-H\|_{L^2}$",
    )
    field_error_axis.plot(
        time,
        positive_magnitudes(diagnostics["field_error_l2"]),
        label="Combined field error",
        linewidth=2.0,
    )
    field_error_axis.plot(
        time,
        positive_magnitudes(diagnostics["energy_density_error_l2"]),
        label=r"$\|w_h-w\|_{L^2}$",
    )
    configure_axis(field_error_axis, "Absolute L2 Errors", "L2 error")
    field_error_axis.set_yscale("log")
    field_error_axis.legend(fontsize="small")

    relative_error_axis.plot(
        time,
        positive_magnitudes(diagnostics["relative_energy_error"]),
        label="Total energy",
    )
    relative_error_axis.plot(
        time,
        positive_magnitudes(diagnostics["electric_relative_error"]),
        label="Electric field",
    )
    relative_error_axis.plot(
        time,
        positive_magnitudes(diagnostics["magnetic_relative_error"]),
        label="Magnetic field",
    )
    relative_error_axis.plot(
        time,
        positive_magnitudes(diagnostics["field_relative_error"]),
        label="Combined field",
        linewidth=2.0,
    )
    relative_error_axis.plot(
        time,
        positive_magnitudes(diagnostics["energy_density_relative_error"]),
        label="Energy density",
    )
    configure_axis(
        relative_error_axis,
        "Absolute Relative Errors",
        "Absolute relative error",
    )
    relative_error_axis.set_yscale("log")
    relative_error_axis.legend(fontsize="small")

    density_axis.plot(
        time,
        diagnostics["energy_density_l2"],
        label=r"$\|w_h\|_{L^2}$",
    )
    density_axis.plot(
        time,
        diagnostics["exact_energy_density_l2"],
        "--",
        label=r"$\|w\|_{L^2}$",
    )
    configure_axis(
        density_axis,
        "Electromagnetic Energy-Density L2 Norm",
        "L2 norm",
    )
    density_axis.legend(fontsize="small")

    for axis in axes[-1, :]:
        axis.set_xlabel("Time")

    cubature_orders = sorted(set(diagnostics["cubature_order"]))
    cubature_label = ", ".join(f"{order:g}" for order in cubature_orders)
    figure.suptitle(
        "DiscoGMPI Poisson-Bracket Maxwell Diagnostics\n"
        f"Jaskowiec-Sukumar cubature order: {cubature_label}",
        fontsize=15,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def plot_charge_and_momentum_diagnostics(
    diagnostics: dict[str, list[float]],
    output: Path,
    dpi: int,
) -> None:
    time = diagnostics["time"]
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(14, 13),
        sharex=True,
        constrained_layout=True,
    )
    (
        electric_charge_axis,
        magnetic_charge_axis,
        linear_momentum_axis,
        linear_error_axis,
        angular_momentum_axis,
        angular_error_axis,
    ) = axes.flat

    electric_charge_axis.plot(
        time,
        diagnostics["electric_charge"],
        label=r"$Q_{e,h}$",
    )
    electric_charge_axis.plot(
        time,
        diagnostics["exact_electric_charge"],
        "--",
        label=r"$Q_e$ exact",
    )
    electric_charge_axis.axhline(0.0, color="black", linewidth=0.8)
    configure_axis(
        electric_charge_axis,
        "Electric Charge",
        r"$\int_\Omega \nabla\!\cdot(\varepsilon E)\,dV$",
    )
    electric_charge_axis.legend(fontsize="small")

    magnetic_charge_axis.plot(
        time,
        diagnostics["magnetic_charge"],
        label=r"$Q_{m,h}$",
    )
    magnetic_charge_axis.plot(
        time,
        diagnostics["exact_magnetic_charge"],
        "--",
        label=r"$Q_m$ exact",
    )
    magnetic_charge_axis.axhline(0.0, color="black", linewidth=0.8)
    configure_axis(
        magnetic_charge_axis,
        "Magnetic Charge",
        r"$\int_\Omega \nabla\!\cdot(\mu H)\,dV$",
    )
    magnetic_charge_axis.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
    )
    magnetic_charge_axis.legend(fontsize="small")

    component_colors = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}
    for component, color in component_colors.items():
        linear_momentum_axis.plot(
            time,
            diagnostics[f"linear_momentum_{component}"],
            color=color,
            label=rf"$P_{{{component},h}}$",
        )
        linear_momentum_axis.plot(
            time,
            diagnostics[f"exact_linear_momentum_{component}"],
            "--",
            color=color,
            label=rf"$P_{component}$ exact",
        )
    linear_momentum_axis.axhline(0.0, color="black", linewidth=0.8)
    configure_axis(
        linear_momentum_axis,
        "Electromagnetic Linear Momentum",
        "Momentum",
    )
    linear_momentum_axis.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
    )
    linear_momentum_axis.legend(ncol=2, fontsize="small")

    linear_errors = {}
    for component, color in component_colors.items():
        error = component_error(
            diagnostics,
            f"linear_momentum_{component}",
            f"exact_linear_momentum_{component}",
        )
        linear_errors[component] = error
        linear_error_axis.plot(
            time,
            error,
            color=color,
            label=rf"$P_{{{component},h}}-P_{component}$",
        )
    linear_error_axis.plot(
        time,
        vector_magnitude(
            linear_errors["x"],
            linear_errors["y"],
            linear_errors["z"],
        ),
        color="black",
        linewidth=2.0,
        label="Vector error magnitude",
    )
    linear_error_axis.axhline(0.0, color="black", linewidth=0.8)
    configure_axis(
        linear_error_axis,
        "Linear-Momentum Error",
        "Momentum error",
    )
    linear_error_axis.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
    )
    linear_error_axis.legend(fontsize="small")

    for component, color in component_colors.items():
        angular_momentum_axis.plot(
            time,
            diagnostics[f"angular_momentum_{component}"],
            color=color,
            label=rf"$L_{{{component},h}}$",
        )
        angular_momentum_axis.plot(
            time,
            diagnostics[f"exact_angular_momentum_{component}"],
            "--",
            color=color,
            label=rf"$L_{component}$ exact",
        )
    angular_momentum_axis.axhline(0.0, color="black", linewidth=0.8)
    configure_axis(
        angular_momentum_axis,
        "Electromagnetic Angular Momentum",
        "Angular momentum",
    )
    angular_momentum_axis.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
    )
    angular_momentum_axis.legend(ncol=2, fontsize="small")

    angular_errors = {}
    for component, color in component_colors.items():
        error = component_error(
            diagnostics,
            f"angular_momentum_{component}",
            f"exact_angular_momentum_{component}",
        )
        angular_errors[component] = error
        angular_error_axis.plot(
            time,
            error,
            color=color,
            label=rf"$L_{{{component},h}}-L_{component}$",
        )
    angular_error_axis.plot(
        time,
        vector_magnitude(
            angular_errors["x"],
            angular_errors["y"],
            angular_errors["z"],
        ),
        color="black",
        linewidth=2.0,
        label="Vector error magnitude",
    )
    angular_error_axis.axhline(0.0, color="black", linewidth=0.8)
    configure_axis(
        angular_error_axis,
        "Angular-Momentum Error",
        "Angular-momentum error",
    )
    angular_error_axis.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
    )
    angular_error_axis.legend(fontsize="small")

    for axis in axes[-1, :]:
        axis.set_xlabel("Time")

    cubature_orders = sorted(set(diagnostics["cubature_order"]))
    cubature_label = ", ".join(f"{order:g}" for order in cubature_orders)
    figure.suptitle(
        "DiscoGMPI Charge and Momentum Diagnostics\n"
        "Angular momentum measured about the coordinate origin; "
        f"cubature order: {cubature_label}",
        fontsize=15,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output = args.output or args.csv_path.with_name(
        "quadrature_diagnostics.png"
    )
    charge_momentum_output = (
        args.charge_momentum_output
        or args.csv_path.with_name("charge_momentum_diagnostics.png")
    )
    diagnostics = read_diagnostics(args.csv_path)
    plot_diagnostics(diagnostics, output, args.dpi)
    plot_charge_and_momentum_diagnostics(
        diagnostics,
        charge_momentum_output,
        args.dpi,
    )
    print(output)
    print(charge_momentum_output)


if __name__ == "__main__":
    main()
