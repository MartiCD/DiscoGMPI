#!/usr/bin/env python3

"""Post-process DiscoGMPI distributed Maxwell convergence results."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator


params = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["TeX Gyre Schola"],
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    "axes.linewidth": 1.0,
    "axes.grid": True,
    "axes.grid.which": "both",
    "grid.color": "0.85",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,

    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "lines.markeredgewidth": 0.8,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,

    "legend.frameon": True,
    "legend.handlelength": 2.5,
    "legend.handletextpad": 0.4,

    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}

plt.rcParams.update(params)
# warnings.filterwarnings("ignore")


REQUIRED_COLUMNS = {
    "mpi_ranks",
    "order",
    "esprk_order",
    "cubature_order",
    "mesh_level",
    "cells_per_axis",
    "nelements",
    "min_owned_elements",
    "max_owned_elements",
    "characteristic_h",
    "dt",
    "nsteps",
    "elapsed_seconds",
    "l2_electric_error",
    "l2_magnetic_error",
    "l2_total_error",
    "relative_total_error",
    "linf_electric_error",
    "linf_magnetic_error",
    "linf_total_error",
    "energy_error",
    "relative_energy_error",
    "electric_charge",
    "magnetic_charge",
    "rate_electric",
    "rate_magnetic",
    "rate_total",
}

INTEGER_COLUMNS = {
    "mpi_ranks",
    "order",
    "esprk_order",
    "cubature_order",
    "mesh_level",
    "cells_per_axis",
    "nelements",
    "min_owned_elements",
    "max_owned_elements",
    "nsteps",
}

TABLE_COLUMNS = (
    "mpi_ranks",
    "boundary_condition",
    "order",
    "esprk_order",
    "mesh_level",
    "cells_per_axis",
    "nelements",
    "characteristic_h",
    "dt",
    "nsteps",
    "l2_electric_error",
    "rate_electric",
    "l2_magnetic_error",
    "rate_magnetic",
    "l2_total_error",
    "rate_total",
    "linf_electric_error",
    "linf_magnetic_error",
    "linf_total_error",
    "relative_total_error",
    "energy_error",
    "relative_energy_error",
    "electric_charge",
    "magnetic_charge",
    "elapsed_seconds",
)


@dataclass(frozen=True)
class Result:
    values: dict[str, float | int | str | None]

    def integer(self, name: str) -> int:
        value = self.values[name]
        assert isinstance(value, int)
        return value

    def number(self, name: str) -> float:
        value = self.values[name]
        assert isinstance(value, (int, float))
        return float(value)

    def optional_number(self, name: str) -> float | None:
        value = self.values[name]
        assert value is None or isinstance(value, (int, float))
        return None if value is None else float(value)

    def text(self, name: str) -> str:
        value = self.values[name]
        assert isinstance(value, str)
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge and post-process distributed Poisson-bracket Maxwell "
            "convergence CSV files."
        )
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        type=Path,
        default=[
            Path(
                "output/convergence_distributed_poisson_bracket.csv"
            )
        ],
        help=(
            "One or more convergence CSV files. Results are merged and "
            "duplicate cases must agree."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for generated files. Default: "
            "<first CSV directory>/convergence_postprocessing"
        ),
    )
    parser.add_argument(
        "--prefix",
        default="distributed_poisson_bracket",
        help="Generated filename prefix (default: distributed_poisson_bracket)",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated plot formats (default: png,pdf)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Raster plot resolution in dots per inch (default: 200)",
    )
    parser.add_argument(
        "--title",
        default="Distributed Poisson-Bracket Maxwell Convergence",
        help="Title used in plots and the LaTeX table caption.",
    )
    return parser.parse_args()


def parse_value(name: str, text: str, path: Path, row_number: int):
    text = text.strip()
    if not text:
        return None
    try:
        if name in INTEGER_COLUMNS:
            return int(text)
        return float(text)
    except ValueError as error:
        raise ValueError(
            f"Invalid value for {name!r} in {path} at row {row_number}: "
            f"{text!r}"
        ) from error


def read_results(paths: Iterable[Path]) -> list[Result]:
    merged: dict[tuple[str, int, int, int, int], Result] = {}

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Convergence CSV not found: {path}")

        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            available = set(reader.fieldnames or ())
            missing = sorted(REQUIRED_COLUMNS - available)
            if missing:
                raise ValueError(
                    f"{path} is missing required columns: "
                    f"{', '.join(missing)}"
                )

            for row_number, row in enumerate(reader, start=2):
                values = {
                    name: parse_value(name, row[name], path, row_number)
                    for name in REQUIRED_COLUMNS
                }
                boundary_condition = row.get(
                    "boundary_condition",
                    "pec",
                ).strip().lower()
                if boundary_condition not in {"pec", "pmc", "periodic"}:
                    raise ValueError(
                        "Invalid boundary_condition in "
                        f"{path} at row {row_number}: "
                        f"{boundary_condition!r}"
                    )
                values["boundary_condition"] = boundary_condition
                result = Result(values)
                key = (
                    result.text("boundary_condition"),
                    result.integer("mpi_ranks"),
                    result.integer("order"),
                    result.integer("mesh_level"),
                    result.integer("cells_per_axis"),
                )
                previous = merged.get(key)
                if previous is not None and previous != result:
                    raise ValueError(
                        "Conflicting duplicate convergence case "
                        f"{key} in {path}"
                    )
                merged[key] = result

    if not merged:
        raise ValueError("No convergence results were found.")

    results = sorted(
        merged.values(),
        key=lambda result: (
            result.text("boundary_condition"),
            result.integer("mpi_ranks"),
            result.integer("order"),
            result.integer("mesh_level"),
        ),
    )
    validate_results(results)
    return results


def validate_results(results: list[Result]) -> None:
    groups = group_results(results)
    rank_counts = {result.integer("mpi_ranks") for result in results}
    if len(rank_counts) != 1:
        raise ValueError(
            "All input results must use the same MPI rank count; found "
            f"{sorted(rank_counts)}"
        )
    boundary_conditions = {
        result.text("boundary_condition") for result in results
    }
    if len(boundary_conditions) != 1:
        raise ValueError(
            "All input results must use the same boundary condition; found "
            f"{sorted(boundary_conditions)}"
        )

    for order, group in groups.items():
        if len(group) < 2:
            raise ValueError(
                f"DG order {order} has only one mesh level; at least two "
                "are needed for convergence post-processing."
            )
        h_values = [result.number("characteristic_h") for result in group]
        if any(value <= 0.0 for value in h_values):
            raise ValueError(f"DG order {order} contains non-positive h.")
        if not all(
            fine < coarse
            for coarse, fine in zip(h_values, h_values[1:])
        ):
            raise ValueError(
                f"DG order {order} is not ordered from coarse to fine mesh."
            )


def group_results(results: list[Result]) -> dict[int, list[Result]]:
    groups: dict[int, list[Result]] = {}
    for result in results:
        groups.setdefault(result.integer("order"), []).append(result)
    for group in groups.values():
        group.sort(key=lambda result: result.integer("mesh_level"))
    return dict(sorted(groups.items()))


def format_csv_value(
    name: str,
    value: float | int | str | None,
) -> str:
    if value is None:
        return ""
    if name == "boundary_condition":
        return str(value)
    if name in INTEGER_COLUMNS:
        return str(int(value))
    if name.startswith("rate_"):
        return f"{float(value):.6f}"
    if name == "elapsed_seconds":
        return f"{float(value):.6f}"
    return f"{float(value):.12e}"


def write_table_csv(path: Path, results: list[Result]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(TABLE_COLUMNS)
        for result in results:
            writer.writerow(
                [
                    format_csv_value(name, result.values[name])
                    for name in TABLE_COLUMNS
                ]
            )


def latex_number(value: float) -> str:
    if value == 0.0:
        return r"$0$"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10.0**exponent
    if -2 <= exponent <= 2:
        return f"${value:.4g}$"
    return rf"${mantissa:.3f}\times 10^{{{exponent}}}$"


def latex_rate(value: float | None) -> str:
    return "--" if value is None else f"${value:.3f}$"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def write_latex_table(
    path: Path,
    results: list[Result],
    caption: str,
) -> None:
    groups = group_results(results)
    rank_count = results[0].integer("mpi_ranks")
    boundary_condition = results[0].text("boundary_condition").upper()
    lines = [
        "% Requires \\usepackage{amsmath,booktabs}",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{rrrrrrrrrr}",
        r"    \toprule",
        (
            r"    $N$ & Cells & $N_e$ & $h$ & "
            r"$\lVert E-E_h\rVert_2$ & Rate & "
            r"$\lVert H-H_h\rVert_2$ & Rate & "
            r"$\lVert U-U_h\rVert_2$ & Rate \\"
        ),
        r"    \midrule",
    ]

    orders = list(groups)
    for group_index, order in enumerate(orders):
        for result in groups[order]:
            lines.append(
                "    "
                + " & ".join(
                    (
                        str(order),
                        str(result.integer("cells_per_axis")),
                        str(result.integer("nelements")),
                        latex_number(result.number("characteristic_h")),
                        latex_number(result.number("l2_electric_error")),
                        latex_rate(
                            result.optional_number("rate_electric")
                        ),
                        latex_number(result.number("l2_magnetic_error")),
                        latex_rate(
                            result.optional_number("rate_magnetic")
                        ),
                        latex_number(result.number("l2_total_error")),
                        latex_rate(result.optional_number("rate_total")),
                    )
                )
                + r" \\"
            )
        if group_index != len(orders) - 1:
            lines.append(r"    \midrule")

    lines.extend(
        (
            r"    \bottomrule",
            r"  \end{tabular}",
            (
                "  \\caption{"
                + latex_escape(caption)
                + f" using {rank_count} MPI ranks and "
                + f"{boundary_condition} boundaries."
                + "}"
            ),
            r"  \label{tab:distributed-maxwell-convergence}",
            r"\end{table}",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def positive(value: float) -> float:
    return abs(value) if value != 0.0 else float("nan")


def final_rate_label(result: Result, rate_name: str) -> str:
    rate = result.optional_number(rate_name)
    return "" if rate is None else f", final rate {rate:.2f}"


def configure_log_axis(
    axis: plt.Axes,
    title: str,
    ylabel: str,
    mesh_sizes: list[float],
) -> None:
    axis.set_title(title)
    axis.set_xlabel(r"Characteristic mesh size $h$")
    axis.set_ylabel(ylabel)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.invert_xaxis()
    axis.xaxis.set_major_locator(FixedLocator(mesh_sizes))
    axis.xaxis.set_major_formatter(
        FixedFormatter([f"{mesh_size:.3f}" for mesh_size in mesh_sizes])
    )
    axis.xaxis.set_minor_locator(NullLocator())
    axis.grid(True, which="both", alpha=0.3)


def plot_error_family(
    groups: dict[int, list[Result]],
    output_stem: Path,
    formats: list[str],
    dpi: int,
    title: str,
    prefix: str,
    error_columns: tuple[str, str, str],
    rate_columns: tuple[str, str, str] | None,
) -> None:
    labels = (
        r"Electric-field error",
        r"Magnetic-field error",
        r"Combined-field error",
    )
    ylabels = (
        r"$\|E-E_h\|$",
        r"$\|H-H_h\|$",
        r"$\|U-U_h\|$",
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.8),
        constrained_layout=True,
    )
    mesh_sizes = sorted(
        {
            result.number("characteristic_h")
            for results in groups.values()
            for result in results
        }
    )

    for axis, label, ylabel, error_name, rate_name in zip(
        axes,
        labels,
        ylabels,
        error_columns,
        rate_columns or (None, None, None),
    ):
        for order, results in groups.items():
            h_values = [
                result.number("characteristic_h") for result in results
            ]
            errors = [positive(result.number(error_name)) for result in results]
            suffix = (
                final_rate_label(results[-1], rate_name)
                if rate_name is not None
                else ""
            )
            axis.plot(
                h_values,
                errors,
                marker="o",
                linewidth=1.8,
                label=f"$N={order}$" + suffix,
            )
        configure_log_axis(axis, label, ylabel, mesh_sizes)
        axis.legend(fontsize="small")

    figure.suptitle(f"{title}: {prefix}")
    save_figure(figure, output_stem, formats, dpi)


def plot_observed_rates(
    groups: dict[int, list[Result]],
    output_stem: Path,
    formats: list[str],
    dpi: int,
    title: str,
) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.8),
        sharey=True,
        constrained_layout=True,
    )
    specifications = (
        ("rate_electric", "Electric-field L2 rate"),
        ("rate_magnetic", "Magnetic-field L2 rate"),
        ("rate_total", "Combined-field L2 rate"),
    )

    for axis, (rate_name, subtitle) in zip(axes, specifications):
        mesh_levels = sorted(
            {
                result.integer("mesh_level")
                for results in groups.values()
                for result in results
                if result.optional_number(rate_name) is not None
            }
        )
        for order, results in groups.items():
            rated = [
                result
                for result in results
                if result.optional_number(rate_name) is not None
            ]
            levels = [
                result.integer("mesh_level") for result in rated
            ]
            rates = [
                result.number(rate_name) for result in rated
            ]
            axis.plot(
                levels,
                rates,
                marker="o",
                linewidth=1.8,
                label=f"$N={order}$",
            )
            axis.axhline(
                order + 1,
                linestyle="--",
                linewidth=1.0,
                alpha=0.45,
            )
        axis.set_title(subtitle)
        axis.set_xlabel("Fine-mesh level")
        axis.set_xticks(mesh_levels)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")

    axes[0].set_ylabel("Observed convergence rate")
    figure.suptitle(f"{title}: observed L2 rates")
    save_figure(figure, output_stem, formats, dpi)


def plot_conservation(
    groups: dict[int, list[Result]],
    output_stem: Path,
    formats: list[str],
    dpi: int,
    title: str,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        constrained_layout=True,
    )
    specifications = (
        ("energy_error", "Absolute energy error", r"$|\Delta\mathcal{E}|$"),
        (
            "relative_energy_error",
            "Relative energy drift",
            r"$|\Delta\mathcal{E}|/\mathcal{E}_0$",
        ),
        ("electric_charge", "Electric charge", r"$|Q_E|$"),
        ("magnetic_charge", "Magnetic charge", r"$|Q_H|$"),
    )
    mesh_sizes = sorted(
        {
            result.number("characteristic_h")
            for results in groups.values()
            for result in results
        }
    )

    for axis, (column, subtitle, ylabel) in zip(
        axes.flat,
        specifications,
    ):
        for order, results in groups.items():
            axis.plot(
                [
                    result.number("characteristic_h")
                    for result in results
                ],
                [positive(result.number(column)) for result in results],
                marker="o",
                linewidth=1.8,
                label=f"$N={order}$",
            )
        configure_log_axis(axis, subtitle, ylabel, mesh_sizes)
        axis.legend(fontsize="small")

    figure.suptitle(f"{title}: conservation diagnostics")
    save_figure(figure, output_stem, formats, dpi)


def plot_runtime(
    groups: dict[int, list[Result]],
    output_stem: Path,
    formats: list[str],
    dpi: int,
    title: str,
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.8),
        constrained_layout=True,
    )
    runtime_axis, imbalance_axis = axes

    for order, results in groups.items():
        elements = [result.integer("nelements") for result in results]
        runtime_axis.plot(
            elements,
            [result.number("elapsed_seconds") for result in results],
            marker="o",
            linewidth=1.8,
            label=f"$N={order}$",
        )
        imbalance_axis.plot(
            elements,
            [
                (
                    result.integer("max_owned_elements")
                    - result.integer("min_owned_elements")
                )
                / result.integer("max_owned_elements")
                for result in results
            ],
            marker="o",
            linewidth=1.8,
            label=f"$N={order}$",
        )

    runtime_axis.set_xscale("log")
    runtime_axis.set_yscale("log")
    element_counts = sorted(
        {
            result.integer("nelements")
            for results in groups.values()
            for result in results
        }
    )

    runtime_axis.set_title("Reported solve time (may include JIT effects)")
    runtime_axis.set_xlabel("Number of tetrahedra")
    runtime_axis.set_ylabel("Elapsed time [s]")
    runtime_axis.xaxis.set_major_locator(FixedLocator(element_counts))
    runtime_axis.xaxis.set_major_formatter(
        FixedFormatter([str(count) for count in element_counts])
    )
    runtime_axis.xaxis.set_minor_locator(NullLocator())
    runtime_axis.grid(True, which="both", alpha=0.3)
    runtime_axis.legend(fontsize="small")

    imbalance_axis.set_xscale("log")
    imbalance_axis.set_title("Element-count partition imbalance")
    imbalance_axis.set_xlabel("Number of tetrahedra")
    imbalance_axis.set_ylabel(r"$(N_{\max}-N_{\min})/N_{\max}$")
    imbalance_axis.xaxis.set_major_locator(FixedLocator(element_counts))
    imbalance_axis.xaxis.set_major_formatter(
        FixedFormatter([str(count) for count in element_counts])
    )
    imbalance_axis.xaxis.set_minor_locator(NullLocator())
    imbalance_axis.grid(True, which="both", alpha=0.3)
    imbalance_axis.legend(fontsize="small")

    figure.suptitle(f"{title}: computational diagnostics")
    save_figure(figure, output_stem, formats, dpi)


def save_figure(
    figure: plt.Figure,
    output_stem: Path,
    formats: list[str],
    dpi: int,
) -> None:
    for file_format in formats:
        figure.savefig(
            output_stem.with_suffix(f".{file_format}"),
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(figure)


def parse_formats(text: str) -> list[str]:
    formats = [
        entry.strip().lower()
        for entry in text.split(",")
        if entry.strip()
    ]
    if not formats:
        raise ValueError("At least one plot format is required.")
    supported = set(plt.gcf().canvas.get_supported_filetypes())
    plt.close()
    unsupported = sorted(set(formats) - supported)
    if unsupported:
        raise ValueError(
            "Unsupported plot formats: " + ", ".join(unsupported)
        )
    return formats


def generated_paths(
    output_dir: Path,
    prefix: str,
    formats: list[str],
) -> list[Path]:
    paths = [
        output_dir / f"{prefix}_convergence_table.csv",
        output_dir / f"{prefix}_convergence_table.tex",
    ]
    for plot_name in (
        "l2_convergence",
        "linf_convergence",
        "observed_rates",
        "conservation",
        "runtime",
    ):
        paths.extend(
            output_dir / f"{prefix}_{plot_name}.{file_format}"
            for file_format in formats
        )
    return paths


def main() -> None:
    args = parse_args()
    formats = parse_formats(args.formats)
    results = read_results(args.csv_paths)
    groups = group_results(results)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.csv_paths[0].parent / "convergence_postprocessing"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    write_table_csv(
        output_dir / f"{args.prefix}_convergence_table.csv",
        results,
    )
    write_latex_table(
        output_dir / f"{args.prefix}_convergence_table.tex",
        results,
        args.title,
    )
    plot_error_family(
        groups,
        output_dir / f"{args.prefix}_l2_convergence",
        formats,
        args.dpi,
        args.title,
        r"$L^2$ errors",
        (
            "l2_electric_error",
            "l2_magnetic_error",
            "l2_total_error",
        ),
        ("rate_electric", "rate_magnetic", "rate_total"),
    )
    plot_error_family(
        groups,
        output_dir / f"{args.prefix}_linf_convergence",
        formats,
        args.dpi,
        args.title,
        r"$L^\infty$ errors",
        (
            "linf_electric_error",
            "linf_magnetic_error",
            "linf_total_error",
        ),
        None,
    )
    plot_observed_rates(
        groups,
        output_dir / f"{args.prefix}_observed_rates",
        formats,
        args.dpi,
        args.title,
    )
    plot_conservation(
        groups,
        output_dir / f"{args.prefix}_conservation",
        formats,
        args.dpi,
        args.title,
    )
    plot_runtime(
        groups,
        output_dir / f"{args.prefix}_runtime",
        formats,
        args.dpi,
        args.title,
    )

    print(
        f"Processed {len(results)} cases for DG orders "
        f"{list(groups)} from {len(args.csv_paths)} input file(s)."
    )
    print(f"Generated files in: {output_dir.resolve()}")
    for path in generated_paths(output_dir, args.prefix, formats):
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
