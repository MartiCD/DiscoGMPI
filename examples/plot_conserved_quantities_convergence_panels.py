#!/usr/bin/env python3
"""Plot six conserved Maxwell quantities for p- and h-convergence runs.

For each convergence family, this script writes two publication-style 3x2
figures in PDF, PNG, and SVG format:

* signed quantities versus time on linear axes;
* absolute errors versus time on a linear x-axis and logarithmic y-axis.

Linear and angular momentum are vector-valued.  Their signed panels display
their Euclidean magnitudes, while their log-log panels display the Euclidean
norm of the error relative to the exact vector.

The panels labelled as electric and magnetic charge use, respectively, the
summed integrated-field diagnostics integrated_d_x/y/z and integrated_b_x/y/z.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    AutoMinorLocator,
    LogFormatterSciNotation,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    ScalarFormatter,
)
import numpy as np


SCALAR_QUANTITIES = (
    ("total_energy", "exact_total_energy", r"Electromagnetic energy $\mathcal{H}_h$"),
    ("optical_chirality", "exact_optical_chirality", r"Optical chirality $\chi_h$"),
)
VECTOR_QUANTITIES = (
    ("linear_momentum", r"Linear momentum $\|\mathbf{P}_h\|_2$"),
    ("angular_momentum", r"Angular momentum $\|\mathbf{L}_h\|_2$"),
)
INTEGRATED_FIELD_QUANTITIES = (
    ("d", r"Electric charge $Q_{E,h}$"),
    ("b", r"Magnetic charge $Q_{H,h}$"),
)
PANEL_LABELS = tuple(f"({letter})" for letter in "abcdef")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
COMPONENTS = "xyz"


def required_columns() -> set[str]:
    columns = {"time"}
    for numerical, exact, _ in SCALAR_QUANTITIES:
        columns.update((numerical, exact))
    for prefix, _ in VECTOR_QUANTITIES:
        for component in COMPONENTS:
            columns.update((f"{prefix}_{component}", f"exact_{prefix}_{component}"))
    for field, _ in INTEGRATED_FIELD_QUANTITIES:
        columns.update(f"integrated_{field}_{component}" for component in COMPONENTS)
    return columns


@dataclass(frozen=True)
class Series:
    identifier: int
    label: str
    path: Path
    data: dict[str, np.ndarray]

    @property
    def time(self) -> np.ndarray:
        return self.data["time"]

    def scalar(self, name: str) -> np.ndarray:
        return self.data[name]

    def vector(self, prefix: str, exact: bool = False) -> np.ndarray:
        stem = f"exact_{prefix}" if exact else prefix
        return np.column_stack([self.data[f"{stem}_{component}"] for component in COMPONENTS])

    def integrated_field_sum(self, field: str) -> np.ndarray:
        return sum(self.data[f"integrated_{field}_{component}"] for component in COMPONENTS)


def comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(entry.strip()) for entry in value.split(",") if entry.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list.") from error
    if not values or len(set(values)) != len(values) or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("Values must be unique non-negative integers.")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output"))
    parser.add_argument("--p-prefix", default="periodic_cuboid_cf_structured_p")
    parser.add_argument("--p-orders", type=comma_separated_ints, default=(1, 2, 3, 4))
    parser.add_argument("--h-prefix", default="periodic_cuboid_cf_structured_p1_h")
    parser.add_argument("--h-levels", type=comma_separated_ints, default=(0, 1, 2, 3))
    parser.add_argument(
        "--family",
        choices=("both", "p", "h"),
        default="both",
        help="Convergence family to process. Default: both.",
    )
    parser.add_argument(
        "--diagnostics-file",
        default="quadrature_diagnostics.csv",
        help="Filename in the run directory or its diagnostics subdirectory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/conserved_quantities_convergence_panels"),
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Maximum rendered samples per run; use 0 to read every row.",
    )
    parser.add_argument("--figure-width", type=float, default=7.2)
    parser.add_argument("--figure-height", type=float, default=8.3)
    parser.add_argument("--title-prefix", default="")
    args = parser.parse_args(argv)
    if args.dpi < 72:
        parser.error("--dpi must be at least 72.")
    if args.max_points < 0 or args.max_points == 1:
        parser.error("--max-points must be 0 or at least 2.")
    if args.figure_width <= 0 or args.figure_height <= 0:
        parser.error("Figure dimensions must be positive.")
    return args


def diagnostics_path(case_dir: Path, filename: str) -> Path:
    candidates = (case_dir / "diagnostics" / filename, case_dir / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing diagnostics file; checked: {', '.join(map(str, candidates))}")


def read_series(
    path: Path,
    identifier: int,
    label: str,
    max_points: int | None = None,
) -> Series:
    """Read one diagnostics series, optionally sampling it for rendering.

    Long-time invariance runs can contain several million rows.  Passing a
    positive ``max_points`` keeps uniformly spaced rows plus the final row,
    which bounds plotting memory without modifying the source diagnostics.
    """
    if max_points is not None and max_points < 2:
        raise ValueError("max_points must be at least 2 or None")

    row_count = None
    stride = 1
    if max_points is not None:
        with path.open(newline="", encoding="utf-8") as stream:
            row_count = sum(1 for _ in csv.DictReader(stream))
        stride = max(1, (row_count + max_points - 1) // max_points)

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        available = set(reader.fieldnames or ())
        missing = sorted(required_columns() - available)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        columns = {name: [] for name in required_columns()}
        for row_index, row in enumerate(reader):
            if stride > 1 and row_index % stride != 0 and row_index != row_count - 1:
                continue
            row_number = row_index + 2
            try:
                for name in columns:
                    columns[name].append(float(row[name]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric value in {path} at row {row_number}.") from error
    data = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    if data["time"].size < 2:
        raise ValueError(f"{path} must contain at least two samples.")
    if not all(np.all(np.isfinite(values)) for values in data.values()):
        raise ValueError(f"{path} contains non-finite values.")
    if np.any(np.diff(data["time"]) <= 0):
        raise ValueError(f"Times in {path} must be strictly increasing.")
    return Series(identifier, label, path, data)


def load_family(
    root: Path,
    prefix: str,
    identifiers: Sequence[int],
    labels: Sequence[str],
    diagnostics_file: str,
    max_points: int | None = None,
) -> list[Series]:
    return [
        read_series(
            diagnostics_path(root / f"{prefix}{identifier}", diagnostics_file),
            identifier,
            label,
            max_points,
        )
        for identifier, label in zip(identifiers, labels)
    ]


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.2,
        "axes.labelsize": 9.8,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def panel_axis(axis: plt.Axes, label: str, ylabel: str, bottom: bool) -> None:
    axis.set_ylabel(ylabel)
    axis.set_xlabel(r"Time $t$" if bottom else "")
    axis.grid(which="major", color="0.86", linewidth=0.52)
    axis.grid(which="minor", color="0.93", linewidth=0.3)
    axis.set_axisbelow(True)
    axis.tick_params(which="both", top=True, right=True)
    axis.margins(x=0.015)
    axis.text(
        0.025, 0.96, label, transform=axis.transAxes,
        va="top", fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="0.82", alpha=0.92, pad=1.5),
    )


def linear_format(axis: plt.Axes) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    formatter.set_useOffset(False)
    axis.yaxis.set_major_formatter(formatter)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.xaxis.set_minor_locator(AutoMinorLocator())


def positive_log_values(time: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (values > 0.0) & np.isfinite(values)
    return time[mask], values[mask]


def log_format(axis: plt.Axes) -> None:
    axis.set_yscale("log")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    # Label every decade explicitly in scientific notation. This is
    # especially important for the charge panels, whose errors can be O(1e-14).
    axis.yaxis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=True))
    axis.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    axis.yaxis.set_minor_formatter(NullFormatter())


def set_relevant_log_ylim(axis: plt.Axes, values: Sequence[np.ndarray]) -> None:
    """Set a decade-aligned y-range spanning at least five orders of magnitude."""
    positive_arrays = [array[array > 0.0] for array in values if np.any(array > 0.0)]
    if not positive_arrays:
        return
    positive = np.concatenate(positive_arrays)
    # Leave one full decade above the largest sample so the curve is not
    # pressed against the top spine.
    upper = 10.0 ** (np.ceil(np.log10(np.max(positive))) + 1.0)
    lower = min(
        upper / 10.0**5,
        10.0 ** np.floor(np.log10(np.min(positive))),
    )
    axis.set_ylim(lower, upper)


def scalar_error(series: Series, numerical: str, exact: str) -> np.ndarray:
    return np.abs(series.scalar(numerical) - series.scalar(exact))


def vector_error(series: Series, prefix: str) -> np.ndarray:
    return np.linalg.norm(series.vector(prefix) - series.vector(prefix, exact=True), axis=1)


def plot_signed(
    series: Sequence[Series],
    output_stem: Path,
    figure_size: tuple[float, float],
    title: str,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=figure_size, sharex=True)
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.065, top=0.89,
                           wspace=0.30, hspace=0.16)
    axes = axes.ravel()
    panel_specs: tuple[tuple[str, str, str], ...] = (
        ("scalar", SCALAR_QUANTITIES[0][0], SCALAR_QUANTITIES[0][2]),
        ("scalar", SCALAR_QUANTITIES[1][0], SCALAR_QUANTITIES[1][2]),
        ("vector", VECTOR_QUANTITIES[0][0], VECTOR_QUANTITIES[0][1]),
        ("vector", VECTOR_QUANTITIES[1][0], VECTOR_QUANTITIES[1][1]),
        ("integrated", INTEGRATED_FIELD_QUANTITIES[0][0],
         INTEGRATED_FIELD_QUANTITIES[0][1]),
        ("integrated", INTEGRATED_FIELD_QUANTITIES[1][0],
         INTEGRATED_FIELD_QUANTITIES[1][1]),
    )
    for panel, (kind, key, ylabel) in enumerate(panel_specs):
        axis = axes[panel]
        if kind == "scalar":
            for run_index, item in enumerate(series):
                axis.plot(item.time, item.scalar(key), color=COLORS[run_index % len(COLORS)],
                          lw=1.25, label=item.label)
        elif kind == "vector":
            for run_index, item in enumerate(series):
                magnitude = np.linalg.norm(item.vector(key), axis=1)
                axis.plot(item.time, magnitude,
                          color=COLORS[run_index % len(COLORS)],
                          lw=1.25, label=item.label)
        else:
            for run_index, item in enumerate(series):
                axis.plot(item.time, item.integrated_field_sum(key),
                          color=COLORS[run_index % len(COLORS)],
                          lw=1.25, label=item.label)
        axis.axhline(0.0, color="0.5", lw=0.5, zorder=0)
        linear_format(axis)
        panel_axis(axis, PANEL_LABELS[panel], ylabel, panel >= 4)
    add_run_legend(figure, series, title)
    save_all_formats(figure, output_stem, dpi)


def plot_logy(
    series: Sequence[Series],
    output_stem: Path,
    figure_size: tuple[float, float],
    title: str,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=figure_size, sharex=True)
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.065, top=0.89,
                           wspace=0.30, hspace=0.16)
    axes = axes.ravel()
    specs: tuple[tuple[str, str, str], ...] = (
        ("scalar", "total_energy", r"Energy error $|\mathcal{H}_h-\mathcal{H}|$"),
        ("scalar", "optical_chirality", r"Chirality error $|\chi_h-\chi|$"),
        ("vector", "linear_momentum", r"Momentum error $\|\mathbf{P}_h-\mathbf{P}\|_2$"),
        ("vector", "angular_momentum", r"Angular-momentum error $\|\mathbf{L}_h-\mathbf{L}\|_2$"),
        ("integrated", "d", r"Electric-charge error $|Q_{E,h}-Q_E|$"),
        ("integrated", "b", r"Magnetic-charge error $|Q_{H,h}-Q_H|$"),
    )
    exact_names = {numerical: exact for numerical, exact, _ in SCALAR_QUANTITIES}
    for panel, (kind, key, ylabel) in enumerate(specs):
        axis = axes[panel]
        plotted = False
        panel_values: list[np.ndarray] = []
        for run_index, item in enumerate(series):
            if kind == "scalar":
                values = scalar_error(item, key, exact_names[key])
            elif kind == "vector":
                values = vector_error(item, key)
            else:
                values = np.abs(item.integrated_field_sum(key))
            time, positive_values = positive_log_values(item.time, values)
            if time.size:
                panel_values.append(positive_values)
                axis.plot(time, positive_values, color=COLORS[run_index % len(COLORS)],
                          lw=1.25, label=item.label)
                plotted = True
        log_format(axis)
        set_relevant_log_ylim(axis, panel_values)
        if not plotted:
            axis.text(0.5, 0.5, "All sampled errors are zero", transform=axis.transAxes,
                      ha="center", va="center", fontsize=8)
        panel_axis(axis, PANEL_LABELS[panel], ylabel, panel >= 4)
    add_run_legend(figure, series, title)
    save_all_formats(figure, output_stem, dpi)


def add_run_legend(
    figure: plt.Figure,
    series: Sequence[Series],
    title: str,
) -> None:
    handles = [Line2D([0], [0], color=COLORS[index % len(COLORS)], lw=1.5,
                      label=item.label) for index, item in enumerate(series)]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
                  ncol=min(4, len(series)), frameon=True, title=title,
                  columnspacing=1.2, handlelength=2.5)


def save_all_formats(figure: plt.Figure, stem: Path, dpi: int = 600) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        options = {"dpi": dpi} if extension == "png" else {}
        figure.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight", **options)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_style()
    p_labels = tuple(rf"$p={order}$" for order in args.p_orders)
    h_labels = tuple(r"$h_0$" if level == 0 else rf"$h_0/{2 ** level}$"
                     for level in args.h_levels)
    size = (args.figure_width, args.figure_height)
    prefix = f"{args.title_prefix} " if args.title_prefix else ""
    outputs = []
    if args.family in ("both", "p"):
        p_series = load_family(args.root, args.p_prefix, args.p_orders, p_labels,
                               args.diagnostics_file, args.max_points or None)
        outputs.extend((
            (plot_signed, p_series, "p_convergence_signed_quantities", prefix + r"$p$ convergence"),
            (plot_logy, p_series, "p_convergence_logy_errors", prefix + r"$p$ convergence"),
        ))
    if args.family in ("both", "h"):
        h_series = load_family(args.root, args.h_prefix, args.h_levels, h_labels,
                               args.diagnostics_file, args.max_points or None)
        outputs.extend((
            (plot_signed, h_series, "h_convergence_signed_quantities", prefix + r"$h$ convergence"),
            (plot_logy, h_series, "h_convergence_logy_errors", prefix + r"$h$ convergence"),
        ))
    for plotter, series, stem, legend_title in outputs:
        plotter(series, args.output_dir / stem, size, legend_title, args.dpi)
    for _, _, stem, _ in outputs:
        for extension in ("pdf", "png", "svg"):
            print(f"Wrote {args.output_dir / f'{stem}.{extension}'}")


if __name__ == "__main__":
    main()
