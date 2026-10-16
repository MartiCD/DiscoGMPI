#!/usr/bin/env python3

"""Create publication-style invariant panels for PEC alternating-flux runs."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Avoid slow/warning-prone Matplotlib cache creation on shared filesystems.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.ticker import (
    AutoMinorLocator,
    LogFormatterMathtext,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    ScalarFormatter,
)


REQUIRED_COLUMNS = {
    "time",
    "total_energy",
    "exact_total_energy",
    "optical_chirality",
    "exact_optical_chirality",
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
DIVERGENCE_COLUMNS = {
    "electric_divergence_l2",
    "magnetic_divergence_l2",
    "electric_divergence_linf",
    "magnetic_divergence_linf",
}

DEFAULT_ORDERS = (1, 2, 3, 4)
DEFAULT_FIGURE_WIDTH = 7.2
DEFAULT_FIGURE_HEIGHT = 8.15

# Okabe-Ito palette: colorblind-safe and robust in print.
COLOR_CYCLE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)
# Data curves are line-only; colors distinguish the series.
MARKER_CYCLE = (None,)
PANEL_LABELS = tuple(f"({letter})" for letter in "abcdefghijklmnopqrstuvwxyz")
# Dash-only line patterns. The h-convergence wrapper reuses this module, so
# these styles distinguish either DG orders p or mesh levels h.
LINESTYLE_CYCLE = (
    (0, (10.0, 2.5)),
    (0, (7.0, 2.0)),
    (0, (4.5, 1.6)),
    (0, (12.0, 2.2, 5.0, 2.2)),
    (0, (8.0, 1.8, 4.0, 1.8)),
    (0, (14.0, 2.5, 3.0, 2.5)),
)


@dataclass(frozen=True)
class DiagnosticsSeries:
    order: int
    path: Path
    columns: dict[str, list[float]]
    label: str

    @property
    def time(self) -> list[float]:
        return self.columns["time"]

    def values(self, name: str) -> list[float]:
        return self.columns[name]

    def magnitude(self, prefix: str) -> list[float]:
        x_values = self.columns[f"{prefix}_x"]
        y_values = self.columns[f"{prefix}_y"]
        z_values = self.columns[f"{prefix}_z"]
        return [
            math.sqrt(x_value * x_value + y_value * y_value + z_value * z_value)
            for x_value, y_value, z_value in zip(x_values, y_values, z_values)
        ]

    def absolute_error(self, numerical: str, exact: str) -> list[float]:
        numerical_values = self.columns[numerical]
        exact_values = self.columns[exact]
        return [
            abs(numerical_value - exact_value)
            for numerical_value, exact_value in zip(
                numerical_values,
                exact_values,
            )
        ]

    def vector_error(self, prefix: str) -> list[float]:
        x_values = self.columns[f"{prefix}_x"]
        y_values = self.columns[f"{prefix}_y"]
        z_values = self.columns[f"{prefix}_z"]
        exact_x_values = self.columns[f"exact_{prefix}_x"]
        exact_y_values = self.columns[f"exact_{prefix}_y"]
        exact_z_values = self.columns[f"exact_{prefix}_z"]
        return [
            math.sqrt(
                (x_value - exact_x_value) ** 2
                + (y_value - exact_y_value) ** 2
                + (z_value - exact_z_value) ** 2
            )
            for (
                x_value,
                y_value,
                z_value,
                exact_x_value,
                exact_y_value,
                exact_z_value,
            ) in zip(
                x_values,
                y_values,
                z_values,
                exact_x_values,
                exact_y_values,
                exact_z_values,
            )
        ]


@dataclass(frozen=True)
class PlotQuantity:
    ylabel: str
    getter: Callable[[DiagnosticsSeries], list[float]]
    symlog: bool = False
    log: bool = False


def parse_int_list(value: str) -> tuple[int, ...]:
    entries = [entry.strip() for entry in value.split(",")]
    try:
        values = tuple(int(entry) for entry in entries if entry)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Expected a comma-separated integer list, got '{value}'."
        ) from error

    if not values:
        raise argparse.ArgumentTypeError("At least one order is required.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Orders must be unique.")
    if any(order < 0 for order in values):
        raise argparse.ArgumentTypeError("Orders must be non-negative.")
    return values


def parse_string_list(value: str) -> tuple[str, ...]:
    entries = tuple(entry.strip() for entry in value.split(",") if entry.strip())
    if not entries:
        raise argparse.ArgumentTypeError("At least one label is required.")
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-ready invariant and momentum-component "
            "panels from DiscoGMPI quadrature_diagnostics.csv files."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output"),
        help="Root directory containing per-order output directories.",
    )
    parser.add_argument(
        "--prefix",
        default="pec_box_af_p",
        help="Directory prefix before the polynomial order.",
    )
    parser.add_argument(
        "--orders",
        type=parse_int_list,
        default=DEFAULT_ORDERS,
        help="Comma-separated DG orders to plot. Default: 1,2,3,4.",
    )
    parser.add_argument(
        "--series-labels",
        type=parse_string_list,
        default=None,
        help=(
            "Comma-separated legend labels. When omitted, labels are "
            "formatted as $p=N$."
        ),
    )
    parser.add_argument(
        "--legend-title",
        default="DG order",
        help="Legend title. Default: DG order.",
    )
    parser.add_argument(
        "--series-id-label",
        default="p",
        help=(
            "Short label used when reporting loaded series. "
            "Default: p."
        ),
    )
    parser.add_argument(
        "--diagnostics-file",
        default="quadrature_diagnostics.csv",
        help="Diagnostics CSV filename inside each order directory.",
    )
    parser.add_argument(
        "--summary-output",
        "--output",
        dest="summary_output",
        type=Path,
        default=Path("output/pec_box_af_invariants_panel.pdf"),
        help=(
            "Output path for the 2x3 invariants panel. "
            "Default: output/pec_box_af_invariants_panel.pdf."
        ),
    )
    parser.add_argument(
        "--components-output",
        type=Path,
        default=Path("output/pec_box_af_momentum_components_panel.pdf"),
        help=(
            "Output path for the 2x3 momentum-components panel. "
            "Default: output/pec_box_af_momentum_components_panel.pdf."
        ),
    )
    parser.add_argument(
        "--error-output",
        type=Path,
        default=Path("output/pec_box_af_invariants_error_panel.pdf"),
        help=(
            "Output path for the 2x3 conserved-invariant error panel. "
            "Default: output/pec_box_af_invariants_error_panel.pdf."
        ),
    )
    parser.add_argument(
        "--divergence-output",
        type=Path,
        default=Path("output/pec_box_af_divergence_constraints_panel.pdf"),
        help=(
            "Output path for the 2x2 divergence-constraint diagnostics panel. "
            "Default: output/pec_box_af_divergence_constraints_panel.pdf."
        ),
    )
    parser.add_argument(
        "--summary-png-output",
        type=Path,
        default=None,
        help="Optional PNG path for the invariants panel.",
    )
    parser.add_argument(
        "--components-png-output",
        type=Path,
        default=None,
        help="Optional PNG path for the momentum-components panel.",
    )
    parser.add_argument(
        "--error-png-output",
        type=Path,
        default=None,
        help="Optional PNG path for the conserved-invariant error panel.",
    )
    parser.add_argument(
        "--divergence-png-output",
        type=Path,
        default=None,
        help="Optional PNG path for the divergence-constraint diagnostics panel.",
    )
    parser.add_argument(
        "--skip-divergence",
        action="store_true",
        help=(
            "Do not write the divergence-constraint panel. Use this for old "
            "CSV files generated before the divergence diagnostics were added."
        ),
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Only write vector outputs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution in dots per inch. Default: 600.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=DEFAULT_FIGURE_WIDTH,
        help=(
            "Figure width in inches. Default: "
            f"{DEFAULT_FIGURE_WIDTH}, tuned for two-column journal figures."
        ),
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=DEFAULT_FIGURE_HEIGHT,
        help=(
            "Figure height in inches. Default: "
            f"{DEFAULT_FIGURE_HEIGHT}."
        ),
    )
    parser.add_argument(
        "--legend-columns",
        type=int,
        default=0,
        help=(
            "Number of columns in the figure-level legend. "
            "Default: automatic."
        ),
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=1.45,
        help="Line width in points. Default: 1.45.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=3.9,
        help="Marker size in points. Default: 3.9.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help=(
            "Optional figure title. Omit for manuscript-ready panels where "
            "the caption carries the description."
        ),
    )
    parser.add_argument(
        "--charge-scale",
        choices=("linear", "symlog"),
        default="linear",
        help="Y-axis scale for electric/magnetic charge panels. Default: linear.",
    )
    parser.add_argument(
        "--momentum-scale",
        choices=("linear", "symlog"),
        default="linear",
        help="Y-axis scale for momentum-component panels. Default: linear.",
    )
    parser.add_argument(
        "--error-scale",
        choices=("log",),
        default="log",
        help="Y-axis scale for invariant-error panels. Fixed to log.",
    )
    parser.add_argument(
        "--usetex",
        action="store_true",
        help="Use LaTeX for text rendering. Requires a working LaTeX install.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures interactively after writing files.",
    )
    return parser.parse_args()


def configure_matplotlib(usetex: bool) -> None:
    plt.rcParams.update(
        {
            "text.usetex": usetex,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 9.4,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.0,
            "legend.fontsize": 8.9,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "axes.linewidth": 0.9,
            "axes.axisbelow": True,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "figure.dpi": 150,
            "lines.linewidth": 1.45,
            "lines.markersize": 3.9,
            "lines.markeredgewidth": 0.75,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "butt",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.8,
            "ytick.major.size": 4.8,
            "xtick.minor.size": 2.6,
            "ytick.minor.size": 2.6,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.75,
            "ytick.minor.width": 0.75,
            "legend.frameon": True,
            "legend.framealpha": 0.96,
            "legend.edgecolor": "0.75",
            "legend.handlelength": 2.3,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_series(path: Path, order: int, label: str) -> DiagnosticsSeries:
    if not path.is_file():
        raise FileNotFoundError(f"Missing diagnostics file: {path}")

    columns = {name: [] for name in REQUIRED_COLUMNS}

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        available = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - available)
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(missing)}"
            )
        for name in sorted(DIVERGENCE_COLUMNS & available):
            columns[name] = []

        for row_number, row in enumerate(reader, start=2):
            try:
                for name in columns:
                    columns[name].append(float(row[name]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value in {path} at row {row_number}."
                ) from error

    if not columns["time"]:
        raise ValueError(f"No data rows found in {path}.")

    return DiagnosticsSeries(order=order, path=path, columns=columns, label=label)


def resolve_diagnostics_path(case_dir: Path, diagnostics_file: str) -> Path:
    diagnostics_path = Path(diagnostics_file)
    candidates = [case_dir / diagnostics_path]
    if len(diagnostics_path.parts) == 1:
        candidates.append(case_dir / "diagnostics" / diagnostics_path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_all_series(
    root: Path,
    prefix: str,
    orders: tuple[int, ...],
    diagnostics_file: str,
    labels: tuple[str, ...] | None,
) -> list[DiagnosticsSeries]:
    if labels is None:
        labels = tuple(rf"$p={order}$" for order in orders)
    if len(labels) != len(orders):
        raise ValueError(
            "The number of --series-labels entries must match --orders."
        )
    return [
        read_series(
            resolve_diagnostics_path(root / f"{prefix}{order}", diagnostics_file),
            order,
            label,
        )
        for order, label in zip(orders, labels)
    ]


def require_columns(
    series: list[DiagnosticsSeries],
    columns: set[str],
    panel_name: str,
) -> None:
    for item in series:
        missing = sorted(column for column in columns if column not in item.columns)
        if missing:
            raise ValueError(
                f"Cannot write {panel_name}: {item.path} is missing "
                f"columns {', '.join(missing)}. Regenerate the run with a "
                "DiscoGMPI version that writes divergence diagnostics, or pass "
                "--skip-divergence."
            )


def marker_spacing(values: list[float]) -> int:
    return max(1, len(values) // 12)


def apply_scientific_formatter(axis: plt.Axes) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    formatter.set_useOffset(False)
    axis.yaxis.set_major_formatter(formatter)


def style_axis(
    axis: plt.Axes,
    panel_label: str,
    ylabel: str,
    show_xlabel: bool,
) -> None:
    axis.set_xlabel(r"Time $t$" if show_xlabel else "")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="major", color="0.86", linestyle="-", linewidth=0.58)
    axis.grid(True, which="minor", color="0.93", linestyle="-", linewidth=0.32)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    if axis.get_yscale() == "log":
        axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        axis.yaxis.set_minor_locator(
            LogLocator(
                base=10.0,
                subs=(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
                numticks=150,
            )
        )
        axis.yaxis.set_minor_formatter(NullFormatter())
    else:
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    if axis.get_yscale() == "linear":
        axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.tick_params(top=True, right=True, pad=2.5)
    axis.margins(x=0.01)
    if axis.get_yscale() != "log":
        apply_scientific_formatter(axis)
    for spine in axis.spines.values():
        spine.set_linewidth(0.9)
    axis.text(
        0.025,
        0.955,
        panel_label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "0.82",
            "alpha": 0.96,
        },
    )


def plot_quantity(
    axis: plt.Axes,
    series: list[DiagnosticsSeries],
    quantity: PlotQuantity,
    panel_label: str,
    show_xlabel: bool,
    line_width: float,
    marker_size: float,
) -> None:
    series_values = [
        (item, quantity.getter(item))
        for item in series
    ]
    log_floor = 1.0e-16
    if quantity.log:
        positive_values = [
            value
            for _, values in series_values
            for value in values
            if math.isfinite(value) and value > 0.0
        ]
        if positive_values:
            log_floor = 10.0 ** (math.floor(math.log10(min(positive_values))) - 1)

    plotted_values: list[float] = []
    for index, item in enumerate(series):
        color = COLOR_CYCLE[index % len(COLOR_CYCLE)]
        marker = MARKER_CYCLE[index % len(MARKER_CYCLE)]
        linestyle = LINESTYLE_CYCLE[index % len(LINESTYLE_CYCLE)]
        values = series_values[index][1]
        if quantity.log:
            values = [
                value if math.isfinite(value) and value > 0.0 else log_floor
                for value in values
            ]
            plotted_values.extend(values)
        axis.plot(
            item.time,
            values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=marker_spacing(item.time),
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.85,
            markersize=marker_size,
            linewidth=line_width,
            alpha=0.98,
            zorder=2.0 + 0.05 * index,
            label=item.label,
        )
    if quantity.symlog:
        axis.set_yscale("symlog", linthresh=1.0e-12)
    if quantity.log:
        axis.set_yscale("log")
        finite_positive_values = [
            value
            for value in plotted_values
            if math.isfinite(value) and value > 0.0
        ]
        if finite_positive_values:
            ymin = 10.0 ** math.floor(math.log10(min(finite_positive_values)))
            ymax = 10.0 ** math.ceil(math.log10(max(finite_positive_values)))
            if ymin >= ymax:
                ymin /= 10.0
                ymax *= 10.0
            axis.set_ylim(ymin, ymax)
    style_axis(axis, panel_label, quantity.ylabel, show_xlabel)


def write_figure(
    figure: plt.Figure,
    output: Path,
    png_output: Path | None,
    dpi: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        facecolor="white",
        metadata={"Creator": "DiscoGMPI invariant-panel postprocessor"},
    )

    if png_output is not None:
        png_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(png_output, dpi=dpi, facecolor="white")


def summary_quantities(charge_scale: str) -> tuple[PlotQuantity, ...]:
    charges_symlog = charge_scale == "symlog"
    return (
        PlotQuantity(
            r"Total energy $\mathcal{H}_h$",
            lambda item: item.values("total_energy"),
        ),
        PlotQuantity(
            r"Optical chirality $\chi_h$",
            lambda item: item.values("optical_chirality"),
        ),
        PlotQuantity(
            r"Linear momentum $\|\mathbf{P}_h\|_2$",
            lambda item: item.magnitude("linear_momentum"),
        ),
        PlotQuantity(
            r"Angular momentum $\|\mathbf{L}_h\|_2$",
            lambda item: item.magnitude("angular_momentum"),
        ),
        PlotQuantity(
            r"Electric charge $\int_{\Omega}\nabla\!\cdot(\varepsilon E_h)\,dx$",
            lambda item: item.values("electric_charge"),
            symlog=charges_symlog,
        ),
        PlotQuantity(
            r"Magnetic charge $\int_{\Omega}\nabla\!\cdot(\mu H_h)\,dx$",
            lambda item: item.values("magnetic_charge"),
            symlog=charges_symlog,
        ),
    )


def component_quantities(momentum_scale: str) -> tuple[PlotQuantity, ...]:
    symlog = momentum_scale == "symlog"
    return (
        PlotQuantity(
            r"Linear momentum $P_x$",
            lambda item: item.values("linear_momentum_x"),
            symlog=symlog,
        ),
        PlotQuantity(
            r"Angular momentum $L_x$",
            lambda item: item.values("angular_momentum_x"),
            symlog=symlog,
        ),
        PlotQuantity(
            r"Linear momentum $P_y$",
            lambda item: item.values("linear_momentum_y"),
            symlog=symlog,
        ),
        PlotQuantity(
            r"Angular momentum $L_y$",
            lambda item: item.values("angular_momentum_y"),
            symlog=symlog,
        ),
        PlotQuantity(
            r"Linear momentum $P_z$",
            lambda item: item.values("linear_momentum_z"),
            symlog=symlog,
        ),
        PlotQuantity(
            r"Angular momentum $L_z$",
            lambda item: item.values("angular_momentum_z"),
            symlog=symlog,
        ),
    )


def error_quantities(error_scale: str) -> tuple[PlotQuantity, ...]:
    log_scale = error_scale == "log"
    return (
        PlotQuantity(
            r"Energy error $|\mathcal{H}-\mathcal{H}_h|$",
            lambda item: item.absolute_error("total_energy", "exact_total_energy"),
            log=log_scale,
        ),
        PlotQuantity(
            r"Chirality error $|\chi-\chi_h|$",
            lambda item: item.absolute_error(
                "optical_chirality",
                "exact_optical_chirality",
            ),
            log=log_scale,
        ),
        PlotQuantity(
            r"Linear momentum error $\|\mathbf{P}-\mathbf{P}_h\|_2$",
            lambda item: item.vector_error("linear_momentum"),
            log=log_scale,
        ),
        PlotQuantity(
            r"Angular momentum error $\|\mathbf{L}-\mathbf{L}_h\|_2$",
            lambda item: item.vector_error("angular_momentum"),
            log=log_scale,
        ),
        PlotQuantity(
            r"Electric-charge error $|Q_E-Q_{E,h}|$",
            lambda item: item.absolute_error(
                "electric_charge",
                "exact_electric_charge",
            ),
            log=log_scale,
        ),
        PlotQuantity(
            r"Magnetic-charge error $|Q_H-Q_{H,h}|$",
            lambda item: item.absolute_error(
                "magnetic_charge",
                "exact_magnetic_charge",
            ),
            log=log_scale,
        ),
    )


def divergence_quantities() -> tuple[PlotQuantity, ...]:
    return (
        PlotQuantity(
            r"$\|\nabla_h\!\cdot(\varepsilon\mathbf{E}_h)\|_{L^2(\Omega)}$",
            lambda item: item.values("electric_divergence_l2"),
            log=True,
        ),
        PlotQuantity(
            r"$\|\nabla_h\!\cdot(\mu\mathbf{H}_h)\|_{L^2(\Omega)}$",
            lambda item: item.values("magnetic_divergence_l2"),
            log=True,
        ),
        PlotQuantity(
            r"$\|\nabla_h\!\cdot(\varepsilon\mathbf{E}_h)\|_{L^\infty(\Omega)}$",
            lambda item: item.values("electric_divergence_linf"),
            log=True,
        ),
        PlotQuantity(
            r"$\|\nabla_h\!\cdot(\mu\mathbf{H}_h)\|_{L^\infty(\Omega)}$",
            lambda item: item.values("magnetic_divergence_linf"),
            log=True,
        ),
    )


def plot_grid_panel(
    series: list[DiagnosticsSeries],
    quantities: tuple[PlotQuantity, ...],
    output: Path,
    png_output: Path | None,
    dpi: int,
    show: bool,
    legend_title: str,
    figure_width: float,
    figure_height: float,
    legend_columns: int,
    line_width: float,
    marker_size: float,
    title: str | None,
) -> None:
    nquantities = len(quantities)
    if nquantities == 4:
        nrows, ncols = 2, 2
    elif nquantities == 6:
        nrows, ncols = 3, 2
    else:
        raise ValueError(f"Unsupported panel size: {nquantities} quantities.")

    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figure_width, figure_height),
        sharex=True,
    )
    subplot_top = 0.855 if title else 0.9
    legend_y = 0.935 if title else 0.985
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.065,
        top=subplot_top,
        wspace=0.3,
        hspace=0.16,
    )

    flat_axes = tuple(axes.ravel())
    first_bottom_axis = (nrows - 1) * ncols
    for index, (axis, quantity) in enumerate(zip(flat_axes, quantities)):
        plot_quantity(
            axis,
            series,
            quantity,
            PANEL_LABELS[index],
            index >= first_bottom_axis,
            line_width,
            marker_size,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    columns = legend_columns if legend_columns > 0 else min(max(len(series), 1), 4)
    legend = figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        title=legend_title,
        title_fontsize=9.0,
        ncol=columns,
        columnspacing=1.05,
        handletextpad=0.5,
    )
    legend.get_frame().set_linewidth(0.8)

    if title:
        figure.suptitle(title, y=0.995, fontsize=10.5)

    figure.align_ylabels(flat_axes)

    write_figure(figure, output, png_output, dpi)

    if show:
        plt.show()
    else:
        plt.close(figure)


def png_default(vector_output: Path, enabled: bool, explicit: Path | None) -> Path | None:
    if not enabled:
        return None
    return explicit or vector_output.with_suffix(".png")


def main() -> None:
    args = parse_args()
    configure_matplotlib(args.usetex)
    series = load_all_series(
        args.root,
        args.prefix,
        args.orders,
        args.diagnostics_file,
        args.series_labels,
    )

    write_png = not args.no_png
    summary_png = png_default(
        args.summary_output,
        write_png,
        args.summary_png_output,
    )
    components_png = png_default(
        args.components_output,
        write_png,
        args.components_png_output,
    )
    error_png = png_default(
        args.error_output,
        write_png,
        args.error_png_output,
    )
    divergence_png = png_default(
        args.divergence_output,
        write_png,
        args.divergence_png_output,
    )

    plot_grid_panel(
        series,
        summary_quantities(args.charge_scale),
        args.summary_output,
        summary_png,
        args.dpi,
        args.show,
        args.legend_title,
        args.figure_width,
        args.figure_height,
        args.legend_columns,
        args.line_width,
        args.marker_size,
        args.title,
    )
    plot_grid_panel(
        series,
        component_quantities(args.momentum_scale),
        args.components_output,
        components_png,
        args.dpi,
        args.show,
        args.legend_title,
        args.figure_width,
        args.figure_height,
        args.legend_columns,
        args.line_width,
        args.marker_size,
        args.title,
    )
    plot_grid_panel(
        series,
        error_quantities(args.error_scale),
        args.error_output,
        error_png,
        args.dpi,
        args.show,
        args.legend_title,
        args.figure_width,
        args.figure_height,
        args.legend_columns,
        args.line_width,
        args.marker_size,
        args.title,
    )
    if not args.skip_divergence:
        require_columns(series, DIVERGENCE_COLUMNS, "divergence panel")
        divergence_height = args.figure_height * (2.0 / 3.0)
        plot_grid_panel(
            series,
            divergence_quantities(),
            args.divergence_output,
            divergence_png,
            args.dpi,
            args.show,
            args.legend_title,
            args.figure_width,
            divergence_height,
            args.legend_columns,
            args.line_width,
            args.marker_size,
            args.title,
        )

    print(f"Wrote {args.summary_output}")
    if summary_png is not None:
        print(f"Wrote {summary_png}")
    print(f"Wrote {args.components_output}")
    if components_png is not None:
        print(f"Wrote {components_png}")
    print(f"Wrote {args.error_output}")
    if error_png is not None:
        print(f"Wrote {error_png}")
    if not args.skip_divergence:
        print(f"Wrote {args.divergence_output}")
        if divergence_png is not None:
            print(f"Wrote {divergence_png}")
    for item in series:
        print(f"Read {args.series_id_label}={item.order}: {item.path}")


if __name__ == "__main__":
    main()
