#!/usr/bin/env python3

"""Post-process periodic cuboid alternating-flux isoresolution runs.

The expected run layout is produced by
``run_periodic_cuboid_alternating_flux_isoresolution_convergence.sh``:

    output/convergence/periodic_cuboid_af_p4_m1
    output/convergence/periodic_cuboid_af_p2_m1
    output/convergence/periodic_cuboid_af_p1_m1
    ...

For each available case, this script reads ``quadrature_diagnostics.csv``,
computes

    max_t || q(t) - q_h(t) ||_{L^2(\Omega)}

for the field variables and their components, writes CSV/LaTeX convergence
tables, and generates publication-style figures.  Convergence rates are based
only on these time-maximum spatial L2 field errors.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter


DEFAULT_ORDERS = (4, 2, 1)
DEFAULT_MESH_LEVELS = (1, 2, 3, 4)
BASE_NX = 4
BASE_NY = 2
BASE_NZ = 2
LX = 2.0
LY = 1.0
LZ = 1.0

AGGREGATE_FIELD_ERROR_NAMES = ("electric_l2", "magnetic_l2", "field_l2")
COMPONENT_FIELD_ERROR_NAMES = (
    "ex_l2",
    "ey_l2",
    "ez_l2",
    "hx_l2",
    "hy_l2",
    "hz_l2",
)
FIELD_ERROR_NAMES = AGGREGATE_FIELD_ERROR_NAMES + COMPONENT_FIELD_ERROR_NAMES

FIELD_ERROR_COLUMN_ALIASES = {
    "electric_l2": (
        "electric_error_l2",
        "linf_time_l2_space_electric_error",
        "final_l2_space_electric_error",
    ),
    "magnetic_l2": (
        "magnetic_error_l2",
        "linf_time_l2_space_magnetic_error",
        "final_l2_space_magnetic_error",
    ),
    "field_l2": (
        "field_error_l2",
        "total_error_l2",
        "linf_time_l2_space_total_error",
        "final_l2_space_total_error",
    ),
    "ex_l2": (
        "ex_error_l2",
        "electric_x_error_l2",
        "E_x_error_l2",
        "linf_time_l2_space_ex_error",
        "final_l2_space_ex_error",
    ),
    "ey_l2": (
        "ey_error_l2",
        "electric_y_error_l2",
        "E_y_error_l2",
        "linf_time_l2_space_ey_error",
        "final_l2_space_ey_error",
    ),
    "ez_l2": (
        "ez_error_l2",
        "electric_z_error_l2",
        "E_z_error_l2",
        "linf_time_l2_space_ez_error",
        "final_l2_space_ez_error",
    ),
    "hx_l2": (
        "hx_error_l2",
        "magnetic_x_error_l2",
        "H_x_error_l2",
        "linf_time_l2_space_hx_error",
        "final_l2_space_hx_error",
    ),
    "hy_l2": (
        "hy_error_l2",
        "magnetic_y_error_l2",
        "H_y_error_l2",
        "linf_time_l2_space_hy_error",
        "final_l2_space_hy_error",
    ),
    "hz_l2": (
        "hz_error_l2",
        "magnetic_z_error_l2",
        "H_z_error_l2",
        "linf_time_l2_space_hz_error",
        "final_l2_space_hz_error",
    ),
}

FIELD_ERROR_LABELS = {
    "electric_l2": r"$\max_t\|E-E_h\|_{L^2(\Omega)}$",
    "magnetic_l2": r"$\max_t\|H-H_h\|_{L^2(\Omega)}$",
    "field_l2": r"$\max_t\|(E,H)-(E_h,H_h)\|_{L^2(\Omega)}$",
    "ex_l2": r"$\max_t\|E_x-E_{x,h}\|_{L^2(\Omega)}$",
    "ey_l2": r"$\max_t\|E_y-E_{y,h}\|_{L^2(\Omega)}$",
    "ez_l2": r"$\max_t\|E_z-E_{z,h}\|_{L^2(\Omega)}$",
    "hx_l2": r"$\max_t\|H_x-H_{x,h}\|_{L^2(\Omega)}$",
    "hy_l2": r"$\max_t\|H_y-H_{y,h}\|_{L^2(\Omega)}$",
    "hz_l2": r"$\max_t\|H_z-H_{z,h}\|_{L^2(\Omega)}$",
}

FIELD_ERROR_TABLE_LABELS = {
    "electric_l2": r"$E$",
    "magnetic_l2": r"$H$",
    "field_l2": r"$U$",
    "ex_l2": r"$E_x$",
    "ey_l2": r"$E_y$",
    "ez_l2": r"$E_z$",
    "hx_l2": r"$H_x$",
    "hy_l2": r"$H_y$",
    "hz_l2": r"$H_z$",
}

FIELD_ERROR_CSV_NAMES = {
    "electric_l2": "electric",
    "magnetic_l2": "magnetic",
    "field_l2": "total",
    "ex_l2": "ex",
    "ey_l2": "ey",
    "ez_l2": "ez",
    "hx_l2": "hx",
    "hy_l2": "hy",
    "hz_l2": "hz",
}

COLORS = {
    4: "#0072B2",
    2: "#D55E00",
    1: "#009E73",
}
MARKERS = {
    4: "o",
    2: "s",
    1: "^",
}
LINESTYLES = {
    4: "-",
    2: "--",
    1: "-.",
}


@dataclass(frozen=True)
class CaseKey:
    order: int
    mesh_level: int

    @property
    def label(self) -> str:
        return f"P{self.order}M{self.mesh_level}"


@dataclass
class CaseResult:
    key: CaseKey
    output_dir: Path
    diagnostics_path: Path
    final_time: float
    target_final_time: float
    complete: bool
    nx: int
    ny: int
    nz: int
    nelements: int
    nodes_per_element: int | None
    mpi_ranks: int | None
    dt: float | None
    steps: int | None
    h_min: float | None
    h_iso: float
    dofs: int | None
    final_errors: dict[str, float]
    time_linf_errors: dict[str, float]
    rates: dict[str, float | None]

    def error(self, name: str) -> float:
        return self.time_linf_errors[name]

    def has_error(self, name: str) -> bool:
        return name in self.time_linf_errors

    def rate(self, name: str) -> float | None:
        return self.rates.get(name)


@dataclass(frozen=True)
class Quantity:
    name: str
    label: str
    table_label: str
    getter: Callable[[CaseResult], float]


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        entries = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Expected a comma-separated integer list, got {value!r}."
        ) from error
    if not entries:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return entries


def parse_formats(value: str) -> tuple[str, ...]:
    formats = tuple(item.strip().lstrip(".") for item in value.split(",") if item.strip())
    if not formats:
        raise argparse.ArgumentTypeError("At least one output format is required.")
    return formats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process periodic cuboid alternating-flux isoresolution "
            "convergence results."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/convergence"),
        help="Root directory containing per-case output folders.",
    )
    parser.add_argument(
        "--prefix",
        default="periodic_cuboid_af",
        help="Case directory prefix. Default: periodic_cuboid_af.",
    )
    parser.add_argument(
        "--orders",
        type=parse_int_list,
        default=DEFAULT_ORDERS,
        help="Comma-separated polynomial orders. Default: 4,2,1.",
    )
    parser.add_argument(
        "--mesh-levels",
        type=parse_int_list,
        default=DEFAULT_MESH_LEVELS,
        help="Comma-separated M levels. Default: 1,2,3,4.",
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=1.0,
        help="Expected final time used to mark runs complete. Default: 1.0.",
    )
    parser.add_argument(
        "--time-tol",
        type=float,
        default=1.0e-10,
        help="Absolute tolerance for final-time completeness. Default: 1e-10.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing case directories instead of failing.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Include cases whose final diagnostic time is not --final-time.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/convergence/postprocess_periodic_cuboid_af_isoresolution"),
        help="Directory for generated tables and figures.",
    )
    parser.add_argument(
        "--output-prefix",
        default="periodic_cuboid_af_isoresolution",
        help="Generated filename prefix.",
    )
    parser.add_argument(
        "--formats",
        type=parse_formats,
        default=("pdf", "png"),
        help="Comma-separated plot formats. Default: pdf,png.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Raster output resolution. Default: 400.",
    )
    parser.add_argument(
        "--usetex",
        action="store_true",
        help="Use LaTeX text rendering. Requires a working LaTeX install.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after writing files.",
    )
    return parser.parse_args()


def configure_matplotlib(usetex: bool) -> None:
    plt.rcParams.update(
        {
            "text.usetex": usetex,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "axes.labelsize": 11.0,
            "axes.titlesize": 11.0,
            "legend.fontsize": 9.2,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.0,
            "lines.markeredgewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "xtick.minor.size": 2.8,
            "ytick.minor.size": 2.8,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.75,
            "ytick.minor.width": 0.75,
            "legend.frameon": True,
            "legend.framealpha": 0.96,
            "legend.edgecolor": "0.75",
            "legend.handlelength": 2.4,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
        }
    )


def order_refinement_offset(order: int) -> int:
    if order == 4:
        return 0
    if order == 2:
        return 1
    if order == 1:
        return 2
    raise ValueError(f"Unsupported isoresolution order {order}; expected 4, 2, or 1.")


def case_resolution(order: int, mesh_level: int) -> tuple[int, int, int, float]:
    exponent = mesh_level - 1 + order_refinement_offset(order)
    nx = BASE_NX << exponent
    ny = BASE_NY << exponent
    nz = BASE_NZ << exponent
    hx = LX / nx
    h_iso = hx / order
    return nx, ny, nz, h_iso


def read_float(row: dict[str, str], column: str, path: Path, row_number: int) -> float:
    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid value for {column!r} in {path} at row {row_number}."
        ) from error


def resolve_field_error_columns(
    available: set[str],
    path: Path,
) -> dict[str, str]:
    columns: dict[str, str] = {}
    for name, aliases in FIELD_ERROR_COLUMN_ALIASES.items():
        for column in aliases:
            if column in available:
                columns[name] = column
                break

    missing_required = [
        name for name in AGGREGATE_FIELD_ERROR_NAMES if name not in columns
    ]
    if missing_required:
        missing_aliases = [
            "/".join(FIELD_ERROR_COLUMN_ALIASES[name])
            for name in missing_required
        ]
        raise ValueError(
            f"{path} is missing required aggregate field error columns: "
            + ", ".join(missing_aliases)
        )

    return columns


def row_errors(
    row: dict[str, str],
    path: Path,
    row_number: int,
    columns: dict[str, str],
) -> dict[str, float]:
    return {
        name: abs(read_float(row, column, path, row_number))
        for name, column in columns.items()
    }


def read_diagnostics(
    path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, float]], dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing diagnostics file: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        available = set(reader.fieldnames or ())
        if "time" not in available:
            raise ValueError(f"{path} is missing required column: time")
        field_error_columns = resolve_field_error_columns(available, path)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No data rows found in {path}.")
    errors = [
        row_errors(row, path, row_number, field_error_columns)
        for row_number, row in enumerate(rows, start=2)
    ]
    return rows, errors, field_error_columns


def parse_simple_toml_value(text: str):
    text = text.strip()
    if not text:
        return None
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        return text
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        return float(text)
    except ValueError:
        return text


def read_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    values: dict[str, object] = {}
    section = ""
    pattern = re.compile(r"^([A-Za-z0-9_]+)\s*=\s*(.+)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.strip("[]")
            continue
        match = pattern.match(stripped)
        if not match:
            continue
        key, value = match.groups()
        full_key = f"{section}.{key}" if section else key
        parsed = parse_simple_toml_value(value)
        values[key] = parsed
        values[full_key] = parsed
    return values


def metadata_float(metadata: dict[str, object], *keys: str) -> float | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def metadata_int(metadata: dict[str, object], *keys: str) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def read_case(
    root: Path,
    prefix: str,
    order: int,
    mesh_level: int,
    target_final_time: float,
    time_tol: float,
) -> CaseResult:
    key = CaseKey(order, mesh_level)
    output_dir = root / f"{prefix}_p{order}_m{mesh_level}"
    diagnostics_path = output_dir / "quadrature_diagnostics.csv"
    metadata_path = output_dir / "run_metadata.toml"
    rows, errors, _ = read_diagnostics(diagnostics_path)
    metadata = read_metadata(metadata_path)

    final_row = rows[-1]
    final_errors = errors[-1]
    time_linf_errors = {
        name: max(error[name] for error in errors)
        for name in final_errors
    }
    final_time = read_float(final_row, "time", diagnostics_path, len(rows) + 1)
    complete = abs(final_time - target_final_time) <= time_tol
    nx, ny, nz, h_iso = case_resolution(order, mesh_level)
    nelements = nx * ny * nz * 6
    h_min = metadata_float(metadata, "global_hmin", "runtime.global_hmin")
    if h_min is not None:
        h_iso = h_min / order
    nodes_per_element = metadata_int(metadata, "nodes_per_element")
    mpi_ranks = metadata_int(metadata, "mpi_ranks")
    dt = metadata_float(metadata, "used_dt", "runtime.used_dt")
    if dt is None and len(rows) > 1:
        dt = read_float(rows[1], "time", diagnostics_path, 3) - read_float(
            rows[0],
            "time",
            diagnostics_path,
            2,
        )
    steps = metadata_int(metadata, "final_step", "runtime.final_step")
    if steps is None:
        try:
            steps = int(float(final_row["step"]))
        except (KeyError, ValueError):
            steps = len(rows) - 1
    dofs = None
    if nodes_per_element is not None:
        dofs = nelements * nodes_per_element * 6

    return CaseResult(
        key=key,
        output_dir=output_dir,
        diagnostics_path=diagnostics_path,
        final_time=final_time,
        target_final_time=target_final_time,
        complete=complete,
        nx=nx,
        ny=ny,
        nz=nz,
        nelements=nelements,
        nodes_per_element=nodes_per_element,
        mpi_ranks=mpi_ranks,
        dt=dt,
        steps=steps,
        h_min=h_min,
        h_iso=h_iso,
        dofs=dofs,
        final_errors=final_errors,
        time_linf_errors=time_linf_errors,
        rates={},
    )


def load_results(args: argparse.Namespace) -> list[CaseResult]:
    results: list[CaseResult] = []
    missing: list[str] = []
    incomplete: list[str] = []
    for mesh_level in args.mesh_levels:
        for order in args.orders:
            try:
                result = read_case(
                    args.root,
                    args.prefix,
                    order,
                    mesh_level,
                    args.final_time,
                    args.time_tol,
                )
            except FileNotFoundError:
                missing.append(f"P{order}M{mesh_level}")
                continue
            if not result.complete:
                incomplete.append(
                    f"{result.key.label}: final time {result.final_time:g}"
                )
            results.append(result)

    if missing and not args.allow_missing:
        raise FileNotFoundError(
            "Missing isoresolution cases: "
            + ", ".join(missing)
            + ". Use --allow-missing to post-process partial data."
        )
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(
            "Incomplete isoresolution cases: "
            + "; ".join(incomplete)
            + ". Use --allow-incomplete to include them."
        )
    if not results:
        raise RuntimeError("No isoresolution cases were loaded.")

    compute_rates(results)
    return sorted(results, key=lambda item: (item.key.order, item.key.mesh_level))


def safe_rate(coarse_error: float, fine_error: float, coarse_h: float, fine_h: float):
    if coarse_error <= 0.0 or fine_error <= 0.0 or coarse_h <= fine_h or fine_h <= 0.0:
        return None
    return math.log(coarse_error / fine_error) / math.log(coarse_h / fine_h)


def compute_rates(results: list[CaseResult]) -> None:
    names = [
        name
        for name in FIELD_ERROR_NAMES
        if any(result.has_error(name) for result in results)
    ]
    by_order: dict[int, list[CaseResult]] = {}
    for result in results:
        by_order.setdefault(result.key.order, []).append(result)
    for order_results in by_order.values():
        order_results.sort(key=lambda item: item.key.mesh_level)
        previous: CaseResult | None = None
        for result in order_results:
            result.rates = {
                name: None
                for name in names
                if result.has_error(name)
            }
            if previous is not None:
                for name in names:
                    if previous.has_error(name) and result.has_error(name):
                        result.rates[name] = safe_rate(
                            previous.error(name),
                            result.error(name),
                            previous.h_iso,
                            result.h_iso,
                        )
            previous = result


def format_float(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return ""
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}g}"


def available_field_error_names(
    results: Iterable[CaseResult],
    candidates: Iterable[str] = FIELD_ERROR_NAMES,
) -> list[str]:
    return [
        name
        for name in candidates
        if any(result.has_error(name) for result in results)
    ]


def write_summary_csv(path: Path, results: list[CaseResult]) -> None:
    error_names = available_field_error_names(results)
    columns = [
        "case",
        "order",
        "mesh_level",
        "complete",
        "final_time",
        "target_final_time",
        "nx",
        "ny",
        "nz",
        "nelements",
        "nodes_per_element",
        "dofs",
        "mpi_ranks",
        "h_min",
        "h_iso",
        "dt",
        "steps",
    ]
    for name in error_names:
        csv_name = FIELD_ERROR_CSV_NAMES[name]
        columns.append(f"linf_time_l2_space_{csv_name}_error")
        columns.append(f"rate_linf_time_l2_space_{csv_name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in sorted(results, key=lambda item: (item.key.mesh_level, -item.key.order)):
            row = {
                "case": result.key.label,
                "order": result.key.order,
                "mesh_level": result.key.mesh_level,
                "complete": int(result.complete),
                "final_time": format_float(result.final_time),
                "target_final_time": format_float(result.target_final_time),
                "nx": result.nx,
                "ny": result.ny,
                "nz": result.nz,
                "nelements": result.nelements,
                "nodes_per_element": result.nodes_per_element or "",
                "dofs": result.dofs or "",
                "mpi_ranks": result.mpi_ranks or "",
                "h_min": format_float(result.h_min),
                "h_iso": format_float(result.h_iso),
                "dt": format_float(result.dt),
                "steps": result.steps or "",
            }
            for name in error_names:
                csv_name = FIELD_ERROR_CSV_NAMES[name]
                row[f"linf_time_l2_space_{csv_name}_error"] = (
                    format_float(result.error(name)) if result.has_error(name) else ""
                )
                row[f"rate_linf_time_l2_space_{csv_name}"] = (
                    format_float(result.rate(name), 3) if result.has_error(name) else ""
                )
            writer.writerow(row)


def latex_scientific(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "--"
    if value == 0.0:
        return "$0$"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10.0 ** exponent)
    return rf"${mantissa:.{digits}f}\times 10^{{{exponent}}}$"


def latex_rate(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2f}"


def write_latex_table(path: Path, results: list[CaseResult]) -> None:
    rows = sorted(results, key=lambda item: (item.key.mesh_level, -item.key.order))
    error_names = available_field_error_names(results)
    column_spec = "cccc" + "cc" * len(error_names)
    header = [
        "$P$",
        "$M$",
        "$N_x\\times N_y\\times N_z$",
        "$h_{\\mathrm{iso}}$",
    ]
    for name in error_names:
        header.append(FIELD_ERROR_TABLE_LABELS[name])
        header.append("$r$")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("\\begin{table}[t]\n")
        stream.write("\\centering\n")
        stream.write("\\scriptsize\n")
        stream.write(
            "\\caption{Periodic cuboid alternating-flux isoresolution convergence. "
            "Errors are $\\max_t\\|q(t)-q_h(t)\\|_{L^2(\\Omega)}$.}\n"
        )
        stream.write("\\label{tab:periodic-cuboid-af-isoresolution}\n")
        stream.write(f"\\begin{{tabular}}{{{column_spec}}}\n")
        stream.write("\\hline\n")
        stream.write(" & ".join(header) + " \\\\\n")
        stream.write("\\hline\n")
        for result in rows:
            entries = [
                str(result.key.order),
                str(result.key.mesh_level),
                f"{result.nx}\\times {result.ny}\\times {result.nz}",
                latex_scientific(result.h_iso),
            ]
            for name in error_names:
                if result.has_error(name):
                    entries.append(latex_scientific(result.error(name)))
                    entries.append(latex_rate(result.rate(name)))
                else:
                    entries.append("--")
                    entries.append("--")
            stream.write(" & ".join(entries) + " \\\\\n")
        stream.write("\\hline\n")
        stream.write("\\end{tabular}\n")
        stream.write("\\end{table}\n")


def set_log_axis(axis: plt.Axes) -> None:
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    axis.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    axis.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=80)
    )
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    axis.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=80)
    )
    axis.yaxis.set_minor_formatter(NullFormatter())


def style_log_axis(axis: plt.Axes, xlabel: str, ylabel: str) -> None:
    set_log_axis(axis)
    axis.grid(True, which="major", color="0.84", linestyle="-", linewidth=0.65)
    axis.grid(True, which="minor", color="0.92", linestyle=":", linewidth=0.45)
    axis.tick_params(top=True, right=True)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.invert_xaxis()


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Iterable[str],
    dpi: int,
    show: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        if fmt.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
            figure.savefig(path, dpi=dpi)
        else:
            figure.savefig(path)
        paths.append(path)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return paths


def results_by_order(results: list[CaseResult]) -> dict[int, list[CaseResult]]:
    grouped: dict[int, list[CaseResult]] = {}
    for result in results:
        grouped.setdefault(result.key.order, []).append(result)
    for order_results in grouped.values():
        order_results.sort(key=lambda item: item.key.mesh_level)
    return grouped


def plot_field_errors(
    results: list[CaseResult],
    output_dir: Path,
    prefix: str,
    formats: Iterable[str],
    dpi: int,
    show: bool,
) -> list[Path]:
    quantities = (
        Quantity(
            "electric_l2",
            FIELD_ERROR_LABELS["electric_l2"],
            "E",
            lambda item: item.error("electric_l2"),
        ),
        Quantity(
            "magnetic_l2",
            FIELD_ERROR_LABELS["magnetic_l2"],
            "H",
            lambda item: item.error("magnetic_l2"),
        ),
        Quantity(
            "field_l2",
            FIELD_ERROR_LABELS["field_l2"],
            "Total",
            lambda item: item.error("field_l2"),
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), constrained_layout=True)
    grouped = results_by_order(results)
    for axis, quantity in zip(axes, quantities):
        for order in sorted(grouped, reverse=True):
            order_results = grouped[order]
            x_values = [item.h_iso for item in order_results]
            y_values = [max(quantity.getter(item), 1.0e-300) for item in order_results]
            label = rf"$p={order}$"
            if len(order_results) >= 2:
                last_rate = order_results[-1].rate(quantity.name)
                if last_rate is not None:
                    label += rf"  ($r={last_rate:.2f}$)"
            axis.plot(
                x_values,
                y_values,
                color=COLORS.get(order, "0.2"),
                marker=MARKERS.get(order, "o"),
                linestyle=LINESTYLES.get(order, "-"),
                markerfacecolor="white",
                markeredgecolor=COLORS.get(order, "0.2"),
                label=label,
            )
        style_log_axis(axis, r"$h_{\mathrm{iso}}$", quantity.label)
    axes[0].legend(title="DG order", loc="best", title_fontsize=9.2)
    return save_figure(
        figure,
        output_dir,
        f"{prefix}_field_errors",
        formats,
        dpi,
        show,
    )


def plot_component_field_errors(
    results: list[CaseResult],
    output_dir: Path,
    prefix: str,
    formats: Iterable[str],
    dpi: int,
    show: bool,
) -> list[Path]:
    if not available_field_error_names(results, COMPONENT_FIELD_ERROR_NAMES):
        return []

    figure, axes = plt.subplots(2, 3, figsize=(10.4, 6.2), constrained_layout=True)
    grouped = results_by_order(results)
    for axis, name in zip(axes.ravel(), COMPONENT_FIELD_ERROR_NAMES):
        has_quantity = False
        for order in sorted(grouped, reverse=True):
            order_results = [
                item for item in grouped[order] if item.has_error(name)
            ]
            if not order_results:
                continue
            has_quantity = True
            x_values = [item.h_iso for item in order_results]
            y_values = [max(item.error(name), 1.0e-300) for item in order_results]
            label = rf"$p={order}$"
            if len(order_results) >= 2:
                last_rate = order_results[-1].rate(name)
                if last_rate is not None:
                    label += rf"  ($r={last_rate:.2f}$)"
            axis.plot(
                x_values,
                y_values,
                color=COLORS.get(order, "0.2"),
                marker=MARKERS.get(order, "o"),
                linestyle=LINESTYLES.get(order, "-"),
                markerfacecolor="white",
                markeredgecolor=COLORS.get(order, "0.2"),
                label=label,
            )
        if has_quantity:
            style_log_axis(axis, r"$h_{\mathrm{iso}}$", FIELD_ERROR_LABELS[name])
        else:
            axis.text(
                0.5,
                0.5,
                "missing\ncomponent\ncolumn",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="0.35",
            )
            axis.set_axis_off()
    legend_axis = None
    for axis in axes.ravel():
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            legend_axis = axis
            break
    if legend_axis is not None:
        legend_axis.legend(title="DG order", loc="best", title_fontsize=9.2)
    return save_figure(
        figure,
        output_dir,
        f"{prefix}_component_field_errors",
        formats,
        dpi,
        show,
    )


def plot_error_matrix(
    results: list[CaseResult],
    output_dir: Path,
    prefix: str,
    formats: Iterable[str],
    dpi: int,
    show: bool,
) -> list[Path]:
    orders = sorted({item.key.order for item in results}, reverse=True)
    levels = sorted({item.key.mesh_level for item in results})
    lookup = {(item.key.order, item.key.mesh_level): item for item in results}
    matrix = []
    positive_values = []
    for level in levels:
        row = []
        for order in orders:
            value = lookup.get((order, level))
            entry = math.nan if value is None else value.error("field_l2")
            row.append(entry)
            if math.isfinite(entry) and entry > 0.0:
                positive_values.append(entry)
        matrix.append(row)

    if not positive_values:
        return []

    vmin = 10.0 ** math.floor(math.log10(min(positive_values)))
    vmax = 10.0 ** math.ceil(math.log10(max(positive_values)))

    figure, axis = plt.subplots(figsize=(5.4, 4.2), constrained_layout=True)
    image = axis.imshow(
        matrix,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis",
        aspect="auto",
    )
    axis.set_xticks(range(len(orders)), [rf"$p={order}$" for order in orders])
    axis.set_yticks(range(len(levels)), [rf"$M={level}$" for level in levels])
    axis.set_xlabel("Polynomial order")
    axis.set_ylabel("Mesh level")
    axis.tick_params(top=True, right=True)
    for row_index, level in enumerate(levels):
        for column_index, order in enumerate(orders):
            result = lookup.get((order, level))
            if result is None:
                label = "--"
            else:
                label = f"{result.error('field_l2'):.1e}"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=8.2,
            )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(r"$\|(E,H)-(E_h,H_h)\|_{L^2(\Omega)}$")
    colorbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    return save_figure(
        figure,
        output_dir,
        f"{prefix}_field_error_matrix",
        formats,
        dpi,
        show,
    )


def print_summary(results: list[CaseResult], written_paths: Iterable[Path]) -> None:
    component_names = available_field_error_names(results, COMPONENT_FIELD_ERROR_NAMES)
    print("Loaded isoresolution cases:")
    for result in sorted(results, key=lambda item: (item.key.mesh_level, -item.key.order)):
        status = "complete" if result.complete else f"incomplete tf={result.final_time:g}"
        print(
            f"  {result.key.label}: h_iso={result.h_iso:.6e}, "
            f"linf_time_field_l2={result.error('field_l2'):.6e}, {status}"
        )
    if not component_names:
        print(
            "Warning: no component field error columns were found; "
            "component convergence rates and plots were skipped."
        )
    for path in written_paths:
        print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    configure_matplotlib(args.usetex)
    results = load_results(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.output_prefix}_summary.csv"
    latex_path = args.output_dir / f"{args.output_prefix}_table.tex"
    write_summary_csv(csv_path, results)
    write_latex_table(latex_path, results)

    written_paths: list[Path] = [csv_path, latex_path]
    written_paths += plot_field_errors(
        results,
        args.output_dir,
        args.output_prefix,
        args.formats,
        args.dpi,
        args.show,
    )
    written_paths += plot_component_field_errors(
        results,
        args.output_dir,
        args.output_prefix,
        args.formats,
        args.dpi,
        args.show,
    )
    written_paths += plot_error_matrix(
        results,
        args.output_dir,
        args.output_prefix,
        args.formats,
        args.dpi,
        args.show,
    )
    print_summary(results, written_paths)


if __name__ == "__main__":
    main()
