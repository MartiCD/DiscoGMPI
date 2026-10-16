#!/usr/bin/env python3

"""Compare DiscoGMPI PEC-almond RCS diagnostics with Austin CEM data.

The almond scattering driver writes

    <run-dir>/diagnostics/rcs.csv

with total and spherical-component far-field magnitudes.  The Austin benchmark
files store monostatic RCS in dBsm.  This postprocessor converts both data sets
to either

    sigma / lambda^2

or

    10*log10(sigma/lambda^2)

and overlays the selected Austin reference curve.

Benchmark convention:
  * H corresponds to HH, i.e. the phi-polarized RCS component.
  * V corresponds to VV, i.e. the theta-polarized RCS component.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/discogmpi-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback.
    import tomli as tomllib  # type: ignore[no-redef]


C0 = 299_792_458.0
EPS = np.finfo(float).eps
DEFAULT_REFERENCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "benchmark"
    / "AustinCEMBenchmarks"
    / "Problem_III-Almonds"
    / "IIIA-Reference Data"
)
REFERENCE_KEYS = ("fx1", "fx2", "fx3", "fx4")
POLARIZATION_COMPONENT = {"H": "phi", "V": "theta"}


@dataclass(frozen=True)
class RCSRow:
    theta_degrees: float
    phi_degrees: float
    rcs_total: float
    rcs_db_total: float
    abs_etheta: float
    abs_ephi: float
    samples: int
    window_sum: float


@dataclass(frozen=True)
class ReferenceRow:
    frequency_hz: float
    theta_degrees: float
    phi_degrees: float
    rcs_dbsm: float


@dataclass(frozen=True)
class DGComparisonRow:
    polarization: str
    component: str
    angle_variable: str
    angle_degrees: float
    theta_degrees: float
    phi_degrees: float
    dg_rcs: float
    dg_rcs_dbsm: float
    dg_rcs_over_lambda2: float
    dg_rcs_over_lambda2_db: float
    reference_rcs_over_lambda2_db: float
    error_db: float
    samples: int
    window_sum: float


@dataclass(frozen=True)
class PolarizationResult:
    polarization: str
    component: str
    frequency_key: str
    reference_path: Path
    reference_frequency_hz: float
    reference_wavelength: float
    reference_angle_degrees: np.ndarray
    reference_dbsm: np.ndarray
    reference_normalized_db: np.ndarray
    reference_normalized_linear: np.ndarray
    comparison_rows: list[DGComparisonRow]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess PEC-almond RCS diagnostics and compare against "
            "Austin CEM benchmark reference data."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("output/validation_sequence/pec_almond_rcs_smoke"),
        help=(
            "DiscoGMPI run directory. Default: "
            "output/validation_sequence/pec_almond_rcs_smoke"
        ),
    )
    parser.add_argument(
        "--rcs-csv",
        type=Path,
        default=None,
        help="Numerical RCS CSV. Default: <run-dir>/diagnostics/rcs.csv",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help=(
            "Austin CEM almond reference-data directory. Default: "
            "benchmark/AustinCEMBenchmarks/Problem_III-Almonds/"
            "IIIA-Reference Data"
        ),
    )
    parser.add_argument(
        "--reference-kind",
        choices=("ref", "meas-m1", "meas-m2"),
        default="ref",
        help="Austin data source. Default: ref",
    )
    parser.add_argument(
        "--frequency-key",
        choices=("auto", *REFERENCE_KEYS),
        default="auto",
        help=(
            "Austin frequency key. 'auto' chooses the nearest available "
            "reference frequency to the run wavelength. Default: auto"
        ),
    )
    parser.add_argument(
        "--polarization",
        choices=("H", "V", "both"),
        default="H",
        help=(
            "Austin polarization to compare. H uses E_phi; V uses E_theta; "
            "'both' overlays both component comparisons. Default: H"
        ),
    )
    parser.add_argument(
        "--component",
        choices=("auto", "total", "theta", "phi"),
        default="auto",
        help=(
            "DG RCS component. Default auto maps H->phi and V->theta. "
            "Use total only for non-polarized diagnostic plots."
        ),
    )
    parser.add_argument(
        "--dg-angle",
        choices=("auto", "theta", "phi"),
        default="auto",
        help=(
            "DG angle to use as the horizontal axis. Default auto chooses "
            "the angle with the larger range after filters."
        ),
    )
    parser.add_argument(
        "--theta-filter",
        type=float,
        default=None,
        help="Keep only DG rows with theta equal to this value.",
    )
    parser.add_argument(
        "--phi-filter",
        type=float,
        default=None,
        help="Keep only DG rows with phi equal to this value.",
    )
    parser.add_argument(
        "--filter-tolerance-degrees",
        type=float,
        default=1.0e-8,
        help="Tolerance for --theta-filter/--phi-filter. Default: 1e-8",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=None,
        help="DG wavelength used for sigma/lambda^2. Default: resolved config.",
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=None,
        help="Physical frequency used to set wavelength = c0/f.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=None,
        help="Incident amplitude. Default: resolved config or 1.",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=None,
        help=(
            "Output comparison CSV. Default: "
            "<run-dir>/diagnostics/pec_almond_rcs_comparison.csv"
        ),
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=None,
        help=(
            "Output LaTeX summary table. Default: "
            "<run-dir>/tables/pec_almond_rcs_comparison.tex"
        ),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help=(
            "Output normalized-dB plot. Default: "
            "<run-dir>/plots/pec_almond_rcs_normalized_db_comparison.pdf"
        ),
    )
    parser.add_argument(
        "--linear-plot-output",
        type=Path,
        default=None,
        help=(
            "Output normalized-linear plot. Default: "
            "<run-dir>/plots/pec_almond_rcs_normalized_linear_comparison.pdf"
        ),
    )
    parser.add_argument(
        "--write-png",
        action="store_true",
        help="Also write PNG copies next to PDF plots.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title.",
    )
    return parser.parse_args(argv)


def read_resolved_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config" / "resolved_config.toml"
    if not config_path.is_file():
        return {}
    with config_path.open("rb") as stream:
        data = tomllib.load(stream)
    config = data.get("configuration", {})
    if not isinstance(config, dict):
        return {}
    return config


def read_rcs_rows(path: Path) -> list[RCSRow]:
    if not path.is_file():
        raise FileNotFoundError(f"RCS CSV not found: {path}")
    rows: list[RCSRow] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            rows.append(
                RCSRow(
                    theta_degrees=float(row["theta_degrees"]),
                    phi_degrees=float(row["phi_degrees"]),
                    rcs_total=float(row["rcs"]),
                    rcs_db_total=float(row["rcs_db"]),
                    abs_etheta=float(row["abs_Etheta"]),
                    abs_ephi=float(row["abs_Ephi"]),
                    samples=int(row["samples"]),
                    window_sum=float(row["window_sum"]),
                )
            )
    if not rows:
        raise ValueError(f"RCS CSV contains no data rows: {path}")
    return rows


def apply_angle_filters(rows: list[RCSRow], args: argparse.Namespace) -> list[RCSRow]:
    filtered = rows
    tol = args.filter_tolerance_degrees
    if args.theta_filter is not None:
        filtered = [
            row
            for row in filtered
            if abs(row.theta_degrees - args.theta_filter) <= tol
        ]
    if args.phi_filter is not None:
        filtered = [
            row for row in filtered if abs(row.phi_degrees - args.phi_filter) <= tol
        ]
    if not filtered:
        raise ValueError("No RCS rows remain after applying angle filters.")
    return filtered


def choose_dg_angle(rows: list[RCSRow], requested: str) -> str:
    if requested != "auto":
        return requested
    theta_values = np.array([row.theta_degrees for row in rows])
    phi_values = np.array([row.phi_degrees for row in rows])
    theta_range = float(np.max(theta_values) - np.min(theta_values))
    phi_range = float(np.max(phi_values) - np.min(phi_values))
    return "theta" if theta_range >= phi_range else "phi"


def dg_angle(row: RCSRow, angle_variable: str) -> float:
    if angle_variable == "theta":
        return row.theta_degrees
    if angle_variable == "phi":
        return row.phi_degrees
    raise ValueError(f"Unsupported DG angle variable: {angle_variable}")


def rcs_db(value: float) -> float:
    return 10.0 * math.log10(value) if value > 0.0 else -math.inf


def normalized_db_from_dbsm(dbsm: np.ndarray | float, wavelength: float):
    return dbsm - 20.0 * np.log10(wavelength)


def reference_file(reference_dir: Path, kind: str, key: str, polarization: str) -> Path:
    if kind == "ref":
        name = f"ref_rcs.III.A.s1.{key}.{polarization}.txt"
    elif kind == "meas-m1":
        name = f"meas.m1_rcs.III.A.s1.{key}.{polarization}.txt"
    elif kind == "meas-m2":
        name = f"meas.m2_rcs.III.A.s1.{key}.{polarization}.txt"
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"Unsupported reference kind: {kind}")
    return reference_dir / name


def read_reference_rows(path: Path) -> list[ReferenceRow]:
    if not path.is_file():
        raise FileNotFoundError(f"Austin reference file not found: {path}")
    rows: list[ReferenceRow] = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 4:
                raise ValueError(f"Malformed reference row {line_number} in {path}")
            rows.append(
                ReferenceRow(
                    frequency_hz=float(fields[0]),
                    theta_degrees=float(fields[1]),
                    phi_degrees=float(fields[2]),
                    rcs_dbsm=float(fields[3]),
                )
            )
    if not rows:
        raise ValueError(f"Reference file contains no rows: {path}")
    return rows


def reference_frequency(reference_dir: Path, kind: str, key: str, polarization: str) -> float:
    rows = read_reference_rows(reference_file(reference_dir, kind, key, polarization))
    return float(np.mean([row.frequency_hz for row in rows]))


def resolve_wavelength(args: argparse.Namespace, config: dict[str, Any]) -> float | None:
    if args.wavelength is not None:
        if args.wavelength <= 0.0:
            raise ValueError("--wavelength must be positive.")
        return args.wavelength
    if args.frequency_hz is not None:
        if args.frequency_hz <= 0.0:
            raise ValueError("--frequency-hz must be positive.")
        return C0 / args.frequency_hz
    value = config.get("wavelength")
    if value is not None:
        wavelength = float(value)
        if wavelength > 0.0:
            return wavelength
    return None


def resolve_amplitude(args: argparse.Namespace, config: dict[str, Any]) -> float:
    if args.amplitude is not None:
        amplitude = args.amplitude
    else:
        amplitude = float(config.get("amplitude", 1.0))
    if amplitude == 0.0:
        raise ValueError("Incident amplitude must be non-zero.")
    return abs(amplitude)


def resolve_frequency_key(
    args: argparse.Namespace,
    wavelength: float | None,
    polarization: str,
) -> str:
    if args.frequency_key != "auto":
        return args.frequency_key
    if wavelength is None:
        return "fx1"
    target_frequency = C0 / wavelength
    distances: list[tuple[float, str]] = []
    for key in REFERENCE_KEYS:
        try:
            freq = reference_frequency(args.reference_dir, args.reference_kind, key, polarization)
        except FileNotFoundError:
            continue
        distances.append((abs(freq - target_frequency), key))
    if not distances:
        raise FileNotFoundError(
            f"No Austin reference files found in {args.reference_dir}"
        )
    return min(distances)[1]


def read_reference_data(
    args: argparse.Namespace,
    frequency_key: str,
    polarization: str,
) -> tuple[Path, np.ndarray, np.ndarray, float, float]:
    path = reference_file(
        args.reference_dir,
        args.reference_kind,
        frequency_key,
        polarization,
    )
    rows = read_reference_rows(path)
    angle = np.array([row.phi_degrees for row in rows], dtype=float)
    dbsm = np.array([row.rcs_dbsm for row in rows], dtype=float)
    frequency_hz = float(np.mean([row.frequency_hz for row in rows]))
    wavelength = C0 / frequency_hz
    order = np.argsort(angle)
    return path, angle[order], dbsm[order], frequency_hz, wavelength


def resolve_component(polarization: str, requested: str) -> str:
    if requested == "auto":
        return POLARIZATION_COMPONENT[polarization]
    return requested


def dg_component_rcs(row: RCSRow, component: str, amplitude: float) -> float:
    if component == "total":
        return row.rcs_total
    if component == "theta":
        return 4.0 * math.pi * row.abs_etheta**2 / amplitude**2
    if component == "phi":
        return 4.0 * math.pi * row.abs_ephi**2 / amplitude**2
    raise ValueError(f"Unsupported component: {component}")


def build_polarization_result(
    rows: list[RCSRow],
    args: argparse.Namespace,
    config: dict[str, Any],
    polarization: str,
    wavelength: float,
    amplitude: float,
    angle_variable: str,
) -> PolarizationResult:
    frequency_key = resolve_frequency_key(args, wavelength, polarization)
    (
        ref_path,
        ref_angle,
        ref_dbsm,
        ref_frequency,
        ref_wavelength,
    ) = read_reference_data(args, frequency_key, polarization)
    ref_norm_db = normalized_db_from_dbsm(ref_dbsm, ref_wavelength)
    ref_norm_linear = np.power(10.0, ref_norm_db / 10.0)
    component = resolve_component(polarization, args.component)

    comparison: list[DGComparisonRow] = []
    for row in sorted(rows, key=lambda item: dg_angle(item, angle_variable)):
        sigma = dg_component_rcs(row, component, amplitude)
        sigma_dbsm = rcs_db(sigma)
        sigma_norm = sigma / wavelength**2 if sigma > 0.0 else 0.0
        sigma_norm_db = rcs_db(sigma_norm)
        angle = dg_angle(row, angle_variable)
        if ref_angle[0] <= angle <= ref_angle[-1]:
            ref_interp = float(np.interp(angle, ref_angle, ref_norm_db))
            error_db = sigma_norm_db - ref_interp
        else:
            ref_interp = math.nan
            error_db = math.nan
        comparison.append(
            DGComparisonRow(
                polarization=polarization,
                component=component,
                angle_variable=angle_variable,
                angle_degrees=angle,
                theta_degrees=row.theta_degrees,
                phi_degrees=row.phi_degrees,
                dg_rcs=sigma,
                dg_rcs_dbsm=sigma_dbsm,
                dg_rcs_over_lambda2=sigma_norm,
                dg_rcs_over_lambda2_db=sigma_norm_db,
                reference_rcs_over_lambda2_db=ref_interp,
                error_db=error_db,
                samples=row.samples,
                window_sum=row.window_sum,
            )
        )

    return PolarizationResult(
        polarization=polarization,
        component=component,
        frequency_key=frequency_key,
        reference_path=ref_path,
        reference_frequency_hz=ref_frequency,
        reference_wavelength=ref_wavelength,
        reference_angle_degrees=ref_angle,
        reference_dbsm=ref_dbsm,
        reference_normalized_db=ref_norm_db,
        reference_normalized_linear=ref_norm_linear,
        comparison_rows=comparison,
    )


def finite_values(values: Iterable[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def rms(values: Iterable[float]) -> float:
    finite = finite_values(values)
    if not finite:
        return math.nan
    return math.sqrt(sum(value * value for value in finite) / len(finite))


def max_abs(values: Iterable[float]) -> float:
    finite = finite_values(values)
    if not finite:
        return math.nan
    return max(abs(value) for value in finite)


def default_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    comparison_csv = args.comparison_csv or (
        args.run_dir / "diagnostics" / "pec_almond_rcs_comparison.csv"
    )
    latex_output = args.latex_output or (
        args.run_dir / "tables" / "pec_almond_rcs_comparison.tex"
    )
    plot_output = args.plot_output or (
        args.run_dir / "plots" / "pec_almond_rcs_normalized_db_comparison.pdf"
    )
    linear_plot_output = args.linear_plot_output or (
        args.run_dir / "plots" / "pec_almond_rcs_normalized_linear_comparison.pdf"
    )
    return comparison_csv, latex_output, plot_output, linear_plot_output


def write_comparison_csv(path: Path, results: list[PolarizationResult]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "polarization",
                "component",
                "frequency_key",
                "reference_file",
                "reference_frequency_hz",
                "angle_variable",
                "angle_degrees",
                "theta_degrees",
                "phi_degrees",
                "dg_rcs",
                "dg_rcs_dbsm",
                "dg_rcs_over_lambda2",
                "dg_rcs_over_lambda2_db",
                "reference_interp_over_lambda2_db",
                "error_db",
                "samples",
                "window_sum",
            ]
        )
        for result in results:
            for row in result.comparison_rows:
                writer.writerow(
                    [
                        row.polarization,
                        row.component,
                        result.frequency_key,
                        result.reference_path,
                        f"{result.reference_frequency_hz:.16e}",
                        row.angle_variable,
                        f"{row.angle_degrees:.16e}",
                        f"{row.theta_degrees:.16e}",
                        f"{row.phi_degrees:.16e}",
                        f"{row.dg_rcs:.16e}",
                        f"{row.dg_rcs_dbsm:.16e}",
                        f"{row.dg_rcs_over_lambda2:.16e}",
                        f"{row.dg_rcs_over_lambda2_db:.16e}",
                        f"{row.reference_rcs_over_lambda2_db:.16e}",
                        f"{row.error_db:.16e}",
                        row.samples,
                        f"{row.window_sum:.16e}",
                    ]
                )
    return path


def latex_scientific(value: float) -> str:
    if not math.isfinite(value):
        return r"\mathrm{n/a}"
    if value == 0.0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.3g}\times 10^{{{exponent}}}"


def write_latex_summary(path: Path, results: list[PolarizationResult]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write("% Generated by examples/postprocess_pec_almond_rcs.py\n")
        stream.write("\\begin{tabular}{llrrrr}\n")
        stream.write("\\hline\n")
        stream.write(
            "Pol. & Component & Samples & "
            "$\\max |\\Delta|$ [dB] & RMS $\\Delta$ [dB] & "
            "$\\max \\sigma_h/\\lambda^2$ \\\\\n"
        )
        stream.write("\\hline\n")
        for result in results:
            errors = [row.error_db for row in result.comparison_rows]
            values = [row.dg_rcs_over_lambda2 for row in result.comparison_rows]
            stream.write(
                f"{result.polarization} & {result.component} & "
                f"{len(result.comparison_rows)} & "
                f"{max_abs(errors):.3g} & {rms(errors):.3g} & "
                f"${latex_scientific(max(finite_values(values) or [math.nan]))}$ \\\\\n"
            )
        stream.write("\\hline\n")
        stream.write("\\end{tabular}\n")
    return path


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.9,
            "lines.linewidth": 2.0,
            "savefig.bbox": "tight",
        }
    )


def angle_label(angle_variable: str) -> str:
    if angle_variable == "phi":
        return r"Observation azimuth $\phi$ [deg]"
    return r"Observation polar angle $\theta$ [deg]"


def result_marker(polarization: str) -> str:
    return "o" if polarization == "H" else "s"


def result_color(index: int) -> str:
    colors = ["#3f2e7e", "#d95f02", "#1b9e77", "#7570b3"]
    return colors[index % len(colors)]


def write_plots(
    db_path: Path,
    linear_path: Path,
    results: list[PolarizationResult],
    angle_variable: str,
    title: str | None,
    write_png: bool,
) -> tuple[Path, Path]:
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for index, result in enumerate(results):
        color = result_color(index)
        dg_angle_values = np.array(
            [row.angle_degrees for row in result.comparison_rows], dtype=float
        )
        dg_values = np.array(
            [row.dg_rcs_over_lambda2_db for row in result.comparison_rows],
            dtype=float,
        )
        ax.plot(
            result.reference_angle_degrees,
            result.reference_normalized_db,
            color=color,
            linestyle="-",
            label=f"Austin {result.polarization} ({result.frequency_key})",
        )
        ax.plot(
            dg_angle_values,
            dg_values,
            color=color,
            linestyle="none",
            marker=result_marker(result.polarization),
            markerfacecolor="white",
            markeredgewidth=1.5,
            markersize=5.0,
            label=f"DiscoGMPI {result.polarization} ({result.component})",
        )

    ax.set_xlabel(angle_label(angle_variable))
    ax.set_ylabel(r"$10\log_{10}(\sigma/\lambda^2)$ [dB]")
    if title:
        ax.set_title(title)
    ax.grid(True, which="major", color="0.86", linewidth=0.9)
    ax.grid(True, which="minor", color="0.93", linewidth=0.5)
    ax.minorticks_on()
    ax.legend(frameon=True, framealpha=0.95)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(db_path)
    if write_png:
        fig.savefig(db_path.with_suffix(".png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for index, result in enumerate(results):
        color = result_color(index)
        dg_angle_values = np.array(
            [row.angle_degrees for row in result.comparison_rows], dtype=float
        )
        dg_values = np.array(
            [row.dg_rcs_over_lambda2 for row in result.comparison_rows],
            dtype=float,
        )
        ax.semilogy(
            result.reference_angle_degrees,
            result.reference_normalized_linear,
            color=color,
            linestyle="-",
            label=f"Austin {result.polarization} ({result.frequency_key})",
        )
        ax.semilogy(
            dg_angle_values,
            dg_values,
            color=color,
            linestyle="none",
            marker=result_marker(result.polarization),
            markerfacecolor="white",
            markeredgewidth=1.5,
            markersize=5.0,
            label=f"DiscoGMPI {result.polarization} ({result.component})",
        )

    ax.set_xlabel(angle_label(angle_variable))
    ax.set_ylabel(r"$\sigma/\lambda^2$")
    if title:
        ax.set_title(title)
    ax.grid(True, which="major", color="0.86", linewidth=0.9)
    ax.grid(True, which="minor", color="0.93", linewidth=0.5)
    ax.legend(frameon=True, framealpha=0.95)
    linear_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(linear_path)
    if write_png:
        fig.savefig(linear_path.with_suffix(".png"), dpi=300)
    plt.close(fig)

    return db_path, linear_path


def print_summary(
    results: list[PolarizationResult],
    outputs: tuple[Path, Path, Path, Path],
    wavelength: float,
    angle_variable: str,
) -> None:
    comparison_csv, latex_output, plot_output, linear_plot_output = outputs
    print("PEC almond RCS postprocessing")
    print("-----------------------------")
    print(f"DG wavelength:        {wavelength:.8e}")
    print(f"DG angle variable:    {angle_variable}")
    for result in results:
        errors = [row.error_db for row in result.comparison_rows]
        print(
            f"{result.polarization}: component={result.component}, "
            f"reference={result.reference_path.name}, "
            f"freq={result.reference_frequency_hz:.6e} Hz, "
            f"samples={len(result.comparison_rows)}, "
            f"max|error|={max_abs(errors):.3e} dB, "
            f"rms={rms(errors):.3e} dB"
        )
    print(f"Wrote comparison CSV: {comparison_csv}")
    print(f"Wrote LaTeX table:    {latex_output}")
    print(f"Wrote dB plot:        {plot_output}")
    print(f"Wrote linear plot:    {linear_plot_output}")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir
    rcs_csv = args.rcs_csv or (run_dir / "diagnostics" / "rcs.csv")
    config = read_resolved_config(run_dir)
    rows = apply_angle_filters(read_rcs_rows(rcs_csv), args)
    angle_variable = choose_dg_angle(rows, args.dg_angle)
    wavelength = resolve_wavelength(args, config)
    if wavelength is None:
        # Temporary value; it will be replaced by the selected Austin reference
        # wavelength after the frequency key is resolved for the first result.
        wavelength = C0 / reference_frequency(args.reference_dir, args.reference_kind, "fx1", "H")
    amplitude = resolve_amplitude(args, config)

    polarizations = ["H", "V"] if args.polarization == "both" else [args.polarization]
    results = [
        build_polarization_result(
            rows,
            args,
            config,
            polarization,
            wavelength,
            amplitude,
            angle_variable,
        )
        for polarization in polarizations
    ]

    outputs = default_outputs(args)
    write_comparison_csv(outputs[0], results)
    write_latex_summary(outputs[1], results)
    write_plots(
        outputs[2],
        outputs[3],
        results,
        angle_variable,
        args.title,
        args.write_png,
    )
    print_summary(results, outputs, wavelength, angle_variable)

    if angle_variable != "phi":
        print(
            "NOTE: Austin almond reference files are phi sweeps at theta=90 deg. "
            "This run is being plotted against the DG theta angle. For a "
            "like-for-like benchmark comparison, run DiscoGMPI with theta fixed "
            "at 90 deg and sweep phi from 0 to 180 deg.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
