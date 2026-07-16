#!/usr/bin/env python3

"""Compare DiscoGMPI metallic-sphere RCS against the exact PEC Mie series.

The scattering driver writes a numerical RCS table to

    <run-dir>/diagnostics/rcs.csv

with the normalization

    sigma(theta, phi) = 4*pi*|E_infinity_scat(theta, phi)|^2 / |E_inc|^2.

This script evaluates the exact perfect-electric-conductor sphere Mie solution
at the same angles, writes a comparison CSV, writes a compact LaTeX summary
table, and generates a publication-style comparison plot.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/discogmpi-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
from scipy.special import spherical_jn, spherical_yn

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback.
    import tomli as tomllib  # type: ignore[no-redef]


EPS = np.finfo(float).eps


@dataclass(frozen=True)
class RCSRow:
    theta_degrees: float
    phi_degrees: float
    numerical_rcs: float
    numerical_rcs_db: float | None


@dataclass(frozen=True)
class ComparisonRow:
    theta_degrees: float
    phi_degrees: float
    numerical_rcs: float
    mie_rcs: float
    absolute_error: float
    relative_error: float
    numerical_rcs_db: float
    mie_rcs_db: float
    db_error: float
    numerical_normalized: float
    mie_normalized: float
    absolute_normalized_error: float
    abs_e_infinity_mie: float
    abs_etheta_mie: float
    abs_ephi_mie: float


@dataclass(frozen=True)
class PhiSummary:
    phi_degrees: float
    samples: int
    max_absolute_normalized_error: float
    max_relative_error: float
    rms_relative_error: float
    max_db_error: float


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare DiscoGMPI PEC-sphere RCS diagnostics against the exact "
            "Mie-series solution."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("output/metallic_sphere_scattering"),
        help="DiscoGMPI run directory. Default: output/metallic_sphere_scattering",
    )
    parser.add_argument(
        "--rcs-csv",
        type=Path,
        default=None,
        help="Numerical RCS CSV. Default: <run-dir>/diagnostics/rcs.csv",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Sphere radius. Default: read from resolved run config.",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=None,
        help="Incident wavelength. Default: read from resolved run config.",
    )
    parser.add_argument(
        "--ka",
        type=float,
        default=None,
        help="Size parameter k*a. Overrides --radius/--wavelength if set.",
    )
    parser.add_argument(
        "--terms",
        type=int,
        default=0,
        help="Mie-series truncation order. Default: Wiscombe-style automatic.",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=None,
        help=(
            "Output comparison CSV. Default: "
            "<run-dir>/diagnostics/mie_rcs_comparison.csv"
        ),
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=None,
        help=(
            "Output LaTeX summary table. Default: "
            "<run-dir>/tables/mie_rcs_comparison.tex"
        ),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help=(
            "Output comparison plot. Default: "
            "<run-dir>/plots/mie_rcs_comparison.pdf"
        ),
    )
    parser.add_argument(
        "--theta0-phi-plot-output",
        type=Path,
        default=None,
        help=(
            "Output theta=0 phi-cut plot. Default: "
            "<run-dir>/plots/mie_rcs_theta0_phi_cut.pdf"
        ),
    )
    parser.add_argument(
        "--theta0-phi-samples",
        type=int,
        default=181,
        help="Number of exact Mie samples for the theta=0, 0<=phi<=180 cut.",
    )
    parser.add_argument(
        "--theta0-tolerance-degrees",
        type=float,
        default=1.0e-10,
        help=(
            "Angular tolerance used to overlay numerical rows on the theta=0 "
            "phi-cut plot. Default: 1e-10 degrees."
        ),
    )
    parser.add_argument(
        "--polar-plot-output",
        type=Path,
        default=None,
        help=(
            "Output polar RCS plot. Default: "
            "<run-dir>/plots/mie_rcs_polar.pdf"
        ),
    )
    parser.add_argument(
        "--polar-mie-samples",
        type=int,
        default=721,
        help=(
            "Number of exact Mie samples per phi cut in the polar plot. "
            "Default: 721."
        ),
    )
    parser.add_argument(
        "--forward-backward-plot-output",
        type=Path,
        default=None,
        help=(
            "Output standalone forward-to-backward RCS plot. Default: "
            "<run-dir>/plots/mie_rcs_forward_backward.pdf"
        ),
    )
    parser.add_argument(
        "--forward-backward-mie-samples",
        type=int,
        default=721,
        help=(
            "Number of exact Mie samples per phi cut in the standalone "
            "forward-to-backward plot. Default: 721."
        ),
    )
    parser.add_argument(
        "--title",
        default="PEC sphere RCS: DiscoGMPI vs exact Mie series",
        help="Figure title.",
    )
    return parser.parse_args(argv)


def load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as stream:
        return tomllib.load(stream)


def run_configuration(run_dir: Path) -> dict[str, Any]:
    candidates = (
        run_dir / "config" / "resolved_config.toml",
        run_dir / "config" / "run_metadata.toml",
        run_dir / "run_metadata.toml",
    )
    for path in candidates:
        data = load_toml(path)
        if data:
            return dict(data.get("configuration", {}))
    return {}


def resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    run_dir = args.run_dir
    rcs_csv = args.rcs_csv or run_dir / "diagnostics" / "rcs.csv"
    comparison_csv = (
        args.comparison_csv
        or run_dir / "diagnostics" / "mie_rcs_comparison.csv"
    )
    latex_output = (
        args.latex_output or run_dir / "tables" / "mie_rcs_comparison.tex"
    )
    plot_output = args.plot_output or run_dir / "plots" / "mie_rcs_comparison.pdf"
    theta0_phi_plot_output = (
        args.theta0_phi_plot_output
        or run_dir / "plots" / "mie_rcs_theta0_phi_cut.pdf"
    )
    polar_plot_output = (
        args.polar_plot_output or run_dir / "plots" / "mie_rcs_polar.pdf"
    )
    forward_backward_plot_output = (
        args.forward_backward_plot_output
        or run_dir / "plots" / "mie_rcs_forward_backward.pdf"
    )
    return (
        comparison_csv,
        latex_output,
        plot_output,
        theta0_phi_plot_output,
        polar_plot_output,
        forward_backward_plot_output,
    )


def positive_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Could not resolve positive {name}.") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise SystemExit(f"{name} must be positive; got {number}.")
    return number


def resolve_geometry_and_size(
    args: argparse.Namespace,
    configuration: dict[str, Any],
) -> tuple[float, float, float]:
    radius = args.radius
    wavelength = args.wavelength
    if radius is None:
        radius = configuration.get("radius")
    if wavelength is None:
        wavelength = configuration.get("wavelength")

    if args.ka is not None:
        ka = positive_number(args.ka, "ka")
        if radius is None:
            radius = 1.0
        radius = positive_number(radius, "radius")
        wavelength = 2.0 * math.pi * radius / ka
        return radius, wavelength, ka

    radius = positive_number(radius, "radius")
    wavelength = positive_number(wavelength, "wavelength")
    ka = 2.0 * math.pi * radius / wavelength
    return radius, wavelength, ka


def read_rcs_csv(path: Path) -> list[RCSRow]:
    if not path.is_file():
        raise SystemExit(f"RCS CSV not found: {path}")

    rows: list[RCSRow] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"theta_degrees", "phi_degrees", "rcs"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise SystemExit(
                f"{path} is missing required columns: {sorted(missing)}"
            )
        for raw in reader:
            rcs = float(raw["rcs"])
            rcs_db = (
                float(raw["rcs_db"])
                if raw.get("rcs_db") not in (None, "", "-Inf")
                else None
            )
            rows.append(
                RCSRow(
                    theta_degrees=float(raw["theta_degrees"]),
                    phi_degrees=float(raw["phi_degrees"]),
                    numerical_rcs=rcs,
                    numerical_rcs_db=rcs_db,
                )
            )

    if not rows:
        raise SystemExit(f"{path} contains no RCS rows.")
    return rows


def automatic_mie_terms(size_parameter: float) -> int:
    return max(
        8,
        int(math.ceil(size_parameter + 4.0 * size_parameter ** (1.0 / 3.0) + 2.0)),
    )


def pec_mie_coefficients(size_parameter: float, terms: int) -> tuple[np.ndarray, np.ndarray]:
    """Return PEC sphere Mie coefficients for the selected size parameter."""
    n = np.arange(1, terms + 1)
    x = float(size_parameter)
    jn = spherical_jn(n, x)
    jn_prime = spherical_jn(n, x, derivative=True)
    yn = spherical_yn(n, x)
    yn_prime = spherical_yn(n, x, derivative=True)
    hn = jn + 1j * yn
    hn_prime = jn_prime + 1j * yn_prime

    psi = x * jn
    psi_prime = jn + x * jn_prime
    xi = x * hn
    xi_prime = hn + x * hn_prime

    # PEC sphere Mie coefficients.
    electric = -psi_prime / xi_prime
    magnetic = -psi / xi
    return electric, magnetic


def angular_functions(mu: float, terms: int) -> tuple[np.ndarray, np.ndarray]:
    pi_values = np.empty(terms, dtype=float)
    tau_values = np.empty(terms, dtype=float)
    if terms <= 0:
        return pi_values, tau_values

    # Standard Mie angular functions:
    # pi_1 = 1, pi_2 = 3*mu,
    # pi_n = ((2n - 1)/(n - 1))*mu*pi_{n-1} - (n/(n - 1))*pi_{n-2}.
    pi_values[0] = 1.0
    if terms > 1:
        pi_values[1] = 3.0 * mu
        for index in range(2, terms):
            n = index + 1
            pi_values[index] = (
                ((2.0 * n - 1.0) / (n - 1.0)) * mu * pi_values[index - 1]
                - (n / (n - 1.0)) * pi_values[index - 2]
            )

    for index, n in enumerate(range(1, terms + 1)):
        pi_previous = 0.0 if index == 0 else pi_values[index - 1]
        tau_values[index] = n * mu * pi_values[index] - (n + 1.0) * pi_previous

    return pi_values, tau_values


def mie_amplitudes(
    theta_degrees: float,
    electric_coefficients: np.ndarray,
    magnetic_coefficients: np.ndarray,
) -> tuple[complex, complex]:
    terms = len(electric_coefficients)
    theta = math.radians(theta_degrees)
    mu = math.cos(theta)
    pi_values, tau_values = angular_functions(mu, terms)
    n = np.arange(1, terms + 1, dtype=float)
    weights = (2.0 * n + 1.0) / (n * (n + 1.0))

    s1 = np.sum(
        weights
        * (electric_coefficients * pi_values + magnetic_coefficients * tau_values)
    )
    s2 = np.sum(
        weights
        * (electric_coefficients * tau_values + magnetic_coefficients * pi_values)
    )
    return complex(s1), complex(s2)


def exact_pec_sphere_rcs(
    theta_degrees: float,
    phi_degrees: float,
    wavenumber: float,
    electric_coefficients: np.ndarray,
    magnetic_coefficients: np.ndarray,
) -> tuple[float, float, float, float]:
    """Evaluate exact PEC-sphere RCS for the driver convention.

    The incident wave propagates in +z and is x-polarized, matching
    `distributed_metallic_sphere_scattering.jl`.
    """
    s1, s2 = mie_amplitudes(
        theta_degrees,
        electric_coefficients,
        magnetic_coefficients,
    )
    phi = math.radians(phi_degrees)

    # Incident E is x-polarized and the wave propagates in +z. The Mie
    # amplitude functions are dimensionless; E_infinity has dimensions of
    # length through the 1/k factor.
    e_theta = s2 * math.cos(phi) / wavenumber
    e_phi = -s1 * math.sin(phi) / wavenumber
    abs_e_theta = abs(e_theta)
    abs_e_phi = abs(e_phi)
    abs_e = math.sqrt(abs_e_theta * abs_e_theta + abs_e_phi * abs_e_phi)
    rcs = 4.0 * math.pi * abs_e * abs_e
    return rcs, abs_e, abs_e_theta, abs_e_phi


def db(value: float) -> float:
    return 10.0 * math.log10(value) if value > 0.0 else -math.inf


def compare_rows(
    numerical_rows: list[RCSRow],
    radius: float,
    wavelength: float,
    terms: int,
) -> list[ComparisonRow]:
    """Pair numerical DiscoGMPI RCS samples with exact Mie values."""
    wavenumber = 2.0 * math.pi / wavelength
    size_parameter = wavenumber * radius
    electric_coefficients, magnetic_coefficients = pec_mie_coefficients(
        size_parameter,
        terms,
    )
    normalization = math.pi * radius * radius

    comparison: list[ComparisonRow] = []
    for row in numerical_rows:
        mie_rcs, abs_e, abs_etheta, abs_ephi = exact_pec_sphere_rcs(
            row.theta_degrees,
            row.phi_degrees,
            wavenumber,
            electric_coefficients,
            magnetic_coefficients,
        )
        numerical = row.numerical_rcs
        absolute_error = abs(numerical - mie_rcs)
        relative_error = absolute_error / max(abs(mie_rcs), EPS)
        numerical_db = row.numerical_rcs_db if row.numerical_rcs_db is not None else db(numerical)
        mie_db = db(mie_rcs)
        db_error = abs(numerical_db - mie_db)
        comparison.append(
            ComparisonRow(
                theta_degrees=row.theta_degrees,
                phi_degrees=row.phi_degrees,
                numerical_rcs=numerical,
                mie_rcs=mie_rcs,
                absolute_error=absolute_error,
                relative_error=relative_error,
                numerical_rcs_db=numerical_db,
                mie_rcs_db=mie_db,
                db_error=db_error,
                numerical_normalized=numerical / normalization,
                mie_normalized=mie_rcs / normalization,
                absolute_normalized_error=absolute_error / normalization,
                abs_e_infinity_mie=abs_e,
                abs_etheta_mie=abs_etheta,
                abs_ephi_mie=abs_ephi,
            )
        )
    return comparison


def write_comparison_csv(path: Path, rows: list[ComparisonRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "theta_degrees",
                "phi_degrees",
                "rcs_numeric",
                "rcs_mie",
                "absolute_error",
                "relative_error",
                "rcs_numeric_db",
                "rcs_mie_db",
                "db_error",
                "rcs_numeric_over_pi_a2",
                "rcs_mie_over_pi_a2",
                "absolute_error_over_pi_a2",
                "abs_E_infinity_mie",
                "abs_Etheta_mie",
                "abs_Ephi_mie",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    f"{row.theta_degrees:.16e}",
                    f"{row.phi_degrees:.16e}",
                    f"{row.numerical_rcs:.16e}",
                    f"{row.mie_rcs:.16e}",
                    f"{row.absolute_error:.16e}",
                    f"{row.relative_error:.16e}",
                    f"{row.numerical_rcs_db:.16e}",
                    f"{row.mie_rcs_db:.16e}",
                    f"{row.db_error:.16e}",
                    f"{row.numerical_normalized:.16e}",
                    f"{row.mie_normalized:.16e}",
                    f"{row.absolute_normalized_error:.16e}",
                    f"{row.abs_e_infinity_mie:.16e}",
                    f"{row.abs_etheta_mie:.16e}",
                    f"{row.abs_ephi_mie:.16e}",
                )
            )


def group_by_phi(rows: list[ComparisonRow]) -> dict[float, list[ComparisonRow]]:
    groups: dict[float, list[ComparisonRow]] = {}
    for row in rows:
        groups.setdefault(row.phi_degrees, []).append(row)
    for group in groups.values():
        group.sort(key=lambda item: item.theta_degrees)
    return dict(sorted(groups.items()))


def summarize_by_phi(rows: list[ComparisonRow]) -> list[PhiSummary]:
    summaries: list[PhiSummary] = []
    for phi, group in group_by_phi(rows).items():
        relative_errors = np.array([item.relative_error for item in group])
        summaries.append(
            PhiSummary(
                phi_degrees=phi,
                samples=len(group),
                max_absolute_normalized_error=max(
                    item.absolute_normalized_error for item in group
                ),
                max_relative_error=float(np.max(relative_errors)),
                rms_relative_error=float(np.sqrt(np.mean(relative_errors**2))),
                max_db_error=max(item.db_error for item in group),
            )
        )
    return summaries


def latex_scientific(value: float, digits: int = 3) -> str:
    if value == 0.0:
        return "$0$"
    if not math.isfinite(value):
        return r"$\infty$" if value > 0 else r"$-\infty$"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10.0**exponent
    if -2 <= exponent <= 2:
        return f"${value:.{digits}g}$"
    return rf"${mantissa:.{digits}f}\times 10^{{{exponent}}}$"


def write_latex_table(
    path: Path,
    summaries: list[PhiSummary],
    *,
    radius: float,
    wavelength: float,
    ka: float,
    terms: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Requires \\usepackage{booktabs}",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        (
            r"  \caption{PEC sphere RCS comparison against the exact Mie "
            rf"series for $a={radius:.6g}$, $\lambda={wavelength:.6g}$, "
            rf"$ka={ka:.6g}$, using {terms} Mie terms." + "}"
        ),
        r"  \label{tab:pec-sphere-mie-rcs}",
        r"  \begin{tabular}{rrrrr}",
        r"    \toprule",
        (
            r"    $\phi$ & Samples & "
            r"$\max |\Delta\sigma|/(\pi a^2)$ & "
            r"$\max |\Delta\sigma|/|\sigma_{\mathrm{Mie}}|$ & "
            r"$\max |\Delta\sigma_{\mathrm{dB}}|$ \\"
        ),
        r"    \midrule",
    ]
    for summary in summaries:
        lines.append(
            "    "
            + " & ".join(
                (
                    f"${summary.phi_degrees:.3g}^\\circ$",
                    str(summary.samples),
                    latex_scientific(summary.max_absolute_normalized_error),
                    latex_scientific(summary.max_relative_error),
                    f"${summary.max_db_error:.3g}$",
                )
            )
            + r" \\"
        )
    lines.extend(
        (
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_comparison(
    path: Path,
    rows: list[ComparisonRow],
    *,
    title: str,
    radius: float,
    wavelength: float,
    ka: float,
    terms: int,
) -> None:
    """Create the two-panel RCS/Mie and relative-error validation figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = group_by_phi(rows)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(groups), 1)))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.0),
        sharex=True,
        constrained_layout=True,
    )
    ax_rcs, ax_error = axes

    for color, (phi, group) in zip(colors, groups.items()):
        theta = np.array([item.theta_degrees for item in group])
        numeric_db = np.array(
            [db(max(item.numerical_normalized, EPS)) for item in group]
        )
        mie_db = np.array([db(max(item.mie_normalized, EPS)) for item in group])
        relative_error = np.array([item.relative_error for item in group])
        label = rf"$\phi={phi:.3g}^\circ$"

        ax_rcs.plot(theta, mie_db, color=color, lw=2.0, label=f"Mie {label}")
        ax_rcs.plot(
            theta,
            numeric_db,
            linestyle="none",
            marker="o",
            markersize=4.0,
            markerfacecolor="none",
            markeredgewidth=1.1,
            color=color,
            label=f"DG {label}",
        )
        ax_error.semilogy(
            theta,
            np.maximum(relative_error, EPS),
            color=color,
            lw=1.8,
            marker="o",
            markersize=3.5,
            markerfacecolor="none",
            label=label,
        )

    ax_rcs.set_ylabel(r"$10\log_{10}(\sigma/(\pi a^2))$")
    ax_error.set_ylabel(r"$|\sigma_h-\sigma_{\rm Mie}|/|\sigma_{\rm Mie}|$")
    ax_error.set_xlabel(r"Scattering angle $\theta$ [deg]")
    ax_error.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax_error.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax_error.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=80)
    )
    ax_error.yaxis.set_minor_formatter(NullFormatter())

    for axis in axes:
        axis.grid(True, which="major", color="0.85", linewidth=0.8)
        axis.grid(True, which="minor", color="0.93", linewidth=0.5)
        axis.tick_params(direction="in", top=True, right=True)

    ax_rcs.legend(ncol=2, fontsize=8, frameon=True)
    ax_error.legend(ncol=2, fontsize=8, frameon=True)
    ax_rcs.set_title(
        f"{title}\n"
        rf"$a={radius:.4g}$, $\lambda={wavelength:.4g}$, "
        rf"$ka={ka:.4g}$, $N_\mathrm{{Mie}}={terms}$"
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_theta0_phi_cut(
    path: Path,
    rows: list[ComparisonRow],
    *,
    title: str,
    radius: float,
    wavelength: float,
    ka: float,
    terms: int,
    phi_samples: int,
    theta_tolerance_degrees: float,
) -> None:
    """Create the theta=0 RCS cut over 0<=phi<=180 degrees.

    At theta=0 the spherical azimuth is geometrically degenerate, but this cut
    is still useful for checking that the RCS extraction is insensitive to the
    observation-basis convention at the forward-scattering direction.
    """
    if phi_samples < 2:
        raise SystemExit("--theta0-phi-samples must be at least 2.")
    if theta_tolerance_degrees < 0.0:
        raise SystemExit("--theta0-tolerance-degrees must be non-negative.")

    path.parent.mkdir(parents=True, exist_ok=True)
    wavenumber = 2.0 * math.pi / wavelength
    normalization = math.pi * radius * radius
    electric_coefficients, magnetic_coefficients = pec_mie_coefficients(ka, terms)

    phi_exact = np.linspace(0.0, 180.0, phi_samples)
    mie_normalized = np.array(
        [
            exact_pec_sphere_rcs(
                0.0,
                float(phi),
                wavenumber,
                electric_coefficients,
                magnetic_coefficients,
            )[0]
            / normalization
            for phi in phi_exact
        ]
    )
    mie_db = np.array([db(max(value, EPS)) for value in mie_normalized])

    numerical_rows = sorted(
        (
            row
            for row in rows
            if abs(row.theta_degrees) <= theta_tolerance_degrees
            and 0.0 <= row.phi_degrees <= 180.0
        ),
        key=lambda item: item.phi_degrees,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.0),
        sharex=True,
        constrained_layout=True,
    )
    ax_rcs, ax_error = axes

    ax_rcs.plot(
        phi_exact,
        mie_db,
        color="C0",
        lw=2.0,
        label=r"Mie $\theta=0^\circ$",
    )

    if numerical_rows:
        phi_numeric = np.array([row.phi_degrees for row in numerical_rows])
        numeric_db = np.array(
            [db(max(row.numerical_normalized, EPS)) for row in numerical_rows]
        )
        relative_error = np.array(
            [max(row.relative_error, EPS) for row in numerical_rows]
        )
        ax_rcs.plot(
            phi_numeric,
            numeric_db,
            linestyle="none",
            marker="o",
            markersize=4.0,
            markerfacecolor="none",
            markeredgewidth=1.1,
            color="C1",
            label=r"DG $\theta=0^\circ$",
        )
        ax_error.semilogy(
            phi_numeric,
            relative_error,
            color="C1",
            lw=1.8,
            marker="o",
            markersize=3.5,
            markerfacecolor="none",
            label=r"$\theta=0^\circ$",
        )
        ax_error.legend(fontsize=8, frameon=True)
    else:
        ax_error.text(
            0.5,
            0.5,
            "No numerical theta=0 phi samples found in the RCS CSV.",
            ha="center",
            va="center",
            transform=ax_error.transAxes,
        )
        ax_error.set_yscale("log")

    ax_rcs.set_ylabel(r"$10\log_{10}(\sigma/(\pi a^2))$")
    ax_error.set_ylabel(r"$|\sigma_h-\sigma_{\rm Mie}|/|\sigma_{\rm Mie}|$")
    ax_error.set_xlabel(r"Azimuthal angle $\phi$ [deg]")
    ax_error.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax_error.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax_error.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=80)
    )
    ax_error.yaxis.set_minor_formatter(NullFormatter())

    for axis in axes:
        axis.set_xlim(0.0, 180.0)
        axis.grid(True, which="major", color="0.85", linewidth=0.8)
        axis.grid(True, which="minor", color="0.93", linewidth=0.5)
        axis.tick_params(direction="in", top=True, right=True)

    ax_rcs.legend(fontsize=8, frameon=True)
    ax_rcs.set_title(
        f"{title}: " + r"$\theta=0^\circ$ phi cut" + "\n"
        rf"$a={radius:.4g}$, $\lambda={wavelength:.4g}$, "
        rf"$ka={ka:.4g}$, $N_\mathrm{{Mie}}={terms}$"
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)


def db_tick_bounds(values: np.ndarray) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -40.0, 0.0, 10.0
    lower = math.floor(float(np.min(finite)) / 10.0) * 10.0
    upper = math.ceil(float(np.max(finite)) / 10.0) * 10.0
    if math.isclose(lower, upper):
        lower -= 10.0
        upper += 10.0
    span = upper - lower
    step = 5.0 if span <= 30.0 else 10.0
    return lower, upper, step


def plot_polar_rcs(
    path: Path,
    rows: list[ComparisonRow],
    *,
    title: str,
    radius: float,
    wavelength: float,
    ka: float,
    terms: int,
    mie_samples: int,
) -> None:
    """Create a polar RCS pattern in normalized dB units.

    Matplotlib polar axes cannot represent negative dB values as negative
    radii without flipping directions.  The plotted radius is therefore
    shifted by a common dB floor, while radial tick labels show the physical
    value of 10*log10(sigma/(pi*a^2)).
    """
    if mie_samples < 2:
        raise SystemExit("--polar-mie-samples must be at least 2.")

    path.parent.mkdir(parents=True, exist_ok=True)
    groups = group_by_phi(rows)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(groups), 1)))
    wavenumber = 2.0 * math.pi / wavelength
    normalization = math.pi * radius * radius
    electric_coefficients, magnetic_coefficients = pec_mie_coefficients(ka, terms)

    numeric_db_values: list[float] = []
    mie_db_values: list[float] = []
    dense_by_phi: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for phi, group in groups.items():
        theta_min = min(item.theta_degrees for item in group)
        theta_max = max(item.theta_degrees for item in group)
        theta_dense = np.linspace(theta_min, theta_max, mie_samples)
        mie_normalized = np.array(
            [
                exact_pec_sphere_rcs(
                    float(theta),
                    phi,
                    wavenumber,
                    electric_coefficients,
                    magnetic_coefficients,
                )[0]
                / normalization
                for theta in theta_dense
            ]
        )
        mie_db = np.array([db(max(value, EPS)) for value in mie_normalized])
        dense_by_phi[phi] = (theta_dense, mie_db)
        mie_db_values.extend(float(value) for value in mie_db)
        numeric_db_values.extend(
            db(max(item.numerical_normalized, EPS)) for item in group
        )

    all_db_values = np.array(mie_db_values + numeric_db_values, dtype=float)
    db_floor, db_ceiling, db_step = db_tick_bounds(all_db_values)

    fig = plt.figure(figsize=(7.0, 6.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")

    for color, (phi, group) in zip(colors, groups.items()):
        theta_dense, mie_db = dense_by_phi[phi]
        theta_numeric = np.array([item.theta_degrees for item in group])
        numeric_db = np.array(
            [db(max(item.numerical_normalized, EPS)) for item in group]
        )
        label = rf"$\phi={phi:.3g}^\circ$"

        ax.plot(
            np.deg2rad(theta_dense),
            np.maximum(mie_db - db_floor, 0.0),
            color=color,
            lw=2.0,
            label=f"Mie {label}",
        )
        ax.plot(
            np.deg2rad(theta_numeric),
            np.maximum(numeric_db - db_floor, 0.0),
            linestyle="none",
            marker="o",
            markersize=4.0,
            markerfacecolor="none",
            markeredgewidth=1.1,
            color=color,
            label=f"DG {label}",
        )

    theta_values = [item.theta_degrees for item in rows]
    if min(theta_values) >= 0.0 and max(theta_values) <= 180.0:
        ax.set_thetamin(0.0)
        ax.set_thetamax(180.0)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
    ax.set_xlabel(r"Scattering angle $\theta$")
    ax.grid(True, color="0.85", linewidth=0.8)

    radial_ticks_db = np.arange(db_floor, db_ceiling + 0.5 * db_step, db_step)
    radial_ticks = radial_ticks_db - db_floor
    ax.set_ylim(0.0, max(db_ceiling - db_floor, db_step))
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([rf"${tick:.0f}$ dB" for tick in radial_ticks_db])
    ax.set_rlabel_position(135.0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.12), fontsize=8)
    ax.set_title(
        f"{title}: polar RCS pattern\n"
        rf"$10\log_{{10}}(\sigma/(\pi a^2))$, "
        rf"$a={radius:.4g}$, $\lambda={wavelength:.4g}$, "
        rf"$ka={ka:.4g}$, $N_\mathrm{{Mie}}={terms}$"
    )

    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_forward_backward_rcs(
    path: Path,
    rows: list[ComparisonRow],
    *,
    title: str,
    radius: float,
    wavelength: float,
    ka: float,
    terms: int,
    mie_samples: int,
) -> None:
    """Create the usual forward-to-backward RCS curve.

    This is the paper-style E-plane plot: normalized RCS in dB versus the
    polar scattering angle theta, with theta=0 deg forward and theta=180 deg
    backscatter for the default +z incident plane wave.
    """
    if mie_samples < 2:
        raise SystemExit("--forward-backward-mie-samples must be at least 2.")

    path.parent.mkdir(parents=True, exist_ok=True)
    groups = group_by_phi(rows)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(groups), 1)))
    wavenumber = 2.0 * math.pi / wavelength
    normalization = math.pi * radius * radius
    electric_coefficients, magnetic_coefficients = pec_mie_coefficients(ka, terms)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)

    for color, (phi, group) in zip(colors, groups.items()):
        theta_numeric = np.array([item.theta_degrees for item in group])
        numeric_db = np.array(
            [db(max(item.numerical_normalized, EPS)) for item in group]
        )
        theta_dense = np.linspace(
            float(np.min(theta_numeric)),
            float(np.max(theta_numeric)),
            mie_samples,
        )
        mie_normalized = np.array(
            [
                exact_pec_sphere_rcs(
                    float(theta),
                    phi,
                    wavenumber,
                    electric_coefficients,
                    magnetic_coefficients,
                )[0]
                / normalization
                for theta in theta_dense
            ]
        )
        mie_db = np.array([db(max(value, EPS)) for value in mie_normalized])
        label = rf"$\phi={phi:.3g}^\circ$"

        ax.plot(
            theta_dense,
            mie_db,
            color=color,
            lw=2.0,
            label=f"Mie {label}",
        )
        ax.plot(
            theta_numeric,
            numeric_db,
            linestyle="none",
            marker="o",
            markersize=4.0,
            markerfacecolor="none",
            markeredgewidth=1.1,
            color=color,
            label=f"DG {label}",
        )

    ax.set_xlabel(r"Scattering angle $\theta$ [deg]")
    ax.set_ylabel(r"$10\log_{10}(\sigma/(\pi a^2))$")
    ax.set_xlim(
        min(item.theta_degrees for item in rows),
        max(item.theta_degrees for item in rows),
    )
    ax.grid(True, which="major", color="0.85", linewidth=0.8)
    ax.grid(True, which="minor", color="0.93", linewidth=0.5)
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(ncol=2, fontsize=8, frameon=True)
    ax.set_title(
        f"{title}: forward-to-backward RCS\n"
        rf"$a={radius:.4g}$, $\lambda={wavelength:.4g}$, "
        rf"$ka={ka:.4g}$, $N_\mathrm{{Mie}}={terms}$"
    )

    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    configuration = run_configuration(args.run_dir)
    (
        comparison_csv,
        latex_output,
        plot_output,
        theta0_phi_plot_output,
        polar_plot_output,
        forward_backward_plot_output,
    ) = resolve_paths(args)
    rcs_csv = args.rcs_csv or args.run_dir / "diagnostics" / "rcs.csv"

    radius, wavelength, ka = resolve_geometry_and_size(args, configuration)
    terms = args.terms if args.terms > 0 else automatic_mie_terms(ka)

    numerical_rows = read_rcs_csv(rcs_csv)
    comparison_rows = compare_rows(
        numerical_rows,
        radius,
        wavelength,
        terms,
    )
    summaries = summarize_by_phi(comparison_rows)

    write_comparison_csv(comparison_csv, comparison_rows)
    write_latex_table(
        latex_output,
        summaries,
        radius=radius,
        wavelength=wavelength,
        ka=ka,
        terms=terms,
    )
    plot_comparison(
        plot_output,
        comparison_rows,
        title=args.title,
        radius=radius,
        wavelength=wavelength,
        ka=ka,
        terms=terms,
    )
    plot_theta0_phi_cut(
        theta0_phi_plot_output,
        comparison_rows,
        title=args.title,
        radius=radius,
        wavelength=wavelength,
        ka=ka,
        terms=terms,
        phi_samples=args.theta0_phi_samples,
        theta_tolerance_degrees=args.theta0_tolerance_degrees,
    )
    plot_polar_rcs(
        polar_plot_output,
        comparison_rows,
        title=args.title,
        radius=radius,
        wavelength=wavelength,
        ka=ka,
        terms=terms,
        mie_samples=args.polar_mie_samples,
    )
    plot_forward_backward_rcs(
        forward_backward_plot_output,
        comparison_rows,
        title=args.title,
        radius=radius,
        wavelength=wavelength,
        ka=ka,
        terms=terms,
        mie_samples=args.forward_backward_mie_samples,
    )

    print("PEC sphere Mie RCS comparison")
    print("-----------------------------")
    print(f"run directory:     {args.run_dir}")
    print(f"input RCS CSV:     {rcs_csv}")
    print(f"radius:            {radius}")
    print(f"wavelength:        {wavelength}")
    print(f"ka:                {ka}")
    print(f"Mie terms:         {terms}")
    print(f"comparison CSV:    {comparison_csv}")
    print(f"LaTeX table:       {latex_output}")
    print(f"plot:              {plot_output}")
    print(f"theta=0 phi plot:  {theta0_phi_plot_output}")
    print(f"polar RCS plot:    {polar_plot_output}")
    print(f"forward-back plot: {forward_backward_plot_output}")
    for summary in summaries:
        print(
            f"phi={summary.phi_degrees:g} deg: "
            f"max rel={summary.max_relative_error:.3e}, "
            f"max dB={summary.max_db_error:.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
