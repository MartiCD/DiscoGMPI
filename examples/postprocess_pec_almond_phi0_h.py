#!/usr/bin/env python3

"""Postprocess one Austin monostatic PEC-almond case: phi_0_H by default.

This script intentionally ignores the aggregate monostatic sweep CSV.  It reads
one case directory directly:

    <case-dir>/diagnostics/rcs.csv
    <case-dir>/diagnostics/pec_boundary_residual.csv
    <case-dir>/config/resolved_config.toml

For an H case it uses the phi-polarized far-field component.  For a V case it
uses the theta-polarized far-field component.  The result is compared against
the Austin CEM Problem IIIA reference curve.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/discogmpi-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


C0 = 299_792_458.0
DEFAULT_CASE_DIR = Path(
    "output/MN5/pec_almond_austin_monostatic_debug/cases/phi_0_H"
)
DEFAULT_REFERENCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "benchmark"
    / "AustinCEMBenchmarks"
    / "Problem_III-Almonds"
    / "IIIA-Reference Data"
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one Austin monostatic almond case, e.g. cases/phi_0_H, "
            "and compare its RCS point against the Austin CEM reference curve."
        )
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=DEFAULT_CASE_DIR,
        help=f"Case directory. Default: {DEFAULT_CASE_DIR}",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help=f"Austin reference-data directory. Default: {DEFAULT_REFERENCE_DIR}",
    )
    parser.add_argument(
        "--frequency-key",
        choices=("fx1", "fx2", "fx3", "fx4"),
        default="fx1",
        help="Austin frequency key. Default: fx1.",
    )
    parser.add_argument(
        "--reference-kind",
        choices=("ref", "meas-m1", "meas-m2"),
        default="ref",
        help="Austin reference source. Default: ref.",
    )
    parser.add_argument(
        "--polarization",
        choices=("auto", "H", "V"),
        default="auto",
        help="Case polarization. Default: infer from case directory suffix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <case-dir>/plots.",
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


def infer_polarization(case_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    name = case_dir.name.upper()
    if name.endswith("_H"):
        return "H"
    if name.endswith("_V"):
        return "V"
    raise ValueError(
        "Could not infer polarization from case directory name. "
        "Use --polarization=H or --polarization=V."
    )


def reference_file(reference_dir: Path, kind: str, key: str, polarization: str) -> Path:
    if kind == "ref":
        name = f"ref_rcs.III.A.s1.{key}.{polarization}.txt"
    elif kind == "meas-m1":
        name = f"meas.m1_rcs.III.A.s1.{key}.{polarization}.txt"
    else:
        name = f"meas.m2_rcs.III.A.s1.{key}.{polarization}.txt"
    return reference_dir / name


def read_one_csv_row(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}; found {len(rows)}.")
    return rows[0]


def read_resolved_config(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    config: dict[str, str] = {}
    in_configuration = False
    with path.open() as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                in_configuration = stripped == "[configuration]"
                continue
            if not in_configuration or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            config[key.strip()] = value.strip().strip('"')
    return config


def config_float(config: dict[str, str], key: str, default: float) -> float:
    raw = config.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def read_residuals(path: Path) -> tuple[float, float]:
    if not path.is_file():
        return math.nan, math.nan
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return math.nan, math.nan
    values = [float(row["relative_rms_norm"]) for row in rows]
    return values[-1], max(values)


def read_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Austin reference file not found: {path}")
    phi: list[float] = []
    dbsm: list[float] = []
    frequencies: list[float] = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 4:
                raise ValueError(f"Malformed row {line_number} in {path}")
            frequencies.append(float(fields[0]))
            phi.append(float(fields[2]))
            dbsm.append(float(fields[3]))
    if not phi:
        raise ValueError(f"No Austin reference data found in {path}")

    frequency_hz = float(np.mean(frequencies))
    wavelength = C0 / frequency_hz
    phi_array = np.asarray(phi, dtype=float)
    dbsm_array = np.asarray(dbsm, dtype=float)
    order = np.argsort(phi_array)
    phi_array = phi_array[order]
    dbsm_array = dbsm_array[order]
    normalized_db = dbsm_array - 20.0 * np.log10(wavelength)
    normalized_linear = np.power(10.0, normalized_db / 10.0)
    return phi_array, normalized_db, normalized_linear, wavelength


def safe_db(value: float) -> float:
    return 10.0 * math.log10(value) if value > 0.0 else math.nan


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


def save_plot(fig: plt.Figure, path: Path, write_png: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if write_png:
        fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def write_comparison_csv(
    path: Path,
    polarization: str,
    phi_degrees: float,
    dg_sigma: float,
    dg_sigma_norm: float,
    dg_sigma_norm_db: float,
    ref_norm_db: float,
    final_residual: float,
    max_residual: float,
    samples: int,
    window_sum: float,
    reference_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "polarization",
                "phi_degrees",
                "dg_rcs",
                "dg_rcs_over_lambda2",
                "dg_rcs_over_lambda2_db",
                "austin_interp_over_lambda2_db",
                "error_db",
                "samples",
                "window_sum",
                "final_pec_relative_rms",
                "max_pec_relative_rms",
                "reference_file",
            ]
        )
        writer.writerow(
            [
                polarization,
                f"{phi_degrees:.16e}",
                f"{dg_sigma:.16e}",
                f"{dg_sigma_norm:.16e}",
                f"{dg_sigma_norm_db:.16e}",
                f"{ref_norm_db:.16e}",
                f"{dg_sigma_norm_db - ref_norm_db:.16e}",
                samples,
                f"{window_sum:.16e}",
                f"{final_residual:.16e}",
                f"{max_residual:.16e}",
                reference_path,
            ]
        )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    case_dir = args.case_dir
    polarization = infer_polarization(case_dir, args.polarization)
    output_dir = args.output_dir or (case_dir / "plots")

    rcs_path = case_dir / "diagnostics" / "rcs.csv"
    residual_path = case_dir / "diagnostics" / "pec_boundary_residual.csv"
    config_path = case_dir / "config" / "resolved_config.toml"
    reference_path = reference_file(
        args.reference_dir,
        args.reference_kind,
        args.frequency_key,
        polarization,
    )

    rcs_row = read_one_csv_row(rcs_path)
    config = read_resolved_config(config_path)
    final_residual, max_residual = read_residuals(residual_path)

    component = "phi" if polarization == "H" else "theta"
    component_key = "abs_Ephi" if polarization == "H" else "abs_Etheta"
    far_field_value = float(rcs_row[component_key])
    amplitude = abs(config_float(config, "amplitude", 1.0))
    wavelength = config_float(config, "wavelength", math.nan)
    if not math.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError(
            f"Could not resolve a positive wavelength from {config_path}."
        )

    phi_degrees = float(rcs_row["phi_degrees"])
    samples = int(rcs_row["samples"])
    window_sum = float(rcs_row["window_sum"])

    dg_sigma = 4.0 * math.pi * far_field_value**2 / max(amplitude**2, np.finfo(float).eps)
    dg_sigma_norm = dg_sigma / wavelength**2
    dg_sigma_norm_db = safe_db(dg_sigma_norm)

    ref_phi, ref_norm_db, ref_norm_linear, ref_wavelength = read_reference(reference_path)
    if ref_phi[0] <= phi_degrees <= ref_phi[-1]:
        ref_interp_db = float(np.interp(phi_degrees, ref_phi, ref_norm_db))
    else:
        ref_interp_db = math.nan

    configure_matplotlib()
    title = args.title or (
        f"NASA PEC almond RCS: {args.frequency_key}, {polarization}, "
        rf"$\phi={phi_degrees:g}^\circ$"
    )

    db_plot = output_dir / f"pec_almond_phi_{phi_degrees:g}_{polarization}_rcs_db.pdf"
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        ref_phi,
        ref_norm_db,
        color="#3f2e7e",
        label=f"Austin {polarization} ({args.frequency_key})",
    )
    ax.plot(
        [phi_degrees],
        [dg_sigma_norm_db],
        color="#d95f02",
        marker="o",
        linestyle="none",
        markerfacecolor="white",
        markeredgewidth=1.8,
        markersize=7.0,
        label=f"DiscoGMPI {polarization} ({component})",
    )
    ax.set_xlabel(r"Observation azimuth $\phi$ [deg]")
    ax.set_ylabel(r"$10\log_{10}(\sigma/\lambda^2)$ [dB]")
    ax.set_title(title)
    ax.grid(True, which="major", color="0.86", linewidth=0.9)
    ax.grid(True, which="minor", color="0.93", linewidth=0.5)
    ax.minorticks_on()
    ax.legend(frameon=True, framealpha=0.95)
    save_plot(fig, db_plot, args.write_png)

    linear_plot = (
        output_dir / f"pec_almond_phi_{phi_degrees:g}_{polarization}_rcs_linear.pdf"
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.semilogy(
        ref_phi,
        ref_norm_linear,
        color="#3f2e7e",
        label=f"Austin {polarization} ({args.frequency_key})",
    )
    ax.semilogy(
        [phi_degrees],
        [dg_sigma_norm],
        color="#d95f02",
        marker="o",
        linestyle="none",
        markerfacecolor="white",
        markeredgewidth=1.8,
        markersize=7.0,
        label=f"DiscoGMPI {polarization} ({component})",
    )
    ax.set_xlabel(r"Observation azimuth $\phi$ [deg]")
    ax.set_ylabel(r"$\sigma/\lambda^2$")
    ax.set_title(title)
    ax.grid(True, which="major", color="0.86", linewidth=0.9)
    ax.grid(True, which="minor", color="0.93", linewidth=0.5)
    ax.minorticks_on()
    ax.legend(frameon=True, framealpha=0.95)
    save_plot(fig, linear_plot, args.write_png)

    comparison_csv = (
        output_dir / f"pec_almond_phi_{phi_degrees:g}_{polarization}_comparison.csv"
    )
    write_comparison_csv(
        comparison_csv,
        polarization,
        phi_degrees,
        dg_sigma,
        dg_sigma_norm,
        dg_sigma_norm_db,
        ref_interp_db,
        final_residual,
        max_residual,
        samples,
        window_sum,
        reference_path,
    )

    print("Single-case Austin PEC almond postprocessing")
    print("--------------------------------------------")
    print(f"case dir:              {case_dir}")
    print(f"polarization/component:{polarization}/{component}")
    print(f"phi:                   {phi_degrees:g} deg")
    print(f"DG sigma/lambda^2:     {dg_sigma_norm:.8e}")
    print(f"DG RCS normalized dB:  {dg_sigma_norm_db:.8e}")
    print(f"Austin interp dB:      {ref_interp_db:.8e}")
    print(f"error dB:              {dg_sigma_norm_db - ref_interp_db:.8e}")
    print(f"final PEC residual:    {final_residual:.8e}")
    print(f"max PEC residual:      {max_residual:.8e}")
    print(f"dB plot:               {db_plot}")
    print(f"linear plot:           {linear_plot}")
    print(f"comparison CSV:        {comparison_csv}")
    print(f"reference wavelength:  {ref_wavelength:.8e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
