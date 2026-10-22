#!/usr/bin/env python3
"""Render the solver-independent FFD comparison CSVs.

The Julia comparison program supplies all physical conversions and errors.
This helper only renders ``samples.csv`` and ``pairs.csv`` using the Python
standard library and matplotlib. It does not read FFD files or fit field data.
Each figure is written as vector PDF/SVG and a 300 dpi PNG.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/discogmpi-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator


# Okabe-Ito colors remain distinguishable in the three-frequency benchmark.
CASE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
SOURCE_STYLES = ("-", "--", ":", "-.")
SAMPLE_NUMBERS = (
    "freq_hz", "angle_rad", "theta_rad", "phi_rad", "rx", "ry", "rz",
    "Etheta_re", "Etheta_im", "Ephi_re", "Ephi_im", "Eco_re", "Eco_im",
    "Ecross_re", "Ecross_im", "sigma_total_m2", "sigma_theta_m2",
    "sigma_phi_m2", "sigma_co_m2", "sigma_cross_m2",
)
PAIR_NUMBERS = ("angle_rad", "sigma_error_normalized", "db_error", "complex_error_normalized")
ERROR_DISPLAY_FLOOR = 1.0e-16


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "text.usetex": False,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.7,
        "axes.axisbelow": True,
        "lines.linewidth": 1.55,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def read_csv(path: Path, labels: tuple[str, ...], numbers: tuple[str, ...]) -> list[dict]:
    """Read and validate the columns used by the renderer, permitting extra columns."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(labels + numbers) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing CSV columns: {', '.join(sorted(missing))}")
        rows = []
        for line_number, values in enumerate(reader, start=2):
            row = {key: values[key] for key in labels}
            if any(not row[key] for key in labels):
                raise ValueError(f"{path}:{line_number}: empty series label")
            for key in numbers:
                try:
                    value = float(values[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: invalid {key}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"{path}:{line_number}: non-finite {key}")
                row[key] = value
            rows.append(row)
    return rows


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.") or "unnamed"


def source_rank(value: str) -> tuple[int, str]:
    name = value.lower()
    if "mie" in name or "exact" in name:
        return 0, name
    if "emagpie" in name:
        return 1, name
    if "discogmpi" in name or "disco" in name:
        return 2, name
    return 3, name


def cut_rank(value: str) -> tuple[int, str]:
    name = value.lower()
    if "phi" in name or "equator" in name:
        return 0, name
    if "theta" in name or "meridian" in name:
        return 1, name
    return 2, name


def cut_title(cut: str) -> str:
    if cut_rank(cut)[0] == 0:
        return r"Equatorial cut: $\theta=90^\circ$"
    if cut_rank(cut)[0] == 1:
        return r"Signed meridional cut: $\phi=0^\circ$"
    return cut.replace("_", " ")


def angle_label(cut: str) -> str:
    if cut_rank(cut)[0] == 0:
        return r"Azimuth $\phi$ (deg)"
    if cut_rank(cut)[0] == 1:
        return r"Signed polar angle $\theta$ (deg)"
    return "Cut angle (deg)"


def rcs_db(value: float, floor: float) -> float:
    if value < 0:
        raise ValueError("RCS must be nonnegative")
    return max(floor, 10.0 * math.log10(value)) if value else floor


def decorate(ax, cut: str, ylabel: str, *, xlabel: bool = True) -> None:
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(angle_label(cut))
    ax.grid(True, which="major")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.margins(x=0.015)


def save_figure(fig, output_dir: Path, name: str) -> list[Path]:
    paths = []
    for extension in ("pdf", "svg", "png"):
        path = output_dir / f"{name}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
        paths.append(path)
    plt.close(fig)
    return paths


def make_styles(samples: list[dict]):
    sources = sorted({row["source"] for row in samples}, key=source_rank)
    cases = sorted({row["case"] for row in samples}, key=lambda case: (
        min(row["freq_hz"] for row in samples if row["case"] == case), case))
    colors = {case: CASE_COLORS[index % len(CASE_COLORS)] for index, case in enumerate(cases)}
    styles = {}
    for index, source in enumerate(sources):
        rank = source_rank(source)[0]
        styles[source] = SOURCE_STYLES[(rank if rank < 3 else index) % len(SOURCE_STYLES)]
    labels = {}
    for case in cases:
        frequencies = {row["freq_hz"] for row in samples if row["case"] == case}
        frequency = min(frequencies)
        labels[case] = f"{case}: {frequency / 1e6:.6g} MHz"
    return sources, cases, colors, styles, labels


def legend_handles(sources, cases, colors, styles, labels):
    return ([Line2D([], [], color=colors[case], label=labels[case]) for case in cases] +
            [Line2D([], [], color="0.25", linestyle=styles[source], label=source)
             for source in sources])


def add_legend(fig, handles, *, bottom: float = -0.01) -> None:
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, bottom),
               ncol=min(3, len(handles)), frameon=False, handlelength=2.8,
               columnspacing=1.6)


def group_samples(samples: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in samples:
        groups[(row["source"], row["case"], row["cut"])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: row["angle_rad"])
    return groups


def render_comparison(samples: list[dict], pairs: list[dict], output_dir: Path,
                      db_floor: float = -100.0) -> list[Path]:
    """Render all figures and return their paths. No fitted transformations are applied."""
    if not samples:
        raise ValueError("samples CSV contains no data rows")
    if not math.isfinite(db_floor):
        raise ValueError("dB floor must be finite")
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources, cases, colors, styles, labels = make_styles(samples)
    cuts = sorted({row["cut"] for row in samples}, key=cut_rank)
    groups = group_samples(samples)
    handles = legend_handles(sources, cases, colors, styles, labels)
    paths = []

    fig, axes = plt.subplots(1, len(cuts), figsize=(3.8 * len(cuts), 3.7), squeeze=False)
    for ax, cut in zip(axes[0], cuts):
        for case in cases:
            for source in sources:
                rows = groups.get((source, case, cut), [])
                if rows:
                    ax.plot([math.degrees(row["angle_rad"]) for row in rows],
                            [rcs_db(row["sigma_total_m2"], db_floor) for row in rows],
                            color=colors[case], linestyle=styles[source])
        ax.set_title(cut_title(cut))
        decorate(ax, cut, r"Total RCS (dBsm)")
    add_legend(fig, handles)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    paths += save_figure(fig, output_dir, "total_rcs")

    components = (("sigma_theta_m2", r"$\theta$ component"),
                  ("sigma_phi_m2", r"$\phi$ component"),
                  ("sigma_co_m2", "Co-polar component"),
                  ("sigma_cross_m2", "Cross-polar component"))
    for cut in cuts:
        fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.6), squeeze=False)
        for ax, (column, title) in zip(axes.flat, components):
            for case in cases:
                for source in sources:
                    rows = groups.get((source, case, cut), [])
                    if rows:
                        ax.plot([math.degrees(row["angle_rad"]) for row in rows],
                                [rcs_db(row[column], db_floor) for row in rows],
                                color=colors[case], linestyle=styles[source])
            ax.set_title(title)
            decorate(ax, cut, "RCS (dBsm)")
        fig.suptitle(cut_title(cut), y=0.99)
        add_legend(fig, handles)
        fig.tight_layout(rect=(0, 0.10, 1, 0.96))
        paths += save_figure(fig, output_dir, f"polarized_rcs_{slug(cut)}")

    complex_columns = (("Eco_re", r"$\operatorname{Re} E_{\rm co}$ (m)"),
                       ("Eco_im", r"$\operatorname{Im} E_{\rm co}$ (m)"),
                       ("co_abs", r"$|E_{\rm co}|$ (m)"),
                       ("cross_abs", r"$|E_{\rm cross}|$ (m)"))
    for case in cases:
        for cut in cuts:
            present = [source for source in sources if (source, case, cut) in groups]
            if not present:
                continue
            fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.2), squeeze=False)
            for ax, (column, ylabel) in zip(axes.flat, complex_columns):
                for source in present:
                    rows = groups[(source, case, cut)]
                    if column == "co_abs":
                        values = [math.hypot(row["Eco_re"], row["Eco_im"]) for row in rows]
                    elif column == "cross_abs":
                        values = [math.hypot(row["Ecross_re"], row["Ecross_im"]) for row in rows]
                    else:
                        values = [row[column] for row in rows]
                    ax.plot([math.degrees(row["angle_rad"]) for row in rows], values,
                            color=colors[case], linestyle=styles[source])
                decorate(ax, cut, ylabel)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
            fig.suptitle(f"{labels[case]} — {cut_title(cut)}", y=0.99)
            add_legend(fig, [Line2D([], [], color=colors[case], linestyle=styles[source],
                                   label=source) for source in present])
            fig.tight_layout(rect=(0, 0.065, 1, 0.96))
            paths += save_figure(fig, output_dir, f"complex_fields_{slug(case)}_{slug(cut)}")

    pair_groups = defaultdict(list)
    for row in pairs:
        pair_groups[(row["source"], row["reference"], row["case"], row["cut"])].append(row)
    for rows in pair_groups.values():
        rows.sort(key=lambda row: row["angle_rad"])
    fig, axes = plt.subplots(3, len(cuts), figsize=(3.8 * len(cuts), 8.0), squeeze=False)
    error_columns = (("sigma_error_normalized", "Normalized RCS error", True),
                     ("db_error", "RCS difference (dB)", False),
                     ("complex_error_normalized", "Normalized complex error", True))
    pair_handles = []
    seen_pairs = set()
    for column_index, cut in enumerate(cuts):
        for row_index, (column, ylabel, logarithmic) in enumerate(error_columns):
            ax = axes[row_index, column_index]
            for (source, reference, case, group_cut), rows in sorted(pair_groups.items()):
                if cut != group_cut:
                    continue
                color = colors.get(case, "0.4")
                linestyle = styles.get(source, "--")
                values = [abs(row[column]) if logarithmic else row[column] for row in rows]
                if logarithmic:
                    values = [max(value, ERROR_DISPLAY_FLOOR) for value in values]
                ax.plot([math.degrees(row["angle_rad"]) for row in rows], values,
                        color=color, linestyle=linestyle)
                pair = (source, reference)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    pair_handles.append(Line2D([], [], color="0.25", linestyle=linestyle,
                                               label=f"{source} − {reference}"))
            if row_index == 0:
                ax.set_title(cut_title(cut))
            if logarithmic:
                ax.set_yscale("log")
            else:
                ax.axhline(0, color="0.5", linewidth=0.65, zorder=0)
            if not any(key[3] == cut for key in pair_groups):
                ax.text(0.5, 0.5, "No matched comparison samples", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9)
            decorate(ax, cut, ylabel, xlabel=row_index == 2)
    error_handles = [Line2D([], [], color=colors[case], label=labels[case]) for case in cases]
    add_legend(fig, error_handles + pair_handles, bottom=0.025)
    fig.text(0.5, 0.005, r"Zero errors are shown at $10^{-16}$; fields use the declared convention.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    paths += save_figure(fig, output_dir, "rcs_errors")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True, help="Comparison samples CSV")
    parser.add_argument("--pairs", type=Path, required=True, help="Pointwise matched errors CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--db-floor", type=float, default=-100.0,
                        help="Display floor in dBsm (default: -100); CSV values are unaffected")
    args = parser.parse_args(argv)
    try:
        samples = read_csv(args.samples, ("source", "case", "cut"), SAMPLE_NUMBERS)
        pairs = read_csv(args.pairs, ("source", "reference", "case", "cut"), PAIR_NUMBERS)
        paths = render_comparison(samples, pairs, args.output_dir, args.db_floor)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Plotting error: {exc}\n")
    print(f"Wrote {len(paths)} plot files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
