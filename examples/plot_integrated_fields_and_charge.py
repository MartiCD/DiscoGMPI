#!/usr/bin/env python3
"""Publication plots of integrated D/B components and divergence-based charges.

Input: a run directory or quadrature_diagnostics.csv. Different dimensional
quantities are kept on separate axes; no relative normalization or detrending.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter, ScalarFormatter
import numpy as np


FIELDS = (
    ("Electric", "d", "D", "electric_charge", "E"),
    ("Magnetic", "b", "B", "magnetic_charge", "H"),
)
COLORS = ("#0072B2", "#D55E00", "#009E73")
STYLES = ("-", (0, (5, 2)), (0, (8, 2, 2, 2)))
REQUIRED = ("time", "electric_charge", "magnetic_charge") + tuple(
    f"integrated_{f}_{c}" for f in ("d", "b") for c in "xyz"
)
DIRECTIONAL = tuple(f"{field}_charge_{c}" for field in ("electric", "magnetic") for c in "xyz")
LEGACY = tuple(f"discog_{field}_charge{variant}{suffix}"
               for field in ("electric", "magnetic")
               for variant in ("", "_strong") for suffix in ("", "_x", "_y", "_z"))


def input_path(path):
    return path / "diagnostics/quadrature_diagnostics.csv" if path.is_dir() else path


def read_data(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}. "
                             "Older histories require a new simulation.")
        present = set(DIRECTIONAL) & set(reader.fieldnames or ())
        if present and len(present) != len(DIRECTIONAL):
            raise ValueError("Incomplete directional charge columns: " +
                             ", ".join(sorted(set(DIRECTIONAL) - present)))
        rows = list(reader)
        legacy = set(LEGACY) & set(reader.fieldnames or ())
        if legacy and len(legacy) != len(LEGACY):
            raise ValueError("Incomplete DiscoG charge columns")
    if len(rows) < 2:
        raise ValueError("At least two diagnostic samples are required.")
    data = {}
    for name in REQUIRED + (DIRECTIONAL if present else ()) + (LEGACY if legacy else ()):
        try:
            data[name] = np.array([float(row[name]) for row in rows])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid numeric data in column {name}") from exc
        if not np.all(np.isfinite(data[name])):
            raise ValueError(f"Nonfinite data in column {name}")
    if np.any(np.diff(data["time"]) <= 0):
        raise ValueError("Times must be strictly increasing; check appended/restarted histories.")
    return data


def drift(values):
    return values - values[0]


def integrated_sum(data, field):
    """Test D or B against (1,1,1); also works with older component-only CSVs."""
    return sum(data[f"integrated_{field}_{c}"] for c in "xyz")


def summary_rows(data):
    data = dict(data)
    for field in ("d", "b"):
        data[f"integrated_{field}_sum"] = integrated_sum(data, field)
    rows = []
    for name in REQUIRED[1:] + ("integrated_d_sum", "integrated_b_sum") + tuple(k for k in DIRECTIONAL + LEGACY if k in data):
        y = data[name]
        dy = drift(y)
        rows.append(dict(quantity=name, reference_time=data["time"][0],
                         final_time=data["time"][-1], samples=len(y),
                         initial_value=y[0], final_value=y[-1],
                         signed_final_drift=dy[-1],
                         max_abs_sampled_drift=np.max(np.abs(dy)),
                         sample_rms_drift=np.linalg.norm(dy) / np.sqrt(len(dy))))
    return rows


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.7, "lines.linewidth": 1.25,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "path.simplify": False,
    })


def decorate(axes):
    for i, ax in enumerate(axes.flat):
        ax.grid(which="major", color="0.86", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.text(0.025, 0.96, f"({chr(97+i)})", transform=ax.transAxes,
                va="top", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
        ax.margins(x=0.02)
    for ax in axes[-1]:
        ax.set_xlabel(r"Time $t$")


def linear_axis(ax):
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, 3))
    fmt.set_useOffset(False)
    ax.yaxis.set_major_formatter(fmt)


def log_axis(ax, arrays):
    # Exact zeros have no logarithm: omit them, never replace with a fake floor.
    positive = np.concatenate([a[a > 0] for a in arrays])
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_formatter(NullFormatter())
    if positive.size:
        lo = np.floor(np.log10(positive.min()))
        hi = np.ceil(np.log10(positive.max()))
        ax.set_ylim(10.0**lo, 10.0**max(hi, lo + 1))
    else:
        ax.set_ylim(1e-16, 1)
        ax.set_yticks([])
        ax.text(0.5, 0.5, "All sampled drifts are exactly zero",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)


def make_figures(data, title):
    style()
    signed, sa = plt.subplots(3, 2, figsize=(7.2, 8.1), sharex=True,
                             constrained_layout=True)
    absolute, aa = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex=True,
                               constrained_layout=True)
    time = data["time"]
    for col, (name, field, symbol, charge, qsymbol) in enumerate(FIELDS):
        for axes in (sa, aa):
            axes[0, col].set_title(name, pad=8)
        absolute_components = []
        for i, component in enumerate("xyz"):
            dy = drift(data[f"integrated_{field}_{component}"])
            label = rf"$\Delta I_{{{symbol},{component}}}$"
            sa[0, col].plot(time, dy, color=COLORS[i], ls=STYLES[i], label=label)
            absdy = np.abs(dy)
            absolute_components.append(absdy)
            aa[0, col].plot(time, np.ma.masked_equal(absdy, 0),
                            color=COLORS[i], ls=STYLES[i], label=label)
        total_drift = drift(integrated_sum(data, field))
        sa[0, col].plot(time, total_drift, color="0.15", lw=1.5, label="sum (1,1,1)")
        aa[0, col].plot(time, np.ma.masked_equal(np.abs(total_drift), 0),
                        color="0.15", lw=1.5, label="sum (1,1,1)")
        absolute_components.append(np.abs(total_drift))
        for axes in (sa, aa):
            axes[0, col].legend(loc="upper right", frameon=True, framealpha=0.95,
                                edgecolor="0.85", ncol=2, handlelength=2.1,
                                columnspacing=0.8)
        sa[0, col].set_ylabel(rf"$\Delta I_{{{symbol},i}}$")
        aa[0, col].set_ylabel(rf"$|\Delta I_{{{symbol},i}}|$")
        q = data[charge]
        dq = drift(q)
        sa[1, col].plot(time, q, color="#332288")
        sa[1, col].set_ylabel(rf"Charge $Q_{{{qsymbol},h}}$")
        sa[2, col].plot(time, dq, color="#332288")
        sa[2, col].set_ylabel(rf"$\Delta Q_{{{qsymbol},h}}$")
        aa[1, col].plot(time, np.ma.masked_equal(np.abs(dq), 0), color="#332288")
        aa[1, col].set_ylabel(rf"$|\Delta Q_{{{qsymbol},h}}|$")
        for ax in sa[:, col]:
            ax.axhline(0, color="0.5", lw=0.5, zorder=0)
            linear_axis(ax)
        log_axis(aa[0, col], absolute_components)
        log_axis(aa[1, col], [np.abs(dq)])
    for fig, axes in ((signed, sa), (absolute, aa)):
        decorate(axes)
        if title:
            fig.suptitle(title, fontsize=11)
    return signed, absolute


def directional_figure(data, title):
    """Compare like-dimensional directional charges and their original total."""
    figure, axes = plt.subplots(3, 2, figsize=(7.2, 8.1), sharex=True,
                                constrained_layout=True)
    for col, (name, _, _, charge, symbol) in enumerate(FIELDS):
        axes[0, col].set_title(name, pad=8)
        absolute = []
        for i, component in enumerate("xyz"):
            q = data[f"{charge}_{component}"]
            dq = drift(q)
            label = rf"$Q_{{{symbol},{component}}}$"
            axes[0, col].plot(data["time"], q, color=COLORS[i], ls=STYLES[i], label=label)
            axes[1, col].plot(data["time"], dq, color=COLORS[i], ls=STYLES[i])
            absolute.append(np.abs(dq))
            axes[2, col].plot(data["time"], np.ma.masked_equal(np.abs(dq), 0),
                              color=COLORS[i], ls=STYLES[i])
        q = data[charge]
        dq = drift(q)
        axes[0, col].plot(data["time"], q, color="0.15", lw=1,
                          label=rf"$Q_{{{symbol},h}}$")
        axes[1, col].plot(data["time"], dq, color="0.15", lw=1)
        axes[2, col].plot(data["time"], np.ma.masked_equal(np.abs(dq), 0),
                          color="0.15", lw=1)
        axes[0, col].legend(loc="upper right", ncol=2, framealpha=0.95,
                            edgecolor="0.85", handlelength=2.1)
        axes[0, col].set_ylabel(rf"$Q_{{{symbol},i}},\ Q_{{{symbol},h}}$")
        axes[1, col].set_ylabel(rf"$\Delta Q_{{{symbol},i}},\ \Delta Q_{{{symbol},h}}$")
        axes[2, col].set_ylabel(rf"$|\Delta Q_{{{symbol},i}}|,\ |\Delta Q_{{{symbol},h}}|$")
        for ax in axes[:2, col]:
            linear_axis(ax)
            ax.axhline(0, color="0.5", lw=0.5, zorder=0)
        log_axis(axes[2, col], absolute + [np.abs(dq)])
    decorate(axes)
    if title:
        figure.suptitle(title, fontsize=11)
    return figure


def reconstruction_rows(data):
    """Report summation roundoff without replacing the original total charge."""
    sums = {field: sum(data[f"{field}_charge_{c}"] for c in "xyz")
            for field in ("electric", "magnetic")}
    rows = []
    for i, time in enumerate(data["time"]):
        row = dict(time=time)
        for field, total in sums.items():
            row[f"{field}_directional_sum"] = total[i]
            row[f"{field}_charge"] = data[f"{field}_charge"][i]
            row[f"{field}_sum_minus_charge"] = total[i] - data[f"{field}_charge"][i]
        rows.append(row)
    return rows


def discog_figures(data, title):
    """Legacy contractions are comparisons, not interchangeable physical charges."""
    total, axes = plt.subplots(3, 2, figsize=(7.2, 8.1), sharex=True,
                               constrained_layout=True)
    directions, da = plt.subplots(2, 2, figsize=(7.2, 6.2), sharex=True,
                                  constrained_layout=True)
    for col, field in enumerate(("electric", "magnetic")):
        axes[0, col].set_title(field.capitalize())
        arrays = []
        for i, (key, label) in enumerate((
            (f"{field}_charge", "Broken divergence"),
            (f"discog_{field}_charge", "DiscoG legacy"),
            (f"discog_{field}_charge_strong", "DiscoG strong"))):
            q, dq = data[key], drift(data[key])
            axes[0, col].plot(data["time"], q, color=COLORS[i], ls=STYLES[i], label=label)
            axes[1, col].plot(data["time"], dq, color=COLORS[i], ls=STYLES[i])
            axes[2, col].plot(data["time"], np.ma.masked_equal(abs(dq), 0),
                              color=COLORS[i], ls=STYLES[i])
            arrays.append(abs(dq))
        axes[0, col].legend(fontsize=7)
        for row, label in enumerate(("Diagnostic value", "Signed change", "Absolute change")):
            axes[row, col].set_ylabel(label)
        for ax in axes[:2, col]:
            linear_axis(ax)
        log_axis(axes[2, col], arrays)
        for row, variant in enumerate(("", "_strong")):
            ax = da[row, col]
            ax.set_title(field.capitalize() + (" legacy" if row == 0 else " strong"))
            for i, component in enumerate("xyz"):
                key = f"discog_{field}_charge{variant}_{component}"
                ax.plot(data["time"], drift(data[key]), color=COLORS[i],
                        ls=STYLES[i], label=component)
            ax.plot(data["time"], drift(data[f"discog_{field}_charge{variant}"]),
                    color="0.15", lw=1, label="total")
            ax.set_ylabel("Signed change")
            ax.legend(ncol=2, fontsize=7)
            linear_axis(ax)
    for fig, axs in ((total, axes), (directions, da)):
        decorate(axs)
        fig.suptitle(title or "DiscoG compatibility diagnostics", fontsize=11)
    return total, directions


def _integrated_fields_panel(data, title="", components=("sum", "x", "y", "z")):
    """Plot raw signed D/B integrals for the requested components."""
    style()
    figure, axes = plt.subplots(len(components), 2,
                                figsize=(7.2, 2.4 * len(components)),
                                sharex=True, constrained_layout=True,
                                squeeze=False)
    for col, (field, symbol, quantity) in enumerate((("d", "D", rf"$\left(\varepsilon \mathbf{{E}}_h,\nabla (x+y+z)\right)_{{\mathcal{{T}}_h}}$"), ("b", "B", rf"$\left(\mu \mathbf{{H}}_h,\nabla (x+y+z)\right)_{{\mathcal{{T}}_h}}$"))):
        axes[0, col].set_title(f"Integrated {symbol}", pad=8)
        for row, component in enumerate(components):
            values = (integrated_sum(data, field) if component == "sum"
                      else data[f"integrated_{field}_{component}"])
            ax = axes[row, col]
            ax.plot(data["time"], values,
                    color="0.15" if row == 0 else COLORS[row - 1],
                    ls="-" if row == 0 else STYLES[row - 1])
            ax.set_ylabel(f"{quantity}" if row == 0 else rf"$\Delta I_{{{symbol},{component}}}$$")
            ax.axhline(0, color="0.5", lw=0.5, zorder=0)
            linear_axis(ax)
    decorate(axes)
    if title:
        figure.suptitle(title, fontsize=11)
    return figure


def integrated_fields_panel(data, title=""):
    """Raw signed integrals: D/B columns and sum/x/y/z rows."""
    return _integrated_fields_panel(data, title)


def integrated_fields_sum_panel(data, title=""):
    """Raw signed field sums and integrated electric/magnetic charges."""
    style()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True,
                                constrained_layout=True)
    field_labels = (
        ("d", "D", r"$(\varepsilon\mathbf{E}_h,\nabla(x+y+z))_{\mathcal{T}_h}$", COLORS[0]),
        ("b", "B", r"$(\mu\mathbf{H}_h,\nabla(x+y+z))_{\mathcal{T}_h}$", COLORS[1]),
    )
    charge_labels = (
        ("electric_charge", "Electric charge",
         r"$(\nabla\!\cdot(\varepsilon\mathbf{E}_h),1)_{\mathcal{T}_h}$", COLORS[2]),
        ("magnetic_charge", "Magnetic charge",
         r"$(\nabla\!\cdot(\mu\mathbf{H}_h),1)_{\mathcal{T}_h}$", "#CC79A7"),
    )
    for col, (field, symbol, ylabel, color) in enumerate(field_labels):
        axes[0, col].set_title(f"Integrated {symbol}", pad=8)
        axes[0, col].plot(data["time"], integrated_sum(data, field),
                           color=color)
        axes[0, col].set_ylabel(ylabel)
        axes[0, col].axhline(0, color="0.5", lw=0.5, zorder=0)
        linear_axis(axes[0, col])
    for col, (key, title_text, ylabel, color) in enumerate(charge_labels):
        axes[1, col].set_title(title_text, pad=8)
        axes[1, col].plot(data["time"], data[key], color=color)
        axes[1, col].set_ylabel(ylabel)
        axes[1, col].axhline(0, color="0.5", lw=0.5, zorder=0)
        linear_axis(axes[1, col])
    decorate(axes)
    if title:
        figure.suptitle(title, fontsize=11)
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Run directory or quadrature CSV")
    parser.add_argument("--output-dir", type=Path, help="Default: run/plots/integrated_fields")
    parser.add_argument("--title", default="", help="Optional figure title")
    parser.add_argument("--dpi", type=int, default=400, help="PNG resolution (default: 400)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    source = input_path(args.input)
    data = read_data(source)
    run = source.parent.parent if source.parent.name == "diagnostics" else source.parent
    output = args.output_dir or run / "plots/integrated_fields"
    stems = ("integrated_fields_charge_signed", "integrated_fields_charge_absolute",
             "integrated_fields_sum_components_panel", "integrated_fields_sum_panel")
    has_directional = all(k in data for k in DIRECTIONAL)
    if has_directional:
        stems += ("directional_charge_comparison",)
    has_legacy = all(k in data for k in LEGACY)
    if has_legacy:
        stems += ("discog_charge_comparison", "discog_charge_directional_drift")
    targets = [output / f"{stem}.{ext}" for stem in stems for ext in ("pdf", "png")]
    targets += [output / "drift_summary.csv", output / "plot_metadata.json"]
    if has_directional:
        targets.append(output / "directional_charge_reconstruction.csv")
    collisions = [str(p) for p in targets if p.exists()]
    if collisions and not args.overwrite:
        raise FileExistsError("Use --overwrite to replace existing outputs: " + ", ".join(collisions))
    output.mkdir(parents=True, exist_ok=True)
    figures = make_figures(data, args.title) + (
        integrated_fields_panel(data, args.title),
        integrated_fields_sum_panel(data, args.title),
    )
    if has_directional:
        figures += (directional_figure(data, args.title),)
    else:
        print("No directional charge columns: retaining the original plots only. "
              "Rerun the simulation with the updated diagnostics to plot directional charges.")
    if has_legacy:
        figures += discog_figures(data, args.title)
    for fig, stem in zip(figures, stems):
        for ext in ("pdf", "png"):
            fig.savefig(output / f"{stem}.{ext}", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
    rows = summary_rows(data)
    with (output / "drift_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if has_directional:
        rows = reconstruction_rows(data)
        with (output / "directional_charge_reconstruction.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    metadata = dict(input=str(source.resolve()), input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    baseline_time=float(data["time"][0]), samples=len(data["time"]),
                    zero_log_policy="Exact zeros masked; no clipping, smoothing or detrending.",
                    comparison="Separate axes: integrated fields and charges have different dimensions. No ratios.",
                    units="Input simulation units, without volume normalization.",
                    numpy=np.__version__, matplotlib=matplotlib.__version__,
                    title=args.title, dpi=args.dpi, directional_charge_available=has_directional,
                    discog_charge_available=has_legacy,
                    discog_convention="Legacy contractions, not equivalent charge definitions; "
                    "electric legacy differentiates partition of unity; strong uses stiffness "
                    "without mass inverse; magnetic legacy uses transposed stiffness and ignores mu.")
    (output / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote PDF/PNG panels, drift_summary.csv and plot_metadata.json to {output}")
    print(f"Drift reference is the first recorded sample: t_ref={data['time'][0]:g}.")


if __name__ == "__main__":
    main()
