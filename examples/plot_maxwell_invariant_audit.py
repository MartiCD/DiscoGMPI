#!/usr/bin/env python3
"""Plot raw signed invariant drift and write slope/residual estimates.

No detrending or absolute-value substitution in signed panels. Quantities are
in the benchmark's nondimensional units. Fits are summaries, not causal tests.
"""
import argparse
import csv
from pathlib import Path
try:
    import tomllib
except ImportError:
    import tomli as tomllib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_history(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty history: {path}")
    return {key: np.array([float(r[key]) for r in rows])
            for key in rows[0] if key != "shortened"}


def fit(x, y):
    if len(x) < 2 or np.ptp(x) == 0:
        return np.nan, np.nan, np.nan
    xc = x - np.mean(x)
    slope = np.dot(xc, y - np.mean(y)) / np.dot(xc, xc)
    intercept = np.mean(y) - slope * np.mean(x)
    residual = y - (slope * x + intercept)
    return slope, intercept, np.sqrt(np.mean(residual ** 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    out = args.output_dir or args.run_dir / "plots"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite plots: {out}")
    paths = sorted(args.run_dir.glob("*/history.csv"))
    if not paths:
        raise FileNotFoundError(f"No case histories below {args.run_dir}")
    out.mkdir(parents=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "pdf.fonttype": 42})
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    local_fig, la = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True)
    moment_fig, ma = plt.subplots(3, 2, figsize=(10, 9), constrained_layout=True)
    records = []
    for path in paths:
        d = read_history(path)
        with (path.parent / "metadata.toml").open("rb") as stream:
            meta = tomllib.load(stream)
        label = path.parent.name + " (Float64)"
        for f, (field, updates) in enumerate((("e", "electric_updates"), ("h", "magnetic_updates"))):
            for col, xkey in enumerate(("time", updates)):
                ax[f, col].plot(d[xkey], d[f"dq{field}"], label=label, lw=1.2)
                slope, intercept, rms = fit(d[xkey], d[f"dq{field}"])
                records.append({"case": path.parent.name, "field": field, "abscissa": xkey,
                                "slope": slope, "intercept": intercept, "residual_rms": rms,
                                "dt_limit": meta["dt_limit"], "samples": len(d[xkey]),
                                "initial_sha256": meta["initial_sha256"]})
            la[0, f].plot(d["time"], d[f"div_{field}_l2"], label=label, lw=1.1)
            jump = np.hypot(d[f"jump_{field}_interior_l2"], d[f"jump_{field}_periodic_l2"])
            la[1, f].plot(d["time"], jump, lw=1.1)
            la[2, f].plot(d["time"], d[f"rhs_q{field}"], lw=1.1)
        for j, component in enumerate("xyz"):
            ma[j, 0].plot(d["time"], d[f"p{component}"], label=label, lw=1.1)
            ma[j, 1].plot(d["time"], d[f"rhs_p{component}"], lw=1.1)
    for f, field in enumerate(("E", "H")):
        for c in range(2):
            ax[f, c].set(xlabel="Time t" if c == 0 else f"{field} updates",
                         ylabel=rf"$Q_{{{field}}}(t)-Q_{{{field}}}(0)$")
            ax[f, c].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        la[0, f].set_ylabel(rf"$\|\nabla_h\cdot U_{field}\|_2$")
        la[1, f].set_ylabel(rf"$\|[U_{field}\cdot n]\|_{{2,\mathcal{{F}}}}$")
        la[2, f].set_ylabel(rf"$\dot Q_{field}$ (actual RHS)")
        for r in range(3):
            la[r, f].set_xlabel("Time t")
            if r < 2:
                la[r, f].set_yscale("log")
    for j, component in enumerate("xyz"):
        ma[j, 0].set(xlabel="Time t", ylabel=rf"$P_{component}$")
        ma[j, 1].set(xlabel="Time t", ylabel=rf"$\dot P_{component}$ (actual RHS)")
    for axes in (ax, la, ma):
        for a in axes.flat:
            a.grid(alpha=0.2)
    for figure, axes, name in ((fig, ax, "signed_charge_drift"),
                                (local_fig, la, "local_constraints"),
                                (moment_fig, ma, "momentum_components")):
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=6, ncol=2)
        figure.suptitle("Periodic Maxwell invariant audit (nondimensional)", fontsize=12, y=1.14)
        for ext in ("pdf", "png"):
            figure.savefig(out / f"{name}.{ext}", bbox_inches="tight")
        plt.close(figure)
    with (out / "drift_fits.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Raw signed plots and least-squares slope/residual table: {out}")


if __name__ == "__main__":
    main()
