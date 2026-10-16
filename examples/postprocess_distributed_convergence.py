#!/usr/bin/env python3
"""Compare Maxwell convergence using sampled Linf(time) L2(space) errors.

Reads the common schema v2 and legacy CSVs containing time maxima and h_max.
Final-time errors are never substituted for maximum-in-time errors.
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext

QUANTITIES = ("electric", "magnetic", "total", "ex", "ey", "ez", "hx", "hy", "hz")
LABELS = (r"\mathbf{E}", r"\mathbf{H}", r"\mathbf{U}", "E_x", "E_y", "E_z", "H_x", "H_y", "H_z")
ERROR = {name: f"linf_time_l2_space_{name}_error" for name in QUANTITIES}
SERIES_FIELDS = ("case", "boundary_condition", "flux", "mesh_family", "nodal_set", "magnetic_initialization", "electric_initialization", "error_mode", "error_every",
                 "final_time", "effective_cfl", "epsilon", "mu", "wave_number", "jitter", "seed")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6A51A3")
STYLES = ("-", "--", (0, (6, 2)), (0, (3, 2)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", nargs="*", type=Path,
                        default=[Path("output/convergence_distributed_poisson_bracket.csv")])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prefix", default="distributed_poisson_bracket")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default="Distributed Maxwell convergence")
    parser.add_argument("--usetex", action="store_true", help="Use an installed LaTeX renderer.")
    return parser.parse_args(argv)


def number(row, key, required=False):
    value = row.get(key, "")
    if value in ("", None):
        if required:
            raise ValueError(f"Missing required numeric column: {key}")
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value in {key}: {value}")
    return result


def series_key(row):
    return tuple(row.get(name, "") for name in SERIES_FIELDS) + (int(row["order"]),)


def normalize_row(raw, path):
    row = dict(raw)
    current = row.get("schema_version") == "2"
    if current and (row.get("error_norm") != "linf_time_l2_space" or row.get("rate_h") != "h_max"):
        raise ValueError(f"{path}: incompatible error norm or mesh-size definition.")
    defaults = {
        "mesh_level": row.get("level", ""), "nodal_set": "equispaced",
        "magnetic_initialization": "interpolate",
        "electric_initialization": "interpolate",
        "boundary_condition": "periodic" if "nx_target" in row else "pec",
        "flux": "centered", "mesh_family": "legacy", "case": "legacy",
        "rate_tolerance": "0.5", "rate_error_floor": "1e-12",
        "mpi_ranks": row.get("active_ranks", ""),
        "relative_energy_drift": row.get("relative_energy_error", ""),
    }
    for key, value in defaults.items():
        row.setdefault(key, value)
    row["nodal_set"] = row["nodal_set"].replace("-", "_")
    for name in QUANTITIES:
        key = ERROR[name]
        if key not in row:
            legacy_name = "u" if name == "total" else name
            aliases = (f"linf_l2_{legacy_name}_error", f"max_l2_{name}_error")
            row[key] = next((row[a] for a in aliases if a in row), "")
        value = number(row, key, required=current or name in QUANTITIES[:3])
        if value is not None and value < 0:
            raise ValueError(f"{path}: negative error in {key}.")
    if number(row, "h_max", required=True) <= 0:
        raise ValueError(f"{path}: h_max must be positive.")
    for name in ("order", "mesh_level"):
        row[name] = str(int(row[name]))
    row["error_norm"], row["rate_h"] = "linf_time_l2_space", "h_max"
    row["source_csv"] = str(path)
    return row


def expected_rate(row, name):
    if row.get("schema_version") == "2":
        return number(row, f"expected_rate_{name}")
    order = int(row["order"])
    if name == "electric":
        return order + 1
    if name in ("magnetic", "total"):
        return order
    active = {"periodic": ("ez", "hy"), "pec": ("ex", "ez", "hx", "hy", "hz"),
              "pmc": ("ex", "ey", "ez", "hx", "hz")}[row["boundary_condition"]]
    return (order + 1 if name.startswith("e") else order) if name in active else None


def rate_check(coarse, fine, coarse_h, fine_h, expected, tolerance, floor):
    if not (coarse_h > fine_h > 0):
        raise ValueError("Mesh levels must strictly decrease measured h_max.")
    rate = math.log(coarse / fine) / math.log(coarse_h / fine_h) if coarse > floor and fine > 0 else None
    if coarse <= floor and fine <= floor:
        status = "SKIP_ROUNDOFF"
    elif expected is None:
        status = "INFO"
    elif coarse <= floor:
        status = "FAIL"
    elif fine <= floor:
        status = "PASS_FLOOR"
    else:
        status = "PASS" if rate is not None and rate >= expected - tolerance else "FAIL"
    return rate, status


def group_results(rows):
    groups = {}
    for row in rows:
        groups.setdefault(series_key(row), []).append(row)
    for group in groups.values():
        group.sort(key=lambda r: -number(r, "h_max", required=True))
        for level, row in enumerate(group):
            row["mesh_level"] = str(level)
    return dict(sorted(groups.items(), key=lambda item: repr(item[0])))


def read_results(paths):
    merged = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            records = list(csv.DictReader(stream))
        if not records:
            raise ValueError(f"No convergence rows in {path}.")
        for raw in records:
            row = normalize_row(raw, path)
            mesh_identity = row.get("mesh_parameter") or row.get("nx_target") or row.get("cells_per_axis") or row["h_max"]
            key = series_key(row) + (mesh_identity,)
            if key in merged:
                previous = {k: v for k, v in merged[key].items() if k != "source_csv"}
                current = {k: v for k, v in row.items() if k != "source_csv"}
                if previous != current:
                    raise ValueError(f"Conflicting duplicate case in {path}: {key}. Keep separate studies in separate invocations.")
            merged[key] = row
        if row.get("schema_version") != "2":
            warnings.warn(f"{path}: legacy CSV; unavailable maximum-in-time component errors remain blank.")
    for group in group_results(merged.values()).values():
        previous = None
        for row in group:
            statuses = []
            for name in QUANTITIES:
                value, expected = number(row, ERROR[name]), expected_rate(row, name)
                rate, status = None, "NO_PAIR"
                if previous is not None and value is not None and number(previous, ERROR[name]) is not None:
                    rate, status = rate_check(
                        number(previous, ERROR[name]), value,
                        number(previous, "h_max"), number(row, "h_max"), expected,
                        number(row, "rate_tolerance"), number(row, "rate_error_floor"))
                elif value is None:
                    status = "UNAVAILABLE"
                stored = number(row, f"rate_{name}")
                if row.get("schema_version") == "2" and (stored is not None or rate is not None):
                    if stored is None or rate is None or not math.isclose(stored, rate, rel_tol=1e-9, abs_tol=1e-9):
                        raise ValueError(f"Stored rate_{name} disagrees with time maxima and h_max in {row['source_csv']}.")
                row[f"rate_{name}"] = "" if rate is None else f"{rate:.12g}"
                row[f"expected_rate_{name}"] = "" if expected is None else str(expected)
                row[f"status_{name}"] = status
                statuses.append(status)
            row["rate_verdict"] = "NO_PAIR" if previous is None else "FAIL" if "FAIL" in statuses else "PASS"
            previous = row
    return [row for group in group_results(merged.values()).values() for row in group]


def write_table_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(text):
    replacements = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "{": r"\{",
                    "}": r"\}", "$": r"\$", "\\": r"\textbackslash{}"}
    return "".join(replacements.get(c, c) for c in str(text))


def latex_number(value):
    if value is None:
        return "--"
    if value == 0:
        return "$0$"
    exponent = math.floor(math.log10(abs(value)))
    return "$" + rf"{value / 10**exponent:.3f}\times 10^{{{exponent}}}$"


def series_description(row):
    return (f"p={row['order']}, {row['nodal_set'].replace('_', '-')}, "
            f"{row['boundary_condition']}, {row['flux']}, {row['mesh_family']}, "
            f"error={row.get('error_mode', 'max-time')}, "
            f"H-init={row.get('magnetic_initialization', 'interpolate').replace('_', '-')}, "
            f"E-init={row.get('electric_initialization', 'interpolate').replace('_', '-')}")


def write_latex_table(path, rows, caption):
    with path.open("w", encoding="utf-8") as stream:
        stream.write("% Requires amsmath and booktabs. Rates use measured h_max.\n")
        for group in group_results(rows).values():
            for names in (QUANTITIES[:3], QUANTITIES[3:6], QUANTITIES[6:]):
                stream.write("\\begin{table}[htbp]\n\\centering\\small\n")
                stream.write(r"\setlength{\tabcolsep}{3pt}" + "\n")
                norm_text = (r"Errors are $\|u_h(T)-u(T)\|_{L^2(\Omega)}$"
                             if group[0].get("error_mode") == "final" else
                             r"Errors are sampled $\max_n\|u_h(t_n)-u(t_n)\|_{L^2(\Omega)}$")
                stream.write(r"\caption{" + latex_escape(caption + ": " + series_description(group[0])) +
                             ". " + norm_text + ".}\n")
                stream.write(r"\begin{tabular}{rr" + "rr" * len(names) + "}\n\\toprule\n")
                header = ["Level", r"$h_{\max}$"]
                for name in names:
                    header.extend(["$" + LABELS[QUANTITIES.index(name)] + "$ error", "Rate"])
                stream.write(" & ".join(header) + r" \\" + "\n\\midrule\n")
                for row in group:
                    entries = [row["mesh_level"], latex_number(number(row, "h_max"))]
                    for name in names:
                        rate = number(row, f"rate_{name}")
                        entries.extend([latex_number(number(row, ERROR[name])), "--" if rate is None else f"{rate:.3f}"])
                    stream.write(" & ".join(entries) + r" \\" + "\n")
                stream.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")


def plot_errors(rows, names, output, args, rates=False):
    ncols = 3 if len(names) == 3 else 2
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.8 * nrows), squeeze=False)
    groups = list(group_results(rows).values())
    for axis, name in zip(axes.flat, names):
        plotted = False
        for index, group in enumerate(groups):
            points = [(number(r, "h_max"), number(r, f"rate_{name}" if rates else ERROR[name])) for r in group]
            points = [(x, y) for x, y in points if y is not None and (rates or y > 0)]
            x, y = zip(*points) if points else ([], [])
            plotted = plotted or bool(points)
            axis.plot(x, y, color=COLORS[index % len(COLORS)],
                      linestyle=STYLES[index % len(STYLES)], marker="o", markersize=4,
                      markerfacecolor="white", label=series_description(group[0]))
        axis.set_xscale("log")
        axis.set_xlabel(r"$h_{\max}$")
        if not rates and plotted:
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(LogLocator(base=10))
            axis.yaxis.set_major_formatter(LogFormatterMathtext())
        elif not plotted:
            axis.text(0.5, 0.5, "No positive data", ha="center", va="center",
                      transform=axis.transAxes, color="0.45", fontsize=9)
        label = LABELS[QUANTITIES.index(name)]
        final_only = all(row.get("error_mode") == "final" for row in rows)
        error_label = (rf"$\|{{{label}}}_h(T)-{label}(T)\|_{{L^2}}$" if final_only else
                       rf"$\max_n\|{{{label}}}_h-{label}\|_{{L^2}}$")
        axis.set_ylabel("Observed rate" if rates else error_label)
        axis.set_title("$" + label + "$")
        axis.grid(which="major", color="0.86", linewidth=0.5)
        axis.tick_params(which="both", direction="in", top=True, right=True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=min(2, len(groups)), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, max(0.60, 0.93 - 0.035 * math.ceil(len(groups) / 2))))
    for extension in args.formats.split(","):
        fig.savefig(output.with_suffix("." + extension.strip()), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    rows = read_results(args.csv_paths)
    if not rows:
        raise ValueError("No convergence data.")
    output = args.output_dir or args.csv_paths[0].parent / "convergence_postprocessing"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "text.usetex": args.usetex,
                         "axes.linewidth": 0.8, "lines.linewidth": 1.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    write_table_csv(output / f"{args.prefix}_convergence_table.csv", rows)
    write_latex_table(output / f"{args.prefix}_convergence_table.tex", rows, args.title)
    plot_errors(rows, QUANTITIES[:3], output / f"{args.prefix}_l2_convergence", args)
    plot_errors(rows, QUANTITIES[3:], output / f"{args.prefix}_component_convergence", args)
    plot_errors(rows, QUANTITIES[:3], output / f"{args.prefix}_observed_rates", args, rates=True)
    for group in group_results(rows).values():
        print(series_description(group[0]), "finest pair:", group[-1]["rate_verdict"])
    print(f"Wrote CSV, LaTeX tables, and plots to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
