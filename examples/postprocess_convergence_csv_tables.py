#!/usr/bin/env python3
"""Create publication-ready LaTeX convergence tables from convergence.csv.

The output matches ``postprocess_convergence_slurm_tables.py``: one table for
each requested physical case and error mode, with centered and alternating
fluxes side by side. Rates are recomputed from the two-decimal scientific
values displayed in each table.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


CASES = ("plane_wave", "pec_cavity")
ERROR_MODES = ("initial", "final", "max-time")
FLUXES = ("centered", "alternating")


def is_mixed_csv(path):
    """Detect and validate mixed metadata before labeling published tables."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if "p_E" not in (reader.fieldnames or ()):
            return False
        for row in reader:
            p = int(row["p_E"])
            if int(row["order"]) != p or int(row["p_H"]) != p+1 or int(row["esprk_order"]) != p+2:
                raise ValueError("Invalid mixed polynomial/integrator orders")
    return True


ERROR_COLUMNS = {
    "initial": (
        "initial_l2_space_electric_error",
        "initial_l2_space_magnetic_error",
    ),
    "final": (
        "final_l2_space_electric_error",
        "final_l2_space_magnetic_error",
    ),
    "max-time": (
        "linf_time_l2_space_electric_error",
        "linf_time_l2_space_magnetic_error",
    ),
}


@dataclass(frozen=True)
class Row:
    order: int
    level: int
    nodes: str
    h_max: str
    dt: str
    electric_error: str | None
    magnetic_error: str | None


def case_name(record: dict[str, str]) -> str:
    campaign = record.get("campaign_mode", "").strip()
    if campaign:
        return campaign
    return {
        "periodic": "plane_wave",
        "cavity": "pec_cavity",
    }.get(record.get("case", "").strip(), record.get("case", "").strip())


def load_rows(path: Path, nodal_set: str, mesh_family: str, time_factor: int,
              h_metric: str, error_modes: list[str]) -> dict[tuple[str, str], dict[tuple[int, int], dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        required = {
            "flux", "nodal_set", "order", "mesh_level", h_metric, "dt",
            *(field for mode in error_modes for field in ERROR_COLUMNS[mode]),
        }
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        grouped: dict[tuple[str, str], dict[tuple[int, int], dict[str, str]]] = {}
        for line_number, record in enumerate(reader, start=2):
            if record["nodal_set"].strip().replace("_", "-") != nodal_set.replace("_", "-"):
                continue
            record_family = record.get("campaign_mesh_family", "").strip() or record.get("mesh_family", "").strip()
            if record_family != mesh_family:
                continue
            if int(record.get("time_factor", "1") or "1") != time_factor:
                continue
            case = case_name(record)
            flux = record["flux"].strip()
            if case not in CASES or flux not in FLUXES:
                continue
            try:
                key = (int(record["order"]), int(record["mesh_level"]))
                for field in (h_metric, "dt"):
                    value = float(record[field])
                    if not math.isfinite(value) or value <= 0.0:
                        raise ValueError(f"invalid {field}={record[field]!r}")
            except ValueError as error:
                raise ValueError(f"Invalid convergence row in {path}:{line_number}: {error}") from error
            destination = grouped.setdefault((case, flux), {})
            record = dict(record)
            record["selected_h"] = record[h_metric]
            if key in destination:
                raise ValueError(f"Duplicate {case}/{flux}/p={key[0]}/level={key[1]} in {path}")
            destination[key] = record
    return grouped


def paired_rows(
    grouped: dict[tuple[str, str], dict[tuple[int, int], dict[str, str]]],
    case: str,
    mode: str,
    strict: bool,
) -> tuple[list[Row | None], list[Row | None], list[str]]:
    centered = grouped.get((case, "centered"), {})
    alternating = grouped.get((case, "alternating"), {})
    keys = sorted(centered.keys() | alternating.keys())
    unpaired = sorted(centered.keys() ^ alternating.keys())
    if unpaired and strict:
        details = ", ".join(f"p={order}/level={level}" for order, level in unpaired)
        raise ValueError(f"Unpaired centered/alternating rows for {case}: {details}")
    electric_column, magnetic_column = ERROR_COLUMNS[mode]
    messages = []
    output: dict[str, list[Row | None]] = {flux: [] for flux in FLUXES}

    def available_row(source, flux: str, key: tuple[int, int]) -> Row | None:
        if source is None:
            return None
        try:
            errors = (float(source[electric_column]), float(source[magnetic_column]))
        except (KeyError, TypeError, ValueError):
            errors = ()
        if len(errors) != 2 or not all(math.isfinite(value) and value > 0.0 for value in errors):
            if strict:
                raise ValueError(
                    f"Missing {mode} error values for {case}/{flux}, "
                    f"p={key[0]}, level={key[1]}"
                )
            messages.append(
                f"{case}/{mode}/{flux}/p={key[0]}/level={key[1]}: "
                "error values are not available yet; rendered as --"
            )
            electric_error = magnetic_error = None
        else:
            electric_error = source[electric_column]
            magnetic_error = source[magnetic_column]
        return Row(key[0], key[1], source["nodal_set"], source["selected_h"],
                   source["dt"], electric_error, magnetic_error)

    for key in keys:
        centered_source = centered.get(key)
        alternating_source = alternating.get(key)
        if centered_source is not None and alternating_source is not None:
            for field in ("selected_h", "dt"):
                if not math.isclose(float(centered_source[field]), float(alternating_source[field]),
                                    rel_tol=1e-12, abs_tol=0.0):
                    raise ValueError(f"Centered/alternating {field} mismatch for {case}, "
                                     f"p={key[0]}, level={key[1]}")
        output["centered"].append(available_row(centered_source, "centered", key))
        output["alternating"].append(available_row(alternating_source, "alternating", key))

    if unpaired:
        details = ", ".join(f"p={order}/level={level}" for order, level in unpaired)
        messages.append(f"{case}/{mode}: missing flux results are rendered as --: {details}")
    if not keys:
        messages.append(f"{case}/{mode}: no completed results are available yet")
    return output["centered"], output["alternating"], messages


def case_title(case: str) -> str:
    return {"plane_wave": "periodic traveling plane wave", "pec_cavity": "PEC cavity"}[case]


def mode_title(mode: str) -> str:
    return {"initial": "initial interpolation", "final": "final-time",
            "max-time": "sampled maximum-in-time"}[mode]


def rounded_scientific(value: str) -> tuple[float, str]:
    rendered = f"{float(value):.2e}"
    mantissa, exponent = rendered.split("e")
    return float(rendered), rf"${mantissa}\times 10^{{{int(exponent)}}}$"


def recomputed_rate(current: Row | None, previous: Row | None, field: str) -> str:
    if (current is None or previous is None or previous.order != current.order or
            current.level != previous.level + 1):
        return "--"
    if getattr(current, field) is None or getattr(previous, field) is None:
        return "--"
    previous_h, _ = rounded_scientific(previous.h_max)
    current_h, _ = rounded_scientific(current.h_max)
    previous_error, _ = rounded_scientific(getattr(previous, field))
    current_error, _ = rounded_scientific(getattr(current, field))
    if previous_h == current_h or previous_error <= 0.0 or current_error <= 0.0:
        return "--"
    return f"{math.log(previous_error/current_error) / math.log(previous_h/current_h):.2f}"


def formatted_error(row: Row | None, field: str) -> str:
    value = None if row is None else getattr(row, field)
    return "--" if value is None else rounded_scientific(value)[1]


def render_table(case: str, mode: str, centered: list[Row | None], alternating: list[Row | None],
                 source: Path, mesh_family: str, h_metric: str,
                 time_factor: int, mixed: bool = False) -> str:
    nodes = sorted({row.nodes for row in centered + alternating if row is not None})
    node_text = ", ".join(nodes).replace("_", r"\_")
    h_symbol = r"$h_{\max}$" if h_metric == "h_max" else r"$h_{\mathrm{eff}}$"
    caption = (
        f"Mesh-refinement convergence for the {case_title(case)} using "
        f"{mode_title(mode)} spatial $L^2$ errors. Centered and alternating "
        f"fluxes are compared at identical {h_symbol} and time step. "
        f"The nodal set is {node_text}, the mesh family is {mesh_family}, and "
        f"the time-step factor is {time_factor}; rates use the measured "
        + (r"$h_{\max}$." if h_metric == "h_max" else r"$h_{\mathrm{eff}}=(|\Omega|/N_e)^{1/3}$.")
    )
    if mixed:
        caption += r" Mixed spaces use $p_E=p$, $p_H=p+1$ and ESPRK order $p+2$."
    lines = [
        "% Generated by examples/postprocess_convergence_csv_tables.py.",
        f"% Source: {source.as_posix()}",
        "% Requires: \\usepackage{amsmath,booktabs,graphicx}",
        r"\begin{table}[p]", r"  \centering", r"  \caption{" + caption + "}",
        r"  \label{tab:convergence-" + case.replace("_", "-") + "-" +
        mesh_family + "-" + h_metric.replace("_", "-") + "-" + mode + "}",
        r"  \setlength{\tabcolsep}{3.5pt}", r"  \renewcommand{\arraystretch}{1.08}",
        r"  \resizebox{\linewidth}{!}{%", r"  \begin{tabular}{@{}rccrrrrrrrr@{}}",
        r"    \toprule",
        r"    & & & \multicolumn{4}{c}{Centered flux} & \multicolumn{4}{c}{Alternating flux} \\",
        r"    \cmidrule(lr){4-7}\cmidrule(l){8-11}",
        r"    $p$ & " + (r"$h_{\max}$" if h_metric == "h_max" else r"$h_{\mathrm{eff}}$") + r" & $\Delta t$ & $\lVert e_E\rVert_2$ & $r_E$ & "
        r"$\lVert e_H\rVert_2$ & $r_H$ & $\lVert e_E\rVert_2$ & $r_E$ & "
        r"$\lVert e_H\rVert_2$ & $r_H$ \\", r"    \midrule",
    ]
    previous_order = None
    centered_previous = alternating_previous = None
    group_row = 0
    order_counts: dict[int, int] = {}
    available_pairs = [
        (c_row, a_row)
        for c_row, a_row in zip(centered, alternating)
        if c_row is not None or a_row is not None
    ]
    for c_row, a_row in available_pairs:
        reference = c_row if c_row is not None else a_row
        if reference is None:  # Defensive: guarded by available_pairs.
            continue
        order_counts[reference.order] = order_counts.get(reference.order, 0) + 1
    if not available_pairs:
        lines.append(r"    \multicolumn{11}{c}{No completed cases available} \\")
    for c_row, a_row in available_pairs:
        reference = c_row if c_row is not None else a_row
        if reference is None:  # Defensive: guarded by available_pairs.
            continue
        if previous_order is not None and reference.order != previous_order:
            lines.extend((r"    \midrule", r"    \addlinespace[2pt]"))
            centered_previous = alternating_previous = None
            group_row = 0
        values = (
            rounded_scientific(reference.h_max)[1],
            rounded_scientific(reference.dt)[1],
            formatted_error(c_row, "electric_error"),
            formatted_error(c_row, "magnetic_error"),
            formatted_error(a_row, "electric_error"),
            formatted_error(a_row, "magnetic_error"),
        )
        order_row = min(2, order_counts[reference.order] - 1)
        order_cell = str(reference.order) if group_row == order_row else ""
        lines.append(
            f"    {order_cell} & {values[0]} & {values[1]} & {values[2]} & "
            f"{recomputed_rate(c_row, centered_previous, 'electric_error')} & {values[3]} & "
            f"{recomputed_rate(c_row, centered_previous, 'magnetic_error')} & {values[4]} & "
            f"{recomputed_rate(a_row, alternating_previous, 'electric_error')} & {values[5]} & "
            f"{recomputed_rate(a_row, alternating_previous, 'magnetic_error')} \\\\"
        )
        previous_order = reference.order
        centered_previous, alternating_previous = c_row, a_row
        group_row += 1
    lines.extend((r"    \bottomrule", r"  \end{tabular}%", r"  }", r"\end{table}", ""))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("convergence.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/convergence/latex_tables_from_csv"))
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--error-modes", nargs="+", choices=ERROR_MODES,
                        default=["final", "max-time"])
    parser.add_argument("--nodal-set", default="equispaced")
    parser.add_argument("--mesh-family", choices=("unstructured", "structured"),
                        default="unstructured")
    parser.add_argument("--time-factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--h-metric", choices=("h_max", "h_eff"), default="h_max")
    parser.add_argument("--strict", action="store_true", help="Fail if either flux is missing a matching row.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {args.input}")
    mixed = is_mixed_csv(args.input)
    grouped = load_rows(args.input, args.nodal_set, args.mesh_family,
                        args.time_factor, args.h_metric, args.error_modes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for case in args.cases:
        for mode in args.error_modes:
            centered, alternating, messages = paired_rows(grouped, case, mode, args.strict)
            for message in messages:
                print(f"warning: {message}")
            output = args.output_dir / f"{case}_{mode}_convergence_table.tex"
            output.write_text(render_table(
                case, mode, centered, alternating, args.input,
                args.mesh_family, args.h_metric, args.time_factor, mixed), encoding="utf-8")
            outputs.append(output)
    print(f"Wrote {len(outputs)} LaTeX tables:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
