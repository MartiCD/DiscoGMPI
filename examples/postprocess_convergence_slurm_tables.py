#!/usr/bin/env python3
"""Convert MN5 convergence stdout logs into publication-ready LaTeX tables.

The default input is ``output/convergence/mn5/slurm``.  Four tables are
written by default: plane-wave and PEC-cavity convergence, each for final-time
and sampled max-in-time errors.  Every table places centered and alternating
flux data side by side.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CASES = ("plane_wave", "pec_cavity")
ERROR_MODES = ("final", "max-time")
FLUXES = ("centered", "alternating")

SETTING_RE = re.compile(
    r"^(CONVERGENCE_CASE|FLUX|ERROR_MODE)\s*=\s*(\S+)\s*$"
)
ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\S+)\s+"
    r"([+\-0-9.eE]+)\s+([+\-0-9.eE]+)\s+"
    r"([+\-0-9.eE]+)\s+([+\-0-9.eE]+)\s+"
    r"(-|[+\-0-9.eE]+)\s+(-|[+\-0-9.eE]+)\s+"
    r"(NO_PAIR|PASS|FAIL)\s*$"
)
JOB_ID_RE = re.compile(r"_(\d+)\.out$")


@dataclass(frozen=True)
class Row:
    order: int
    level: int
    nodes: str
    h_max: str
    dt: str
    electric_error: str
    magnetic_error: str
    electric_rate: str
    magnetic_rate: str
    verdict: str


@dataclass(frozen=True)
class Run:
    path: Path
    convergence_case: str
    flux: str
    error_mode: str
    error_description: str
    rows: tuple[Row, ...]
    overall_verdict: str
    job_id: int


def parse_log(path: Path) -> Run | None:
    settings: dict[str, str] = {}
    rows: list[Row] = []
    error_description = ""
    overall_verdict = ""

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        setting = SETTING_RE.match(line)
        if setting:
            settings[setting.group(1)] = setting.group(2)
            continue
        if line.startswith("Errors:"):
            error_description = line.removeprefix("Errors:").strip()
            continue
        match = ROW_RE.match(line)
        if match:
            values = match.groups()
            rows.append(
                Row(
                    order=int(values[0]),
                    level=int(values[1]),
                    nodes=values[2],
                    h_max=values[3],
                    dt=values[4],
                    electric_error=values[5],
                    magnetic_error=values[6],
                    electric_rate=values[7],
                    magnetic_rate=values[8],
                    verdict=values[9],
                )
            )
            continue
        if line.startswith("Overall expected-rate convergence:"):
            remainder = line.split(":", 1)[1].strip()
            overall_verdict = remainder.split()[0]

    required = {"CONVERGENCE_CASE", "FLUX", "ERROR_MODE"}
    if not required.issubset(settings):
        return None
    if not rows or not overall_verdict:
        return None

    job_match = JOB_ID_RE.search(path.name)
    job_id = int(job_match.group(1)) if job_match else -1
    return Run(
        path=path,
        convergence_case=settings["CONVERGENCE_CASE"],
        flux=settings["FLUX"],
        error_mode=settings["ERROR_MODE"],
        error_description=error_description,
        rows=tuple(rows),
        overall_verdict=overall_verdict,
        job_id=job_id,
    )


def newest_runs(paths: Iterable[Path]) -> tuple[dict[tuple[str, str, str], Run], list[str]]:
    selected: dict[tuple[str, str, str], Run] = {}
    messages: list[str] = []
    for path in sorted(paths):
        run = parse_log(path)
        if run is None:
            continue
        key = (run.convergence_case, run.error_mode, run.flux)
        previous = selected.get(key)
        if previous is None or (run.job_id, path.stat().st_mtime) > (
            previous.job_id,
            previous.path.stat().st_mtime,
        ):
            if previous is not None:
                messages.append(
                    f"Replacing {previous.path.name} with newer {path.name} for {key}."
                )
            selected[key] = run
        else:
            messages.append(f"Ignoring older duplicate {path.name} for {key}.")
    return selected, messages


def case_title(case: str) -> str:
    return {
        "plane_wave": "periodic traveling plane wave",
        "pec_cavity": "PEC cavity",
    }.get(case, case.replace("_", " "))


def mode_title(mode: str) -> str:
    return {
        "final": "final-time",
        "max-time": "sampled maximum-in-time",
    }.get(mode, mode.replace("_", " "))


def rounded_scientific(value: str) -> tuple[float, str]:
    """Return the displayed two-decimal value and its LaTeX representation."""
    rendered = f"{float(value):.2e}"
    mantissa, exponent = rendered.split("e")
    rounded_value = float(rendered)
    return rounded_value, rf"${mantissa}\times 10^{{{int(exponent)}}}$"


def recomputed_rate(current: Row, previous: Row | None, field: str) -> str:
    """Compute a rate from the same rounded values shown in the table."""
    if previous is None or previous.order != current.order:
        return "--"
    previous_h, _ = rounded_scientific(previous.h_max)
    current_h, _ = rounded_scientific(current.h_max)
    previous_error, _ = rounded_scientific(getattr(previous, field))
    current_error, _ = rounded_scientific(getattr(current, field))
    rate = math.log(previous_error / current_error) / math.log(
        previous_h / current_h
    )
    return f"{rate:.2f}"


def validate_pair(centered: Run, alternating: Run) -> None:
    if len(centered.rows) != len(alternating.rows):
        raise ValueError(
            f"Row-count mismatch between {centered.path} and {alternating.path}."
        )
    for centered_row, alternating_row in zip(centered.rows, alternating.rows):
        common_centered = (
            centered_row.order,
            centered_row.level,
            centered_row.nodes,
            centered_row.h_max,
            centered_row.dt,
        )
        common_alternating = (
            alternating_row.order,
            alternating_row.level,
            alternating_row.nodes,
            alternating_row.h_max,
            alternating_row.dt,
        )
        if common_centered != common_alternating:
            raise ValueError(
                "Centered/alternating common-column mismatch: "
                f"{common_centered} != {common_alternating}."
            )


def render_table(case: str, mode: str, centered: Run, alternating: Run) -> str:
    validate_pair(centered, alternating)
    nodes = sorted({row.nodes for row in centered.rows + alternating.rows})
    node_text = ", ".join(nodes).replace("_", r"\_")
    caption = (
        f"Mesh-refinement convergence for the {case_title(case)} using "
        f"{mode_title(mode)} spatial $L^2$ errors. Centered and alternating "
        f"fluxes are compared at identical $h_{{\max}}$ and time step. "
        f"The nodal set is {node_text}; rates use the measured $h_{{\max}}$."
    )
    label = f"tab:convergence-{case.replace('_', '-')}-{mode}"

    lines = [
        "% Generated by examples/postprocess_convergence_slurm_tables.py.",
        f"% Centered source: {centered.path.as_posix()}",
        f"% Alternating source: {alternating.path.as_posix()}",
        "% Requires: \\usepackage{amsmath,booktabs,graphicx}",
        r"\begin{table}[p]",
        r"  \centering",
        r"  \caption{" + caption + "}",
        r"  \label{" + label + "}",
        r"  \setlength{\tabcolsep}{3.5pt}",
        r"  \renewcommand{\arraystretch}{1.08}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{@{}rccrrrrrrrr@{}}",
        r"    \toprule",
        r"    & & & \multicolumn{4}{c}{Centered flux} & "
        r"\multicolumn{4}{c}{Alternating flux} \\",
        r"    \cmidrule(lr){4-7}\cmidrule(l){8-11}",
        r"    $p$ & $h_{\max}$ & $\Delta t$ & "
        r"$\lVert e_E\rVert_2$ & $r_E$ & $\lVert e_H\rVert_2$ & $r_H$ & "
        r"$\lVert e_E\rVert_2$ & $r_E$ & $\lVert e_H\rVert_2$ & $r_H$ \\",
        r"    \midrule",
    ]

    previous_order: int | None = None
    centered_previous: Row | None = None
    alternating_previous: Row | None = None
    group_row = 0
    for centered_row, alternating_row in zip(centered.rows, alternating.rows):
        if previous_order is not None and centered_row.order != previous_order:
            lines.append(r"    \midrule")
            lines.append(r"    \addlinespace[2pt]")
            group_row = 0
            centered_previous = None
            alternating_previous = None
        _, h_max_latex = rounded_scientific(centered_row.h_max)
        _, dt_latex = rounded_scientific(centered_row.dt)
        _, centered_e_latex = rounded_scientific(centered_row.electric_error)
        _, centered_h_latex = rounded_scientific(centered_row.magnetic_error)
        _, alternating_e_latex = rounded_scientific(alternating_row.electric_error)
        _, alternating_h_latex = rounded_scientific(alternating_row.magnetic_error)
        order_cell = str(centered_row.order) if group_row == 2 else ""
        lines.append(
            "    "
            f"{order_cell} & {h_max_latex} & {dt_latex} & "
            f"{centered_e_latex} & {recomputed_rate(centered_row, centered_previous, 'electric_error')} & "
            f"{centered_h_latex} & {recomputed_rate(centered_row, centered_previous, 'magnetic_error')} & "
            f"{alternating_e_latex} & {recomputed_rate(alternating_row, alternating_previous, 'electric_error')} & "
            f"{alternating_h_latex} & {recomputed_rate(alternating_row, alternating_previous, 'magnetic_error')} \\\\"
        )
        previous_order = centered_row.order
        centered_previous = centered_row
        alternating_previous = alternating_row
        group_row += 1

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}%",
            r"  }",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse convergence Slurm .out files and create one LaTeX table "
            "per convergence case and error mode."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/convergence/mn5/slurm"),
        help="Directory containing convergence .out files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/convergence/mn5/latex_tables"),
        help="Directory for the four generated .tex tables.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(CASES),
        help="Convergence cases to include.",
    )
    parser.add_argument(
        "--error-modes",
        nargs="+",
        default=list(ERROR_MODES),
        help="Error modes to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    selected, messages = newest_runs(args.input_dir.glob("conv_*.out"))
    for message in messages:
        print(f"warning: {message}")

    missing: list[tuple[str, str, str]] = []
    for case in args.cases:
        for mode in args.error_modes:
            for flux in FLUXES:
                key = (case, mode, flux)
                if key not in selected:
                    missing.append(key)
    if missing:
        formatted = "\n".join(f"  {case} / {mode} / {flux}" for case, mode, flux in missing)
        raise RuntimeError(f"Missing complete convergence logs:\n{formatted}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for case in args.cases:
        for mode in args.error_modes:
            centered = selected[(case, mode, "centered")]
            alternating = selected[(case, mode, "alternating")]
            output = args.output_dir / f"{case}_{mode}_convergence_table.tex"
            output.write_text(
                render_table(case, mode, centered, alternating), encoding="utf-8"
            )
            written.append(output)

    print(f"Wrote {len(written)} LaTeX tables:")
    for output in written:
        print(f"  {output}")


if __name__ == "__main__":
    main()
