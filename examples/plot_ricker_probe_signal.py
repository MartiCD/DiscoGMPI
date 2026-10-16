#!/usr/bin/env python3

"""Plot 3D Ricker-pulse probe electromagnetic-field components.

The distributed Ricker driver writes the point-probe history to

    <run-dir>/diagnostics/probe_signal.h5

when ``--probe-format=hdf5`` is used. This script reads that file and creates a
publication-style 3 x 2 panel: electric components in the left column and
magnetic components in the right column.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/discogmpi-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter

try:
    import h5py
except ModuleNotFoundError as exc:  # pragma: no cover - dependency error path.
    h5py = None
    H5PY_IMPORT_ERROR = exc
else:
    H5PY_IMPORT_ERROR = None


FIELD_COLUMNS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
EXACT_FIELD_COLUMNS = (
    "exact_Ex",
    "exact_Ey",
    "exact_Ez",
    "exact_Hx",
    "exact_Hy",
    "exact_Hz",
)
COMPONENT_ROWS = (
    (r"$E_x$", "Ex", "exact_Ex", r"$H_x$", "Hx", "exact_Hx"),
    (r"$E_y$", "Ey", "exact_Ey", r"$H_y$", "Hy", "exact_Hy"),
    (r"$E_z$", "Ez", "exact_Ez", r"$H_z$", "Hz", "exact_Hz"),
)
DEFAULT_RUN_DIR = Path("output/ricker3d_p4")


@dataclass(frozen=True)
class ProbeMetadata:
    point: tuple[float, float, float] | None
    frequency: float | None
    dt: float | None
    nsteps: int | None
    probe_every: int | None
    layout: str | None
    n_samples: int
    stride: int


@dataclass(frozen=True)
class ProbeData:
    time: np.ndarray
    values: dict[str, np.ndarray]
    metadata: ProbeMetadata


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a journal-style 3 x 2 panel of Ex, Ey, Ez, Hx, Hy, Hz "
            "from a DiscoGMPI Ricker probe_signal.h5 file."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Run directory. Default: {DEFAULT_RUN_DIR}",
    )
    parser.add_argument(
        "--probe-h5",
        type=Path,
        default=None,
        help="Probe HDF5 path. Default: <run-dir>/diagnostics/probe_signal.h5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "PDF output path. Default: "
            "<run-dir>/plots/ricker_probe_em_fields_panel.pdf"
        ),
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        default=None,
        help=(
            "PNG output path. Default: PDF path with .png suffix. "
            "Use --no-png to disable."
        ),
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Do not write a PNG companion file.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--no-exact",
        action="store_true",
        help="Do not overlay exact probe components even if present in the HDF5 file.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help="Read every STRIDE-th HDF5 sample. Default: automatic from --max-points.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200_000,
        help="Maximum number of plotted samples when --stride=0. Default: 200000.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=7.2,
        help="Figure width in inches. Default: 7.2.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=6.7,
        help="Figure height in inches. Default: 6.7.",
    )
    parser.add_argument(
        "--usetex",
        action="store_true",
        help="Use LaTeX text rendering if the local system supports it.",
    )
    return parser.parse_args(argv)


def configure_matplotlib(usetex: bool) -> None:
    plt.rcParams.update(
        {
            "text.usetex": usetex,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 9.2,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.3,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "axes.linewidth": 0.95,
            "axes.axisbelow": True,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "figure.dpi": 160,
            "lines.linewidth": 1.55,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "butt",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.8,
            "ytick.major.size": 4.8,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.75,
            "ytick.minor.width": 0.75,
            "legend.frameon": True,
            "legend.framealpha": 0.96,
            "legend.edgecolor": "0.72",
            "legend.handlelength": 2.4,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def default_probe_h5(run_dir: Path) -> Path:
    return run_dir / "diagnostics" / "probe_signal.h5"


def default_pdf_output(run_dir: Path) -> Path:
    return run_dir / "plots" / "ricker_probe_em_fields_panel.pdf"


def decode_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def read_scalar(group, name: str) -> float | None:
    if name not in group:
        return None
    value = np.asarray(group[name])
    if value.size == 0:
        return None
    return float(value.reshape(-1)[0])


def read_int_scalar(group, name: str) -> int | None:
    value = read_scalar(group, name)
    return None if value is None else int(round(value))


def read_tuple3(group, name: str) -> tuple[float, float, float] | None:
    if name not in group:
        return None
    value = np.asarray(group[name], dtype=float).reshape(-1)
    if value.size != 3:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def read_layout(group) -> str | None:
    if "layout" not in group:
        return None
    return decode_string(group["layout"][()])


def read_columns(group) -> list[str]:
    if "columns" not in group:
        raise ValueError("HDF5 probe group is missing dataset 'columns'.")
    return [decode_string(value) for value in group["columns"][()]]


def selected_columns(columns: list[str], include_exact: bool) -> list[str]:
    required = ["time", *FIELD_COLUMNS]
    if include_exact:
        required.extend(name for name in EXACT_FIELD_COLUMNS if name in columns)

    missing = [name for name in ["time", *FIELD_COLUMNS] if name not in columns]
    if missing:
        raise ValueError(
            "Probe HDF5 file is missing required columns: " + ", ".join(missing)
        )
    return required


def sample_axis_and_count(shape: tuple[int, int], ncolumns: int) -> tuple[int, int]:
    if shape[1] == ncolumns:
        return 0, shape[0]
    if shape[0] == ncolumns:
        return 1, shape[1]
    raise ValueError(
        "Cannot infer probe data orientation: "
        f"data shape is {shape}, but columns has length {ncolumns}."
    )


def compute_stride(n_samples: int, requested_stride: int, max_points: int) -> int:
    if requested_stride > 0:
        return requested_stride
    if max_points <= 0:
        raise ValueError("--max-points must be positive when --stride=0.")
    return max(1, int(math.ceil(n_samples / max_points)))


def read_selected_table(dataset, sample_axis: int, indices: list[int], stride: int):
    if sample_axis == 0:
        return np.asarray(dataset[::stride, indices], dtype=float)
    return np.asarray(dataset[indices, ::stride], dtype=float).T


def read_probe_hdf5(path: Path, include_exact: bool, stride: int, max_points: int):
    if h5py is None:
        raise RuntimeError(
            "This script requires h5py to read probe_signal.h5. "
            f"Original import error: {H5PY_IMPORT_ERROR}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Probe HDF5 file not found: {path}")

    with h5py.File(path, "r") as h5:
        if "probe" not in h5:
            raise ValueError(f"{path} does not contain a 'probe' group.")
        group = h5["probe"]
        if "data" not in group:
            raise ValueError(f"{path} does not contain 'probe/data'.")

        columns = read_columns(group)
        names = selected_columns(columns, include_exact)
        indices = [columns.index(name) for name in names]
        dataset = group["data"]
        if len(dataset.shape) != 2:
            raise ValueError(f"Expected a rank-2 probe/data dataset, got {dataset.shape}.")

        sample_axis, n_samples = sample_axis_and_count(dataset.shape, len(columns))
        if n_samples == 0:
            raise ValueError(f"{path} contains no probe samples.")
        actual_stride = compute_stride(n_samples, stride, max_points)
        table = read_selected_table(dataset, sample_axis, indices, actual_stride)

        values = {name: table[:, j] for j, name in enumerate(names)}
        metadata = ProbeMetadata(
            point=read_tuple3(group, "probe_point"),
            frequency=read_scalar(group, "frequency"),
            dt=read_scalar(group, "dt"),
            nsteps=read_int_scalar(group, "nsteps"),
            probe_every=read_int_scalar(group, "probe_every"),
            layout=read_layout(group),
            n_samples=n_samples,
            stride=actual_stride,
        )

    return ProbeData(time=values.pop("time"), values=values, metadata=metadata)


def finite_limits(*arrays: Iterable[float]) -> tuple[float, float]:
    finite = []
    for array in arrays:
        data = np.asarray(array, dtype=float)
        finite.append(data[np.isfinite(data)])
    if not finite:
        return (-1.0, 1.0)
    values = np.concatenate(finite)
    if values.size == 0:
        return (-1.0, 1.0)
    lo = float(values.min())
    hi = float(values.max())
    if lo == hi:
        pad = 1.0 if lo == 0.0 else 0.08 * abs(lo)
        return lo - pad, hi + pad
    pad = 0.07 * (hi - lo)
    return lo - pad, hi + pad


def panel_label(index: int) -> str:
    return f"({chr(ord('a') + index)})"


def metadata_subtitle(metadata: ProbeMetadata) -> str:
    parts = []
    if metadata.point is not None:
        x, y, z = metadata.point
        parts.append(rf"$x_p=({x:.5g},{y:.5g},{z:.5g})$")
    if metadata.frequency is not None:
        parts.append(rf"$f_0={metadata.frequency:.5g}$")
    if metadata.dt is not None:
        parts.append(rf"$\Delta t={metadata.dt:.3e}$")
    if metadata.probe_every is not None:
        parts.append(rf"probe every {metadata.probe_every} step(s)")
    if metadata.stride > 1:
        parts.append(rf"plotted every {metadata.stride} sample(s)")
    return ", ".join(parts)


def add_axis_style(ax) -> None:
    ax.grid(which="major", color="0.86", linewidth=0.7)
    ax.grid(which="minor", color="0.93", linewidth=0.45)
    ax.tick_params(which="both", top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)
    ax.axhline(0.0, color="0.35", linewidth=0.55, alpha=0.7, zorder=0)


def plot_probe_panel(
    data: ProbeData,
    output_pdf: Path,
    output_png: Path | None,
    title: str | None,
    figure_width: float,
    figure_height: float,
    include_exact: bool,
) -> None:
    colors = {
        "Ex": "#1b6da8",
        "Ey": "#1b6da8",
        "Ez": "#1b6da8",
        "Hx": "#b55d00",
        "Hy": "#b55d00",
        "Hz": "#b55d00",
    }
    exact_color = "0.18"

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(figure_width, figure_height),
        sharex=True,
        constrained_layout=True,
    )

    handles = []
    labels = []
    for row, spec in enumerate(COMPONENT_ROWS):
        e_label, e_name, e_exact_name, h_label, h_name, h_exact_name = spec
        for col, (label, name, exact_name) in enumerate(
            ((e_label, e_name, e_exact_name), (h_label, h_name, h_exact_name))
        ):
            ax = axes[row, col]
            line = ax.plot(
                data.time,
                data.values[name],
                color=colors[name],
                linewidth=1.45,
                label="DG probe",
            )[0]
            if row == 0 and col == 0:
                handles.append(line)
                labels.append("DG probe")

            if include_exact and exact_name in data.values:
                exact_line = ax.plot(
                    data.time,
                    data.values[exact_name],
                    color=exact_color,
                    linestyle=(0, (5.0, 2.2)),
                    linewidth=1.1,
                    alpha=0.82,
                    label="exact",
                )[0]
                if row == 0 and col == 0:
                    handles.append(exact_line)
                    labels.append("exact")

            y_arrays = [data.values[name]]
            if include_exact and exact_name in data.values:
                y_arrays.append(data.values[exact_name])
            # ax.set_ylim(*finite_limits(*y_arrays))
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel(label)
            add_axis_style(ax)
            ax.text(
                0.018,
                0.92,
                panel_label(2 * row + col),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.4,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.80",
                    "boxstyle": "round,pad=0.18",
                    "alpha": 0.92,
                },
            )

    axes[0, 0].set_title("Electric field")
    axes[0, 1].set_title("Magnetic field")
    axes[-1, 0].set_xlabel(r"Time $t$")
    axes[-1, 1].set_xlabel(r"Time $t$")

    figure_title = title or "3D Ricker-pulse probe signal"
    fig.suptitle(figure_title, fontsize=12.2, y=1.035)
    subtitle = metadata_subtitle(data.metadata)
    if subtitle:
        fig.text(0.5, 1.0, subtitle, ha="center", va="bottom", fontsize=8.9)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.972),
            ncol=len(handles),
        )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png)
    plt.close(fig)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    configure_matplotlib(args.usetex)

    probe_h5 = args.probe_h5 or default_probe_h5(args.run_dir)
    output_pdf = args.output or default_pdf_output(args.run_dir)
    output_png = None if args.no_png else (args.png_output or output_pdf.with_suffix(".png"))

    data = read_probe_hdf5(
        probe_h5,
        include_exact=not args.no_exact,
        stride=args.stride,
        max_points=args.max_points,
    )
    plot_probe_panel(
        data,
        output_pdf,
        output_png,
        args.title,
        args.figure_width,
        args.figure_height,
        include_exact=not args.no_exact,
    )

    print(f"Read probe HDF5: {probe_h5}")
    print(f"Samples in file: {data.metadata.n_samples}")
    print(f"Plot stride:     {data.metadata.stride}")
    print(f"Wrote PDF:       {output_pdf}")
    if output_png is not None:
        print(f"Wrote PNG:       {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
