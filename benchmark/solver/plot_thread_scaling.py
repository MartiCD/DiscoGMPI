#!/usr/bin/env python3

import csv
import os
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "discompi-matplotlib"),
)

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    print("matplotlib is required to generate the scaling plots.", file=sys.stderr)
    print("Install it with: python3 -m pip install matplotlib", file=sys.stderr)
    raise exc


DEFAULT_INPUT_CSV = BENCHMARK_DIR / "thread_scaling.csv"
SPEEDUP_PNG = BENCHMARK_DIR / "thread_speedup.png"
EFFICIENCY_PNG = BENCHMARK_DIR / "thread_efficiency.png"

REQUIRED_COLUMNS = (
    "threads",
    "speedup_med",
    "efficiency_med",
)


def usage() -> None:
    print("Usage:")
    print("  python3 benchmark/solver/plot_thread_scaling.py [thread_scaling.csv]")
    print()
    print("Default input:")
    print(f"  {DEFAULT_INPUT_CSV}")


def read_scaling_data(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"CSV file not found: {path}. Run benchmark/solver/run_thread_scaling.sh first."
        )

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]

        if missing:
            raise ValueError(
                f"CSV file {path} is missing required columns: {', '.join(missing)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file has no benchmark rows: {path}")

    rows.sort(key=lambda row: int(row["threads"]))

    return rows


def column_as_float(rows: list[dict[str, str]], name: str) -> list[float]:
    return [float(row[name]) for row in rows]


def plot_speedup(rows: list[dict[str, str]]) -> None:
    threads = column_as_float(rows, "threads")
    speedup = column_as_float(rows, "speedup_med")

    xmax = max(threads)
    ymax = max(max(threads), max(speedup))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(threads, threads, "--", color="0.5", label="ideal")
    ax.plot(threads, speedup, "o-", label="measured")
    ax.set_xlabel("Julia threads")
    ax.set_ylabel("Median speedup")
    ax.set_title("DiscoGMPI ThreadedBackend Strong Scaling")
    ax.set_xlim(0.0, 1.05 * xmax)
    ax.set_ylim(0.0, 1.10 * ymax)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(SPEEDUP_PNG, dpi=180)
    plt.close(fig)


def plot_efficiency(rows: list[dict[str, str]]) -> None:
    threads = column_as_float(rows, "threads")
    efficiency = column_as_float(rows, "efficiency_med")

    xmax = max(threads)
    ymax = max(1.0, max(efficiency))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(threads, [1.0] * len(threads), "--", color="0.5", label="ideal")
    ax.plot(threads, efficiency, "o-", label="measured")
    ax.set_xlabel("Julia threads")
    ax.set_ylabel("Median parallel efficiency")
    ax.set_title("DiscoGMPI ThreadedBackend Parallel Efficiency")
    ax.set_xlim(0.0, 1.05 * xmax)
    ax.set_ylim(0.0, 1.10 * ymax)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(EFFICIENCY_PNG, dpi=180)
    plt.close(fig)


def main(argv: list[str]) -> int:
    if any(arg in ("--help", "-h") for arg in argv):
        usage()
        return 0

    if len(argv) > 1:
        print("Expected at most one CSV path argument.", file=sys.stderr)
        usage()
        return 2

    input_csv = Path(argv[0]).resolve() if argv else DEFAULT_INPUT_CSV
    rows = read_scaling_data(input_csv)

    plot_speedup(rows)
    plot_efficiency(rows)

    print(f"Wrote speedup plot:    {SPEEDUP_PNG}")
    print(f"Wrote efficiency plot: {EFFICIENCY_PNG}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
