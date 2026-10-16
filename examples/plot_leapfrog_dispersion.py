#!/usr/bin/env python3
"""Publication figure for exact-mass P1 centered DG + staggered E-H leapfrog.

Dependencies: Python >= 3.9, NumPy, Matplotlib. No LaTeX installation is needed.

Run:
    python3 plot_leapfrog_dispersion.py
    python3 plot_leapfrog_dispersion.py --courant 0.1 0.3 0.49 --output-dir figures
    python3 plot_leapfrog_dispersion.py --width-mm 180 --height-mm 98 --dpi 600

Panel (a): positive semidiscrete and fully discrete dispersion branches.
Panel (b): physical-mode relative phase-speed error and leading-order expansion.

Definitions:
    theta = k*h, r = c*dt/h, Omega = omega*h/c.
    F_phys/add = sqrt(sin(theta)**2 + 6*(1-cos(theta))) -/+ sin(theta).
    Omega_FD = 2/r * arcsin(r*F/2).
    phase_error = Omega_FD_phys/theta - 1.
    long_wave_error = (1/48 + r**2/24) * theta**2.

Only 0 <= theta <= pi is plotted; negative-frequency branches follow by
symmetry. The exact all-wavenumber stability condition is 0 < r < 1/2.
The r=0 curves denote the continuous-time (semidiscrete) limit, not a run
with a zero time step. These formulas assume exact mass integration; a
mass-lumped discretization generally has different branches.

Small-argument series are used ONLY to stabilize floating-point evaluation,
not to replace the dispersion curves by their leading-order asymptotes.
The asymptotes are evaluated separately and drawn as dashed curves.
"""

import argparse
import io
import os
import tempfile
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import matplotlib


COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9")
MARKERS = ("o", "s", "^", "D", "v")
SEMIDISCRETE_COLOR = "#727272"
EXACT_STYLE = (0, (5.0, 3.0))
ADDITIONAL_STYLE = (0, (1.0, 2.1))
ASYMPTOTIC_STYLE = (0, (4.5, 3.0))
PI = np.pi


def spatial_branches(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the positive physical and additional P1 branches on [0, pi].

    The half-angle expression avoids cancellation in 1-cos(theta).
    R = 2*sin(theta/2)*sqrt(4-sin(theta/2)**2).
    """
    theta = np.asarray(theta, dtype=float)
    u = np.sin(0.5 * theta)
    v = np.cos(0.5 * theta)
    root = np.sqrt(4.0 - u * u)
    return 2.0 * u * (root - v), 2.0 * u * (root + v)


def _asin_argument(z: np.ndarray) -> np.ndarray:
    """Reject a genuinely invalid argument; tolerate only roundoff at +/-1."""
    z = np.asarray(z, dtype=float)
    if np.any(~np.isfinite(z)) or np.any(np.abs(z) > 1.0 + 8 * np.finfo(float).eps):
        raise ValueError("The inverse-sine argument lies outside its real domain.")
    return np.clip(z, -1.0, 1.0)


def fully_discrete(branch: np.ndarray, r: float) -> np.ndarray:
    """Apply the staggered leapfrog frequency map; r=0 is the exact-time limit."""
    branch = np.asarray(branch, dtype=float)
    if not np.isfinite(r) or r < 0.0 or r >= 0.5:
        raise ValueError("Use 0 <= r < 0.5; r=0 denotes the semidiscrete limit.")
    if r == 0.0:
        return branch.copy()
    return (2.0 / r) * np.arcsin(_asin_argument(0.5 * r * branch))


def _arcsinc_minus_one(z: np.ndarray) -> np.ndarray:
    """Compute arcsin(z)/z - 1 accurately, including its limiting value at 0."""
    z = _asin_argument(z)
    result = np.empty_like(z)
    small = np.abs(z) < 0.01
    q = z[small] ** 2
    result[small] = q * (1.0 / 6.0 + q * (
        3.0 / 40.0 + q * (5.0 / 112.0 + q * 35.0 / 1152.0)
    ))
    result[~small] = np.arcsin(z[~small]) / z[~small] - 1.0
    return result


def phase_speed_error(theta: np.ndarray, r: float) -> np.ndarray:
    """Evaluate Omega_FD_phys/theta - 1 without subtracting near-equal values.

    Let ds = F_phys/theta - 1 and dt = arcsin(z)/z - 1, z=r*F_phys/2.
    Then error = ds + (1+ds)*dt. For theta < 0.01 the spatial error is
    evaluated through theta**6; its omitted terms are below the precision
    useful for this plot. The function is continuous with value 0 at theta=0.
    """
    theta = np.asarray(theta, dtype=float)
    if np.any(~np.isfinite(theta)) or np.any(theta < 0.0) or np.any(theta > PI):
        raise ValueError("theta must lie in [0, pi].")
    if not np.isfinite(r) or r < 0.0 or r >= 0.5:
        raise ValueError("Use 0 <= r < 0.5.")
    physical, _ = spatial_branches(theta)
    spatial_error = np.empty_like(theta)
    small = theta < 0.01
    t2 = theta[small] ** 2
    spatial_error[small] = t2 * (
        1.0 / 48.0 + t2 * (-7.0 / 15360.0 - t2 * 599.0 / 10321920.0)
    )
    spatial_error[~small] = physical[~small] / theta[~small] - 1.0
    if r == 0.0:
        return spatial_error
    temporal_error = _arcsinc_minus_one(0.5 * r * physical)
    return spatial_error + (1.0 + spatial_error) * temporal_error


def long_wave_error(theta: np.ndarray, r: float) -> np.ndarray:
    return (1.0 / 48.0 + r * r / 24.0) * np.asarray(theta, dtype=float) ** 2


def _style() -> dict:
    return {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.0,
        "axes.linewidth": 0.65,
        "axes.unicode_minus": True,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "legend.fontsize": 8.0,
        "legend.frameon": False,
        "legend.handlelength": 2.3,
        "legend.borderaxespad": 0.6,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # Keep SVG text editable.
        "svg.hashsalt": "p1-centered-dg-staggered-leapfrog",
        "path.simplify": False,
        "text.usetex": False,
    }


def make_figure(courant: Sequence[float], width_mm: float, height_mm: float,
                theta_min: float, points: int):
    """Construct the publication figure and return it without saving."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    from matplotlib.ticker import AutoMinorLocator, FixedLocator, LogLocator, NullFormatter

    theta_star = np.arccos(-3.0 / 5.0)
    theta = np.unique(np.r_[np.linspace(0.0, PI, points), theta_star])
    theta_log = np.geomspace(theta_min, PI, points)
    physical, additional = spatial_branches(theta)

    with plt.rc_context(_style()):
        fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
        grid = fig.add_gridspec(1, 2, left=0.085, right=0.985,
                               bottom=0.225, top=0.905, wspace=0.34)
        ax_disp = fig.add_subplot(grid[0, 0])
        ax_err = fig.add_subplot(grid[0, 1])

        for ax in (ax_disp, ax_err):
            ax.set_axisbelow(True)
            ax.grid(True, which="major", axis="y", color="#DCDCDC", linewidth=0.45)
            ax.tick_params(pad=3.5)

        ax_disp.set_title(r"$\mathbf{(a)}$ Dispersion branches", loc="left", pad=9)
        ax_err.set_title(r"$\mathbf{(b)}$ Physical-mode phase error", loc="left", pad=9)

        # Reference and additional branches: same colors, different line styles.
        ax_disp.plot(theta, physical, color=SEMIDISCRETE_COLOR, lw=1.25, zorder=2)
        ax_disp.plot(theta, additional, color=SEMIDISCRETE_COLOR, lw=1.65,
                     linestyle=ADDITIONAL_STYLE, zorder=2)
        ax_disp.plot(theta, theta, color="#171717", lw=1.2,
                     linestyle=EXACT_STYLE, zorder=5)

        color_handles = [Line2D([], [], color=SEMIDISCRETE_COLOR, lw=1.5,
                               label=r"$r\to0$ (semidiscrete)")]
        max_frequency = float(np.max(additional))
        for index, r in enumerate(courant):
            color = COLORS[index]
            fd_physical = fully_discrete(physical, r)
            fd_additional = fully_discrete(additional, r)
            ax_disp.plot(theta, fd_physical, color=color, lw=1.4, zorder=3)
            ax_disp.plot(theta, fd_additional, color=color, lw=1.7,
                         linestyle=ADDITIONAL_STYLE, zorder=3)
            max_frequency = max(max_frequency, float(np.max(fd_additional)))
            color_handles.append(Line2D([], [], color=color, lw=1.6,
                                        label=rf"$r={r:g}$"))

        ax_disp.set(xlim=(0.0, PI), ylim=(0.0, np.ceil(max_frequency * 1.06)),
                    xlabel=r"$\theta=kh$", ylabel=r"$\Omega=\omega h/c$")
        ax_disp.set_xticks([0.0, PI / 4, PI / 2, 3 * PI / 4, PI],
                           [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
        ax_disp.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax_disp.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax_disp.legend(handles=[
            Line2D([], [], color="#333333", lw=1.4, label="Physical branch"),
            Line2D([], [], color="#333333", lw=1.7,
                   linestyle=ADDITIONAL_STYLE, label="Spurious branch"),
            Line2D([], [], color="#171717", lw=1.2,
                   linestyle=EXACT_STYLE, label=r"Exact: $\Omega=\theta$"),
        ], loc="upper left")
        ax_disp.text(0.98, 0.98, r"$p=1,\quad r<\frac{1}{2}$",
                     ha="right", va="top", transform=ax_disp.transAxes,
                     fontsize=8.5, color="#333333")

        # Curves and asymptotes are independently evaluated.
        errors = []
        for index, r in enumerate([0.0, *courant]):
            color = SEMIDISCRETE_COLOR if index == 0 else COLORS[index - 1]
            err = phase_speed_error(theta_log, r)
            asymptote = long_wave_error(theta_log, r)
            if np.any(err <= 0.0) or not np.all(np.isfinite(err)):
                raise ValueError("Expected positive finite phase errors for this P1 model.")
            errors.extend([err, asymptote])
            ax_err.plot(theta_log, asymptote, color=color, lw=1.0,
                        linestyle=ASYMPTOTIC_STYLE, alpha=0.75, zorder=2)
            kwargs = {} if index == 0 else {
                "marker": MARKERS[index - 1], "markersize": 2.8,
                "markerfacecolor": "white", "markeredgewidth": 0.7,
                "markevery": [int(f * (points - 1)) for f in (0.18, 0.43, 0.68, 0.90)],
            }
            ax_err.plot(theta_log, err, color=color, lw=1.3, zorder=3, **kwargs)

        ymin = 10.0 ** np.floor(np.log10(min(float(a.min()) for a in errors)))
        ymax = 10.0 ** np.ceil(np.log10(max(float(a.max()) for a in errors)))
        ax_err.set(xscale="log", yscale="log", xlim=(theta_min, PI),
                   ylim=(ymin, ymax), xlabel=r"$\theta=kh$",
                   ylabel=r"$c_{\mathrm{ph}}/c-1$")
        decades = 10.0 ** np.arange(int(np.ceil(np.log10(theta_min))), 1)
        xticks = [*decades, PI]
        xlabels = [rf"$10^{{{int(round(np.log10(x)))}}}$" for x in decades] + [r"$\pi$"]
        ax_err.xaxis.set_major_locator(FixedLocator(xticks))
        ax_err.set_xticklabels(xlabels)
        ax_err.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        ax_err.xaxis.set_minor_formatter(NullFormatter())
        ax_err.yaxis.set_major_locator(LogLocator(base=10, numticks=9))
        ax_err.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=30))
        ax_err.yaxis.set_minor_formatter(NullFormatter())
        ax_err.legend(handles=[
            Line2D([], [], color="#333333", lw=1.4, label="Dispersion relation"),
            Line2D([], [], color="#333333", lw=1.0,
                   linestyle=ASYMPTOTIC_STYLE, label="Long-wave expansion"),
        ], loc="upper left")

        # Triangle coordinates are chosen in log space: diagonal slope = 2.
        span = np.log(PI / theta_min)
        x0, x1 = theta_min * np.exp(np.array([0.60, 0.80]) * span)
        y0 = max(5.0 * ymin, 0.015 * float(phase_speed_error(np.array([x0]), 0)[0]))
        y1 = y0 * (x1 / x0) ** 2
        triangle = Polygon([(x0, y0), (x1, y0), (x1, y1)], closed=True,
                           facecolor="#ECECEC", edgecolor="#686868", lw=0.7, zorder=1)
        ax_err.add_patch(triangle)
        ax_err.text((x0 * x1 * x1) ** (1 / 3), (y0 * y0 * y1) ** (1 / 3),
                    r"$2$", ha="center", va="center", fontsize=9, color="#444444")

        fig.legend(handles=color_handles, loc="lower center", bbox_to_anchor=(0.53, 0.035),
                   ncol=min(4, len(color_handles)), fontsize=8.5,
                   handlelength=2.0, columnspacing=1.4, handletextpad=0.55)
        return fig


def write_caption(path: Path, basename: str, courant: Sequence[float]) -> None:
    values = ", ".join(f"{r:g}" for r in courant)
    caption = r"""% Requires graphicx and amsmath. Place alongside the exported figure.
\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{BASENAME.pdf}
  \caption{Dispersion of the one-dimensional Maxwell system discretized by
  centered-flux DG with $p=1$, exact mass integration, and staggered
  electric--magnetic leapfrog time integration. Here $\theta=kh$,
  $\Omega=\omega h/c$, and $r=c\Delta t/h$.
  (a) Positive physical (solid) and additional (dotted) frequency branches
  for $r=VALUES$, together with the semidiscrete limit (gray) and the exact
  relation $\Omega=\theta$ (black dashed). Negative-frequency branches follow
  by symmetry. All displayed Courant numbers satisfy the sharp
  all-wavenumber stability bound $r<1/2$.
  (b) Relative physical-mode phase-speed error from the dispersion relation
  (solid) and its leading-order expansion
  $(1/48+r^2/24)\theta^2$ (dashed).
  The dashed curves are asymptotic predictions as $\theta\to0$;
  their extension to larger $\theta$ illustrates the departure from that limit.
  The triangle indicates a logarithmic slope of two.}
  \label{fig:p1-dg-leapfrog-dispersion}
\end{figure}
"""
    path.write_text(caption.replace("BASENAME", basename).replace("VALUES", values), encoding="utf-8")


def save_figure_atomic(fig, path: Path, extension: str, dpi: int, metadata) -> None:
    """Finish rendering before publishing a file, so interrupted writes stay private."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format=extension, dpi=dpi, metadata=metadata)
    payload = buffer.getvalue()
    if extension == "pdf" and not payload.rstrip().endswith(b"%%EOF"):
        raise RuntimeError("PDF rendering did not produce a complete document.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".figure_",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--courant", type=float, nargs="+", default=[0.1, 0.3, 0.49],
                        help="One to five distinct Courant numbers, all strictly between 0 and 0.5.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--basename", default="dg_p1_leapfrog_dispersion")
    parser.add_argument("--formats", nargs="+", choices=("pdf", "svg", "png"), default=["pdf", "svg", "png"])
    parser.add_argument("--width-mm", type=float, default=180.0, help="Physical figure width (default: 180 mm).")
    parser.add_argument("--height-mm", type=float, default=98.0)
    parser.add_argument("--dpi", type=int, default=600, help="Resolution of PNG export.")
    parser.add_argument("--theta-min", type=float, default=0.01, help="Left endpoint of the log panel; default: 0.01.")
    parser.add_argument("--points", type=int, default=2000)
    parser.add_argument("--show", action="store_true", help="Also display the figure interactively.")
    args = parser.parse_args()
    if not 1 <= len(args.courant) <= len(COLORS):
        parser.error("Choose between one and five Courant numbers.")
    if any(not np.isfinite(r) or not 0 < r < 0.5 for r in args.courant):
        parser.error("Every Courant number must satisfy 0 < r < 0.5; the critical value 0.5 is excluded.")
    if len(set(args.courant)) != len(args.courant):
        parser.error("Courant numbers must be distinct.")
    args.courant = sorted(args.courant)
    if not np.isfinite(args.theta_min) or not 1e-8 <= args.theta_min <= 0.1:
        parser.error("--theta-min must lie between 1e-8 and 0.1.")
    if args.points < 200 or args.points > 100000:
        parser.error("--points must lie between 200 and 100000.")
    if not (np.isfinite(args.width_mm) and np.isfinite(args.height_mm)
            and args.width_mm >= 150 and args.height_mm >= 85):
        parser.error("For this two-panel layout use width >= 150 mm and height >= 85 mm.")
    if args.dpi < 150:
        parser.error("--dpi must be at least 150 (600 is recommended for publication).")
    if not args.basename or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.basename):
        parser.error("--basename may contain only letters, digits, underscores, and hyphens.")
    return args


def main() -> None:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = make_figure(args.courant, args.width_mm, args.height_mm, args.theta_min, args.points)
    # Preserve the requested physical dimensions: do not use bbox_inches='tight'.
    with plt.rc_context(_style()):
        for extension in dict.fromkeys(args.formats):
            path = args.output_dir / f"{args.basename}.{extension}"
            metadata = None
            if extension == "pdf":
                metadata = {"Title": "P1 DG Maxwell: staggered leapfrog dispersion",
                            "Subject": "Fully discrete branches and physical-mode phase-speed error",
                            "Creator": "plot_leapfrog_dispersion.py", "CreationDate": None, "ModDate": None}
            elif extension == "svg":
                metadata = {"Title": "P1 DG Maxwell: staggered leapfrog dispersion", "Date": None}
            save_figure_atomic(fig, path, extension, args.dpi, metadata)
            print(f"Saved: {path.resolve()}")
    if "pdf" in args.formats:
        caption_path = args.output_dir / f"{args.basename}_caption.tex"
        write_caption(caption_path, args.basename, args.courant)
        print(f"Saved: {caption_path.resolve()}")

    print(f"Figure size: {args.width_mm:g} x {args.height_mm:g} mm; PNG: {args.dpi} dpi")
    print("Model: P1 centered-flux DG, exact mass; staggered E-H leapfrog; 0 < r < 1/2.")
    fit_theta = np.geomspace(1e-4, 1e-2, 150)
    for r in [0.0, *args.courant]:
        slope = np.polyfit(np.log(fit_theta), np.log(phase_speed_error(fit_theta, r)), 1)[0]
        print(f"r={r:g}: long-wave log-log slope = {slope:.6f} (expected 2)")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
