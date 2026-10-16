#!/usr/bin/env python3
"""Publication figure for the 1D, p=1, centered-flux DG Maxwell spectrum.

Assumptions: uniform mesh, exact reference mass matrix, positive constant
epsilon and mu, theta = k*h, Omega = omega*h/c, c = 1/sqrt(epsilon*mu).
The four frequency branches are +/- (R - sin(theta)), +/- (R + sin(theta)).
Physical labels denote continuation from theta -> 0+ on [0, pi].

Dependencies: numpy, matplotlib (no external LaTeX installation required).
Usage:
    python3 plot_dispersion_branches.py
    python3 plot_dispersion_branches.py --outdir figures --formats pdf png svg
    python3 plot_dispersion_branches.py --width 3.5 --height 3.4 --font-size 8
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib


def dispersion_branches(theta):
    """Return positive-frequency physical and Spurious Modes on [0, pi]."""
    theta = np.asarray(theta, dtype=float)
    # Stable equivalent of sin(theta)**2 + 6*(1 - cos(theta)); avoids
    # subtracting nearly equal numbers close to theta = 0.
    radicand = np.sin(theta)**2 + 12.0 * np.sin(theta / 2.0)**2
    root = np.sqrt(radicand)
    return root - np.sin(theta), root + np.sin(theta)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outdir', type=Path, default=Path('figures'))
    parser.add_argument('--name', default='dispersion_branches_p1_centered')
    parser.add_argument('--formats', nargs='+', choices=('pdf', 'png', 'svg'),
                        default=['pdf', 'png'])
    parser.add_argument('--points', type=int, default=2001)
    parser.add_argument('--width', type=float, default=7.0, help='Width in inches.')
    parser.add_argument('--height', type=float, default=4.7, help='Height in inches.')
    parser.add_argument('--font-size', type=float, default=11.0)
    parser.add_argument('--dpi', type=int, default=600, help='Raster export resolution.')
    parser.add_argument('--title', default='', help='Optional title; omitted for journal use.')
    parser.add_argument('--show', action='store_true', help='Also open an interactive window.')
    args = parser.parse_args()
    if args.points < 3 or min(args.width, args.height, args.font_size, args.dpi) <= 0:
        parser.error('Use at least 3 points and positive dimensions, font size, and DPI.')
    if Path(args.name).name != args.name or args.name in ('', '.', '..'):
        parser.error('--name must be a filename stem without directory components.')
    return args


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    style = {
        'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': args.font_size,
        'axes.labelsize': args.font_size + 2,
        'axes.titlesize': args.font_size + 1,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'legend.fontsize': args.font_size - 1,
        'legend.frameon': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'savefig.facecolor': 'white',
    }
    theta = np.linspace(0.0, np.pi, args.points)
    physical, additional = dispersion_branches(theta)

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(args.width, args.height), layout='constrained')
        # Colorblind-friendly colors; line patterns retain distinction in grayscale.
        for sign in (1, -1):
            ax.plot(theta, sign * physical, color='#0072B2', linewidth=2.0,
                    label='Physical Mode' if sign == 1 else '_nolegend_', zorder=3)
            ax.plot(theta, sign * additional, color='#D55E00', linewidth=1.8,
                    linestyle=(0, (5, 2.5)),
                    label='Spurious Modes' if sign == 1 else '_nolegend_', zorder=2)
            ax.plot(theta, sign * theta, color='#202020', linewidth=1.25,
                    linestyle=(0, (1.5, 2.2)),
                    label=r'Exact: $\Omega=\pm\theta$' if sign == 1 else '_nolegend_',
                    zorder=4)

        ax.set_xlabel(r'$\theta = kh$')
        ax.set_ylabel(r'$\Omega = \omega h/c$')
        ax.set_xlim(0, np.pi)
        ymax = 1.10 * max(np.max(additional), np.pi)
        ax.set_ylim(-ymax, ymax)
        ax.set_xticks(np.pi * np.arange(5) / 4,
                      ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
        ax.set_yticks(np.arange(-4, 5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_axisbelow(True)
        ax.grid(which='major', color='0.90', linewidth=0.55)
        ax.axhline(0, color='0.65', linewidth=0.65, zorder=1)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.015),
                  ncol=3 if args.width >= 6 else 1, handlelength=2.6,
                  columnspacing=1.4, borderaxespad=0)
        if args.title:
            fig.suptitle(args.title)

        args.outdir.mkdir(parents=True, exist_ok=True)
        for extension in dict.fromkeys(args.formats):
            path = args.outdir / f'{args.name}.{extension}'
            fig.savefig(path, dpi=args.dpi)
            print(f'Saved {path.resolve()}')
        if args.show:
            plt.show()
        plt.close(fig)


if __name__ == '__main__':
    main()
