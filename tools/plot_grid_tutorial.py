#!/usr/bin/env python3
"""Plot the two figures of the parameter grid tutorial from the output of `proteus grid`.

Usage:
    proteus grid -c input/tutorials/tutorial_grid.toml
    python tools/plot_grid_tutorial.py output/tutorial_grid/ --output output_files/grid_plots/

Reads the resolved config (`init_coupler.toml`), the `status` file and
`runtime_helpfile.csv` of every `case_*` folder and writes:
    grid_tracks   - global melt fraction against time, coloured by orbital distance
    grid_outputs  - final melt fraction, final surface temperature and stop time against
                    orbital distance, and final surface pressure against the hydrogen
                    budget, coloured by planet mass

The figures follow the active matplotlib style: series colours from its colour cycle,
neutral colours from its text and surface colours. `--style light` or `--style dark`
activates the proteus-mpl style of the documentation figures.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from proteus.utils.plot import FADE_DARK_GREY, FADE_MID_GREY, style_neutral

STATUS_SOLIDIFIED = 10
STOP_LABELS = {STATUS_SOLIDIFIED: 'solidified', 14: 'energy balance'}


def load_cases(grid_dir: Path) -> list[dict]:
    """Read the swept inputs, the stop and the helpfile of every case of a grid.

    Parameters
    ----------
    grid_dir : Path
        Output folder of `proteus grid`, which holds the `case_*` folders.

    Returns
    -------
    list[dict]
        One record per case: `mass` [M_earth], `distance` [AU], `H_budget` and its
        unit `H_mode`, `status` code, `phi_crit` (None when the solidification stop
        is off) and `hf`, the helpfile rows after the initial stage. A case with a
        missing or empty file, or with no row after the initial stage, is
        skipped and named on the terminal.

    Raises
    ------
    FileNotFoundError
        If `grid_dir` holds no `case_*` folder with output.
    """
    cases = []
    for case_dir in sorted(grid_dir.glob('case_*')):
        files = [case_dir / n for n in ('init_coupler.toml', 'status', 'runtime_helpfile.csv')]
        if not all(f.is_file() and f.stat().st_size for f in files):
            print(f'Skipped {case_dir.name}: an output file is missing or empty')
            continue
        with open(case_dir / 'init_coupler.toml', 'rb') as f:
            cfg = tomllib.load(f)
        solid = cfg['params']['stop']['solid']
        hf = pd.read_csv(case_dir / 'runtime_helpfile.csv', sep=r'\s+')
        hf = hf[hf['Time'] > 0]
        if hf.empty:
            print(f'Skipped {case_dir.name}: no row after the initial stage')
            continue
        cases.append(
            {
                'mass': cfg['planet']['mass_tot'],
                'distance': cfg['orbit']['semimajoraxis'],
                'H_budget': cfg['planet']['elements']['H_budget'],
                'H_mode': cfg['planet']['elements']['H_mode'],
                'status': int((case_dir / 'status').read_text().split()[0]),
                'phi_crit': solid['phi_crit'] if solid['enabled'] else None,
                'hf': hf,
            }
        )
    if not cases:
        raise FileNotFoundError(f'No case_* folder with output in {grid_dir}')
    return cases


def _colours(values) -> dict:
    """Map each distinct value, in rising order, to a colour of the active style cycle."""
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return {v: cycle[i % len(cycle)] for i, v in enumerate(sorted(set(values)))}


def plot_tracks(cases: list[dict], out_path: Path) -> Figure:
    """Plot the global melt fraction of every case against time.

    Parameters
    ----------
    cases : list[dict]
        Records from `load_cases`.
    out_path : Path
        File to write.

    Returns
    -------
    Figure
        The saved figure.
    """
    colours = _colours(c['distance'] for c in cases)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c in cases:
        ax.plot(c['hf']['Time'], c['hf']['Phi_global'], color=colours[c['distance']], lw=1.6)

    # The threshold of the solidification stop, as the resolved configs give it
    for phi_crit in sorted({c['phi_crit'] for c in cases} - {None}):
        ax.axhline(phi_crit, color=style_neutral(FADE_MID_GREY), ls=':', lw=1.2)
        ax.annotate(
            rf'solidification ($\Phi$ = {phi_crit:g})',
            xy=(0.02, phi_crit),
            xycoords=('axes fraction', 'data'),
            xytext=(0, 4),
            textcoords='offset points',
            color=style_neutral(FADE_DARK_GREY),
            fontsize='small',
        )

    ax.set_xscale('log')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel(r'Global melt fraction $\Phi_{\rm global}$')
    handles = [
        Line2D([], [], color=col, lw=2.5, label=f'{d:.3g} AU') for d, col in colours.items()
    ]
    ax.legend(handles=handles, title='Orbital distance', frameon=False, loc='center left')
    fig.tight_layout()
    fig.savefig(out_path)
    return fig


def plot_outputs(cases: list[dict], out_path: Path) -> Figure:
    """Plot the final state of every case in four panels.

    Filled circles are cases that stopped solidified; open squares stopped for
    another reason, which the legend names.

    Parameters
    ----------
    cases : list[dict]
        Records from `load_cases`.
    out_path : Path
        File to write.

    Returns
    -------
    Figure
        The saved figure.
    """
    colours = _colours(c['mass'] for c in cases)
    ink, grey = style_neutral(), style_neutral(FADE_MID_GREY)
    distances = sorted({c['distance'] for c in cases})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [
        (axes[0, 0], 'distance', 'Phi_global', r'(a) Final melt fraction $\Phi_{\rm global}$'),
        (axes[0, 1], 'distance', 'T_surf', r'(b) Final $T_{\rm surf}$ [K]'),
        (axes[1, 0], 'distance', 'Time', '(c) Stop time [yr]'),
        (axes[1, 1], 'H_budget', 'P_surf', r'(d) Final $P_{\rm surf}$ [bar]'),
    ]
    for c in cases:
        col = colours[c['mass']]
        if c['status'] == STATUS_SOLIDIFIED:
            style = {'marker': 'o', 'facecolor': col, 'edgecolor': ink, 'linewidth': 0.8}
        else:
            style = {'marker': 's', 'facecolor': 'none', 'edgecolor': col, 'linewidth': 1.5}
        for ax, xkey, column, _ in panels:
            ax.scatter(c[xkey], c['hf'][column].iloc[-1], s=50, **style)

    for ax, xkey, _, title in panels:
        ax.set_title(title, loc='left')
        if xkey == 'distance':
            ax.set_xscale('log')
            ax.set_xticks(distances, [f'{d:.3g}' for d in distances])
            ax.minorticks_off()
            ax.set_xlabel('Orbital distance [AU]')
        else:
            ax.set_xlabel(f'H budget [{cases[0]["H_mode"]}]')
    axes[1, 1].set_yscale('log')

    mass_handles = [
        Line2D([], [], marker='o', ls='none', color=col, label=rf'{m:g} $M_\oplus$')
        for m, col in colours.items()
    ]
    axes[0, 0].legend(
        handles=mass_handles, title='Planet mass', frameon=False, loc='center left'
    )
    stop_handles = [
        Line2D(
            [],
            [],
            ls='none',
            marker='o' if code == STATUS_SOLIDIFIED else 's',
            markerfacecolor=grey if code == STATUS_SOLIDIFIED else 'none',
            markeredgecolor=ink if code == STATUS_SOLIDIFIED else grey,
            label=STOP_LABELS.get(code, f'status {code}'),
        )
        for code in sorted({c['status'] for c in cases})
    ]
    axes[0, 1].legend(
        handles=stop_handles, title='Stop reason', frameon=False, loc='upper right'
    )
    fig.tight_layout()
    fig.savefig(out_path)
    return fig


def make_tracks(run_dirs: list[Path], out_path: Path) -> Figure:
    """Draw `grid_tracks` for `tools/make_docs_figures.py` from the grid folder `run_dirs[0]`."""
    return plot_tracks(load_cases(run_dirs[0]), out_path)


def make_outputs(run_dirs: list[Path], out_path: Path) -> Figure:
    """Draw `grid_outputs` for `tools/make_docs_figures.py` from the grid folder `run_dirs[0]`."""
    return plot_outputs(load_cases(run_dirs[0]), out_path)


def main(argv: list[str] | None = None):
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('grid_dir', type=Path, help='Output folder of `proteus grid`')
    parser.add_argument('--output', type=Path, default=Path('output_files/grid_plots'))
    parser.add_argument('--format', default='png', help='Figure format (png, pdf, ...)')
    parser.add_argument('--style', choices=['light', 'dark'], help='proteus-mpl style to use')
    args = parser.parse_args(argv)

    cases = load_cases(args.grid_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'savefig.dpi': 200}):
        if args.style:
            try:
                import proteus_mpl
            except ImportError:
                parser.error('--style needs the proteus-mpl package (pip install proteus-mpl)')
            proteus_mpl.use(args.style)
        plot_tracks(cases, args.output / f'grid_tracks.{args.format}')
        plot_outputs(cases, args.output / f'grid_outputs.{args.format}')
    plt.close('all')
    print(f'Wrote grid_tracks.{args.format} and grid_outputs.{args.format} to {args.output}')


if __name__ == '__main__':
    main()
