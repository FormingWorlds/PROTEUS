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
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

# Wong colour-blind palette, without black
WONG = ['#E69F00', '#56B4E9', '#009E73', '#D55E00', '#0072B2', '#CC79A7', '#F0E442']

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
        is off) and `hf`, the helpfile rows after the initial stage.

    Raises
    ------
    FileNotFoundError
        If `grid_dir` holds no `case_*` folder.
    """
    cases = []
    for case_dir in sorted(grid_dir.glob('case_*')):
        with open(case_dir / 'init_coupler.toml', 'rb') as f:
            cfg = tomllib.load(f)
        solid = cfg['params']['stop']['solid']
        hf = pd.read_csv(case_dir / 'runtime_helpfile.csv', sep=r'\s+')
        cases.append(
            {
                'mass': cfg['planet']['mass_tot'],
                'distance': cfg['orbit']['semimajoraxis'],
                'H_budget': cfg['planet']['elements']['H_budget'],
                'H_mode': cfg['planet']['elements']['H_mode'],
                'status': int((case_dir / 'status').read_text().split()[0]),
                'phi_crit': solid['phi_crit'] if solid['enabled'] else None,
                'hf': hf[hf['Time'] > 0],
            }
        )
    if not cases:
        raise FileNotFoundError(f'No case_* folder in {grid_dir}')
    return cases


def _colours(values) -> dict:
    """Map each distinct value, in rising order, to a colour of the palette."""
    return {v: WONG[i % len(WONG)] for i, v in enumerate(sorted(set(values)))}


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
        ax.axhline(phi_crit, color='grey', ls=':', lw=1.2)
        ax.annotate(
            rf'solidification ($\Phi$ = {phi_crit:g})',
            xy=(0.02, phi_crit),
            xycoords=('axes fraction', 'data'),
            xytext=(0, 4),
            textcoords='offset points',
            color='dimgrey',
            fontsize=9,
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
    fig.savefig(out_path, dpi=200)
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
            style = {'marker': 'o', 'facecolor': col, 'edgecolor': 'black', 'linewidth': 0.8}
        else:
            style = {'marker': 's', 'facecolor': 'none', 'edgecolor': col, 'linewidth': 1.5}
        for ax, xkey, column, _ in panels:
            ax.scatter(c[xkey], c['hf'][column].iloc[-1], s=50, **style)

    for ax, xkey, _, title in panels:
        ax.set_title(title, loc='left', fontsize=10)
        if xkey == 'distance':
            ax.set_xscale('log')
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
            markerfacecolor='grey' if code == STATUS_SOLIDIFIED else 'none',
            markeredgecolor='black' if code == STATUS_SOLIDIFIED else 'grey',
            label=STOP_LABELS.get(code, f'status {code}'),
        )
        for code in sorted({c['status'] for c in cases})
    ]
    axes[0, 1].legend(
        handles=stop_handles, title='Stop reason', frameon=False, loc='upper right'
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    return fig


def main(argv: list[str] | None = None):
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('grid_dir', type=Path, help='Output folder of `proteus grid`')
    parser.add_argument('--output', type=Path, default=Path('output_files/grid_plots'))
    parser.add_argument('--format', default='png', help='Figure format (png, pdf, ...)')
    args = parser.parse_args(argv)

    cases = load_cases(args.grid_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    plot_tracks(cases, args.output / f'grid_tracks.{args.format}')
    plot_outputs(cases, args.output / f'grid_outputs.{args.format}')
    plt.close('all')
    print(f'Wrote grid_tracks.{args.format} and grid_outputs.{args.format} to {args.output}')


if __name__ == '__main__':
    main()
