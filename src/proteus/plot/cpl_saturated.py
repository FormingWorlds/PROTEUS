from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.utils.constants import condensate_list, condensate_names
from proteus.utils.plot import get_colour

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger('fwl.' + __name__)

# Activity at and above which a condensate counts as saturated (stable)
SATURATED = 0.999


def plot_saturated(hf_all: pd.DataFrame, output_dir: str, plot_format='pdf'):
    """Plot condensate saturation, condensed mass, and the magma state over time.

    Reads the ``{X}_cd_activity`` and ``{X}_cd_kg`` columns that the
    atmodeller outgassing module writes for each supported condensate. A
    condensate that was never part of the equilibrium is left out, and the
    plot is skipped when none was.
    """
    hf_crop = hf_all.iloc[2:]
    hf_crop = hf_crop[hf_crop['Time'] > 0]
    time = np.array(hf_crop['Time'])

    # Condensates with at least two solved points
    solved = {}
    for c in condensate_list:
        key = f'{c}_cd_activity'
        if key not in hf_crop.columns:
            continue
        activity = np.array(hf_crop[key], dtype=float)
        valid = np.isfinite(activity) & (activity > 0)
        if np.count_nonzero(valid) >= 2:
            solved[c] = (activity, valid)
    if not solved:
        log.debug('No condensate activities to plot; skipping plot_saturated')
        return

    log.info('Plot saturated')

    # mass unit
    M_uval = 1e18  # kg
    M_ulbl = r'$10^{18}$kg'

    # create plot
    lw = 1.2
    scale = 1.2
    fig, axs = plt.subplots(3, 1, figsize=(5 * scale, 6 * scale), sharex=True)
    ax0 = axs[0]  # activity
    ax1 = axs[1]  # condensed mass
    ax2 = axs[2]  # magma temperature and melt fraction
    ax2r = ax2.twinx()

    for c, (activity, valid) in solved.items():
        col = get_colour(c)
        lbl = condensate_names.get(c, c)

        ax0.plot(time[valid], activity[valid], lw=lw, color=col, label=lbl)

        # Below saturation the condensed mass is only the solver's floor, so
        # only saturated points carry a mass
        mass = np.array(hf_crop.get(f'{c}_cd_kg', np.nan), dtype=float)
        mass = np.where(valid & (activity >= SATURATED), mass, 0.0)
        ax1.plot(time, mass / M_uval, lw=lw, color=col)

    # Decorate top plot; the reference sits under a saturated condensate's line
    ax0.axhline(1.0, color='grey', ls='dashed', lw=lw * 0.8, label='Saturated', zorder=1)
    ax0.set_yscale('log')
    ax0.set_ylim(top=2.0)
    ax0.set_ylabel('Activity')
    ax0.legend(
        loc='upper left',
        bbox_to_anchor=(1.0, 1.0),
        fontsize=9,
        labelspacing=0.3,
        handlelength=1.5,
        handletextpad=0.4,
    )

    # Decorate middle plot
    ax1.set_ylabel(rf'Condensed [{M_ulbl}]')
    ax1.set_ylim(bottom=0)

    # Magma temperature and melt fraction
    ax2.plot(time, hf_crop['T_magma'], lw=lw, color='k')
    ax2.set_ylabel(r'T$_\text{magma}$ [K]')
    color = 'tab:red'
    ax2r.plot(time, hf_crop['Phi_global'], lw=lw, color=color, alpha=0.8)
    ax2r.set_ylim(0, 1.05)
    ax2r.tick_params(axis='y', color=color, labelcolor=color)
    ax2r.set_ylabel('Melt fraction', color=color)

    ax2.set_xscale('log')
    ax2.set_xlim(left=np.amin(time), right=np.amax(time))
    ax2.set_xlabel('Time [yr]')

    # Adjust
    fig.subplots_adjust(hspace=0.04)
    fig.align_ylabels()
    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, 'plots', 'plot_saturated.%s' % plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')


def plot_saturated_entry(handler: Proteus):
    from proteus.utils.coupler import read_helpfile_table

    # read helpfile
    hf_all = read_helpfile_table(
        os.path.join(handler.directories['output'], 'runtime_helpfile.csv')
    )

    # make plot
    plot_saturated(
        hf_all=hf_all,
        output_dir=handler.directories['output'],
        plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == '__main__':
    from proteus.plot._cpl_helpers import get_handler_from_argv

    handler = get_handler_from_argv()
    plot_saturated_entry(handler)
