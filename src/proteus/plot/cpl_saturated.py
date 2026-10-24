from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.utils.constants import condensate_density, condensate_list, condensate_names
from proteus.utils.plot import get_colour

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger('fwl.' + __name__)

# Activity at and above which a condensate counts as saturated (stable)
SATURATED = 0.999

# Condensed mass above which a point carries a real layer [kg]. Below saturation the
# solver reports only a floor of order 1e-6 kg; a larger mass at lower activity is a
# layer held by outgas.prevent_desaturation.
HELD_MASS_KG = 1.0


def plot_saturated(hf_all: pd.DataFrame, output_dir: str, plot_format='pdf'):
    """Plot condensate saturation, condensed column, layer thickness, and magma state.

    Reads the ``{X}_cd_activity`` and ``{X}_cd_kg`` columns that the
    atmodeller outgassing module writes for each supported condensate. The
    condensed mass is shown per unit surface area, and as the thickness of a
    uniform layer of the condensate at its ``condensate_density``, which is an
    estimate only. A condensate that was never part of the equilibrium is left
    out, and the plot is skipped when none was.

    With ``outgas.prevent_desaturation`` the activity is that of the inventory left
    in the solve, so it can fall below 1 while a held layer keeps its thickness.
    """
    hf_crop = hf_all.iloc[2:]
    hf_crop = hf_crop[hf_crop['Time'] > 0]
    time = np.array(hf_crop['Time'])

    # Condensates with at least two solved points
    solved = {}
    for c in condensate_list:
        if f'{c}_cd_activity' not in hf_crop.columns or f'{c}_cd_kg' not in hf_crop.columns:
            continue
        activity = np.array(hf_crop[f'{c}_cd_activity'], dtype=float)
        mass = np.array(hf_crop[f'{c}_cd_kg'], dtype=float)
        valid = np.isfinite(activity) & (activity > 0)
        if np.count_nonzero(valid) >= 2:
            solved[c] = (activity, valid, mass)
    if not solved:
        log.debug('No condensate activities to plot; skipping plot_saturated')
        return

    log.info('Plot saturated')

    # column density unit
    C_uval = 1e6  # kg m-2
    C_ulbl = r'$10^{6}$ kg m$^{-2}$'

    # Surface area over which the condensate is spread [m2]
    area = 4.0 * np.pi * np.array(hf_crop['R_int'], dtype=float) ** 2

    # create plot
    lw = 1.2
    scale = 1.2
    fig, axs = plt.subplots(4, 1, figsize=(5 * scale, 7.5 * scale), sharex=True)
    ax0 = axs[0]  # activity
    ax1 = axs[1]  # condensed column density
    ax2 = axs[2]  # condensed layer thickness
    ax3 = axs[3]  # magma temperature and melt fraction
    ax3r = ax3.twinx()

    for c, (activity, valid, mass) in solved.items():
        col = get_colour(c)
        lbl = condensate_names.get(c, c)

        ax0.plot(time[valid], activity[valid], lw=lw, color=col, label=lbl)

        # Below saturation the condensed mass is only the solver's floor, so only
        # saturated points, or a layer held from re-vapourising, carry a mass. A held
        # layer is kept also where the condensate left the network (activity 0).
        keep = (np.isfinite(mass) & (mass > HELD_MASS_KG)) | (valid & (activity >= SATURATED))
        mass = np.where(keep, mass, 0.0)
        column = mass / area
        ax1.plot(time, column / C_uval, lw=lw, color=col)
        ax2.plot(time, column / condensate_density[c], lw=lw, color=col)

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

    # Decorate column density and layer thickness plots
    ax1.set_ylabel(rf'Condensed [{C_ulbl}]')
    ax1.set_ylim(bottom=0)
    ax2.set_ylabel('Layer thickness [m]')
    ax2.set_ylim(bottom=0)

    # Magma temperature and melt fraction
    ax3.plot(time, hf_crop['T_magma'], lw=lw, color='k')
    ax3.set_ylabel(r'T$_\text{magma}$ [K]')
    color = 'tab:red'
    ax3r.plot(time, hf_crop['Phi_global'], lw=lw, color=color, alpha=0.8)
    ax3r.set_ylim(0, 1.05)
    ax3r.tick_params(axis='y', color=color, labelcolor=color)
    ax3r.set_ylabel('Melt fraction', color=color)

    ax3.set_xscale('log')
    ax3.set_xlim(left=np.amin(time), right=np.amax(time))
    ax3.set_xlabel('Time [yr]')

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
