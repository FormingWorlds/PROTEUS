from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.utils.constants import R_earth, element_list, secs_per_year
from proteus.utils.plot import get_colour

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger('fwl.' + __name__)


def plot_escape(hf_all: pd.DataFrame, output_dir: str, plot_format='pdf'):
    time = hf_all['Time']
    if (len(time) < 3) or (np.amax(time) < 2):
        log.debug('Insufficient data to make plot_escape')
        return

    log.info('Plot escape')

    hf_crop = hf_all.iloc[2:]
    time = np.array(hf_crop['Time'])

    # mass unit
    M_uval = 1e18  # kg
    M_ulbl = r'$10^{18}$kg'

    # create plot
    lw = 1.2
    scale = 1.2
    fig, axs = plt.subplots(3, 1, figsize=(5 * scale, 6 * scale), sharex=True)

    # get axes
    ax0 = axs[0]  # inventories
    ax1 = axs[1]  # rates
    ax2 = axs[2]  # Psurf and Rxuv
    axr = ax2.twinx()

    ax2.set_xlim(left=0, right=np.amax(time))
    ax2.set_xlabel('Time [yr]')

    # By element
    total = np.zeros(len(time))
    log.info(element_list)
    for e in element_list:

        log.info(e)

        _lw = lw
        if e == 'H':
            _lw = lw * 1.8
        col = get_colour(e)

        # Plot planetary inventory of this element
        y = np.array(hf_crop[e + '_kg_total']) / M_uval
        total += y
        ax0.plot(time, y, lw=_lw, ls='dotted', color=col)

        # Plot atmospheric inventory of this element
        y = np.array(hf_crop[f'{e}_kg_atm']) / M_uval
        ax0.plot(time, y, lw=_lw, ls='solid', color=col, label=e)

        # Plot escape rate of this element
        y = np.array(hf_crop[f'esc_rate_{e}']) * secs_per_year * 1e6 / M_uval
        ax1.plot(time, y, lw=_lw, ls='solid', color=col)

    # Planetary element sum inventory
    ax0.plot(time, total, lw=lw, ls='dotted', label='Total', c='k')

    # Atmosphere mass
    M_atm = np.array(hf_crop['M_atm']) / M_uval
    ax0.plot(time, M_atm, lw=lw, ls='solid', label='Atm.', c='k')

    # Decorate top plot
    ax0.set_ylabel(rf'Mass [{M_ulbl}]')
    ax0.set_yscale('symlog', linthresh=1e-1)
    ax0.legend(loc='upper left', bbox_to_anchor=(1.0, 0.8), labelspacing=0.4)
    ax0.set_ylim(0, np.amax(total) * 1.5)

    # Plot bulk escape rate (mass per yr)
    y = np.array(hf_crop['esc_rate_total']) * secs_per_year * 1e6 / M_uval
    ax1.plot(time, y, lw=lw, color='k', ls='dotted')

    # Decorate middle plot
    ax1.set_ylabel(rf'Esc rate [{M_ulbl} / Myr]')
    y_max = np.amax(y)
    if y_max <= 0:
        y_max = 1e-10
    ax1.set_ylim(0, y_max)

    # Plot Rxuv
    y = hf_crop['R_xuv'] / R_earth
    ax2.plot(time, y, color='k', lw=lw)
    ax2.set_ylabel(r'XUV radius [R$_\oplus$]')
    ax2.set_ylim(np.amin(hf_crop['R_int'] / R_earth), np.amax(y) + 0.1)

    # Plot surface pressure
    color = 'seagreen'
    alpha = 0.8
    y = np.array(hf_crop['P_surf'])
    axr.plot(time, y, lw=lw, color=color, alpha=alpha)
    axr.set_ylim(0, np.amax(y))
    axr.tick_params(axis='y', color=color, labelcolor=color)
    axr.set_ylabel('Surf pressure [bar]', color=color)

    # Plot vertical line to show surface pressure maximum
    tmax = time[np.argmax(y)]
    for ax in axs:
        if ax is axs[2]:
            lbl = r'Maximum P$_\text{surf}$'
        else:
            lbl = ''
        ax.axvline(tmax, color=color, alpha=alpha, ls='dashdot', label=lbl)

    # Adjust
    fig.subplots_adjust(hspace=0.04)
    fig.align_ylabels()
    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, 'plots', 'plot_escape.%s' % plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')


def plot_escape_entry(handler: Proteus):
    # read helpfile
    hf_all = pd.read_csv(
        os.path.join(handler.directories['output'], 'runtime_helpfile.csv'), sep=r'\s+'
    )

    # make plot
    plot_escape(
        hf_all=hf_all,
        output_dir=handler.directories['output'],
        plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == '__main__':
    from proteus.plot._cpl_helpers import get_handler_from_argv

    handler = get_handler_from_argv()
    plot_escape_entry(handler)
