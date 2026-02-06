from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger('fwl.' + __name__)


def plot_bolometry(
    hf_all: pd.DataFrame, output_dir: str, plot_format: str = 'pdf', t0: float = 100.0
):
    time = np.array(hf_all['Time'])
    if np.amax(time) < 2:
        log.debug('Insufficient data to make plot_bolometry')
        return

    log.info('Plot bolometry')

    # make plot
    lw = 1.2
    scale = 1.1
    fig, (axt, axl) = plt.subplots(2, 1, figsize=(7 * scale, 4 * scale), sharex=True)

    # top axis
    axt.plot(time, hf_all['albedo_pl'] * 100, lw=lw, color='tab:orange', label='albedo_pl')
    axt.plot(time, hf_all['bond_albedo'] * 100, lw=lw, color='tab:blue', label='bond_albedo')
    axt.set(ylabel='Albedo [%]', ylim=(0, 100))
    axt.legend()

    # bottom axis (left spine)
    axl.plot(time, hf_all['transit_depth'] * 1e6, lw=lw, color='k', zorder=2)
    axl.set_ylabel('Transit depth [ppm]')

    # bottom axis (right spine)
    axr = axl.twinx()
    color = 'tab:red'
    axr.plot(time, hf_all['eclipse_depth'] * 1e6, lw=lw, color=color, zorder=3)
    axr.set_ylabel('Eclipse depth [ppm]')
    axr.yaxis.label.set_color(color)
    axr.tick_params(axis='y', colors=color)

    # x-axis
    axl.set_xlabel('Time [yr]')
    axl.set_xscale('log')
    axl.set_xlim(left=t0, right=np.amax(time))
    axt.grid(alpha=0.2, zorder=-2)
    axl.grid(alpha=0.2, zorder=-2)

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, 'plots', 'plot_bolometry.%s' % plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')


def plot_bolometry_entry(handler: Proteus):
    # read helpfile
    hf_all = pd.read_csv(
        os.path.join(handler.directories['output'], 'runtime_helpfile.csv'), sep=r'\s+'
    )

    # make plot
    plot_bolometry(
        hf_all,
        handler.directories['output'],
        plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == '__main__':
    from proteus.plot._cpl_helpers import get_handler_from_argv

    handler = get_handler_from_argv()
    plot_bolometry_entry(handler)
