from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from matplotlib.ticker import LogLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from proteus.atmos_clim.common import read_atmosphere_data
from proteus.utils.plot import sample_output

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)


def plot_atmosphere_cbar(output_dir, times, profiles, plot_format="pdf"):

    if len(times) < 2:
        log.warning("Insufficient data to make plot_atmosphere_cbar")
        return

    log.info("Plot atmosphere temperatures colourbar")

    norm = mpl.colors.LogNorm(vmin=max(times[0],1), vmax=times[-1])
    sm = plt.cm.ScalarMappable(cmap=cm.batlowK_r, norm=norm)
    sm.set_array([])

    # Initialise plot
    scale = 1.1
    alpha = 0.6
    fig,ax = plt.subplots(1,1,figsize=(5*scale,4*scale))
    ax.set_ylabel("Pressure [bar]")
    ax.set_xlabel("Temperature [K]")
    ax.invert_yaxis()
    ax.set_yscale("log")

    tmp_max = 1000.0
    prs_max = 1.0
    for i, t in enumerate( times ):
        prof = profiles[i]
        color = sm.to_rgba(t)
        tmp = prof["t"]
        prs = prof["p"]/1e5

        tmp_max = max(tmp_max, np.amax(tmp))
        prs_max = max(prs_max, np.amax(prs))

        ax.plot(tmp, prs, color=color, alpha=alpha, zorder=3)

    # Grid
    ax.grid(alpha=0.2, zorder=2)
    ax.set_xlim(0,tmp_max+100)
    ax.xaxis.set_minor_locator(MultipleLocator(base=250))

    ax.set_ylim(bottom=prs_max, top=np.amin(prs))
    ax.yaxis.set_major_locator(LogLocator())

    # Plot colourbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label("Time [yr]")

    # Save plot
    fname = os.path.join(output_dir,"plots","plot_atmosphere_cbar.%s"%plot_format)
    fig.savefig(fname, bbox_inches='tight', dpi=200)


def plot_atmosphere_cbar_entry(handler: Proteus):
    plot_times, _ = sample_output(handler, tmin=1e4, extension="_atm.nc", nsamp=40)
    print("Snapshots:", plot_times)
    profiles = read_atmosphere_data(handler.directories["output"], plot_times)

    plot_atmosphere_cbar(
        handler.directories["output"], plot_times, profiles,
        plot_format=handler.config.params.out.plot_fmt)


if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_atmosphere_cbar_entry(handler)
