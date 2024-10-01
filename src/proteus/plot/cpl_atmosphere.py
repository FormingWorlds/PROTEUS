from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from matplotlib.ticker import LogLocator

from proteus.atmos_clim.common import read_ncdfs
from proteus.utils.plot import latex_float, sample_output

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)


def plot_atmosphere( output_dir:str, times:list, profiles:list, plot_format="pdf"):

    if np.amax(times) < 2:
        log.debug("Insufficient data to make plot_atmosphere")
        return

    log.info("Plot atmosphere temperatures")

    norm = mpl.colors.LogNorm(vmin=max(times[0],1), vmax=times[-1])
    sm = plt.cm.ScalarMappable(cmap=cm.batlowK_r, norm=norm)
    sm.set_array([])

    scale = 1.2
    fig,(ax0,ax1) = plt.subplots(2,1, sharex=True, figsize=(5*scale,7*scale))

    for i, t in enumerate( times ):
        prof = profiles[i]
        label = latex_float(t)+" yr"
        color = sm.to_rgba(t)
        ax0.plot( prof["t"], prof["z"]/1e3, color=color, label=label, lw=1.5)
        ax1.plot( prof["t"], prof["p"]/1e5, color=color, label=label, lw=1.5)

    #####  T-Z
    # fig_o.set_myaxes( ax0, xlabel='$T$ (K)', ylabel='$z_\mathrm{atm}$\n(km)', xmin=xmin, xmax=xmax, ymin=0, ymax=ymax_atm_z, xticks=xticks )
    ax0.set_ylabel(r"Height [km]")

    #####  T-P
    # fig_o.set_myaxes( ax1, xlabel='$T$ (K)', ylabel='$P_\mathrm{atm}$\n(bar)', xmin=xmin, xmax=xmax, xticks=xticks )
    ax1.set_xlabel("Temperature [K]")
    ax1.set_ylabel("Pressure [bar]")
    ax1.invert_yaxis()
    ax1.set_yscale("log")

    # Legend
    ax1.legend( fontsize=8, fancybox=True, framealpha=0.5 )
    ax1.yaxis.set_major_locator(LogLocator(numticks=1000))

    fig.subplots_adjust(hspace=0.02)

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plot_atmosphere.%s"%plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')


def plot_atmosphere_entry(handler: Proteus):
    plot_times, _ = sample_output(handler)
    print("Snapshots:", plot_times)

    # Plot fixed set from above
    profiles = read_ncdfs(handler.directories["output"], plot_times)
    plot_atmosphere(
        output_dir=handler.directories["output"],
        times=plot_times,
        profiles=profiles,
        plot_format=handler.config["plot_format"],
   )


if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_atmosphere_entry(handler)
