from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from matplotlib.ticker import LogLocator

from proteus.atmos_clim.common import read_atmosphere_data
from proteus.utils.plot import latex_float, sample_output

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)


def plot_atmosphere( output_dir:str, times:list, profiles:list, plot_format="pdf"):

    if (len(times) < 1) or (np.amax(times) < 2):
        log.debug("Insufficient data to make plot_atmosphere")
        return

    log.info("Plot atmosphere temperatures")

    norm = mpl.colors.LogNorm(vmin=max(times[0],1), vmax=times[-1])
    sm = plt.cm.ScalarMappable(cmap=cm.batlowK_r, norm=norm)
    sm.set_array([])

    scale = 1.15
    fig,(ax0,ax1) = plt.subplots(2,1, sharex=True, figsize=(5*scale,7*scale))

    zmax = 100.0
    pmax = 1.0

    for i, t in enumerate( times ):
        prof = profiles[i]
        label = latex_float(t)+" yr"
        color = sm.to_rgba(t)

        zarr = prof["z"]/1e3
        parr =  prof["p"]/1e5

        zmax = max(zmax, np.amax(zarr))
        pmax = max(pmax, np.amax(parr))

        ax0.plot( prof["t"], zarr, color=color, label=label, lw=1.5, zorder=i+2)
        ax1.plot( prof["t"], parr, color=color, label=label, lw=1.5, zorder=i+2)

    #####  T-Z
    ax0.set_ylabel(r"Height [km]")
    ax0.set_ylim(bottom=0.0, top=zmax)

    #####  T-P
    ax1.set_xlabel("Temperature [K]")
    ax1.set_ylabel("Pressure [bar]")
    ax1.invert_yaxis()
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=pmax, top=np.amin(parr))
    ax1.yaxis.set_major_locator(LogLocator(numticks=1000))

    # Legend
    ax1.legend( fontsize=8, fancybox=True, framealpha=0.5).set_zorder(99)
    fig.subplots_adjust(hspace=0.02)
    fpath = os.path.join(output_dir, "plots", "plot_atmosphere.%s"%plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')


def plot_atmosphere_entry(handler: Proteus):
    plot_times, _ = sample_output(handler, tmin=1e4, extension="_atm.nc")
    print("Snapshots:", plot_times)

    # Plot fixed set from above
    profiles = read_atmosphere_data(handler.directories["output"], plot_times)
    plot_atmosphere(
        output_dir=handler.directories["output"],
        times=plot_times,
        profiles=profiles,
        plot_format=handler.config.params.out.plot_fmt,
   )


if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_atmosphere_entry(handler)
