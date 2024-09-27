from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from matplotlib.ticker import LogLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from proteus.atmos_clim.janus import read_ncdf_profile

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)


def plot_atmosphere_cbar(output_dir, plot_format="pdf", tmin=1e2):

    print("Plot atmosphere temperatures colourbar")

    # Gather data files
    output_files = glob.glob(output_dir+"/data/*_atm.nc")
    output_times = [ int(str(f).split('/')[-1].split('_')[0]) for f in output_files]
    sort_mask = np.argsort(output_times)
    sorted_files = np.array(output_files)[sort_mask]
    sorted_times = np.array(output_times)[sort_mask]

    if np.amax(sorted_times) < tmin:
        tmin = 1

    if np.amax(sorted_times) < 2:
        log.debug("Insufficient data to make plot_atmosphere_cbar")
        return

    # Parse NetCDF files
    stride = 1
    sorted_p = []
    sorted_t = []
    sorted_z = []
    sorted_y = []
    for i in range(0,len(sorted_files),stride):
        time = sorted_times[i]
        if time < tmin:
            continue
        sorted_y.append(sorted_times[i])

        prof = read_ncdf_profile(sorted_files[i])
        sorted_p.append(prof["p"] / 1.0e5)  # Convert Pa -> bar
        sorted_t.append(prof["t"])
        sorted_z.append(prof["z"] / 1.0e3)  # Convert m -> km
    nfiles = len(sorted_p)

    # Initialise plot
    scale = 1.1
    fig,ax = plt.subplots(1,1,figsize=(5*scale,4*scale))
    ax.set_ylabel("Pressure [bar]")
    ax.set_xlabel("Temperature [K]")
    ax.invert_yaxis()
    ax.set_yscale("log")

    # Colour mapping
    norm = mpl.colors.LogNorm(vmin=sorted_y[0], vmax=sorted_y[-1])
    sm = plt.cm.ScalarMappable(cmap=cm.batlowK_r, norm=norm)
    sm.set_array([])

    # Plot data
    for i in range(nfiles):
        a = 0.7
        c = sm.to_rgba(sorted_y[i])
        ax.plot(sorted_t[i], sorted_p[i], color=c, alpha=a, zorder=3)

    # Grid
    ax.grid(alpha=0.2, zorder=2)
    ax.set_xlim(0,np.amax(sorted_t)+100)
    ax.xaxis.set_minor_locator(MultipleLocator(base=250))

    ax.set_ylim(np.amax(sorted_p)*3, np.amin(sorted_p)/3)
    ax.yaxis.set_major_locator(LogLocator())

    # Plot colourbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label("Time [yr]")

    # Save plot
    fname = os.path.join(output_dir,"plot_atmosphere_cbar.%s"%plot_format)
    print(fname)
    fig.savefig(fname, bbox_inches='tight', dpi=200)


def plot_atmosphere_cbar_entry(handler: Proteus):
    # Plot fixed set from above
    plot_atmosphere_cbar(
        output_dir=handler.directories["output"],
        plot_format=handler.config["plot_format"])


if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_atmosphere_cbar_entry(handler)
