from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from cmcrameri import cm
from matplotlib.ticker import LogLocator

from proteus.utils.plot import latex_float, sample_output

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)

def plot_emission(output_dir:str, times:list, plot_format="pdf",
                cumulative=False, xmax=2e4):

    if len(np.unique(times)) < 3:
        log.warning("Insufficient data to make plot_emission")
        return

    log.info("Plot emission spectrum")

    norm = mpl.colors.LogNorm(vmin=max(times[0],1), vmax=times[-1])
    sm = plt.cm.ScalarMappable(cmap=cm.batlowK_r, norm=norm)
    sm.set_array([])

    scale = 1.7
    fig,ax = plt.subplots(1,1, figsize=(4*scale,3*scale))

    # atmosphere level to plot emission from
    atm_lvl = 0

    # plotting
    al = 0.9
    lw = 1.2
    ymax = 1e-20 # updated below
    ymin = 1e20  # updated below

    for i, t in enumerate( times ):

        label = latex_float(t)+" yr"
        color = sm.to_rgba(t)

        atm_file = os.path.join(output_dir, "data", "%.0f_atm.nc"%t)
        ds = nc.Dataset(atm_file)

        x_arr = []
        w_arr = []
        y_arr = np.array(ds["ba_U_LW"][atm_lvl,:]) + np.array(ds["ba_U_SW"][atm_lvl,:])

        # reversed?
        if ds["bandmin"][1] < ds["bandmin"][0]:
            bandmin = np.array(ds["bandmin"][::-1])
            bandmax = np.array(ds["bandmax"][::-1])
            y_arr = y_arr[::-1]
        else:
            bandmin = np.array(ds['bandmin'][:])
            bandmax = np.array(ds['bandmax'][:])

        y_arr = y_arr * 1000.0

        if cumulative:
            # bin edges
            x_arr = [bandmin[0]*1e9]
            for b in bandmax:
                x_arr.append(b*1e9)

            # copy array
            b_arr = np.zeros(len(y_arr))
            b_arr[:] = y_arr[:]

            # calulate cumulative emission
            y_arr = [0.0]
            for y in b_arr:
                y_arr.append(y_arr[-1]+y)

            x_arr = np.array(x_arr)
            y_arr = np.array(y_arr)

        else:
            x_arr = 1e9 * (bandmin + bandmax) / 2.0
            w_arr = 1e9 * (bandmax - bandmin)
            y_arr = y_arr / w_arr

        ds.close()

        # find maximum x value
        imax = np.argmin(np.abs(x_arr-xmax)) + 1
        imax = min(imax, len(x_arr))
        x_arr = x_arr[:imax]
        w_arr = w_arr[:imax]
        y_arr = y_arr[:imax]

        ax.step(x_arr, y_arr, label=label, color=color, alpha=al, lw=lw, where='mid')
        ymax = max(ymax, np.amax(y_arr))
        ymin = min(ymin, np.amin(y_arr))

    # Decorate
    ax.legend( fontsize=8, fancybox=True, framealpha=0.5 )
    ax.set_yscale("log")
    if cumulative:
        ax.set_ylabel(r"Cumulative emission [erg s$^{-1}$ cm$^{-2}$]")
    else:
        ax.set_ylabel(r"Flux density [erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$]")
    ax.yaxis.set_major_locator(LogLocator(numticks=100))

    ax.set_xlabel("Wavelength [nm]")
    ax.set_xscale("log")

    ax.set_xlim(left=np.amin(x_arr), right=np.amax(x_arr))
    ax.set_ylim(bottom=ymin/2, top=ymax*2)

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plots", "plot_emission.%s"%plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')

def plot_emission_entry(handler: Proteus):

    plot_times, _ = sample_output(handler, tmin=1000.0, extension="_atm.nc")
    print("Snapshots:", plot_times)


    # Plot fixed set from above
    plot_emission(
        handler.directories["output"],
        plot_times,
        plot_format=handler.config.params.out.plot_fmt,
   )


if __name__ == '__main__':
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_emission_entry(handler)
