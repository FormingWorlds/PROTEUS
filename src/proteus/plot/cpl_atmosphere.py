#!/usr/bin/env python3
from __future__ import annotations

import glob
import logging
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from cmcrameri import cm
from matplotlib.ticker import LogLocator

from proteus.utils.plot import latex_float

log = logging.getLogger("PROTEUS")

#====================================================================
def plot_atmosphere( output_dir:str, times:list, plot_format="pdf"):

    log.info("Plot atmosphere temperatures")

    norm = mpl.colors.LogNorm(vmin=max(times[0],1), vmax=times[-1])
    sm = plt.cm.ScalarMappable(cmap=cm.batlowK_r, norm=norm)
    sm.set_array([])

    scale = 1.2
    fig,(ax0,ax1) = plt.subplots(2,1, sharex=True, figsize=(5*scale,7*scale))

    for i, t in enumerate( times ):

        atm_file = os.path.join(output_dir, "data", "%d_atm.nc"%t)

        ds = nc.Dataset(atm_file)
        tmp =   np.array(ds.variables["tmpl"][:])
        p   =   np.array(ds.variables["pl"][:])
        z   =   np.array(ds.variables["zl"][:]) * 1e-3  # convert to km
        ds.close()

        label = latex_float(t)+" yr"
        color = sm.to_rgba(t)

        ax0.plot( tmp, z, '-', color=color, label=label, lw=1.5)

        # Atmosphere T-P
        ax1.semilogy(tmp, p/1e5, '-', color=color, label=label, lw=1.5)

    #####  T-Z
    # fig_o.set_myaxes( ax0, xlabel='$T$ (K)', ylabel='$z_\mathrm{atm}$\n(km)', xmin=xmin, xmax=xmax, ymin=0, ymax=ymax_atm_z, xticks=xticks )
    ax0.set_ylabel(r"Height [km]")

    #####  T-P
    # fig_o.set_myaxes( ax1, xlabel='$T$ (K)', ylabel='$P_\mathrm{atm}$\n(bar)', xmin=xmin, xmax=xmax, xticks=xticks )
    ax1.set_xlabel("Temperature [K]")
    ax1.set_ylabel("Pressure [bar]")
    ax1.invert_yaxis()

    # Legend
    ax1.legend( fontsize=8, fancybox=True, framealpha=0.5 )
    ax1.yaxis.set_major_locator(LogLocator(numticks=1000))

    fig.subplots_adjust(hspace=0.02)

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plot_atmosphere.%s"%plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')

#====================================================================
def main():

    if len(sys.argv) == 2:
        cfg = sys.argv[1]
    else:
        cfg = 'init_coupler.cfg'

    # Read in COUPLER input file
    log.info("Read cfg file")
    from utils.coupler import ReadInitFile, SetDirectories
    OPTIONS = ReadInitFile( cfg )

    # Set directories dictionary
    dirs = SetDirectories(OPTIONS)

    files = glob.glob(os.path.join(dirs["output"], "data", "*_atm.nc"))
    times = [int(f.split("/")[-1].split("_")[0]) for f in files]

    plot_times = []
    tmin = max(1,np.amin(times))
    tmax = max(tmin+1, np.amax(times))

    for s in np.logspace(np.log10(tmin),np.log10(tmax),8): # Sample on log-scale

        remaining = list(set(times) - set(plot_times))
        if len(remaining) == 0:
            break

        v,_ = find_nearest(remaining,s) # Find next new sample
        plot_times.append(int(v))

    plot_times = sorted(set(plot_times)) # Remove any duplicates + resort
    print("Snapshots:", plot_times)

    # Plot fixed set from above
    plot_atmosphere( output_dir=dirs["output"], times=plot_times, plot_format=OPTIONS["plot_format"] )

#====================================================================

if __name__ == "__main__":

    main()
