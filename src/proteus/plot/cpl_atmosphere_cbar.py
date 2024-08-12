#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from cmcrameri import cm
from matplotlib.ticker import LogLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from proteus.utils.plot import *

#====================================================================

# Helper function
def _read_nc(nc_fpath):
    ds = nc.Dataset(nc_fpath)

    p = np.array(ds.variables["p"][:])
    pl = np.array(ds.variables["pl"][:])

    t = np.array(ds.variables["tmp"][:])
    tl = np.array(ds.variables["tmpl"][:])

    z = np.array(ds.variables["z"][:])
    zl = np.array(ds.variables["zl"][:])

    nlev = len(p)
    arr_p = [pl[0]]
    arr_t = [tl[0]]
    arr_z = [zl[0]]
    for i in range(nlev):
        arr_p.append(p[i]); arr_p.append(pl[i+1])
        arr_t.append(t[i]); arr_t.append(tl[i+1])
        arr_z.append(z[i]); arr_z.append(zl[i+1])

    ds.close()

    return np.array(arr_p), np.array(arr_t), np.array(arr_z)


# Plotting function
def plot_atmosphere_cbar(output_dir, plot_format="pdf"):

    print("Plot atmosphere temperatures colourbar")

    # Gather data files
    output_files = glob.glob(output_dir+"/data/*_atm.nc")
    output_times = [ int(str(f).split('/')[-1].split('_')[0]) for f in output_files]
    sort_mask = np.argsort(output_times)
    sorted_files = np.array(output_files)[sort_mask]
    sorted_times = np.array(output_times)[sort_mask] / 1e6

    if len(sorted_times) < 3:
        print("WARNING: Too few samples to make atmosphere_cbar plot")
        return

    # Parse NetCDF files
    stride = 1
    sorted_p = []
    sorted_t = []
    sorted_z = []
    for i in range(0,len(sorted_files),stride):
        p,t,z = _read_nc(sorted_files[i])
        sorted_p.append(p / 1.0e5)  # Convert Pa -> bar
        sorted_t.append(t )
        sorted_z.append(z / 1.0e3)  # Convert m -> km
    nfiles = len(sorted_p)

    # Initialise plot
    scale = 1.1
    fig,ax = plt.subplots(1,1,figsize=(5*scale,4*scale))
    ax.set_ylabel("Pressure [bar]")
    ax.set_xlabel("Temperature [K]")
    ax.invert_yaxis()
    ax.set_yscale("log")

    # Colour mapping
    norm = mpl.colors.LogNorm(vmin=max(1,sorted_times[0]), vmax=sorted_times[-1])
    sm = plt.cm.ScalarMappable(cmap=cm.batlowK_r, norm=norm) #
    sm.set_array([])

    # Plot data
    for i in range(nfiles):
        a = 0.7
        c = sm.to_rgba(sorted_times[i])
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
    cbar.ax.set_yticks([round(v,1) for v in np.linspace(sorted_times[0] , sorted_times[-1], 8)])
    cbar.set_label("Time [Myr]")

    # Save plot
    fname = os.path.join(output_dir,"plot_atmosphere_cbar.%s"%plot_format)
    fig.savefig(fname, bbox_inches='tight', dpi=200)

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

    # Plot fixed set from above
    plot_atmosphere_cbar( output_dir=dirs["output"], plot_format=OPTIONS["plot_format"])

#====================================================================

if __name__ == "__main__":
    main()
