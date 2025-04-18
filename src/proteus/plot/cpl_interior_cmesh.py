from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from proteus.interior.wrapper import read_interior_data
from proteus.utils.plot import sample_output

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)

def plot_interior_cmesh(output_dir: str, times: list | np.ndarray, data: list,
                        module: str, use_contour: bool=True,
                        cblevels: int=24, numticks: int=5, plot_format: str="pdf"):

    if len(data) < 3:
        log.warning("Too few samples to make interior_cmesh plot")
        return

    log.info("Plot interior colourmesh")

    # Downsample data?
    if len(data) < 1000:
        stride = int(1)
    else:
        stride = int(15)
    data = data[::stride]
    times = times[::stride]
    nfiles = len(data)

    # Initialise plot
    scale = 1.0
    fig,axs = plt.subplots(4,1, sharex=True, figsize=(5*scale,7*scale))
    ax1,ax2,ax3,ax4 = axs
    for ax in axs:
        ax.set_ylabel("Pressure [GPa]")
    ax4.set_xlabel("Time [yr]")

    # Pressure [GPa] grid
    ds = data[0]
    if module == "aragog":
        arr_yb = ds["pres_b"]
    elif module == "spider":
        arr_yb = ds.get_dict_values(['data','pressure_b']) * 1e-9
    nlev_b   = len(arr_yb)

    # Create 2D arrays
    arr_z1 = np.zeros((nfiles,nlev_b),dtype=float)
    arr_z2 = np.zeros((nfiles,nlev_b),dtype=float)
    arr_z3 = np.zeros((nfiles,nlev_b),dtype=float)
    arr_z4 = np.zeros((nfiles,nlev_b),dtype=float)

    # Populate 2D arrays
    for i in range(nfiles):

        # Get 1D data for this index
        ds = data[i]
        if module == "aragog":
            y_tmp = ds["temp_b"][:]
            y_phi = ds["phi_b"][:]
            y_vis = 10**ds["log10visc_b"][:]
            y_flx = ds["Fconv_b"][:]
        elif module == "spider":
            y_tmp = ds.get_dict_values(['data','temp_b'])
            y_phi = ds.get_dict_values(['data','phi_b'])
            y_vis = ds.get_dict_values(['data','visc_b'])
            y_flx = ds.get_dict_values(['data','Jconv_b'])

        # Store these data in the 2D array
        for j in range(nlev_b):
            arr_z1[i,j] = y_tmp[j]
            arr_z2[i,j] = y_phi[j]
            arr_z3[i,j] = y_vis[j]
            arr_z4[i,j] = y_flx[j] / 1e3

    # Ensure that all values are float
    for a in (arr_z1,arr_z2,arr_z3,arr_z4,arr_yb):
        a = np.array(a,dtype=float)

    # Y-axis ticks
    yticks = np.linspace(np.amin(arr_yb),np.amax(arr_yb),4)
    yticks = [round(v) for v in yticks]
    for ax in axs:
        ax.set_ylim([yticks[-1], yticks[0]])
        ax.set_yticks(yticks)

    # Plot panel label
    panel_labels = ['(a)','(b)','(c)','(d)']
    for i in range(4):
        axs[i].text(0.015,0.97, panel_labels[i], transform=axs[i].transAxes,
                    verticalalignment="top", horizontalalignment="left", fontsize=12,
                    bbox=dict(boxstyle='square,pad=0.1',fc='white', alpha=0.4, linewidth=0))

    # Plot temperature
    cmap = cm.lajolla
    cax = make_axes_locatable(ax1).append_axes('right', size='5%', pad=0.05)
    norm = mpl.colors.Normalize(vmin=np.amin(arr_z1), vmax=np.amax(arr_z1))
    if use_contour:
        cf = ax1.contourf(times, arr_yb, arr_z1.T, cmap=cmap, norm=norm, levels=cblevels)
    else:
        cf = ax1.pcolormesh(times, arr_yb, arr_z1.T, cmap=cmap, norm=norm, rasterized=True)
    cb = fig.colorbar(cf, cax=cax, orientation='vertical')
    cb.set_label("Temperature [K]")
    cb.set_ticks([float(round(v, -2)) for v in np.linspace(np.amin(arr_z1), np.amax(arr_z1), numticks)])


    # Plot melt fraction
    cmap = cm.grayC
    cax = make_axes_locatable(ax2).append_axes('right', size='5%', pad=0.05)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    if use_contour:
        cf = ax2.contourf(times, arr_yb, arr_z2.T, cmap=cmap, norm=norm, levels=np.linspace(0.0, 1.0, cblevels))
    else:
        cf = ax2.pcolormesh(times, arr_yb, arr_z2.T, cmap=cmap, norm=norm, rasterized=True)
    cb = fig.colorbar(cf, cax=cax, orientation='vertical')
    cb.set_label("Melt fraction")
    cb.set_ticks(list(np.linspace(0.0,1.0,numticks)))

    # Plot viscosity
    cmap = cm.imola_r
    cax = make_axes_locatable(ax3).append_axes('right', size='5%', pad=0.05)
    if (np.amax(arr_z3) > 100.0*np.amin(arr_z3)):
        norm = mpl.colors.LogNorm(vmin=np.amin(arr_z3), vmax=np.amax(arr_z3))
    else:
        norm = mpl.colors.Normalize(vmin=np.amin(arr_z3), vmax=np.amax(arr_z3))
    if use_contour:
        cf = ax3.contourf(times, arr_yb, arr_z3.T, cmap=cmap, norm=norm, levels=cblevels)
    else:
        cf = ax3.pcolormesh(times, arr_yb, arr_z3.T, cmap=cmap, norm=norm, rasterized=True)
    cb = fig.colorbar(cf, cax=cax, orientation='vertical')
    cb.set_label("Viscosity [Pa s]")

    # Plot convective heat flux
    cmap = cm.acton
    cmap.set_under("k")
    cax = make_axes_locatable(ax4).append_axes('right', size='5%', pad=0.05)
    arr_z4 = np.clip(arr_z4, 1e2, np.inf)
    norm = mpl.colors.LogNorm(vmin=1e2, vmax=np.amax(arr_z4))
    if use_contour:
        cf = ax4.contourf(times, arr_yb, arr_z4.T, cmap=cmap, norm=norm, levels=cblevels, extend='min')
    else:
        cf = ax4.pcolormesh(times, arr_yb, arr_z4.T, cmap=cmap, norm=norm, rasterized=True, extend='min')
    cb = fig.colorbar(cf, cax=cax, orientation='vertical', extend='under')
    cb.set_label("Convective flux \n [kW m$^{-2}$]")
    ax4.set_xticks(np.linspace(times[0], times[-1], 5))

    fig.subplots_adjust(top=0.98, bottom=0.07, right=0.85, left=0.13, hspace=0.11)

    # Save plot
    fname = os.path.join(output_dir,"plots", "plot_interior_cmesh.%s"%plot_format)
    fig.savefig(fname, dpi=200)


def plot_interior_cmesh_entry(handler: Proteus):

    # Which module was used?
    module = handler.config.interior.module
    if module == "spider":
        extension = ".json"
    elif module == "aragog":
        extension = "_int.nc"
    else:
        log.warning(f"Cannot make interior_cmesh plot for module '{module}'")
        return

    # Get data
    plot_times,_ = sample_output(handler, extension=extension, tmin=1e3, nsamp=99999)
    data = read_interior_data(handler.directories["output"], module, plot_times)

    # Plot fixed set from above
    plot_interior_cmesh(
        output_dir=handler.directories["output"],
        times=plot_times, data=data, module=module,
        plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_interior_cmesh_entry(handler)
