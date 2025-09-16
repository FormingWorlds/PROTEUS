from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

from proteus.utils.helper import natural_sort
from proteus.utils.plot import get_colour

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)


def plot_fluxes_atmosphere(output_dir:str, plot_format="pdf"):

    log.info("Plot atmosphere fluxes")

    files = glob.glob(os.path.join(output_dir, "data", "*_atm.nc"))
    if len(files) == 0:
        log.warning("No atmosphere NetCDF files found in output folder")
        return
    nc_fpath = natural_sort(files)[-1]

    # Read netCDF
    with nc.Dataset(nc_fpath) as ds:

        #  (required variables)
        atm_pl      = np.array(ds.variables['pl'     ][:])
        atm_fl_D_LW = np.array(ds.variables['fl_D_LW'][:])
        atm_fl_U_LW = np.array(ds.variables['fl_U_LW'][:])
        atm_fl_D_SW = np.array(ds.variables['fl_D_SW'][:])
        atm_fl_U_SW = np.array(ds.variables['fl_U_SW'][:])
        atm_fl_D    = np.array(ds.variables['fl_D'   ][:])
        atm_fl_U    = np.array(ds.variables['fl_U'   ][:])
        atm_fl_N    = np.array(ds.variables['fl_N'   ][:])

        #  (optional variables)
        try:
            # dry convection
            atm_fl_C = np.array(ds.variables['fl_cnvct'][:])
        except KeyError:
            atm_fl_C = np.zeros_like(atm_pl)

        try:
            # latent heat flux
            atm_fl_L = np.array(ds.variables['fl_latent'][:])
        except KeyError:
            atm_fl_L = np.zeros_like(atm_pl)

        try:
            # total flux
            atm_fl_T = np.array(ds.variables['fl_tot'][:])
        except KeyError:
            atm_fl_T = np.zeros_like(atm_pl)

        try:
            # surface sensible heat flux (scalar)
            atm_fl_S = float(ds.variables['fl_sens'][0])
        except KeyError:
            atm_fl_S = 0.0


    scale = 1.0
    fig,ax = plt.subplots(1,1, figsize=(7*scale,6*scale ))

    # centre line
    ax.axvline(0,color='black',lw=0.8)

    # pressure array
    pl = atm_pl * 1.e-5

    # legend entries
    w = 1.0
    a = 0.9
    col_r = get_colour("flux_r")
    col_n = get_colour("flux_n")
    ax.plot([], [], ls="dotted", lw=w, alpha=a, color=col_r, label="SW")
    ax.plot([], [], ls="dashed", lw=w, alpha=a, color=col_r, label="LW")
    ax.plot([], [], ls="solid" , lw=w, alpha=a, color=col_r, label="LW+SW")
    ax.plot([], [], ls="solid" , lw=w, alpha=a, color=col_n, label="UP-DN")

    # LW
    ax.plot(atm_fl_U_LW,        pl, alpha=a, color=col_r, ls='dashed',lw=w)
    ax.plot(-1.0*atm_fl_D_LW,   pl, alpha=a, color=col_r, ls='dashed',lw=w)

    # SW
    ax.plot(atm_fl_U_SW,        pl, alpha=a, color=col_r, ls='dotted',lw=w)
    ax.plot(-1.0*atm_fl_D_SW ,  pl, alpha=a, color=col_r, ls='dotted',lw=w)

    # net radiative
    ax.plot(atm_fl_U,       pl, alpha=a, color=col_r, lw=w)
    ax.plot(-1.0*atm_fl_D,  pl, alpha=a, color=col_r, lw=w)
    ax.plot(atm_fl_N,       pl, alpha=a, color=col_n, lw=w)

    # Convection
    ax.plot(atm_fl_C, pl, color=get_colour("flux_c"),lw=w*1.2, zorder=8, label="Convection")

    # Latent heat
    ax.plot(atm_fl_L, pl, color=get_colour("flux_p"),lw=w*1.2, zorder=9, label="Latent")

    # Sensible
    ax.scatter(atm_fl_S, np.amax(pl)*0.9, color=col_r, label="Sensible",
                marker='^', edgecolors='k', s=40, zorder=10)

    # Total
    ax.plot(atm_fl_T, pl, color=get_colour("flux_t"),lw=w, zorder=11, label="Total")

    # Axis limits
    max_fl = max(100.0,
                    np.amax(np.abs(atm_fl_T)),
                    np.amax(atm_fl_U),
                    np.amax(atm_fl_D)
                    ) * 1.1
    ax.set_xlim(left=-max_fl, right=max_fl)

    # Titles
    for tit in zip([1/3,2/3],["Downward","Upward"]):
        ax.text(tit[0], 0.99, tit[1], fontsize=13, zorder=20,
                    ha='center', va='top', transform=ax.transAxes)

    # Decorate
    ax.grid(zorder=-2, alpha=0.2)
    ax.set_xscale("symlog")
    ax.set_xlabel("Upward-directed flux [W m-2]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_yscale("log")
    ax.set_ylim([pl[-1],pl[0]])
    ax.legend(loc='upper left',
                bbox_to_anchor=(1,1), handletextpad=0.4, labelspacing=0.3)

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plots", "plot_fluxes_atmosphere.%s"%plot_format)
    fig.savefig(fpath, bbox_inches='tight', dpi=200)


def plot_fluxes_atmosphere_entry(handler: Proteus):
    plot_fluxes_atmosphere(
        output_dir=handler.directories["output"],
        plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == '__main__':
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_fluxes_atmosphere_entry(handler)
