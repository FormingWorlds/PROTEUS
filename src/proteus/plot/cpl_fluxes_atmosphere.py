from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

from proteus.utils.helper import natural_sort

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)


def plot_fluxes_atmosphere(output_dir:str, plot_format="pdf"):

    log.info("Plot atmosphere fluxes")

    files = glob.glob(os.path.join(output_dir, "data", "*_atm.nc"))
    nc_fpath = natural_sort(files)[-1]

    # Read netCDF
    ds = nc.Dataset(nc_fpath)
    atm_pl      = np.array(ds.variables['pl'     ][:])
    atm_fl_D_LW = np.array(ds.variables['fl_D_LW'][:])
    atm_fl_U_LW = np.array(ds.variables['fl_U_LW'][:])
    atm_fl_D_SW = np.array(ds.variables['fl_D_SW'][:])
    atm_fl_U_SW = np.array(ds.variables['fl_U_SW'][:])
    atm_fl_D    = np.array(ds.variables['fl_D'   ][:])
    atm_fl_U    = np.array(ds.variables['fl_U'   ][:])
    atm_fl_N    = np.array(ds.variables['fl_N'   ][:])
    ds.close()

    scale = 1.0
    fig,ax = plt.subplots(1,1, figsize=(7*scale,6*scale ))

    ax.axvline(0,color='black',lw=0.8)

    pl = atm_pl * 1.e-5

    ax.plot(atm_fl_U      ,pl,color='red',label='UP',lw=1)
    ax.plot(atm_fl_U_SW   ,pl,color='red',label='UP SW',linestyle='dotted',lw=2)
    ax.plot(atm_fl_U_LW   ,pl,color='red',label='UP LW',linestyle='dashed',lw=1)

    ax.plot(-1.0*atm_fl_D      ,pl,color='green',label='DN',lw=2)
    ax.plot(-1.0*atm_fl_D_SW   ,pl,color='green',label='DN SW',linestyle='dotted',lw=3)
    ax.plot(-1.0*atm_fl_D_LW   ,pl,color='green',label='DN LW',linestyle='dashed',lw=2)

    ax.plot(atm_fl_N ,pl,color='black',label='NET')

    ax.set_xscale("symlog")
    ax.set_xlabel("Upward-directed flux [W m-2]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_yscale("log")
    ax.set_ylim([pl[-1],pl[0]])

    ax.legend(loc='upper left')

    plt.close()
    plt.ioff()
    fig.savefig(output_dir+"/plot_fluxes_atmosphere.%s"%plot_format,
                bbox_inches='tight', dpi=200)


def plot_fluxes_atmosphere_entry(handler: Proteus):
    plot_fluxes_atmosphere(
        output_dir=handler.directories["output"],
        plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == '__main__':
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_fluxes_atmosphere_entry(handler)
