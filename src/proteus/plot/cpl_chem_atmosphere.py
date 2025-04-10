from __future__ import annotations

import logging
import os
import glob
from typing import TYPE_CHECKING

import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

from proteus.atmos_clim.common import read_ncdf_profile
from proteus.atmos_chem.common import read_result
from proteus.utils.plot import latexify, get_colour
from proteus.utils.constants import gas_list
from proteus.utils.helper import natural_sort

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)

def plot_chem_atmosphere( output_dir:str, chem_module:str, plot_format="pdf"):

    log.info("Plot atmosphere chemical composition")

    # Get last output NetCDF file
    files = glob.glob(os.path.join(output_dir, "data", "*_atm.nc"))
    nc_fpath = natural_sort(files)[-1]
    atm_profile = read_ncdf_profile(nc_fpath,
                                    extra_keys=["pl", "tmpl", "x_gas"])

    parr = atm_profile["pl"] * 1e-5  # convert to bar
    tarr = atm_profile["tmpl"]

    # Read offline chemistry output if available
    offchem = chem_module and (chem_module != "none")
    if offchem:
        atm_offchem = read_result(output_dir, chem_module)
        print(atm_offchem.columns)
        atm_offchem.drop(columns=["tmp","p","z","Kzz"], inplace=True)

    # init plot
    scale = 1.2
    fig,ax = plt.subplots(1,1, figsize=(8*scale,5*scale))

    # plot netCDF profiles
    ls = 'solid'
    lw = 1.0
    al = 0.9
    plot_gases = []
    for key in atm_profile.keys():
        if "_vmr" in key:
            gas = key.replace("_vmr", "")
            ax.plot(atm_profile[key], parr[1:],
                        label=latexify(gas), color=get_colour(gas), lw=lw, ls=ls, alpha=al)
            plot_gases.append(gas)

    # plot offline chemistry profiles
    if offchem:
        ls = 'dashed'
        for gas in atm_offchem.keys():
            label = ""
            if gas not in plot_gases:
                label = latexify(gas)

            ax.plot(atm_offchem[gas], parr,
                        label=label, color=get_colour(gas), lw=lw, ls=ls, alpha=al)
            plot_gases.append(gas)

    # Decorate
    ax.set_xscale("log")
    ax.set_xlim(left=1e-10, right=1.1)
    ax.set_xlabel("Volume mixing ratio")
    ax.set_ylabel("Pressure [bar]")
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_ylim(bottom=np.amax(parr), top=np.amin(parr))
    ax.yaxis.set_major_locator(LogLocator(numticks=1000))

    # Legend
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1),
              ncols=10, fontsize=9, borderpad=0.4,
              labelspacing=0.3, columnspacing=1.0, handlelength=1.5, handletextpad=0.3)
    fpath = os.path.join(output_dir, "plot_chem_atmosphere.%s"%plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')

def plot_chem_atmosphere_entry(handler: Proteus):
    plot_chem_atmosphere(
        output_dir=handler.directories["output"],
        chem_module=handler.config.atmos_chem.module,
        plot_format=handler.config.params.out.plot_fmt,
   )

if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_chem_atmosphere_entry(handler)
