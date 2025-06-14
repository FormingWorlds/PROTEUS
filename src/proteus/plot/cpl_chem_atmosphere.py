from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

from proteus.atmos_chem.common import read_result
from proteus.atmos_clim.common import read_ncdf_profile
from proteus.utils.constants import vol_list
from proteus.utils.helper import natural_sort
from proteus.utils.plot import get_colour, latexify

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)

GASES_STANDARD = ("C2H4", "CO", "H2O", "H2SO4", "N2", "O2", "O3", "OH", "H", "SO", "CH4",
                  "CO2", "H2", "H2S", "HCN", "NH2", "NH3", "OCS", "S2", "S6", "S8", "SO2",
                  "N2O", "NO", "NO2", "HNO3", "PH3", "C2H2", "NO3", "N2O5", "HONO",
                  "HO2NO2", "H2O2", "C2H6", "CH3", "H2CO", "HO2", "C",
                  "N", "O", "S", "SO", "CS2",)

def plot_chem_atmosphere( output_dir:str, chem_module:str, plot_format="pdf",
                            plot_gases:list=None, plot_offchem:bool=True, xmin:float=1e-14):

    log.info("Plot atmosphere chemical composition")

    # Default species.
    #     Ensure that members of gas_list are first
    if not plot_gases:
        plot_gases = list(vol_list) + list(GASES_STANDARD)

    # Remove duplicates, preserving order
    plot_gases = list(dict.fromkeys(plot_gases))

    # Get last output NetCDF file
    files = glob.glob(os.path.join(output_dir, "data", "*_atm.nc"))
    if len(files) == 0:
        log.warning("No atmosphere NetCDF files found in output folder")
        return
    nc_fpath = natural_sort(files)[-1]
    atm_profile = read_ncdf_profile(nc_fpath,
                                        extra_keys=["pl", "tmpl", "x_gas"])

    parr = atm_profile["pl"] * 1e-5  # convert to bar

    # Check that the arrays are the right way up
    if parr[1] > parr[0]:
        # needs flipping
        for key in atm_profile.keys():
            atm_profile[key] = atm_profile[key][::-1]

    # Get year
    year = float(nc_fpath.split("/")[-1].split("_atm")[0])

    # Read offline chemistry output if available and requested
    if plot_offchem and (chem_module is not None) and (chem_module != "none"):
        atm_offchem = read_result(output_dir, chem_module)
        has_offchem = atm_offchem is not None
        if has_offchem:
            atm_offchem.drop(columns=["tmp","p","z","Kzz"], inplace=True)
    else:
        has_offchem = False

    # init plot
    scale = 1.2
    fig,ax = plt.subplots(1,1, figsize=(6*scale,5*scale))

    # plot species profiles
    lw = 0.9
    al = 0.8
    vmr_surf = []
    for i,gas in enumerate(plot_gases):

        col = get_colour(gas)
        lbl = latexify(gas)
        vmr = 0.0

        _lw = lw
        if gas in vol_list:
            _lw *= 1.25

        # plot from netCDF (dashed lines)
        key = gas+"_vmr"
        if key in atm_profile.keys():
            xarr = list(atm_profile[key])
            xarr = [xarr[0]] + xarr
            if np.amax(xarr) >= xmin:
                vmr = float(xarr[-1])
                ax.plot(xarr, parr, ls = 'dashed', color=col, lw=_lw, alpha=al)

        # plot from offline chemistry, if available (solid lines)
        if has_offchem and (gas in atm_offchem.keys()):
            xarr = list(atm_offchem[gas].values)
            if np.amax(xarr) >= xmin:
                vmr = float(xarr[-1])  # prefer vmr from offline chemistry
                ax.plot(xarr, parr, ls = 'solid', color=col, lw=_lw, alpha=al)

        # create legend entry and store surface vmr
        if vmr > 0.0:
            ax.plot(1e30, 1e30, ls='solid', color=col, lw=_lw, alpha=al, label=lbl)
            vmr_surf.append(vmr)

    # Decorate
    ax.set_xscale("log")
    ax.set_xlim(left=xmin, right=1.1)
    ax.set_xlabel("Volume mixing ratio, at t=%.2e yr"%year)
    ax.xaxis.set_major_locator(LogLocator(numticks=1000))

    ax.set_ylabel("Pressure [bar]")
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_ylim(bottom=np.amax(parr), top=np.amin(parr))
    ax.yaxis.set_major_locator(LogLocator(numticks=1000))

    # Legend (handles sorted by surface vmr)
    handles, labels = ax.get_legend_handles_labels()
    order = np.argsort(vmr_surf)[::-1]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
              loc='lower center', bbox_to_anchor=(0.5, 1),
              ncols=11, fontsize=9, borderpad=0.4,
              labelspacing=0.3, columnspacing=1.0, handlelength=1.5, handletextpad=0.3)

    # Save file
    fpath = os.path.join(output_dir, "plots", "plot_chem_atmosphere.%s"%plot_format)
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
