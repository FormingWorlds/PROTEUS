from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from proteus.atmos_clim.common import read_ncdf_profile
from proteus.utils.helper import natural_sort
from proteus.utils.constants import R_earth
from proteus.utils.visual import cs_srgb, interp_spec

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)


def plot_visual(output_dir:str, plot_format="pdf"):


    log.info("Plot visual")

    # Get last output NetCDF file
    files = glob.glob(os.path.join(output_dir, "data", "*_atm.nc"))
    if len(files) == 0:
        log.warning("No atmosphere NetCDF files found in output folder")
        return

    fpath = natural_sort(files)[-1]
    time  = float(fpath.split("/")[-1].split("_")[0])
    ds    = read_ncdf_profile(fpath, extra_keys=[
                                    "ba_U_LW", "ba_U_SW", "ba_D_SW",
                                    "bandmin", "bandmax",
                                    "pl", "tmpl", "rl"])


    scale = 1.7
    fig,ax = plt.subplots(1,1, figsize=(4*scale,4*scale))

    # read fluxes
    y_arr = np.array(ds["ba_U_LW"][:,:]) + np.array(ds["ba_U_SW"][:,:])
    s_arr = np.array(ds["ba_D_SW"][0,:])

    # reversed?
    reversed = bool(ds["bandmin"][1] < ds["bandmin"][0])
    if reversed:
        bandmin = np.array(ds["bandmin"][::-1])
        bandmax = np.array(ds["bandmax"][::-1])
        y_arr = y_arr[:,::-1]
        s_arr = s_arr[::-1]
    else:
        bandmin = np.array(ds['bandmin'][:])
        bandmax = np.array(ds['bandmax'][:])

    # get spectrum
    wl = 0.5*(bandmin+bandmax)
    fl_nrm = np.amax(y_arr)
    st = s_arr / fl_nrm
    fl = y_arr / fl_nrm

    # radii
    r_arr = ds["rl"] / R_earth
    r_min, r_max = np.amin(r_arr), np.amax(r_arr)
    r_lim = r_max * 1.15
    n_lev = len(r_arr)

    # plot surface
    col = cs_srgb.spec_to_rgb(interp_spec(wl, fl[-1,:]))
    srf = patches.Circle((0,0), radius=r_min, fc=col, zorder=2)
    ax.add_patch(srf)

    # plot shells
    for i in range(n_lev):
        fl_lev = fl[i,:]-fl[-1,:]
        spec = interp_spec(wl, fl_lev)
        print(spec)
        col = cs_srgb.spec_to_rgb(spec)
        rad = r_arr[i]
        alp = 1 / n_lev
        cir = patches.Circle((0,0), radius=rad, fc=col,
                                alpha=alp, zorder=5,)
        ax.add_patch(cir)

    # annotate planet
    ax.text(0,0, "Planet", color='black', fontsize=12, ha='center', va='center')

    # annotate time
    ax.text(-r_max,r_max, "t = %.1f Myr"%(time/1e6), color='white', fontsize=12)

    # plot star
    col = cs_srgb.spec_to_rgb(interp_spec(wl, st))
    r_star = 1.1*(r_lim**2-r_max**2)**0.5
    cir = patches.Circle((r_max,r_max), radius=r_star, fc=col, zorder=2,)
    ax.add_patch(cir)
    ax.text(r_max, r_max, "Star", color='black', fontsize=12,
                ha='center', va='center', zorder=3)

    ax.set_facecolor('k')

    ax.set_xlim(-r_lim, r_lim)
    ax.set_ylim(-r_lim, r_lim)

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plots", "plot_visual.%s"%plot_format)
    fig.savefig(fpath, dpi=250, bbox_inches='tight')

def plot_visual_entry(handler: Proteus):
    plot_visual(
        handler.directories["output"],
        plot_format=handler.config.params.out.plot_fmt,
   )


if __name__ == '__main__':
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_visual_entry(handler)
