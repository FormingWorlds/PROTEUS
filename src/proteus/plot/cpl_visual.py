from __future__ import annotations

import pandas as pd
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


def plot_visual(hf_all:pd.DataFrame, output_dir:str,
                    idx=-1, plot_format="pdf"):

    log.info("Plot visual")

    # Orbital separation
    sep = float(hf_all["separation"].iloc[idx])

    # Set viewing distance
    # obs = R_earth * 12
    obs = hf_all["R_obs"].iloc[idx] * 10

    # Get last output NetCDF file
    time  = hf_all["Time"].iloc[idx]
    fpath = os.path.join(output_dir,"data","%.0f_atm.nc"%time)
    ds    = read_ncdf_profile(fpath, extra_keys=[
                                    "ba_U_LW", "ba_U_SW", "ba_D_SW",
                                    "bandmin", "bandmax",
                                    "p", "tmp", "r"])


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
    fl_nrm = max(np.amax(y_arr), np.amax(s_arr))
    st = s_arr / fl_nrm
    fl = y_arr / fl_nrm

    # radii
    r_arr = ds["r"] / obs
    r_min, r_max = np.amin(r_arr), np.amax(r_arr)
    r_lim = r_max * 1.15
    n_lev = len(r_arr)

    # plot surface
    col = cs_srgb.spec_to_rgb(interp_spec(wl, fl[-1,:]))
    srf = patches.Circle((0,0), radius=r_min, fc=col, zorder=2)
    ax.add_patch(srf)

    # plot shells
    for i in range(n_lev):
        fl_lev = fl[i,:]-fl[i+1,:]
        spec = interp_spec(wl, fl_lev)
        col = cs_srgb.spec_to_rgb(spec)

        alp = 1 / n_lev
        cir = patches.Circle((0,0), radius=r_arr[i], fc=col,
                                alpha=alp, zorder=5,)
        ax.add_patch(cir)

    # annotate planet
    ax.text(0,r_min, "Planet", color='white', fontsize=12, ha='center', va='bottom', zorder=6)

    # annotate time and distance
    ann = r"Viewing from %.1f R$_\oplus$"%(obs/R_earth) + " at %6.1f Myr"%(time/1e6)
    ax.text(0.5, 0.00, ann, color='white', fontsize=12,
                transform=ax.transAxes, ha='center', va='bottom')

    # plot star
    col = cs_srgb.spec_to_rgb(interp_spec(wl, st))
    r_star = hf_all["R_star"].iloc[idx] / (sep + obs)
    cir = patches.Circle((r_max,r_max), radius=r_star, fc=col, zorder=2,)
    ax.add_patch(cir)
    ax.text(r_lim-r_star/2, r_lim-r_star/2, "Star", color='white', fontsize=12,
                ha='right', va='top', zorder=3)

    # decorate
    ax.set_facecolor('k')
    ax.set_xlim(-r_lim, r_lim)
    ax.set_ylim(-r_lim, r_lim)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plots", "plot_visual.%s"%plot_format)
    fig.savefig(fpath, dpi=250, bbox_inches='tight')

def plot_visual_entry(handler: Proteus):

    # read helpfile
    hf_all = pd.read_csv(os.path.join(handler.directories['output'],
                                      "runtime_helpfile.csv"), sep=r"\s+")

    plot_visual(
        hf_all,
        handler.directories["output"],
        plot_format=handler.config.params.out.plot_fmt,
   )


if __name__ == '__main__':
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_visual_entry(handler)
