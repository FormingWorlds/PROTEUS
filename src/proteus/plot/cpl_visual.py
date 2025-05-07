from __future__ import annotations

import pandas as pd
import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from proteus.atmos_clim.common import read_ncdf_profile
from proteus.utils.constants import R_earth
from proteus.utils.visual import interp_spec
from proteus.utils.visual import cs_hdtv as colsys

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)


def plot_visual(hf_all:pd.DataFrame, output_dir:str,
                    idx=-1, osamp=2, plot_format="pdf"):

    log.info("Plot visual")

    osamp = max(osamp,2)

    # Orbital separation
    sep = float(hf_all["separation"].iloc[idx])

    # Set viewing distance
    R_int = float(hf_all["R_int"].iloc[idx])
    obs = R_int * 12

    # Get output NetCDF file
    time  = hf_all["Time"].iloc[idx]
    fpath = os.path.join(output_dir,"data","%.0f_atm.nc"%time)
    ds    = read_ncdf_profile(fpath, extra_keys=[
                                    "ba_U_LW", "ba_U_SW", "ba_D_SW",
                                    "bandmin", "bandmax",
                                    "pl", "tmpl", "rl"])


    scale = 1.7
    fig,ax = plt.subplots(1,1, figsize=(4*scale,4*scale))

    # read fluxes
    sw_arr = np.array(ds["ba_U_SW"][:,:])
    lw_arr = np.array(ds["ba_U_LW"][:,:])
    st_arr = np.array(ds["ba_D_SW"][0,:])

    # reversed?
    reversed = bool(ds["bandmin"][1] < ds["bandmin"][0])
    if reversed:
        bandmin = np.array(ds["bandmin"][::-1])
        bandmax = np.array(ds["bandmax"][::-1])
        sw_arr = sw_arr[:,::-1]
        lw_arr = lw_arr[:,::-1]
        st_arr = st_arr[::-1]
    else:
        bandmin = np.array(ds['bandmin'][:])
        bandmax = np.array(ds['bandmax'][:])

    # get spectrum
    wl = 0.5*(bandmin+bandmax) * 1e9
    st = st_arr
    sw = sw_arr
    lw = lw_arr

    # radii
    r_arr = ds["rl"] / obs
    r_min, r_max = np.amin(r_arr), np.amax(r_arr)
    r_lim = 0.2
    n_lev = len(r_arr)

    # pressures
    p_arr = ds["pl"] / 1e5 # bar
    p_min, p_max = np.amin(p_arr), np.amax(p_arr)

    # plot surface of planet
    col = colsys.spec_to_rgb(interp_spec(wl, lw[-1,:]+sw[-1,:]))
    srf = patches.Circle((0,0), radius=r_min, fc=col, zorder=20)
    ax.add_patch(srf)

    # level opacities
    gamma = 0.05
    a_arr = []
    for i in range(n_lev-1):
        alp = abs(p_arr[i] - p_arr[i+1]) / p_max
        a_arr.append(alp**gamma)
    a_arr /= sum(a_arr)*osamp
    a_arr *= 0.9

    # plot levels
    for i in range(n_lev-2,-1,-1):

        sw_lev = sw[i,:] - sw[i+1,:]
        lw_lev = lw[i,:] - lw[i+1,:]

        rad_c = r_arr[i]
        rad_l = r_arr[i+1]

        spec = interp_spec(wl, sw_lev + lw_lev)
        col = colsys.spec_to_rgb(spec)

        for rad in np.linspace(rad_c, rad_l, osamp):
            cir = patches.Circle((0,0), radius=rad, fc=col,
                                alpha=a_arr[i], zorder=3+i)
            ax.add_patch(cir)

    # annotate planet
    ax.text(0,0.5*R_int/obs, "Planet", color='white', fontsize=12, ha='center', va='center', zorder=999)

    # annotate time and distance
    ann = r"Viewing from %.1f R$_\oplus$"%(obs/R_earth) + " at %6.1f Myr"%(time/1e6)
    ax.text(0.01, 0.99, ann, color='white', fontsize=12, zorder=999,
                transform=ax.transAxes, ha='left', va='top')

    # plot star
    col = colsys.spec_to_rgb(interp_spec(wl, st))
    r_star = hf_all["R_star"].iloc[idx] / (sep + obs)
    x_star = r_lim * 0.75
    cir = patches.Circle((x_star,x_star), radius=r_star, fc=col, zorder=2,)
    ax.add_patch(cir)
    ax.text(x_star, x_star, "Star", color='white', fontsize=12,
                ha='center', va='center', zorder=999)

    # scale bar
    for r in np.arange(0,20,1):
        x = r * R_earth / obs / 2**0.5
        if abs(x) > r_lim:
            break
        ax.scatter(x, -x, s=20, color='w', zorder=999)
        if r > 0:
            ax.text(x, -x, r"  %.0f R$_\oplus$"%r, ha='left', va='center', fontsize=8, color='w', zorder=999)
    ax.plot([0,x],[0,-x],lw=1,color='w',zorder=99)

    # decorate
    ax.set_facecolor('k')
    ax.set_xlim(-r_lim, r_lim)
    ax.set_ylim(-r_lim, r_lim)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # inset spectrum
    axr = ax.inset_axes((0.08, 0.08, 0.32,0.25))
    axr.set_facecolor('k')
    imax = np.argmin(np.abs(wl-2500))
    axr.plot(wl[:imax], lw[0,:imax]+sw[0,:imax], color='w', lw=1.3)
    axr.spines[['bottom','left']].set_color('w')
    axr.spines[['right', 'top']].set_visible(False)
    axr.tick_params(axis='both', colors='w', labelsize=8)
    axr.set_xlabel("Wavelength [nm]", color='w', fontsize=8)
    axr.set_ylabel("Emission", color='w', fontsize=8)
    axr.set_ylim(bottom=0)

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
        idx=-1
   )


if __name__ == '__main__':
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_visual_entry(handler)
